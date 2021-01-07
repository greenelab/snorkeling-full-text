from collections import Counter, OrderedDict, defaultdict
import csv
import gc
import itertools
import lzma
import re

import lxml.etree as ET
import pandas as pd
from pathlib import Path
import spacy
import tqdm


def clear_citations(open_token, close_token, section_text):
    """
    This function is designed to remove in text citations.
    Args:
       - open_token - open citation symbol
       - close_token - closed citation symbol
       - section_text - the text that contains a citation
    """
    parenthesis_index_open = section_text.find(open_token)
    parenthesis_index_close = section_text.find(close_token)

    while parenthesis_index_open != -1 and parenthesis_index_close != -1:
        parenthesis_string = section_text[
            parenthesis_index_open:parenthesis_index_close
        ]
        cleaned_string = re.sub(r"\.", "", parenthesis_string)
        section_text = section_text.replace(parenthesis_string, cleaned_string)
        parenthesis_index_open = section_text.find(
            open_token, parenthesis_index_open + 1
        )
        parenthesis_index_close = section_text.find(
            close_token, parenthesis_index_close + 1
        )

    return section_text


def clear_document(document):
    """
    This function clears the document object to save memory as NLP takes a lot RAM.
    Part of this code was obtained from:
    https://www.ibm.com/developerworks/xml/library/x-hiperfparse/#listing5

    Args:
       - document - lxml document object to be cleared
    """

    document.clear()

    while document.getprevious() is not None:
        del document.getparent()[0]
    return


def get_sql_result(sql_query, conn):
    """
    This helper function executes and returns an sql query.

    Args:
       - sql_query - the sql_query
       - conn - the sqlalchemy connection needed to execute the query
    """
    result_proxy = conn.execute(sql_query)
    return [
        {col: val for col, val in result_row.items()} for result_row in result_proxy
    ]


def insert_entities(
    file_open,
    output_file,
    already_seen=set(),
    pmcid_map=None,
    entity_id_start=1,
    skip_documents=False,
    seen_entity=defaultdict(set),
):
    """
    This function reads in a list of annotions from Pubtator Central and
    returns a single file containing all entities found.

    Args:
       - file_open - the file path to read entites from
       - output_file - the output file containing all the entities
       - already_seen - skips documents that have already been parsed
       - pmcid_map - the mapping of pubmed central ideas to regular pubmed ids
       - entity_id_start - the sql row counter for each entry (makes postgres happy when included)
       - skip_documents - skip documents if already parsed
       - seen_entity - if entity line was already seen skip it (ensure unique constraint for databases operations)
    """
    file_mode = "a" if Path(output_file).exists() else "w"
    with lzma.open(file_open, "rt") as infile, open(output_file, file_mode) as outfile:

        reader = csv.DictReader(infile, delimiter="\t")

        output_fieldnames = [
            "entity_id",
            "document_id",
            "entity_type",
            "entity_cids",
            "start",
            "end",
        ]

        writer = csv.DictWriter(outfile, fieldnames=output_fieldnames, delimiter="\t")

        if file_mode == "w":
            writer.writeheader()

        for line in tqdm.tqdm(reader):

            if pmcid_map is not None:

                query = "PMC" + line["pubmed_id"]
                if query not in pmcid_map:
                    continue

                pubmed_id = int(pmcid_map[query])

            else:
                pubmed_id = int(line["pubmed_id"])

            if pubmed_id in already_seen and skip_documents:
                continue

            if (pubmed_id, int(line["offset"]), int(line["end"])) in seen_entity[
                pubmed_id
            ]:
                continue

            writer.writerow(
                {
                    "entity_id": entity_id_start,
                    "document_id": pubmed_id,
                    "entity_type": line["type"],
                    "entity_cids": line["identifier"],
                    "start": int(line["offset"]),
                    "end": int(line["end"]),
                }
            )

            entity_id_start += 1
            already_seen.add(pubmed_id)
            seen_entity[pubmed_id].add(
                (pubmed_id, int(line["offset"]), int(line["end"]))
            )

    return already_seen, entity_id_start, seen_entity


def parse_document(path_map, fieldnames, data_queue):
    """
    This function reads documents (abstracts and full text) from Pubtator Central and
    returns multiple files that have been parsed by Spacy.

    Args:
       - path_map - xpaths designed to quickly parse xml files
       - fieldnames - the list of fieldnames for the tsv file
       - data_queue - the queue that holds documents in xml form to be parsed (multiprocessing)
    """
    while True:
        document_str = data_queue.get()

        if document_str is None:
            break

        try:
            # Create the document
            document = ET.fromstring(document_str)
            document_id = int(document.xpath(path_map["document_id_path"])[0])

            if not Path(f"output/files/{document_id}.tsv").exists():
                with open(f"output/files/{document_id}.tsv", "w") as outfile:
                    writer = csv.DictWriter(
                        outfile,
                        fieldnames=fieldnames,
                        delimiter="\t",
                        quoting=csv.QUOTE_MINIMAL,
                    )

                    # Load the parser
                    nlp = spacy.load("en")

                    # Section counter
                    section_counter = Counter()
                    for passage in document.xpath(path_map["passage_path"]):

                        section = passage.xpath(path_map["section_path"])
                        offset = passage.xpath(path_map["offset_path"])
                        section_text = passage.xpath(path_map["section_text_path"])

                        if any(
                            [
                                len(section) == 0,
                                len(offset) == 0,
                                len(section_text) == 0,
                            ]
                        ):
                            continue

                        section = str(section[0]).upper()
                        offset = int(offset[0])
                        section_text = str(section_text[0])

                        if section not in [
                            "TITLE",
                            "ABSTRACT",
                            "INTRO",
                            "METHODS",
                            "RESULTS",
                            "CONCL",
                            "SUPPL",
                        ]:
                            continue
                        else:
                            section = (
                                "INTRODUCTION"
                                if section == "INTRO"
                                else "CONCLUSION"
                                if section == "CONCL"
                                else "SUPPLEMENTAL"
                                if section == "SUPPL"
                                else section
                            )

                        # Fix those pesky_citations
                        section_text = clear_citations("(", ")", section_text)
                        section_text = clear_citations("[", "]", section_text)
                        section_text = re.sub(r"\"", "", section_text)

                        doc = nlp(section_text, disable=["ner"])
                        doc_sentences = list(doc.sents)

                        for position, sentence in enumerate(doc_sentences):
                            (word, pos_tag, lemma, dep, char_offset) = list(
                                zip(
                                    *[
                                        (
                                            repr(tok.text),
                                            repr(tok.tag_),
                                            repr(tok.lemma_.lower()),
                                            repr(tok.dep_),
                                            str(tok.idx + offset),
                                        )
                                        for tok in sentence
                                    ]
                                )
                            )

                            writer.writerow(
                                {
                                    "document_id": document_id,
                                    "section": section.lower(),
                                    "position": position
                                    + section_counter[
                                        section
                                    ],  # sections are broken by paragraphs and titles
                                    "text": sentence.text,
                                    "word": "|".join(word),
                                    "pos_tag": "|".join(pos_tag),
                                    "lemma": "|".join(lemma),
                                    "dep": "|".join(dep),
                                    "char_offset": "|".join(
                                        char_offset
                                    ),  # act like title and abstract are one string
                                }
                            )

                        # Update the section counter based on sections
                        section_counter.update({section: len(doc_sentences)})

                    del section_counter
                    del nlp

            data_queue.task_done()

            # Remove subelements from document
            # Thereby freeing memory
            clear_document(document)
            del document

        except Exception as e:
            print(f"This was an exception {e}")

            # Do nothing but tell the queue that the task is done
            data_queue.task_done()

    data_queue.task_done()


def supply_documents(doc_type, batch_iterator, data_queue):
    """
    This function supples the parse_documents function with xml documents to be parsed.

    Args:
       - doc_type - specify full text or abstracts
       - batch_iterator - a generated to read in the documents
       - data_queue - the queue that holds documents in xml form to be parsed (multiprocessing)
    """
    if doc_type == "full":
        for batch_file in tqdm.tqdm(batch_iterator):
            tree = ET.parse(str(batch_file.resolve()))
            root = tree.getroot()

            for document in root.xpath("document"):
                try:
                    data_queue.put(ET.tostring(document))
                except Exception as e:
                    print(e)
                    pass
                finally:
                    root.clear()
                    del tree

    elif doc_type == "abstract":
        for event, document in tqdm.tqdm(batch_iterator):
            try:
                data_queue.put(ET.tostring(document))
            except Exception as e:
                print(e)
                pass
            finally:
                document.clear()
    else:
        raise ValueError("Please provide either full or abstract for doc type")


# Sliding window obtained from:
# https://docs.python.org/release/2.3.5/lib/itertools-example.html
def window(seq, n=2):
    "Returns a sliding window (of width n) over data from the iterable"
    "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
    it = iter(seq)
    result = tuple(itertools.islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result
