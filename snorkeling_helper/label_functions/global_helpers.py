# Global imports for all lable functions
import random
import re

import networkx as nx
import spacy
import tqdm

"""
Global Variables
"""

random.seed(100)
nlp = spacy.load("en_core_web_sm")
stop_word_list = nlp.Defaults.stop_words

"""
Global ENUMS
"""

# Define the label mappings for convenience
ABSTAIN = -1
NEGATIVE = 0
POSITIVE = 1


# Match Dep path and return value
def match_dep_path(dep_path, dep_path_base, cat_code, response_one, response_two):
    if dep_path in dep_path_base:
        if cat_code in dep_path_base[dep_path]:
            return response_one
    return response_two


## Creates a graph using a sentences dependency tree
def generate_dep_path(c):
    entity_one_start, entity_one_end, entity_two_start, entity_two_end = get_members(c)
    sen_text = get_tagged_text(
        c.word,
        c[entity_one_start],
        c[entity_one_end],
        c[entity_two_start],
        c[entity_two_end],
    )
    if c[entity_one_end] < c[entity_two_start]:
        sen_text = re.sub("{{A}}", "start_entity", sen_text)
        sen_text = re.sub("{{B}}", "end_entity", sen_text)
    else:
        sen_text = re.sub("{{B}}", "start_entity", sen_text)
        sen_text = re.sub("{{A}}", "end_entity", sen_text)

    try:
        doc = nlp(sen_text.lower())
        dep_graph = nx.Graph()
        for token in doc:
            for child in token.children:
                dep_graph.add_edge(
                    "{0}".format(token), "{0}".format(child), dep=child.dep_
                )

        path = nx.shortest_path(dep_graph, "start_entity", "end_entity")
        pathGraph = nx.path_graph(path)

        return " ".join(
            [
                f"{ea[1]}|{dep_graph.edges[ea[0], ea[1]]['dep']}|{ea[0]}"
                for ea in pathGraph.edges()
            ]
        )

    except nx.NetworkXNoPath as e:
        print(e)
        return ""
    except nx.NodeNotFound as e:
        print(e)
        return ""


# Extract nodes from the pre-generated dependency paths
def extract_nodes(node_list):
    nodes = []
    for path in node_list:
        attrib = path.split("|")
        # Ignore the dep type as I was using different parsers
        nodes += [attrib[0], attrib[2]]
    return nodes


# Need to organize Functions
def create_dep_mapper(bicluster_dep_df, cat_codes):
    dep_path_mapper = {}
    for idx, row in tqdm.tqdm(
        bicluster_dep_df.fillna(" ").query("spacy_dep_path!=''").iterrows()
    ):
        dep_path = re.sub("-lrb-", "(", row["spacy_dep_path"].lower())
        dep_path = re.sub("-rrb-", ")", dep_path)

        if dep_path == " ":
            continue

        dep_path_mapper[dep_path] = set()
        for cat in cat_codes:
            if row[cat] > 0:
                dep_path_mapper[dep_path].add(cat)
    return dep_path_mapper


# Combines keywords into one
# whole group for regex search
def ltp(tokens):
    return "(" + "|".join(tokens) + ")"


# Gets the dataframe columns needed
# to execute certain label functions
def get_members(c):
    if "disease_start" in c.index.tolist():
        if "gene_start" in c.index.tolist():
            return ["disease_start", "disease_end", "gene_start", "gene_end"]
        else:
            return ["compound_start", "compound_end", "disease_start", "disease_end"]
    if "gene1_start" in c.index.tolist():
        return ["gene1_start", "gene1_end", "gene2_start", "gene2_end"]

    return ["compound_start", "compound_end", "gene_start", "gene_end"]


# Takes the word array and returns list of words with
# entities replaced with tags {{A}} and {{B}}
def get_tagged_text(
    word_array, entity_one_start, entity_one_end, entity_two_start, entity_two_end
):
    if entity_one_end < entity_two_start:
        return " ".join(
            word_array[0:entity_one_start]
            + ["{{A}}"]
            + word_array[entity_one_end + 1 : entity_two_start]
            + ["{{B}}"]
            + word_array[entity_two_end + 1 :]
        )
    else:
        return " ".join(
            word_array[0:entity_two_start]
            + ["{{B}}"]
            + word_array[entity_two_end + 1 : entity_one_start]
            + ["{{A}}"]
            + word_array[entity_one_end + 1 :]
        )


# Takes the word array and returns the slice of words between
# both entities
def get_tokens_between(
    word_array, entity_one_start, entity_one_end, entity_two_start, entity_two_end
):
    # entity one comes before entity two
    if entity_one_end < entity_two_start:
        return word_array[entity_one_end + 1 : entity_two_start]
    else:
        return word_array[entity_two_end + 1 : entity_one_start]


# Gets the left and right tokens of an entity position
def get_token_windows(
    word_array, entity_offset_start, entity_offset_end, window_size=10
):
    return (
        word_array[max(entity_offset_start - 10, 0) : entity_offset_start],
        word_array[
            entity_offset_end + 1 : min(entity_offset_end + 10, len(word_array))
        ],
    )


def get_text_in_windows(c, window_size=10):
    entity_columns = list(c.index)

    correct_columns = (
        {
            "entity_one_start": "disease_start",
            "entity_one_end": "disease_end",
            "entity_two_start": "gene_start",
            "entity_two_end": "gene_end",
        }
        if "gene_start" in entity_columns and "disease_start" in entity_columns
        else {
            "entity_one_start": "compound_start",
            "entity_one_end": "compound_end",
            "entity_two_start": "disease_start",
            "entity_two_end": "disease_end",
        }
        if "compound_start" in entity_columns and "disease_start" in entity_columns
        else {
            "entity_one_start": "compound_start",
            "entity_one_end": "compound_end",
            "entity_two_start": "gene_start",
            "entity_two_end": "gene_end",
        }
        if "gene_start" in entity_columns and "compound_start" in entity_columns
        else {
            "entity_one_start": "gene1_start",
            "entity_one_end": "gene1_end",
            "entity_two_start": "gene2_start",
            "entity_two_end": "gene1_end",
        }
    )

    between_phrases = " ".join(
        get_tokens_between(
            word_array=c.word,
            entity_one_start=c[correct_columns["entity_one_start"]],
            entity_one_end=c[correct_columns["entity_one_end"]],
            entity_two_start=c[correct_columns["entity_two_start"]],
            entity_two_end=c[correct_columns["entity_two_end"]],
        )
    )

    entity_one_left_window, entity_one_right_window = get_token_windows(
        word_array=c.word,
        entity_offset_start=c[correct_columns["entity_one_start"]],
        entity_offset_end=c[correct_columns["entity_one_end"]],
        window_size=window_size,
    )

    entity_two_left_window, entity_two_right_window = get_token_windows(
        word_array=c.word,
        entity_offset_start=c[correct_columns["entity_two_start"]],
        entity_offset_end=c[correct_columns["entity_two_end"]],
        window_size=window_size,
    )

    return {
        "text_between": between_phrases,
        "left_window": (entity_one_left_window, entity_two_left_window),
        "right_window": (entity_one_right_window, entity_two_right_window),
        "entity_columns": correct_columns,
    }
