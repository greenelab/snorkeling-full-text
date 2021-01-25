import tqdm


def encode_lemmas(
    candidate_df,
    stopwords,
    word_mapper,
    entity_fieldnames=["disease_start", "disease_end", "gene_start", "gene_end"],
    entity_one="DISEASE_ENTITY",
    entity_two="GENE_ENTITY",
):

    embedded_sentences = []
    for idx, row in tqdm.tqdm(candidate_df.iterrows()):
        lemma_array = row.lemma.copy()

        if row.disease_end < row.gene_start:
            offset = row[entity_fieldnames[1]] - row[entity_fieldnames[0]]
            lemma_array[row[entity_fieldnames[0]]: row[entity_fieldnames[1]]] = [
                entity_one
            ]
            lemma_array[
                row[entity_fieldnames[2]]
                - offset
                + 1: row[entity_fieldnames[2]]
                - offset
                + 1
            ] = [entity_two]

        else:
            offset = row[entity_fieldnames[3]] - row[entity_fieldnames[2]]
            lemma_array[row[entity_fieldnames[2]]: row[entity_fieldnames[3]]] = [
                entity_two
            ]
            lemma_array[
                row[entity_fieldnames[0]]
                - offset
                + 1: row[entity_fieldnames[1]]
                - offset
                + 1
            ] = [entity_one]

        filtered_lemma = list(filter(lambda x: x not in stopwords, lemma_array))

        encoded_array = list(
            map(lambda x: word_mapper[x] if x in word_mapper else 1, filtered_lemma)
        )

        embedded_sentences.append(
            {
                "encoded_lemmas": "|".join(map(str, encoded_array)),
                "parsed_lemmas": "|".join(filtered_lemma),
                "candidate_id": row.candidate_id,
            }
        )

    return embedded_sentences
