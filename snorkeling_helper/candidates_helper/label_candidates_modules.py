def char_to_word(row, row_fields):
    """
    This function is designed to convert character indicies into word indicies.
    Pubmed Central reports entities as individual character indicies, but it is easier to use word incidies
        parameters:
            row - the individual dataframe row to be converted,
            row_fields - a list that contains the following field names:
                entity one start, entity one end, entity two start, entity two end.
    """
    char_word_mapper = [
        (idx, int(tok)) for idx, tok in enumerate(row["char_offsets"].split("|"))
    ]
    entity_one_start = -1
    entity_one_end = -1
    entity_two_start = -1
    entity_two_end = -1

    for word in char_word_mapper:
        if word[1] >= row[row_fields[0]] and word[1] <= row[row_fields[1]]:
            if word[1] == row[row_fields[0]]:
                entity_one_start = word[0]
            else:
                entity_one_end = word[0]
        if word[1] >= row[row_fields[2]] and word[1] <= row[row_fields[3]]:
            if word[1] == row[row_fields[2]]:
                entity_two_start = word[0]
            else:
                entity_two_end = word[0]

    return (
        entity_one_start,
        entity_one_end if entity_one_end != -1 else entity_one_start + 1,
        entity_two_start,
        entity_two_end if entity_two_end != -1 else entity_two_start + 1,
    )
