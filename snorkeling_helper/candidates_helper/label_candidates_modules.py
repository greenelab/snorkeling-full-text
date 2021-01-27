def char_to_word(row):
    """
    This function is designed to convert character indicies into word indicies.
    Pubmed Central reports entities as individual character indicies, but it is easier to use word incidies
        parameters:
            row - the individual dataframe row to be converted
    """
    char_word_mapper = [
        (idx, int(tok)) for idx, tok in enumerate(row["char_offsets"].split("|"))
    ]
    gene_end = -1
    gene_start = -1
    disease_end = -1
    disease_start = -1
    for word in char_word_mapper:
        if word[1] >= row["disease_start"] and word[1] <= row["disease_end"]:
            if word[1] == row["disease_start"]:
                disease_start = word[0]
            else:
                disease_end = word[0]
        if word[1] >= row["gene_start"] and word[1] <= row["gene_end"]:
            if word[1] == row["gene_start"]:
                gene_start = word[0]
            else:
                gene_end = word[0]

    return (
        disease_start,
        disease_end if disease_end != -1 else disease_start + 1,
        gene_start,
        gene_end if gene_end != -1 else gene_start + 1,
    )
