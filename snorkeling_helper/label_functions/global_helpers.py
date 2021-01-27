# Global imports for all lable functions
from collections import OrderedDict
import pathlib
import random
import re

import pandas as pd
from snorkel.labeling import labeling_function
import spacy

"""
Global Variables
"""

random.seed(100)
nlp = spacy.load("en")
stop_word_list = nlp.Defaults.stop_words

"""
Global ENUMS
"""

# Define the label mappings for convenience
ABSTAIN = -1
NEGATIVE = 0
POSITIVE = 1

"""
Helper functions
"""


# Combines keywords into one
# whole group for regex search
def ltp(tokens):
    return "(" + "|".join(tokens) + ")"


# Takes the word array and returns list of words with
# entities replaced with tags {{A}} and {{B}}
def get_tagged_text(
    word_array, entity_one_start, entity_one_end, entity_two_start, entity_two_end
):
    if entity_one_end < entity_two_start:
        return " ".join(
            word_array[0: entity_one_start]
            + ["{{A}}"]
            + word_array[entity_one_end + 1: entity_two_start]
            + ["{{B}}"]
            + word_array[entity_two_end + 1:]
        )
    else:
        return " ".join(
            word_array[0: entity_two_start]
            + ["{{B}}"]
            + word_array[entity_two_end + 1: entity_one_start]
            + ["{{A}}"]
            + word_array[entity_one_end + 1:]
        )


# Takes the word array and returns the slice of words between
# both entities
def get_tokens_between(
    word_array, entity_one_start, entity_one_end, entity_two_start, entity_two_end
):
    # entity one comes before entity two
    if entity_one_end < entity_two_start:
        return word_array[entity_one_end + 1: entity_two_start]
    else:
        return word_array[entity_two_end + 1: entity_one_start]


# Gets the left and right tokens of an entity position
def get_token_windows(
    word_array, entity_offset_start, entity_offset_end, window_size=10
):
    return (
        word_array[max(entity_offset_start - 10, 0): entity_offset_start],
        word_array[
            entity_offset_end + 1: min(entity_offset_end + 10, len(word_array))
        ],
    )
