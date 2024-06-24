# -*- coding: utf-8 -*-

"""
Module to run the Model training, based on the example:
https://keras.io/examples/nlp/ner_transformers/
"""

from collections import Counter
from datasets import load_dataset
import keras
import numpy as np

def make_tag_lookup_table():
    """
    Creates a lookup table for IOB and NER labels.

    Returns:
        dict: A dictionary mapping integer indices to label strings.
    """
    iob_labels = ["B", "I"]
    ner_labels = ["PER", "ORG", "LOC", "MISC"]
    all_labels = [(label1, label2) for label2 in ner_labels
                        for label1 in iob_labels]
    all_labels = ["-".join([a, b]) for a, b in all_labels]
    all_labels = ["[PAD]", "0"] + all_labels
    return dict(zip(range(0, len(all_labels)+1), all_labels))

if __name__=="__main__":
    conll_data = load_dataset("conll2003", trust_remote_code=True)
    mapping = make_tag_lookup_table()
    all_tokens = sum(conll_data["train"]["tokens"], [])
    all_tokens_array = np.array(list(map(str.lower, all_tokens)))

    counter = Counter(all_tokens_array)
    print(len(counter))

    num_tags = len(mapping)
    VOCAB_SIZE = 20000

    vocabulary = [token for token, count
                        in counter.most_common(VOCAB_SIZE - 2)]
    lookup_layer = keras.layers.StringLookup(vocabulary=vocabulary)

# EOF
