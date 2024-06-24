# -*- coding: utf-8 -*-

"""
Script to download the conll dataset, based on example:
https://keras.io/examples/nlp/ner_transformers/
"""

import os

from datasets import load_dataset

def export_to_file(export_file_path, data):
    """
    Writes the provided data to a file at the specified path.

    This function iterates over the data, which is expected to be a
    list of dictionaries. Each dictionary should have 'ner_tags' and 'tokens'
    as keys. The 'tokens' is a list of words, and 'ner_tags' is a corresponding
    list of Named Entity Recognition (NER) tags. The function writes the length
    of tokens, the tokens themselves, and the NER tags to the file, separated
    by tabs. Each record is written on a new line.

    Args:
        export_file_path (str): The path to the file where the data will be
                written.
        data (list): A list of dictionaries. Each dictionary should have
                'ner_tags' and 'tokens' keys.

    Returns:
        None
    """
    with open(export_file_path, "w", encoding="uft-8") as f:
        for record in data:
            ner_tags = record["ner_tags"]
            tokens = record["tokens"]
            if len(tokens) > 0:
                f.write(
                    str(len(tokens))
                    + "\t"
                    + "\t".join(tokens)
                    + "\t"
                    + "\t".join(map(str, ner_tags))
                    + "\n"
                )

if __name__=="__main__":
    conll_data = load_dataset("conll2003", trust_remote_code=True)
    try:
        os.mkdir("/app/workdir/data")
    except FileExistsError:
        pass
    export_to_file("/app/workdir/data/conll_train.txt", conll_data["train"])
    export_to_file("/app/workdir/data/conll_val.txt", conll_data["validation"])
