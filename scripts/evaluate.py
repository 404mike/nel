import os
import typer
from pathlib import Path

import sys
import spacy
from spacy.tokens import DocBin
from spacy.training import Example

# we need to import this to parse the custom reader from the config
from custom_functions import create_docbin_reader
from datetime import datetime


def main(nlp_dir: Path, dev_set: Path):

    now = datetime.now()
    print("Time now ", now)

    r = 0
    w = 0
    p = 0
    """ Step 4: Evaluate the new Entity Linking component by applying it to unseen text. """
    nlp = spacy.util.load_model_from_path(nlp_dir)
    examples = []

    filepath = os.path.join('corpus','dev')
    diretory = os.fsencode(filepath)

    doc_bin = DocBin()
    # print("docbin len")
    # print(len(doc_bin))

    for files in os.listdir(diretory):
        filename = os.fsdecode(files)
        # print(f"file: {filename}")
        doc_bin_loop = DocBin().from_disk(str(filepath) + '\\' + filename)
        doc_bin.merge(doc_bin_loop)
        # print(len(doc_bin))


    docs = doc_bin.get_docs(nlp.vocab)
    for doc in docs:
        examples.append(Example(nlp(doc.text), doc))


    # with open(dev_set, "rb") as f:
        # doc_bin = DocBin().from_disk(dev_set)
        # docs = doc_bin.get_docs(nlp.vocab)
        # for doc in docs:
        #     examples.append(Example(nlp(doc.text), doc))

    print()
    print("RESULTS ON THE DEV SET:")
    lenTest = len(examples)
    print(f"Number of items: {lenTest}")
    for example in examples:
        try:
            ref = example.reference.ents[0].kb_id_
            for pred in example.predicted.ents:
                if pred.kb_id_ == ref:
                    r += 1
                    pass
        except IndexError:
            p += 1

    print()
    print("RUNNING THE PIPELINE ON UNSEEN TEXT:")
    text = "Kyffin Williams was a painter."
    doc = nlp(text)
    print(text)

    print()
    text2 = "Winston Churchill was a prime minister along with David Lloyd George"
    for ent in doc.ents:
        print(ent.text, ent.label_, ent.kb_id_)
    print()
    print(f"Right: {r}")
    print(f"Wrong: {w}")
    print(f"Passed: {p}")

    score = (r / lenTest) * 100
    print(f"Average of {score}%")


if __name__ == "__main__":
    typer.run(main)
