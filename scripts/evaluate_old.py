import typer
from pathlib import Path

import sys
import spacy
from spacy.tokens import DocBin
from spacy.training import Example

# we need to import this to parse the custom reader from the config
from custom_functions import create_docbin_reader


def main(nlp_dir: Path, dev_set: Path):
    r = 0
    w = 0
    p = 0
    """ Step 4: Evaluate the new Entity Linking component by applying it to unseen text. """
    nlp = spacy.util.load_model_from_path(nlp_dir)
    examples = []
    with open(dev_set, "rb") as f:
        doc_bin = DocBin().from_disk(dev_set)
        docs = doc_bin.get_docs(nlp.vocab)
        for doc in docs:
            examples.append(Example(nlp(doc.text), doc))

    print()
    print("RESULTS ON THE DEV SET:")
    for example in examples:
        try:
            # print(example)
            # print(example.text)
            # print(f"Gold annotation: {example.reference.ents[0].kb_id_}")
            # print(f"Predicted annotation: {example.predicted.ents[0].kb_id_}")
            # print()
            # print(example.reference.ents)

            # print("Loop Through all predicted ents")
            # for blah in example.predicted.ents:
            #     print(blah.kb_id_)
            # # sys.exit()
            # print("no more people")
            # print()
            # print(type(example.predicted.ents))

            ref = xample.reference.ents[0].kb_id_
            for pred in example.predicted.ents:
                if pred.kb_id_ == ref:
                    r += 1
                    pass
                else:
                    pass
            # if example.reference.ents[0].kb_id_ in example.predicted.ents:
            #     r += 1
            # else:
            #     w += 1
                # print(f"{example.reference.ents[0].kb_id_} != {example.predicted.ents[0].kb_id_}")
        except IndexError:
            p += 1

    print()
    print("RUNNING THE PIPELINE ON UNSEEN TEXT:")
    text = "Kyffin Williams was a painter."
    doc = nlp(text)
    print(text)
    for ent in doc.ents:
        print(ent.text, ent.label_, ent.kb_id_)
    print()
    print(f"Right: {r}")
    print(f"Wrong: {w}")
    print(f"Passed: {p}")



if __name__ == "__main__":
    typer.run(main)
