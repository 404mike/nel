from cgi import print_environ
import sys
import typer
import json
import os
import os.path
from collections import Counter
from pathlib import Path
import spacy
import random
import numpy as np
from spacy.tokens import DocBin, Span
import pickle

def main(json_loc: Path, nlp_dir: Path, train_corpus: Path, test_corpus: Path):
    """ Step 2: Once we have done the manual annotations with Prodigy, create corpora in spaCy format. """

    path = os.path.join("pickle", "qids.pkl")
    name_dict = pickle.load(open(path, "rb"))
    qids = name_dict.keys()

    nlp = spacy.load(nlp_dir, exclude="parser, tagger")
    docs = []
    dataset = []
    gold_ids = []
    with json_loc.open("r", encoding="utf8") as jsonfile:
        for line in jsonfile:
            example = json.loads(line)
            sentence = example["text"]
            if example["answer"] == "accept":
                QID = example["accept"][0]
                doc = nlp.make_doc(sentence)
                gold_ids.append(QID)
                offset = (example["spans"][0]["start"], example["spans"][0]["end"])
                links_dict = {QID: 1.0}
                entity_label = example["spans"][0]["label"]
                entities = [(offset[0], offset[1], entity_label)]
                # we assume only 1 annotated span per sentence, and only 1 KB ID per span
                entity = doc.char_span(
                    example["spans"][0]["start"],
                    example["spans"][0]["end"],
                    label=example["spans"][0]["label"],
                    kb_id=QID,
                )
                doc.ents = [entity]
                for i, t in enumerate(doc):
                    doc[i].is_sent_start = i == 0
                docs.append(doc)

                dataset.append((sentence, {"links": {offset: links_dict}, "entities": entities}))

    print("Statistics of manually annotated data:")
    print(Counter(gold_ids))
    print()

    # print(dataset[0])
    train_docs = DocBin()
    test_docs = DocBin()
    temp = []

    for QID in qids:  
        indices = [i for i, j in enumerate(gold_ids) if j == QID]  

        if len(indices) > 1:
            # split indicies
            train, validate, test = np.split(indices, [int(len(indices)*0.8), int(len(indices)*1)])
            # train, test = np.split(indices, [int(len(indices)*0.6), int(len(indices)*1)])
            # print("test ")
            # print(train)
            # print(test)
            # print(validate)
            # print("")
            
            # merge test and validate
            test_data_arr = np.concatenate((test, validate))

            # loop through all train arr and add to train docs
            for index in train:
                train_docs.add(docs[index])

            # loop through all test arr and add to test docs
            for index in test_data_arr:
                test_docs.add(docs[index])

        if len(indices) == 1:
            for index in indices:
                train_docs.add(docs[index])

    # random.shuffle(train_docs)
    # random.shuffle(test_docs)

    print(len(train_docs))
    print(len(test_docs))

    train_docs.to_disk(train_corpus)
    test_docs.to_disk(test_corpus)

    print("Data output")
    print(train_docs)
    sys.exit()


if __name__ == "__main__":
    typer.run(main)
