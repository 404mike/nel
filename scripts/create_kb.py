import typer
import csv
import os
import os.path
from pathlib import Path

import spacy
from spacy.kb import KnowledgeBase

import pickle

def main(entities_loc: Path, vectors_model: str, kb_loc: Path, nlp_dir: Path):
    """ Step 1: create the Knowledge Base in spaCy and write it to file """

    # First: create a simpel model from a model with an NER component
    # To ensure we get the correct entities for this demo, add a simple entity_ruler as well.
    nlp = spacy.load(vectors_model, exclude="parser, tagger, lemmatizer")
    ruler = nlp.add_pipe("entity_ruler", after="ner")
    patterns = [{"label": "PERSON", "pattern": [{"LOWER": "emerson"}]}]
    ruler.add_patterns(patterns)
    nlp.add_pipe("sentencizer", first=True)

    name_dict, desc_dict = _load_entities(entities_loc)

    kb = KnowledgeBase(vocab=nlp.vocab, entity_vector_length=300)

    for qid, desc in desc_dict.items():
        desc_doc = nlp(desc)
        desc_enc = desc_doc.vector
        # Set arbitrary value for frequency
        kb.add_entity(entity=qid, entity_vector=desc_enc, freq=342)

    # store duplicate names
    names_merged = {}

    # loop through names and add qid to list
    for qid, name in name_dict.items():
        if name in names_merged:
            names_merged[name].append(qid)
        else:
            names_merged[name] = [qid]

    # loop through each of the names and create alias for each
    for key, val in names_merged.items():
        name = key
        # get number of qids for this name
        qid_len = len(val)
        # get probabilty for each person
        prob_val = (1/qid_len)
        
        # format probabilty
        if qid_len > 1:
            probs = [prob_val for v in val]
        else:
            probs = [1]
        
        # add person data to alias
        kb.add_alias(alias=name, entities=val, probabilities=probs)

    qids = name_dict.keys()

    # save qids to pickle file
    path = os.path.join("pickle", "qids.pkl")
    with open(path, 'wb') as f:
        pickle.dump(name_dict, f)

    probs = [0.3 for qid in qids]
    # ensure that sum([probs]) <= 1 when setting aliases
    # kb.add_alias(alias="Emerson", entities=qids, probabilities=probs)  #

    # print(f"Entities in the KB: {kb.get_entity_strings()}")
    # print(f"Aliases in the KB: {kb.get_alias_strings()}")
    # print()
    kb.to_disk(kb_loc)
    if not os.path.exists(nlp_dir):
        os.mkdir(nlp_dir)
    nlp.to_disk(nlp_dir)


def _load_entities(entities_loc: Path):
    """ Helper function to read in the pre-defined entities we want to disambiguate to. """
    names = dict()
    descriptions = dict()
    with entities_loc.open("r", encoding="utf8") as csvfile:
        csvreader = csv.reader(csvfile, delimiter=",")
        for row in csvreader:
            qid = row[0]
            name = row[1]
            desc = row[2]
            names[qid] = name
            descriptions[qid] = desc
    return names, descriptions


if __name__ == "__main__":
    typer.run(main)
