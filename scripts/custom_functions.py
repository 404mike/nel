import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from functools import partial
from pathlib import Path
from typing import Iterable, Callable
import spacy
from spacy.training import Example
from spacy.tokens import DocBin


@spacy.registry.readers("MyCorpus.v1")
def create_docbin_reader(file: Path) -> Callable[["Language"], Iterable[Example]]:
    return partial(read_files, file)


def read_files(filepath: Path, nlp: "Language") -> Iterable[Example]:
    # print(f"Opening {filepath}")
    diretory = os.fsencode(filepath)
    # we run the full pipeline and not just nlp.make_doc to ensure we have entities and sentences
    # which are needed during training of the entity linker
    with nlp.select_pipes(disable="entity_linker"):

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
            yield Example(nlp(doc.text), doc)
