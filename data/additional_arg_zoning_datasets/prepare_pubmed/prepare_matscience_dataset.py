# Experiment resources related to the MuLMS-AZ corpus (CODI 2023).
# Copyright (c) 2023 Robert Bosch GmbH
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""
This module is responsible for preparing the materials science
specialized version of PubMed.
"""

import os
import pickle
from argparse import ArgumentParser

import nltk
from lxml import etree

from source.arg_zoning.az_datasets.pubmed_dataset import PubmedDataset

parser: ArgumentParser = ArgumentParser()
parser.add_argument("--input_path", type=str, required=True, help="Path to XML files")
parser.add_argument("--output_path", type=str, required=True, help="Storage Path for Pickle file")

args = parser.parse_args()


def main():
    """
    Entry point.
    """

    dataset: PubmedDataset = PubmedDataset(name="materials_science_pubmed", data_split=(900, 1200))
    dataset.categories = ["BACKGROUND", "OBJECTIVE", "METHODS", "RESULTS", "CONCLUSIONS"]
    dataset.cat2idx = {
        "BACKGROUND": 0,
        "OBJECTIVE": 1,
        "METHODS": 2,
        "RESULTS": 3,
        "CONCLUSIONS": 4,
    }
    dataset.idx2cat = dict([(v, k) for k, v in dataset.cat2idx.items()])

    for file in os.listdir(args.input_path):
        if not file.endswith(".xml"):
            continue

        with open(os.path.join(args.input_path, file), "r", encoding="utf-8") as file:
            tree = etree.parse(file)

        root = tree.getroot()
        pubmed_articles: list = root.findall("PubmedArticle")
        for p in pubmed_articles:
            doc: list = []
            abstract_sentences = p.findall(".//AbstractText")
            article_ids = p.findall(".//ArticleId")
            doi: str = None
            for id in article_ids:
                if id.attrib["IdType"] == "doi":
                    doi = id.text
            for a in abstract_sentences:
                if "NlmCategory" in a.attrib:
                    if a.attrib["NlmCategory"] in dataset.categories:
                        tokens = nltk.word_tokenize(a.text)
                        pos = nltk.pos_tag(tokens)
                        for i in range(len(tokens)):
                            tok = tokens[i]
                            dataset.posVocab.add(pos[i])
                            if tok not in dataset.vocab:
                                dataset.word2idx[tok] = len(dataset.vocab.keys())
                                dataset.idx2word[len(dataset.vocab.keys())] = tok
                            dataset.vocab[tok] += 1
                        # map tokens to indices (to save space)
                        tokens: list = [dataset.word2idx[t] for t in tokens]
                        pos: list = [p[1] for p in pos]
                        doc.append((tokens, pos, a.attrib["NlmCategory"]))
            if len(doc) > 0:
                dataset.documents.append(doc)
                dataset.doc_ids.append(doi)
    with open(os.path.join(args.output_path, dataset.name + ".pickle"), mode="wb") as f:
        pickle.dump(dataset, f)


if __name__ == "__main__":
    main()
