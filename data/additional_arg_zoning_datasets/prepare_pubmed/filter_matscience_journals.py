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
This module is responsible for filtering PubMed download files
w.r.t. materials science journals.
"""
import gzip
import os
import shutil
from argparse import ArgumentParser

from lxml import etree
from tqdm import tqdm

parser: ArgumentParser = ArgumentParser()

parser.add_argument("--input_path", type=str, required=True, help="Path to Pubmed .gz files.")
parser.add_argument("--output_path", type=str, required=True, help="Storage path.")
parser.add_argument("--start_file", type=str, help="File which to start with (in reversed order)")

args = parser.parse_args()

TMP_FILE_NAME: str = "file.xml"

# Retrieved from https://en.wikipedia.org/wiki/List_of_materials_science_journals
MAT_SCIENCE_JOURNALS: list[str] = [
    "Journal of Composite Materials",
    "Nano Research",
    "Journal of Colloid and Interface Science",
    "Journal of Materials Chemistry",
    "Journal of Materials Science: Materials in Electronics",
    "Advanced Functional Materials",
    "Materials and Structures",
    "ACS Nano",
    "Statistics",
    "Sensors and Materials",
    "Small",
    "Macromolecular Reaction Engineering",
    "Nature Reviews Materials",
    "Advanced Materials",
    "Scripta Materialia",
    "Journal of Biomedical Materials Research",
    "Journal of Materials Science: Materials in Medicine",
    "Nano",
    "Beilstein Journal of Nanotechnology",
    "Macromolecular Chemistry and Physics",
    "Functional Materials",
    "Lists of academic journals",
    "Journal of Electronic Materials",
    "Structural and Multidisciplinary Optimization",
    "Materials",
    "Science and Technology of Advanced Materials",
    "Advanced Energy Materials",
    "Nanoscale Horizon",
    "Macromolecular Theory and Simulations",
    "Journal of the American Ceramic Society",
    "Macromolecular Rapid Communications",
    "Biomaterials",
    "Metallurgical and Materials Transactions",
    "Physical Review B",
    "Macromolecular Bioscience",
    "Journal of Physical Chemistry B",
    "Journal of Elastomers and Plastics",
    "Acta Metallurgica",
    "Progress in Polymer Science",
    "Nature Materials",
    "Journal of the European Ceramic Society",
    "Nano Today",
    "Advanced Composite Materials",
    "Annual Review of Materials Research",
    "Carbon",
    "Bulletin of Materials Science",
    "Progress in Materials Science",
    "Journal of Biomaterials Applications",
    "Functional Materials Letters",
    "Computational Materials Science",
    "Acta Biomaterialia",
    "Biomacromolecules",
    "Biofabrication",
    "Acta Crystallographica",
    "Journal of Nuclear Materials",
    "Materials Today",
    "Nano Letters",
    "Advanced Optical Materials",
    "Chemistry of Materials",
    "Journal of Materials Science Letters",
    "Materials Research Letters",
    "Nano Energy",
    "Synthetic Metals",
    "Journal of Materials Science",
    "Materials Chemistry and Physics",
    "Nanoscale",
    "APL Materials",
    "MRS Bulletin",
    "Journal of Bioactive and Compatible Polymers",
    "Advanced Engineering Materials",
    "Metamaterials",
    "Materials science journals",
    "Materials Science and Engineering",
    "Nature Nanotechnology",
    "ACS Applied Materials & Interfaces",
    "Crystal Growth & Design",
    "Journal of Applied Crystallography",
    "Acta Materialia",
    "Macromolecular Materials and Engineering",
    "Journal of Materials Research and Technology",
    "Materials Horizons",
    "Dental Materials",
    "Modelling and Simulation in Materials Science and Engineering",
]

MAT_SCIENCE_JOURNALS = [j.lower() for j in MAT_SCIENCE_JOURNALS]

for filename in tqdm(reversed(sorted(list(os.listdir(args.input_path))))):
    if not filename.endswith(".gz"):
        continue
    if args.start_file is not None:
        if filename > args.start_file:
            continue
    with gzip.open(os.path.join(args.input_path, filename), "rb") as f_in:
        with open(TMP_FILE_NAME, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

    try:
        with open(TMP_FILE_NAME, "r", encoding="utf-8") as file:
            tree = etree.parse(file)
    except FileNotFoundError:
        continue

    root = tree.getroot()
    pubmed_articles: list = root.findall("PubmedArticle")
    for p in pubmed_articles:
        journal = p.find(".//Journal")
        if journal.find("Title").text.lower() not in MAT_SCIENCE_JOURNALS:
            p.getparent().remove(p)
            continue
        abstract = p.find(".//Abstract")
        if abstract is None:
            p.getparent().remove(p)
            continue
        abstract_sentences: list = abstract.findall("AbstractText")
        if len(abstract_sentences) == 1:
            p.getparent().remove(p)

    if len(root.getchildren()) > 0:
        tree.write(os.path.join(args.output_path, filename.split(".gz")[0]), pretty_print=True)

    os.remove(TMP_FILE_NAME)
