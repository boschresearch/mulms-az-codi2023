<!---

    Copyright (c) 2023 Robert Bosch GmbH and its subsidiaries.

-->

# MuLMS-AZ - Experiment Resources

This repository contains the companion material for the following publication:

> Timo Pierre Schrader, Teresa Bürkle, Sophie Henning, Sherry Tan, Matteo Finco, Stefan Grünewald, Maira Indrikova, Felix Hildebrand, Annemarie Friedrich. **MuLMS-AZ: An Argumentative Zoning Dataset for the Materials Science Domain.** CODI 2023.

Please cite this paper if using the dataset or the code, and direct any questions regarding the dataset
to [Annemarie Friedrich](mailto:annemarie.friedrich@uni-a.de), and any questions regarding the code to
[Timo Schrader](mailto:timo.schrader@de.bosch.com).

## Purpose of this Software

This software is a research prototype, solely developed for and published as
part of the publication cited above. It will neither be maintained nor monitored in any way.

## The Multi-Layer Materials Science Argumentative Zoning Corpus (MuLMS-AZ)

The Multi-Layer Materials Science Argumentative Zoning (MuLMS-AZ) corpus consists of 50 documents (licensed CC BY) from the materials science domain, spanning across the following 7 subareas: "Electrolysis", "Graphene", "Polymer Electrolyte Fuel Cell (PEMFC)", "Solid Oxide Fuel Cell (SOFC)", "Polymers", "Semiconductors" and "Steel".
There are annotations on sentence-leven and token-level for several NLP tasks, including Argumentative Zoning (AZ). Every sentence in the dataset is labeled with one or multiple argumetative zones. The dataset can be used to train classifiers and text mining systems on argumentative zoning in the materials science domain.

You can find the information about all papers and their authors in the [MuLMS_Corpus_Metadata.csv](data/mulms_corpus/MuLMS_Corpus_Metadata.csv) files, including links to the copy of their respective license.

## Argumentative Zoning

Argumentative zones (AZ) describe the rethorical function of sentences within scientific publications. Therefore, we model this task on sentence level. We use a BERT-based transformer to generate contextualized embeddings, take the [CLS] token to get the sentence embedding and feed it into a linear classification layer. To compensate for imbalances of classes, we implement the ML-ROS [[1]](#1) algorithm which dynamically clones instances in the training set that belong to so-called minority classes.

On top of that, we provide code for multi-tasking and data augmentation experiments. Whereas our multi-task models use different output heads on top of the language model (one per dataset), data augmentation refers to the procedure of adding additional training samples from another dataset into the same training set by mapping the label set to our own one.

We used the following publicly available AZ datasets:
- AZ-CL [[2,3]](#2)
- DRI [[4,5]](#4)
- ART [[6]](#6)
- Materials science-related subset of PubMed/MEDLINE [[7]](#7)

Due to legal reasons, we cannot provide the files directly in this repo. Please refer to the README.md in `data/additional_arg_zoning_datasets` in order to prepare the additional data.

The following argumentative zones are modeled in our dataset: **Experiment, Results, Exp_Preparation, Exp_Characterization, Background_PriorWork, Explanation, Conclusion, Motivation, Background, Metadata, Caption, Heading, Abstract**

We conduct the following experiments for which there are all necessary scripts in this repository:

* Plain Argumentative Zoning ("AZ"):
* Argumentative Zoning with oversampling ("OS"):
* Multi-tasking with ART dataset (MT)

For dataset statistics and counts, please refer to our paper.

## Data Format

We provide our dataset in the form of a HuggingFace dataset. Therefore, it can be easily loaded using

```
from datasets import Dataset, load_dataset
train_dataset: Dataset = load_dataset(
        MULMS_DATASET_READER_PATH.__str__(),
        data_dir=MULMS_PATH.__str__(),
        data_files=MULMS_FILES,
        name="MuLMS_Corpus",
        split="train",
    )
```
It is constructred from UIMA CAS XMI files that can be used in annotation tools to read in and adapt the data. It operates on a sentence-level basis.

### Data Fields
**Note: There are more fields in the dataset than relevant to the argumentative zoning task. They are part of another work of ours.**

The following fields in the dataset are relevant to argumentative zoning:
* doc_id: Identifier of the document that allows for lookup in the MuLMS_Corpus_Metadata.csv
* sentence: This is the raw sentence string.
* tokens: The pre-tokenized version of the sentence.
* beginOffset: The character offset of the first token in the sentence in the document.
* endOffset: The character offset of the last token in the sentence in the document.
* AZ_labels: List of (multiple) AZ labels that belong to the current instance.
* data_split: The split (train1/2/3/4/5, dev or test) which the current document belongs to.
* category: One of the seven presented categories which the document is part of.

### Split Setup

Our dataset is divided into several splits, please look them up in the paper for further explanation:

* train
* tune1/tune2/tune3/tune4/tune5
* dev
* test

## Setup

Please install all dependencies or the environment as listed in [environment.yml](environment.yml) and make sure to have **Python 3.9** installed (we recommend 3.9.11). You might also add the root folder of this project to the `$PYTHONPATH` environment variable. This enables all scripts to automatically find the imports.

**NOTE: This code really requires Python 3.9. It does **not** support Python 3.8 and below or 3.10 and above due to type hinting and package dependencies.**

## Code

We provide bash scripts in `scripts` for each AZ task separately. Furthermore, for subtaks (e.g., multi-tasking), there are additional scripts that contain all parameters necessary. Use these scripts to reproduce the results from our paper and adapt those if you want to do additional experiments. Moreover, you can check all available settings in each Python file via `python <script_name.py> --help`. Every flag is described individually, some of them are not used in the bash scripts but might still be of interest for further experiments.

### Reproducibility

In order to reproduce the numbers of our papers, we provide the list of all hyperparameters and seeds for each experiment reported in table 5 indivually (only the best run and only with SciBERT). Please note that different GPUs might produce slightly different numbers due to differences in floating-point arithmetic. For further configurations, please refer to our paper. If you need help or have any questions regarding the experiments, feel free to contact us! Also, if you find better configurations, you are invited to let us know!

|                           | AZ  | OS  | MT (+ ART) | MT ( + AZ-CL) |
|---------------------------|-----|-----|------------|---------------|
| Learning Rate             |3e-6 | 2e-6|2e-6        | 2e-6          |
| Batch Size                |32   | 32  |16          | 16            |
| Oversample Percentage     |x    | 0.2 |0.2         | 0.2           |
| Multi-Task Subsample Rate |x    | x   |0.4         | 0.4           |
| Seed                      |3784 | 1848|7491        | 725           |

**Note**: You might obtain slightly different results for multi-tasking depending on you self-prepared training data.

### Transformer-based Models

We use BERT-based language models, namely [BERT](https://huggingface.co/bert-base-uncased), [SciBERT](https://huggingface.co/allenai/scibert_scivocab_uncased) and [MatSciBERT](https://huggingface.co/m3rg-iitd/matscibert), as contextualized transformer LMs as basis of all our models. Moreover, we implement task-specific output layers on top of the LM. All Pytorch models can be found in the `models` subdirectory.

### Evaluation

Use the [aggregate_cv_scores.py](source/arg_zoning/evaluation/aggregate_cv_scores.py) scripts in the `evaluation` subdirectory to evaluate the performance of trained models across all five folds.

## License

This software is open-sourced under the AGPL-3.0 license. See the
[LICENSE](LICENSE) file for details.
The MuLMS-AZ corpus is released under the CC BY-SA 4.0 license. See the [LICENSE](data/mulms_corpus/LICENSE) file for details.
For a list of other open source components included in this project, see the
file [3rd-party-licenses.txt](3rd-party-licenses.txt).

## Citation

If you use our software or dataset in your scientific work, please cite our paper:

```
@inproceedings{schrader-etal-2023-mulms,
    title = "{M}u{LMS}-{AZ}: An Argumentative Zoning Dataset for the Materials Science Domain",
    author = {Schrader, Timo  and
      B{\"u}rkle, Teresa  and
      Henning, Sophie  and
      Tan, Sherry  and
      Finco, Matteo  and
      Gr{\"u}newald, Stefan  and
      Indrikova, Maira  and
      Hildebrand, Felix  and
      Friedrich, Annemarie},
    booktitle = "Proceedings of the 4th Workshop on Computational Approaches to Discourse (CODI 2023)",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.codi-1.1",
    doi = "10.18653/v1/2023.codi-1.1",
    pages = "1--15",
}
```

## References

<a id="1">[1]</a>
[Addressing imbalance in multilabel classification: Measures and random resampling algorithms](https://www.researchgate.net/publication/275723946_Addressing_imbalance_in_multilabel_classification_Measures_and_random_resampling_algorithms) (Charte et al., 2015)

<a id="2">[2]</a>
[An annotation scheme for discourse-level argumentation in research articles](https://aclanthology.org/E99-1015) (Teufel et al., EACL 1999)

<a id="3">[3]</a>
[Discourse-level argumentation in scientific articles: human and automatic annotation](https://aclanthology.org/W99-0311) (Teufel & Moens, 1999)

<a id="4">[4]</a>
[A Multi-Layered Annotated Corpus of Scientific Papers](https://aclanthology.org/L16-1492) (Fisas et al., LREC 2016)

<a id="5">[5]</a>
[On the Discoursive Structure of Computer Graphics Research Papers](https://aclanthology.org/W15-1605) (Fisas et al., LAW 2015)

<a id="6">[6]</a>
[An ontology methodology and CISP-the proposed Core Information about Scientific Papers](https://www.semanticscholar.org/paper/An-ontology-methodology-and-CISP-the-proposed-Core-Soldatova-Liakata/17569fa30cef89c2a5e83ac407a79da937d5eee7) (Soldatova and Liakata, 2007)

<a id="7">[7]</a>
[Using LSTM Encoder-Decoder for Rhetorical Structure Prediction](https://ieeexplore.ieee.org/document/8575626) (de Moura and Feltrim, 2018)
