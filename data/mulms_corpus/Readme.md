# Multi-Layer Materials Science Corpus Dataset

The Multi-Layer Materials Science Corpus HuggingFace Dataset provides an easy way to access all samples and annotations that have been made by the annotators. It can be loaded and postprocessed for training of classifiers.

## Setup

Please install the HuggingFace datasets library via pip:

```
pip install datasets
```

**Note**: Please don't use conda as it seems to not work correctly.

You also need the following Uima CAS Python libraries installed:

- [puima](https://github.com/annefried/puima)

## Prepare Dataset

The HuggingFace dataset class will be loaded during runtime by reading all dataset files. Please make sure to have all XMI annotation files + TypeSystem.xml + MuLMS_Corpus_Metadata.csv available in one directory.

**Note**: Please make sure to only have filenames in the format of _name_.xmi (e.g., "graphene_01.xmi"; no other trailing file endings).

## Load Dataset

Add the following lines to your script to load the dataset:

```
from datasets import load_dataset
from source.constants.constants import MULMS_DATASET_READER_PATH, MULMS_PATH, MULMS_FILES

dataset = load_dataset(
            MULMS_DATASET_READER_PATH.__str__(),
            data_dir=MULMS_PATH.__str__(),
            data_files=MULMS_FILES,
            name="MuLMS_Corpus",
            split="train/validation/test", # Select on of these splits or omit to load all
        )
```

**Note**: Make sure that all scripts in this repository are accessible from outside, otherwise loading the dataset is very likely to fail. The easiest way to do so is to add the root directory of this repository to the `$PYTHONPATH` environment variable.

`data_files` is a list of filenames that should be read from `data_dir`, i.e., all filenames are concatenated with `data_dir` by calling `os.path.join()`.

## Working with the Dataset

Loading the dataset can take one or two minutes. After having finished, all features can be accessed via `dataset[feature]`. The following features are available:

- doc_id: Document IDs
- sentence: Sentence String; dataset is sentence-wise organized
- tokens: List of (pre-processed) tokens corresponding to this sentence
- beginOffset: Beginning index of this sentence within full text string
- endOffset: Ending index within full text string
- AZ_labels: "Argumentative Zoning" label for this sentence
- NER_labels: Entity labels corresponding to tokens in this sentence; **Note**: Each label dict holds 5 lists (text, value, begin, end, tokenIndex) which all are of same length, i.e., all elements at index 0 belong together, ... -> tokenIndex refers to the index of the token in the list of tokens for the whole sentence!
- docFileName: Filename of the corresponding document
- data_split: Corresponding split of the document
- category: As some documents are grouped to related documents, that info is provided in this column

**Note**: Since one sentence **OR** subset sentences can have multiple Argumentative Zoning labels, many sentences or part of them occur twice. Make sure to treat them properly!
