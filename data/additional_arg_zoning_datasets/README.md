# Adding additional AZ Corpora

Due to legal reasons, we cannot directly provide the other AZ corpora used in the multi-task and transfer experiments. Therefore, we provide the necessary instructions to prepare everything.

## Directory Structure

This directory (`data/additional_arg_zoning_datasets`) expects the following files to be present:
* `ART.csv`
* `AZ-CL.csv`
* `DRI.csv`
* `materials_science_pubmed.pickle`

## File Structure

For all 3 CSV files, the following structure **without heading line** is expected:

```
AZ_Label <TAB> Sentence
AZ_Label <TAB> Sentence
.
.
.
```

## Obtain the Datasets

* ART: Data can be found here: https://www.aber.ac.uk/en/cs/research/cb/projects/art/art-corpus/
* AZ-CL: Data can be found here: https://www.cl.cam.ac.uk/~sht25/AZ_corpus.html
* DRI: As of now (July 2023), it seems that the data is no longer available on the server (http://sempub.taln.upf.edu/dricorpus). It might be released in the future again. Alternatively, you could also try to find an archived version of the website.
* PubMed: We provide two preparation scripts in order to prepare the corpus in `data/additional_arg_zoning_datasets/prepare_pubmed`. First, you need to obtain the PubMed `*.xml.gz` files on https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/ and https://ftp.ncbi.nlm.nih.gov/pubmed/updatefiles/. We used the files listed in [pubmed_file_list.txt](prepare_pubmed/pubmed_file_list.txt). You may need to find an archived version of these websites to obtain these files. Secondly, you need to filter the downloaded XML files using [filter_matscience_journals.py](prepare_pubmed/filter_matscience_journals.py). Finally, create the PubMed dataset using [prepare_matscience_dataset.py](prepare_pubmed/prepare_matscience_dataset.py).