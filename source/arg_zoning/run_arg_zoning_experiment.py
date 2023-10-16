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
This module is responsible for the entire training
pipeline of the Argumentative Zoning experiments.
"""
import datetime
import logging
import os
import random
import sys
import time
from argparse import ArgumentParser
from copy import deepcopy

import numpy as np
import torch
from datasets import Dataset, load_dataset
from torch.nn.modules.loss import _Loss
from torch.optim import AdamW
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm
from transformers import (
    BertTokenizer,
    BertTokenizerFast,
    get_constant_schedule_with_warmup,
)

from source.arg_zoning.az_datasets.az_dataset import AZ_Dataset
from source.arg_zoning.az_datasets.mulms_az_dataset import MuLMS_AZDataset
from source.arg_zoning.az_datasets.pubmed_dataset import PubmedDataset  # noqa: F401
from source.arg_zoning.az_datasets.pubmed_dataset_pytorch import PubmedAZDataset
from source.arg_zoning.evaluation.evaluation import calculate_average_scores, evaluate
from source.arg_zoning.models.all_task_az_classifier import AllTaskAZClassifier
from source.arg_zoning.models.az_classifier import AZClassifier
from source.arg_zoning.models.two_task_az_classifier import TwoTaskAZClassifier
from source.arg_zoning.utils.augmented_loss_function import AugmentedLoss
from source.arg_zoning.utils.multi_oversampler import MultiLabelOversampler
from source.constants.constants import (
    ADDITIONAL_AZ_LABELS,
    AZ_DATASET_PATHS,
    CPU,
    CUDA,
    MULMS_DATASET_READER_PATH,
    MULMS_FILES,
    MULMS_PATH,
    az_filtered_id2label,
    az_filtered_label2id,
)
from source.utils.helper_functions import (
    get_executor_device,
    move_to_device,
    print_cmd_args,
)
from source.utils.multitask_dataloader import MultitaskDataloader

parser: ArgumentParser = ArgumentParser()
parser.add_argument(
    "--model_name",
    type=str,
    help="Name of/Path to the pretrained model",
    default="allenai/scibert_scivocab_uncased",
)
parser.add_argument(
    "--output_path", type=str, help="Storage path for fine-tuned model", default="."
)
parser.add_argument(
    "--disable_model_storage",
    help="Disables storage of model parameters in order to save disk space.",
    action="store_true",
)
parser.add_argument("--seed", type=int, help="Random seed", default=213)
parser.add_argument(
    "--disable_cuda", action="store_true", help="Disable CUDA in favour of CPU usage"
)
parser.add_argument("--lr", type=float, help="Learning rate", default=5e-6)
parser.add_argument("--batch_size", type=int, help="Batch size used during training", default=32)
parser.add_argument("--num_epochs", type=int, help="Number of training epochs", default=50)
parser.add_argument("--dropout_rate", type=float, help="Dropout rate during training", default=0.1)
parser.add_argument(
    "--os_percentage",
    type=float,
    help="Determines number of cloned samples (0.2 = + 20% cloned samples)",
    default=0.2,
)
parser.add_argument(
    "--single_label_only",
    action="store_true",
    help="If set, only single-label instances will be cloned",
)
parser.add_argument(
    "--use_weights",
    action="store_true",
    help="Set for weighted dynamic ML-ROS with 20% lower bound",
)
parser.add_argument(
    "--use_fixed_meanIR", action="store_true", help="Set to not update meanIR after each iteration"
)
parser.add_argument(
    "--exclude_new_minlabels",
    action="store_true",
    help="Exclude labels that become minority labels during oversampling",
)
parser.add_argument(
    "--oversample_each_epoch",
    action="store_true",
    help="Whether to oversample after epoch; changes cloned instances every time.",
)
parser.add_argument(
    "--disable_oversampling",
    action="store_true",
    help="Disables oversampling during training entirely. Overrides --oversample_each_epoch.",
)
parser.add_argument(
    "--cv",
    type=int,
    help="If set, the corresponding train set is used as tune set for CV training.",
    choices=[1, 2, 3, 4, 5],
    default=None,
)
parser.add_argument(
    "--enable_multi_tasking",
    help="Whether to train one model with multiple AZ datasets and output heads.",
    action="store_true",
)
parser.add_argument(
    "--multi_task_dataset",
    type=str,
    choices=["PUBMED", "DRI", "AZ-CL", "ART", "all"],
    help="Which of the 4 AZ dataset to combine with MuLMS.",
    default="PUBMED",
)
parser.add_argument(
    "--mt_subsample_rate",
    type=float,
    help="Percentage by which to subsample the multi-task dataset.",
    default=1.0,
)
parser.add_argument(
    "--include_az_label_abstract",
    action="store_true",
    help="If set, the > Abstract < AZ label will be included in the training.",
)
parser.add_argument(
    "--augment_pubmed",
    action="store_true",
    help="Whether to integrate Pubmed samples into the Mulms AZ dataset by mapping labels. Cannot be used together with Multi Task Learning.",
)
parser.add_argument(
    "--distinguish_pubmed_from_az_samples",
    action="store_true",
    help="If set, augmented Pubmed samples will be treated separatly in the loss function.",
)

args = parser.parse_args()

assert not (
    args.enable_multi_tasking and args.augment_pubmed
), "Cannot augment Mulms dataset and enable MT learning together!"

DEVICE: str = get_executor_device(args.disable_cuda)
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
# Set random values
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if DEVICE == CUDA:
    torch.cuda.manual_seed_all(args.seed)

hierarchy: dict[str, list[str]] = {
    "Experiment": ["Exp_Characterization", "Exp_Preparation"],
    "Explanation": ["Explanation_Assumption", "Explanation_Hypothesis"],
    "Background": ["Background_PriorWork"],
    "Caption": ["Captionc"],
    "Metadata": ["References"],
}
labels_to_remove: list[str] = [
    "Explanation_Assumption",
    "Explanation_Hypothesis",
    "None",
    "Captionc",
    "References",
]

if not args.include_az_label_abstract:
    labels_to_remove.append("Abstract")
    az_filtered_label2id.pop("Abstract")
    az_filtered_id2label = dict([(v, k) for k, v in az_filtered_label2id.items()])  # noqa: F811

# Part of the functions were adapted from https://github.com/crux82/AILC-lectures2021-lab/blob/main/AILC_Lectures_2021_Training_BERT_based_models_in_few_lines_of_code.ipynb


def train(
    classifier: AZClassifier,
    tokenizer: BertTokenizer,
    device: str,
    hyperparameters: dict,
    output_model_name: str,
    train_dl: DataLoader,
    tune_dl: DataLoader,
    eval_dl: DataLoader,
    loss_function: _Loss,
    mt_enabled=False,
) -> torch.nn.Module:
    """
    Starts the training loop for the AZ classifier.

    Args:
        classifier (AZClassifier): BERT-based classifier
        tokenizer (BertTokenizer): BERT-based tokenizer
        device (str): Pytorch device
        hyperparameters (dict): Hyperparameters used for training
        output_model_name (str): Output name of trained model
        train_dl (DataLoader): Dataloader containing train set
        tune_dl (DataLoader): Dataloader containing tune set
        eval_dl (DataLoader): Dataloader containing set for final evaluation
        mt_enabled (bool): Whether Multi-Task learning is active. Defaults to False.

    """
    assert (
        classifier is not None
        and tokenizer is not None
        and device is not None
        and hyperparameters is not None
        and train_dl is not None
        and eval_dl is not None
        and tune_dl is not None
        and loss_function is not None
    ), "Please verify that all variables were passed correctly."
    batch_size: int = hyperparameters["batch_size"]
    dropout_rate: float = hyperparameters["dropout_rate"]
    num_epochs: int = hyperparameters["num_epochs"]
    learning_rate: float = hyperparameters["learning_rate"]
    warmup_proportion: float = hyperparameters["warmup_proportion"]

    num_training_steps: int = int(len(train_dl) / batch_size * num_epochs)
    num_warmup_steps: int = int(num_training_steps * warmup_proportion)
    logging.log(
        logging.INFO, f"Warmup steps: {num_warmup_steps}, total steps: {num_training_steps}"
    )

    optimizer: AdamW = AdamW(classifier.parameters(), lr=learning_rate)
    scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps)

    best_epoch: int = -1
    best_tune_f1: float = -1
    training_stats: list = []
    total_t0: str = time.time()

    best_model: torch.nn.Module = None
    early_stopping_counter: int = 0

    for epoch in range(0, num_epochs):
        logging.log(logging.INFO, f"Starting epoch {epoch + 1}/{args.num_epochs}")

        t0: float = time.time()

        train_loss: float = 0

        classifier = classifier.train()

        for batch in tqdm(train_dl):

            if device != CPU:
                (
                    batch["tensor"]["input_ids"],
                    batch["tensor"]["attention_mask"],
                    batch["tensor"]["token_type_ids"],
                    batch["label"],
                ) = move_to_device(
                    device,
                    batch["tensor"]["input_ids"],
                    batch["tensor"]["attention_mask"],
                    batch["tensor"]["token_type_ids"],
                    batch["label"],
                )

            optimizer.zero_grad()

            if mt_enabled:
                train_logits, _, _ = classifier(
                    input_ids=batch["tensor"]["input_ids"],
                    attention_mask=batch["tensor"]["attention_mask"],
                    token_type_ids=batch["tensor"]["token_type_ids"],
                    dataset_flag=batch["dataset"],
                )
            else:
                train_logits, _, _ = classifier(
                    input_ids=batch["tensor"]["input_ids"],
                    attention_mask=batch["tensor"]["attention_mask"],
                    token_type_ids=batch["tensor"]["token_type_ids"],
                )

            if args.enable_multi_tasking:
                if "isAugmented" in batch:
                    loss: torch.Tensor = loss_function[batch["dataset"]](
                        train_logits, batch["label"], batch["isAugmented"]
                    )
                else:
                    loss: torch.Tensor = loss_function[batch["dataset"]](
                        train_logits, batch["label"]
                    )

            else:
                if "isAugmented" in batch:
                    loss: torch.Tensor = loss_function(
                        train_logits, batch["label"], batch["isAugmented"]
                    )
                else:
                    loss: torch.Tensor = loss_function(
                        train_logits, batch["label"], torch.zeros(len(batch["label"]))
                    )

            if device != CPU:
                (
                    batch["tensor"]["input_ids"],
                    batch["tensor"]["attention_mask"],
                    batch["tensor"]["token_type_ids"],
                    batch["label"],
                ) = move_to_device(
                    CPU,
                    batch["tensor"]["input_ids"],
                    batch["tensor"]["attention_mask"],
                    batch["tensor"]["token_type_ids"],
                    batch["label"],
                )

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            scheduler.step()

        avg_train_loss: float = train_loss / len(train_dl)
        training_time: str = format_time(time.time() - t0)

        logging.log(logging.INFO, "Running Epoch Test on Tune set:")
        t0 = time.time()
        classifier = classifier.eval()

        if args.enable_multi_tasking:
            eval_results: dict = evaluate(
                tune_dl,
                classifier,
                loss_function[0],
                device,
                tokenizer,
                hierarchy,
                az_filtered_id2label,
                az_filtered_label2id,
                print_classification_output=False,
                eval_mt_dataset=0,
            )
        else:
            eval_results: dict = evaluate(
                tune_dl,
                classifier,
                loss_function,
                device,
                tokenizer,
                hierarchy,
                az_filtered_id2label,
                az_filtered_label2id,
                print_classification_output=False,
                eval_mt_dataset=-1,
            )
        avg_tune_loss: float = eval_results["avg_loss"]
        micro_hierarchical_f1: float = eval_results["micro_h_f1"]
        test_time: str = format_time(time.time() - t0)
        logging.log(logging.INFO, f"Micro Hierarchical F1: {micro_hierarchical_f1}")
        logging.log(logging.INFO, f"Tune Loss: {avg_tune_loss}")
        training_stats.append(
            {
                "epoch": epoch + 1,
                "Training Loss": avg_train_loss,
                "Tune Loss": avg_tune_loss,
                "Tune micro hierarchical f1": micro_hierarchical_f1,
                "Training Time": training_time,
                "Test Time": test_time,
            }
        )

        if micro_hierarchical_f1 > best_tune_f1:
            best_tune_f1 = micro_hierarchical_f1
            best_epoch = epoch + 1
            if not args.disable_model_storage:
                torch.save(classifier.state_dict(), output_model_name)
            best_model = deepcopy(classifier)
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
        if early_stopping_counter == 3:
            logging.log(
                logging.WARN,
                f"Model has not improved during the last {early_stopping_counter} epochs, training is stopped now.",
            )
            break

    train_losses: list = []
    tune_losses: list = []
    tune_f1: list = []

    for stat in training_stats:
        train_losses.append(stat["Training Loss"])
        tune_losses.append(stat["Tune Loss"])
        tune_f1.append(stat["Tune micro hierarchical f1"])
        logging.log(logging.INFO, stat)

    logging.log(
        logging.INFO,
        "Fold complete! Training took {:} (h:mm:ss)".format(format_time(time.time() - total_t0)),
    )

    #   Final evaluation of the best model on the dev set
    if args.enable_multi_tasking:
        eval_results = evaluate(
            eval_dl,
            best_model,
            loss_function[0],
            device,
            tokenizer,
            hierarchy,
            az_filtered_id2label,
            az_filtered_label2id,
            print_classification_output=True,
            eval_mt_dataset=0,
        )
    else:
        eval_results = evaluate(
            eval_dl,
            best_model,
            loss_function,
            device,
            tokenizer,
            hierarchy,
            az_filtered_id2label,
            az_filtered_label2id,
            print_classification_output=True,
            eval_mt_dataset=-1,
        )
    logging.log(
        logging.INFO,
        f"Evaluating best model from epoch {best_epoch} with Hyperparameters batch size {batch_size}, dropout {dropout_rate}, learning reate {learning_rate}, warmup {hyperparameters['warmup_proportion']} on dev set",
    )
    logging.log(logging.INFO, f"Dev micro hierarchical p: {round(eval_results['micro_h_p'], 3)}")
    logging.log(logging.INFO, f"Dev micro hierarchical r: {round(eval_results['micro_h_r'], 3)}")
    logging.log(logging.INFO, f"Dev micro hierarchical f1: {round(eval_results['micro_h_f1'], 3)}")
    logging.log(logging.INFO, f"Dev Loss: {round(eval_results['avg_loss'], 3)}")

    return best_model


def format_time(elapsed: float) -> str:
    """
    Takes a time in seconds and returns a string hh:mm:ss

    Args:
        elapsed (float): Floating point number indicating time (in s) elapsed

    Returns:
        str: Time string
    """
    # Round to the nearest second.
    elapsed_rounded: int = int(round((elapsed)))
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


def main():
    """
    Entry point.
    """

    print_cmd_args(args)

    tune_fold: int = args.cv
    use_tune_fold: bool = args.cv is not None

    if use_tune_fold:
        args.output_path = os.path.join(args.output_path, f"cv_{tune_fold}")
        os.makedirs(args.output_path, exist_ok=True)

    train_dataset: Dataset = load_dataset(
        MULMS_DATASET_READER_PATH.__str__(),
        data_dir=MULMS_PATH.__str__(),
        data_files=MULMS_FILES,
        name="MuLMS_Corpus",
        split="train",
    )
    dev_dataset: Dataset = load_dataset(
        MULMS_DATASET_READER_PATH.__str__(),
        data_dir=MULMS_PATH.__str__(),
        data_files=MULMS_FILES,
        name="MuLMS_Corpus",
        split="validation",
    )
    test_dataset: Dataset = load_dataset(
        MULMS_DATASET_READER_PATH.__str__(),
        data_dir=MULMS_PATH.__str__(),
        data_files=MULMS_FILES,
        name="MuLMS_Corpus",
        split="test",
    )

    embedding_model_name: str = args.model_name

    # Hyperparameters
    hyperparameters: dict = {
        "batch_size": args.batch_size,
        "dropout_rate": args.dropout_rate,
        "learning_rate": args.lr,
        "num_epochs": args.num_epochs,
        "warmup_proportion": 0.15,
        "max_seq_length": 128,
    }

    # Oversampling Parameters
    os_percentage: float = args.os_percentage
    single_label_only: bool = args.single_label_only
    use_weights: bool = args.use_weights
    use_fixed_meanIR: bool = args.use_fixed_meanIR
    include_new_minlabels: bool = not args.exclude_new_minlabels
    oversample_each_epoch: bool = args.oversample_each_epoch

    classifier: torch.nn.Module = None

    if args.enable_multi_tasking:
        if args.multi_task_dataset == "all":
            classifier = AllTaskAZClassifier(
                model_path=embedding_model_name,
                num_labels_mulms=len(az_filtered_label2id),
                num_labels_art=ADDITIONAL_AZ_LABELS["ART"]["num_labels"],
                num_labels_az_cl=ADDITIONAL_AZ_LABELS["AZ-CL"]["num_labels"],
                num_labels_dri=ADDITIONAL_AZ_LABELS["DRI"]["num_labels"],
                device=DEVICE,
            )
        else:
            classifier = TwoTaskAZClassifier(
                model_path=embedding_model_name,
                num_labels_mulms=len(az_filtered_label2id),
                num_labels_second_task=ADDITIONAL_AZ_LABELS[args.multi_task_dataset]["num_labels"],
                device=DEVICE,
            )
    else:
        classifier = AZClassifier(
            embedding_model_name,
            num_labels=len(az_filtered_label2id),
            dropout_rate=hyperparameters["dropout_rate"],
        )

    if DEVICE != CPU:
        classifier = classifier.to(DEVICE)

    tokenizer: BertTokenizerFast = BertTokenizerFast.from_pretrained(
        embedding_model_name, max_length=classifier._encoder.config.max_position_embeddings
    )

    mulms_train_set: MuLMS_AZDataset = None
    if not use_tune_fold:
        mulms_train_set = MuLMS_AZDataset(
            dataset=train_dataset,
            split="train",
            tokenizer_model_name=args.model_name,
            hyperparameters=hyperparameters,
            hierarchy=hierarchy,
            labels_to_remove=labels_to_remove,
        )
    else:
        mulms_train_set = MuLMS_AZDataset(
            dataset=train_dataset,
            split="train",
            tokenizer_model_name=args.model_name,
            hyperparameters=hyperparameters,
            hierarchy=hierarchy,
            labels_to_remove=labels_to_remove,
            tune_id=tune_fold,
        )
        mulms_tune_set: MuLMS_AZDataset = MuLMS_AZDataset(
            dataset=train_dataset,
            split="tune",
            tokenizer_model_name=args.model_name,
            hyperparameters=hyperparameters,
            hierarchy=hierarchy,
            labels_to_remove=labels_to_remove,
            tune_id=tune_fold,
        )

    mulms_dev_set: MuLMS_AZDataset = MuLMS_AZDataset(
        dataset=dev_dataset,
        split="validation",
        tokenizer_model_name=args.model_name,
        hyperparameters=hyperparameters,
        hierarchy=hierarchy,
        labels_to_remove=labels_to_remove,
    )
    mulms_test_set: MuLMS_AZDataset = MuLMS_AZDataset(
        dataset=test_dataset,
        split="test",
        tokenizer_model_name=args.model_name,
        hyperparameters=hyperparameters,
        hierarchy=hierarchy,
        labels_to_remove=labels_to_remove,
    )

    if args.augment_pubmed:
        pubmed_train_dl: DataLoader = DataLoader(
            PubmedAZDataset(args.model_name, "train"), batch_size=1, shuffle=False
        )
        mulms_train_set.augment_data_from_pubmed(
            pubmed_train_dl,
            mask_non_pubmed_labels=args.distinguish_pubmed_from_az_samples,
            percentage_to_add=args.mt_subsample_rate,
        )

    logging.log(logging.INFO, f"There are {len(mulms_train_set)} train samples")

    train_sampler: MultiLabelOversampler = None
    if not args.disable_oversampling:
        train_sampler = MultiLabelOversampler(
            mulms_train_set,
            num_labels=len(az_filtered_label2id),
            percentage=os_percentage,
            use_probabilities=use_weights,
            fix_meanIR=use_fixed_meanIR,
            include_new_minlabels=include_new_minlabels,
            single_label_only=single_label_only,
            sublabel_indices=[az_filtered_label2id["Exp_Preparation"]],
            oversample_each_epoch=oversample_each_epoch,
        )

    mulms_train_dl: DataLoader = DataLoader(
        mulms_train_set, sampler=train_sampler, batch_size=hyperparameters["batch_size"]
    )

    mulms_dev_dl: DataLoader = DataLoader(
        mulms_dev_set,
        sampler=RandomSampler(mulms_dev_set),
        batch_size=hyperparameters["batch_size"],
    )
    mulms_test_dl: DataLoader = DataLoader(
        mulms_test_set,
        sampler=RandomSampler(mulms_test_set),
        batch_size=hyperparameters["batch_size"],
    )

    if use_tune_fold:
        mulms_tune_dl: DataLoader = DataLoader(
            mulms_tune_set,
            sampler=RandomSampler(mulms_tune_set),
            batch_size=hyperparameters["batch_size"],
        )

    if args.enable_multi_tasking:

        mt_train_set: MultitaskDataloader = None
        mt_dev_set: MultitaskDataloader = None
        mt_test_set: MultitaskDataloader = None
        mt_tune_set: MultitaskDataloader = None

        if args.multi_task_dataset == "PUBMED":
            pubmed_train_dl: DataLoader = DataLoader(
                PubmedAZDataset(
                    args.model_name,
                    "train",
                    subsample_rate=args.mt_subsample_rate,
                    max_seq_length=classifier._encoder.config.max_position_embeddings,
                ),
                batch_size=hyperparameters["batch_size"],
                shuffle=True,
            )
            pubmed_dev_dl: DataLoader = DataLoader(
                PubmedAZDataset(
                    args.model_name,
                    "validation",
                    max_seq_length=classifier._encoder.config.max_position_embeddings,
                ),
                batch_size=hyperparameters["batch_size"],
            )
            pubmed_test_dl: DataLoader = DataLoader(
                PubmedAZDataset(
                    args.model_name,
                    "test",
                    max_seq_length=classifier._encoder.config.max_position_embeddings,
                ),
                batch_size=hyperparameters["batch_size"],
            )

            mt_train_set = MultitaskDataloader(mulms_train_dl, pubmed_train_dl)
            mt_dev_set = MultitaskDataloader(mulms_dev_dl, pubmed_dev_dl)
            mt_test_set = MultitaskDataloader(mulms_test_dl, pubmed_test_dl)
            if use_tune_fold:
                mt_tune_set = MultitaskDataloader(mulms_tune_dl, pubmed_dev_dl)

        elif args.multi_task_dataset == "all":
            art_dataloader: DataLoader = DataLoader(
                AZ_Dataset(
                    "ART",
                    AZ_DATASET_PATHS["ART"],
                    args.model_name,
                    ADDITIONAL_AZ_LABELS["ART"]["label2id"],
                    ADDITIONAL_AZ_LABELS["ART"]["id2label"],
                    subsample_rate=args.mt_subsample_rate,
                    max_seq_length=classifier._encoder.config.max_position_embeddings,
                ),
                batch_size=hyperparameters["batch_size"],
                shuffle=True,
            )
            az_cl_dataloader: DataLoader = DataLoader(
                AZ_Dataset(
                    "AZ-CL",
                    AZ_DATASET_PATHS["AZ-CL"],
                    args.model_name,
                    ADDITIONAL_AZ_LABELS["AZ-CL"]["label2id"],
                    ADDITIONAL_AZ_LABELS["AZ-CL"]["id2label"],
                    subsample_rate=args.mt_subsample_rate,
                    max_seq_length=classifier._encoder.config.max_position_embeddings,
                ),
                batch_size=hyperparameters["batch_size"],
                shuffle=True,
            )
            dri_dataloader: DataLoader = DataLoader(
                AZ_Dataset(
                    "DRI",
                    AZ_DATASET_PATHS["DRI"],
                    args.model_name,
                    ADDITIONAL_AZ_LABELS["DRI"]["label2id"],
                    ADDITIONAL_AZ_LABELS["DRI"]["id2label"],
                    subsample_rate=args.mt_subsample_rate,
                    max_seq_length=classifier._encoder.config.max_position_embeddings,
                ),
                batch_size=hyperparameters["batch_size"],
                shuffle=True,
            )

            mt_train_set = MultitaskDataloader(
                mulms_train_dl, art_dataloader, az_cl_dataloader, dri_dataloader
            )
            mt_dev_set = MultitaskDataloader(
                mulms_dev_dl, art_dataloader, az_cl_dataloader, dri_dataloader
            )
            mt_test_set = MultitaskDataloader(
                mulms_test_dl, art_dataloader, az_cl_dataloader, dri_dataloader
            )
            if use_tune_fold:
                mt_tune_set = MultitaskDataloader(
                    mulms_tune_dl, art_dataloader, az_cl_dataloader, dri_dataloader
                )

        else:
            additional_az_dataloader: DataLoader = DataLoader(
                AZ_Dataset(
                    args.multi_task_dataset,
                    AZ_DATASET_PATHS[args.multi_task_dataset],
                    args.model_name,
                    ADDITIONAL_AZ_LABELS[args.multi_task_dataset]["label2id"],
                    ADDITIONAL_AZ_LABELS[args.multi_task_dataset]["id2label"],
                    subsample_rate=args.mt_subsample_rate,
                    max_seq_length=classifier._encoder.config.max_position_embeddings,
                ),
                batch_size=hyperparameters["batch_size"],
                shuffle=True,
            )

            mt_train_set = MultitaskDataloader(mulms_train_dl, additional_az_dataloader)
            mt_dev_set = MultitaskDataloader(mulms_dev_dl, additional_az_dataloader)
            mt_test_set = MultitaskDataloader(mulms_test_dl, additional_az_dataloader)
            if use_tune_fold:
                mt_tune_set = MultitaskDataloader(mulms_tune_dl, additional_az_dataloader)

    model_output_name: str = os.path.join(
        args.output_path,
        f"best_scibert_{hyperparameters['batch_size']}_{str(hyperparameters['dropout_rate']).replace('.', '')}_{hyperparameters['learning_rate']}os_{os_percentage}.pt",
    )

    logging.log(logging.INFO, "Start Training Model")
    best_model: torch.nn.Module = None

    nll_loss = None

    if args.enable_multi_tasking:
        if args.multi_task_dataset == "all":
            nll_loss = [
                AugmentedLoss(),
                torch.nn.CrossEntropyLoss(),
                torch.nn.CrossEntropyLoss(),
                torch.nn.CrossEntropyLoss(),
            ]
        else:
            nll_loss = [AugmentedLoss(), torch.nn.CrossEntropyLoss()]

    else:
        nll_loss: AugmentedLoss = AugmentedLoss(
            distinguish_augmented_samples=args.distinguish_pubmed_from_az_samples,
            loss_weights=[1.0, 1.0],
        )

    if args.enable_multi_tasking:
        if use_tune_fold:
            best_model = train(
                classifier,
                tokenizer,
                DEVICE,
                hyperparameters,
                model_output_name,
                mt_train_set,
                mt_tune_set,
                mt_dev_set,
                nll_loss,
                mt_enabled=True,
            )
        else:
            best_model = train(
                classifier,
                tokenizer,
                DEVICE,
                hyperparameters,
                model_output_name,
                mt_train_set,
                mt_dev_set,
                mt_test_set,
                nll_loss,
                mt_enabled=True,
            )
    else:
        if use_tune_fold:
            best_model = train(
                classifier,
                tokenizer,
                DEVICE,
                hyperparameters,
                model_output_name,
                mulms_train_dl,
                mulms_tune_dl,
                mulms_dev_dl,
                nll_loss,
            )
        else:
            best_model = train(
                classifier,
                tokenizer,
                DEVICE,
                hyperparameters,
                model_output_name,
                mulms_train_dl,
                mulms_dev_dl,
                mulms_test_dl,
                nll_loss,
            )
    logging.log(logging.INFO, f"Training complete for tune fold {tune_fold}/5")

    if args.enable_multi_tasking:
        dev_results: dict = evaluate(
            mulms_dev_dl,
            best_model,
            nll_loss[0],
            DEVICE,
            tokenizer,
            hierarchy,
            az_filtered_id2label,
            az_filtered_label2id,
            print_classification_output=False,
            eval_mt_dataset=0,
        )
        test_results: dict = evaluate(
            mulms_test_dl,
            best_model,
            nll_loss[0],
            DEVICE,
            tokenizer,
            hierarchy,
            az_filtered_id2label,
            az_filtered_label2id,
            print_classification_output=False,
            eval_mt_dataset=0,
        )
    else:
        dev_results: dict = evaluate(
            mulms_dev_dl,
            best_model,
            nll_loss,
            DEVICE,
            tokenizer,
            hierarchy,
            az_filtered_id2label,
            az_filtered_label2id,
            print_classification_output=False,
            eval_mt_dataset=-1,
        )
        test_results: dict = evaluate(
            mulms_test_dl,
            best_model,
            nll_loss,
            DEVICE,
            tokenizer,
            hierarchy,
            az_filtered_id2label,
            az_filtered_label2id,
            print_classification_output=False,
            eval_mt_dataset=-1,
        )

    np.savez(os.path.join(args.output_path, "scores_dev.npz"), dev_results)
    np.savez(os.path.join(args.output_path, "scores_test.npz"), test_results)

    avg_results: dict = calculate_average_scores([dev_results])
    logging.log(logging.INFO, "Final avg results are: ")
    logging.log(
        logging.INFO, f"Average micro hierarchical precision: {round(avg_results['micro_h_p'], 3)}"
    )
    logging.log(
        logging.INFO, f"Average micro hierarchical recall: {round(avg_results['micro_h_r'], 3)}"
    )
    logging.log(
        logging.INFO, f"Average micro hierarchical f1: {round(avg_results['micro_h_f1'], 3)}"
    )
    logging.log(
        logging.INFO, f"Average macro hierarchical f1: {round(avg_results['macro_h_f1'], 3)}"
    )
    logging.log(logging.INFO, "Average labelwise scores:")
    print("\nLABEL\tPREC.\tRECALL\tF1")
    print("----------------------------------------")
    for label in avg_results["p_labelwise"]:
        print(
            f"{label[:7]}\t{round(avg_results['p_labelwise'][label], 3)}\t{round(avg_results['r_labelwise'][label], 3)}\t{round(avg_results['f1_labelwise'][label], 3)}"
        )

    print(
        f"\nTrained for {hyperparameters['num_epochs']} epochs with hyperparameters: batch size {hyperparameters['batch_size']}, dropout {hyperparameters['dropout_rate']}, learning rate {hyperparameters['learning_rate']} and warmup {hyperparameters['warmup_proportion']}.\nOversampling percentage was {os_percentage} and use probabilities was {use_weights}."
    )


if __name__ == "__main__":
    main()
