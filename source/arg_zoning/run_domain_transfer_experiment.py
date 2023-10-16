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
This module is responsible for the training
the binary AZ transfer classifiers.
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
from sklearn.metrics import precision_recall_fscore_support
from torch.nn.modules.loss import BCEWithLogitsLoss
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    BertTokenizer,
    BertTokenizerFast,
    get_constant_schedule_with_warmup,
)

from source.arg_zoning.az_datasets.mulms_az_dataset import MuLMS_AZDataset
from source.arg_zoning.az_datasets.transfer_dataset import AZ_TransferDataset
from source.arg_zoning.models.binary_az_classifier import BinaryAZClassifier
from source.constants.constants import (
    CPU,
    CUDA,
    MULMS_DATASET_READER_PATH,
    MULMS_FILES,
    MULMS_PATH,
    az_filtered_label2id,
)
from source.utils.helper_functions import (
    get_executor_device,
    move_to_device,
    print_cmd_args,
)

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
    "--include_az_label_abstract",
    action="store_true",
    help="If set, the > Abstract < AZ label will be included in the training.",
)
parser.add_argument(
    "--az_target_label",
    required=True,
    type=str,
    choices=["Motivation", "Background", "Experiment", "Results", "Conclusion"],
    help="Which AZ label to train for.",
)
parser.add_argument(
    "--train_on_mulms",
    action="store_true",
    help="Whether to train on MuLMS instead of the other corpora.",
)

args = parser.parse_args()

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

AZ_MAPPING_TABLE: dict = {
    "Motivation": {
        "PUBMED": ["OBJECTIVE"],
        "ART": ["Hyp", "Mot", "Goa"],
        "AZ-CL": ["AIM"],
        "DRI": ["DRI_Challenge"],
        "targets": ["Motivation"],
    },
    "Background": {
        "PUBMED": ["BACKGROUND"],
        "ART": ["Bac"],
        "AZ-CL": ["BKG", "CTR", "BAS"],
        "DRI": ["DRI_Background"],
        "targets": ["Background", "Background_PriorWork"],
    },
    "Experiment": {
        "PUBMED": ["METHOD"],
        "ART": ["Obj", "Met", "Mod", "Exp", "Obs"],
        "DRI": ["DRI_Approach"],
        "targets": ["Experiment", "Exp_Preparation", "Exp_Characterization", "Explanation"],
    },
    "Results": {
        "PUBMED": ["RESULT"],
        "ART": ["Res"],
        "DRI": ["DRI_Outcome"],
        "targets": ["Results", "Explanation"],
    },
    "Conclusion": {
        "PUBMED": ["CONCLUSION"],
        "ART": ["Con"],
        "DRI": ["DRI_Outcome"],
        "targets": ["Conclusion"],
    },
}

# code adapted from https://github.com/crux82/AILC-lectures2021-lab/blob/main/AILC_Lectures_2021_Training_BERT_based_models_in_few_lines_of_code.ipynb


def train(
    classifier: BinaryAZClassifier,
    tokenizer: BertTokenizer,
    device: str,
    hyperparameters: dict,
    output_model_name: str,
    transfer_dl: DataLoader,
    dev_dl: DataLoader,
    test_dl: DataLoader,
    loss_function: BCEWithLogitsLoss,
    mulms_az_target_label: str,
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
        and transfer_dl is not None
        and test_dl is not None
        and dev_dl is not None
        and loss_function is not None
    ), "Please verify that all variables were passed correctly."
    batch_size: int = hyperparameters["batch_size"]
    num_epochs: int = hyperparameters["num_epochs"]
    learning_rate: float = hyperparameters["learning_rate"]
    warmup_proportion: float = hyperparameters["warmup_proportion"]

    num_training_steps: int = int(len(transfer_dl) / batch_size * num_epochs)
    num_warmup_steps: int = int(num_training_steps * warmup_proportion)
    logging.log(
        logging.INFO, f"Warmup steps: {num_warmup_steps}, total steps: {num_training_steps}"
    )

    optimizer: AdamW = AdamW(classifier.parameters(), lr=learning_rate)
    scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps)

    best_f1: float = -1
    total_t0: str = time.time()

    best_model: torch.nn.Module = None
    early_stopping_counter: int = 0

    for epoch in range(0, num_epochs):
        logging.log(logging.INFO, f"Starting epoch {epoch + 1}/{args.num_epochs}")

        train_loss: float = 0

        classifier = classifier.train()

        for batch in tqdm(transfer_dl):

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

            train_logits, _, _ = classifier(
                input_ids=batch["tensor"]["input_ids"],
                attention_mask=batch["tensor"]["attention_mask"],
                token_type_ids=batch["tensor"]["token_type_ids"],
            )

            loss: torch.Tensor = loss_function(
                train_logits, torch.unsqueeze(batch["label"], dim=1)
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

        logging.log(logging.INFO, "Running Epoch Test on Tune set:")
        classifier = classifier.eval()

        eval_results: dict = binary_az_evaluate(dev_dl, classifier, device, mulms_az_target_label)

        f1_score: float = eval_results[mulms_az_target_label]["F1"]

        logging.log(
            logging.INFO,
            f"F1 for label {mulms_az_target_label}: {eval_results[mulms_az_target_label]['F1']}",
        )

        if f1_score > best_f1:
            best_f1 = f1_score
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

    logging.log(
        logging.INFO,
        "Fold complete! Training took {:} (h:mm:ss)".format(format_time(time.time() - total_t0)),
    )

    return best_model


def binary_az_evaluate(
    dataloader: DataLoader, model: BinaryAZClassifier, device: str, mulms_az_target_label: str
) -> dict:
    """
    Evaluates the performance of a trained binary classifier on the AZ transfer experiment.

    Args:
        dataloader (DataLoader): The dataloader that contains the evaluation dataset.
        model (BinaryAZClassifier): The trained classifier to evaluate.
        device (str): CPU or GPU backend.
        mulms_az_target_label (str): The target AZ label to evaluate.

    Returns:
        dict: Result metrics in terms of P, R and F1 for target and OTHER.
    """
    logging.log(logging.INFO, "Evaluating.")

    binary_predictions: list[int] = []
    binary_gold_labels: list[int] = []

    az_label_indices_tensor: torch.Tensor = torch.tensor(
        [az_filtered_label2id[id] for id in AZ_MAPPING_TABLE[mulms_az_target_label]["targets"]]
    )

    for batch in dataloader:
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

        _, normalized_scores, _ = model(
            input_ids=batch["tensor"]["input_ids"],
            attention_mask=batch["tensor"]["attention_mask"],
            token_type_ids=batch["tensor"]["token_type_ids"],
        )

        binary_predictions.extend(torch.squeeze((normalized_scores > 0.5).int()).tolist())
        # Checking if at least one of the target labels is 1 by taking the sum of all relevant labels and checking if > 0
        binary_gold_labels.extend(
            (batch["label"][:, az_label_indices_tensor].sum(axis=1) > 0).int().tolist()
        )

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

    precision, recall, f1, _ = precision_recall_fscore_support(
        binary_gold_labels, binary_predictions
    )

    result_scores: dict = {
        mulms_az_target_label: {"P": precision[1], "R": recall[1], "F1": f1[1]},
        "OTHER": {"P": precision[0], "R": recall[0], "F1": f1[0]},
    }

    return result_scores


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

    tokenizer: BertTokenizerFast = BertTokenizerFast.from_pretrained(embedding_model_name)

    # Hyperparameters
    hyperparameters: dict = {
        "batch_size": args.batch_size,
        "dropout_rate": args.dropout_rate,
        "learning_rate": args.lr,
        "num_epochs": args.num_epochs,
        "warmup_proportion": 0.15,
        "max_seq_length": 128,
    }

    transfer_dataset: AZ_TransferDataset = AZ_TransferDataset(
        args.model_name,
        args.az_target_label,
        AZ_MAPPING_TABLE,
        train_on_mulms=args.train_on_mulms,
    )
    transfer_dataloader: DataLoader = DataLoader(
        transfer_dataset, batch_size=args.batch_size, shuffle=True
    )

    classifier: torch.nn.module = BinaryAZClassifier(
        embedding_model_name, dropout_rate=hyperparameters["dropout_rate"]
    )

    if DEVICE != CPU:
        classifier = classifier.to(DEVICE)

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

    mulms_dev_dl: DataLoader = DataLoader(mulms_dev_set, batch_size=hyperparameters["batch_size"])
    mulms_test_dl: DataLoader = DataLoader(
        mulms_test_set, batch_size=hyperparameters["batch_size"]
    )

    model_output_name: str = os.path.join(args.output_path, "binary_az_model.pt")

    logging.log(logging.INFO, "Start Training Model")
    best_model: torch.nn.Module = None

    loss_function: BCEWithLogitsLoss = BCEWithLogitsLoss()

    best_model = train(
        classifier,
        tokenizer,
        DEVICE,
        hyperparameters,
        model_output_name,
        transfer_dataloader,
        mulms_dev_dl,
        mulms_test_dl,
        loss_function=loss_function,
        mulms_az_target_label=args.az_target_label,
    )

    dev_results: dict = binary_az_evaluate(
        mulms_dev_dl, best_model, DEVICE, mulms_az_target_label=args.az_target_label
    )
    test_results: dict = binary_az_evaluate(
        mulms_test_dl, best_model, DEVICE, mulms_az_target_label=args.az_target_label
    )

    logging.log(logging.INFO, f"Dev results: {dev_results}")
    logging.log(logging.INFO, f"Test results: {dev_results}")

    np.savez(os.path.join(args.output_path, "scores_dev.npz"), dev_results)
    np.savez(os.path.join(args.output_path, "scores_test.npz"), test_results)

    logging.log(logging.INFO, "Finished training.")


if __name__ == "__main__":
    main()
