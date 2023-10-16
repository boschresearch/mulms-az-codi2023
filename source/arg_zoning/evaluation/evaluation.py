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
This module contains functions to evaluate performance on the AZ task.
"""
from types import FunctionType

import numpy as np
import torch
import torch.nn.functional as F
from sklearn import metrics
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer

from source.arg_zoning.models.az_classifier import AZClassifier
from source.constants.constants import CPU, az_filtered_id2label
from source.utils.helper_functions import move_to_device


def evaluate(
    dataloader: DataLoader,
    classifier: AZClassifier,
    nll_loss: FunctionType,
    device: str,
    tokenizer: PreTrainedTokenizer,
    hierarchy: dict,
    id2label: dict,
    label2id: dict,
    print_classification_output=False,
    eval_mt_dataset=-1,
) -> dict:
    """
    Evaluation method which will be applied to development and test datasets.
    It returns the pair (average loss, hierarchical f1)

    Args:
        dataloader (DataLoader): Contains samples to be classified
        classifier (AZClassifier): BERT-based AZ classifier
        nll_loss (FunctionType): Loss function
        device (str): Pytorch device
        tokenizer (PreTrainedTokenizer): BERT-based tokenizer
        hierarchy (dict): Label hierarchy used to calculate hierarchical F1 score
        id2label (dict): ID to AZ label mapping
        label2id (dict): AZ label to ID mapping
        print_classification_output (bool, optional): Whether to print the classification details of each sample. Defaults to False.
        eval_mt_dataset(int, optional): When set, only the corresponding dataset will be used for evaluation when using multi-task training. Defaults to -1.

    Returns:
        dict: All metric scores
    """
    total_loss: float = 0
    gold_labels: list = []
    label_id_array: list = []
    gold_labels_binary: list = []

    if print_classification_output:
        print("\n------------------------")
        print("  Classification outcomes")
        print("is_correct\tgold_label\tsystem_label\ttext")
        print("------------------------")

    for batch in dataloader:
        # Only use correct evaluation dataset here
        if "dataset" in batch and batch["dataset"] != eval_mt_dataset:
            continue

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

        with torch.no_grad():
            if eval_mt_dataset != -1:
                logits, normalized_scores, _ = classifier(
                    input_ids=batch["tensor"]["input_ids"],
                    attention_mask=batch["tensor"]["attention_mask"],
                    token_type_ids=batch["tensor"]["token_type_ids"],
                    dataset_flag=eval_mt_dataset,
                )
            else:
                logits, normalized_scores, _ = classifier(
                    input_ids=batch["tensor"]["input_ids"],
                    attention_mask=batch["tensor"]["attention_mask"],
                    token_type_ids=batch["tensor"]["token_type_ids"],
                )
            total_loss += nll_loss(logits, batch["label"], batch["isAugmented"])

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

        # transform binary predictions (e.g. [1., 0., 0., ... 0.]) to labellist (e.g. [1, 15])
        predictions: torch.Tensor = normalized_scores.detach().to(CPU)
        predictions_binary: torch.Tensor = (predictions > 0.5).float()
        batch_label_predictions: list = []
        for i, sample_preds_binary in enumerate(predictions_binary):
            sample_pred_labels: list = np.argwhere(sample_preds_binary == 1).tolist()[0]
            # Alternative if no score is > 0.5 -> take maximum one
            if len(sample_pred_labels) == 0:
                sample_pred_labels.append(np.argmax(predictions[i].numpy()))
            # add superlabels
            for superlabel in hierarchy:
                for sample_pred_label in sample_pred_labels:
                    if (
                        id2label[sample_pred_label] in hierarchy[superlabel]
                        and label2id[superlabel] not in sample_pred_labels
                    ):
                        sample_pred_labels.append(label2id[superlabel])
            batch_label_predictions.append(sample_pred_labels)
            label_ids: torch.Tensor = torch.tensor(sample_pred_labels, dtype=torch.long)
            label_onehot: torch.Tensor = F.one_hot(label_ids, num_classes=len(label2id))
            labels_onehot_acc: torch.Tensor = torch.zeros(len(label2id))
            for label_tensor in label_onehot:
                labels_onehot_acc.add_(label_tensor)
            label_id_array.append(labels_onehot_acc)

        # transform binary golds (e.g. [1., 0., 0., ... 0.]) to labellist (e.g. [1, 15])
        batch_gold_labels_binary: torch.Tensor = batch["label"]
        gold_labels_binary += batch_gold_labels_binary
        batch_gold_labels: list = []
        for sample_gold_binary in batch_gold_labels_binary:
            sample_gold_labels: list = np.argwhere(sample_gold_binary == 1).tolist()[0]
            batch_gold_labels.append(sample_gold_labels)
        gold_labels += batch_gold_labels

        # Print the output of the classification for each input element
        if print_classification_output:
            for i in range(len(batch_gold_labels)):
                input_strings: str = tokenizer.decode(
                    batch["tensor"]["input_ids"][i], skip_special_tokens=True
                )
                # convert class id to the real label
                predicted_labels = batch_label_predictions[i]
                predicted_labels_string: list = []
                gold_labels_string: list = []
                for predicted_label in predicted_labels:
                    predicted_labels_string.append(id2label[predicted_label])
                gold_labels: list = batch_gold_labels[i]
                for gold_label in gold_labels:
                    gold_labels_string.append(id2label[gold_label])
                # put the prefix "[OK]" if the classification is correct
                if (
                    len(set(predicted_labels).intersection(set(gold_labels)))
                    == len(set(predicted_labels))
                    == len(set(gold_labels))
                ):
                    output = "[OK]"
                elif len(set(predicted_labels).intersection(set(gold_labels))) > 0:
                    output = "[PA]"
                else:
                    output = "[NO]"
                print(
                    f"{output}\t[{(', '.join(gold_labels_string))}]\t[{(', '.join(predicted_labels_string))}]\t{input_strings}"
                )

    gold_labels_binary: torch.Tensor = torch.stack(gold_labels_binary)
    predicted_labels_binary: torch.Tensor = torch.stack(label_id_array)
    l_precision, l_recall, l_fscore, l_support = metrics.precision_recall_fscore_support(
        gold_labels_binary.numpy().astype(int),
        predicted_labels_binary.numpy().astype(int),
        average=None,
    )
    (
        micro_hierarchical_p,
        micro_hierarchical_r,
        micro_hierarchical_f1,
        support,
    ) = metrics.precision_recall_fscore_support(
        gold_labels_binary.numpy().astype(int),
        predicted_labels_binary.numpy().astype(int),
        average="micro",
    )
    (
        macro_hierarchical_p,
        macro_hierarchical_r,
        macro_hierarchical_f1,
        _,
    ) = metrics.precision_recall_fscore_support(
        gold_labels_binary.numpy().astype(int),
        predicted_labels_binary.numpy().astype(int),
        average="macro",
    )
    # Calculate the average loss over all of the batches.
    avg_loss = total_loss / len(dataloader)
    avg_loss = avg_loss.item()
    return {
        "avg_loss": avg_loss,
        "micro_h_f1": micro_hierarchical_f1,
        "micro_h_r": micro_hierarchical_r,
        "micro_h_p": micro_hierarchical_p,
        "macro_h_f1": macro_hierarchical_f1,
        "macro_h_p": macro_hierarchical_p,
        "macro_h_r": macro_hierarchical_r,
        "p_labelwise": l_precision,
        "r_labelwise": l_recall,
        "f1_labelwise": l_fscore,
    }


def calculate_average_scores(scores: list[dict]) -> dict:
    """
    Calculates average scores over multiple CVs for AZ predictions

    Args:
        scores (list[dict]): List containing dicts with scores; one dict per CV

    Returns:
        dict: Averaged scores
    """
    assert scores is not None and len(scores) > 0, "Please provide a valid list of scores."
    average_scores: dict = {}
    for key in scores[0].keys():
        average_scores[key] = np.mean([rs[key] for rs in scores], axis=0)
        if type(average_scores[key]) == np.ndarray:
            result_dict: dict = {}
            for i, v in enumerate(average_scores[key]):
                result_dict[az_filtered_id2label[i]] = v
            average_scores[key] = result_dict
    average_scores["micro_h_f1_std"] = np.std([s["micro_h_f1"] for s in scores])
    average_scores["macro_h_f1_std"] = np.std([s["macro_h_f1"] for s in scores])
    return average_scores
