#!/bin/bash

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

echo -e '\033[1;31mExecute this script from the directory containing this file! \033[0m'

PROJECT_ROOT=$(realpath "..")

export PYTHONPATH="$PYTHONPATH:${PROJECT_ROOT}"

### Training Hyperparameters (add additional if needed) ###

lr=1e-6
numEpochs=50
batchSize=32
seed="$(shuf -i 1-100000 -n 1)" # Random Seed
bertModel="allenai/scibert_scivocab_uncased"
targetLabel="Motivation"

######

### Path Settings ###

outputDir="$PROJECT_ROOT/output/az_domain_transfer"

######

mkdir -p $outputDir

cd $PROJECT_ROOT/source/arg_zoning

mkdir -p $outputDir/cv_$f

python run_domain_transfer_experiment.py \
--model_name $bertModel \
--num_epochs $numEpochs \
--output_path $outputDir \
--num_epochs $numEpochs \
--seed $seed \
--lr $lr \
--az_target_label $targetLabel \
--train_on_mulms
# ^ Remove to train on the other corpora instead

### Evaluation Parameters ###

numFolds=5 # Number of folds to iterate over
split="test" # Must be one of ["dev", "test"]

######

cd $PROJECT_ROOT/source/arg_zoning/evaluation

python aggregate_cv_scores.py \
--input_path $outputDir \
--set $split \
--num_folds $numFolds
