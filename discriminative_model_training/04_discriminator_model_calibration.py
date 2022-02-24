# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.9.1+dev
#   kernelspec:
#     display_name: Python [conda env:snorkeling_full_text]
#     language: python
#     name: conda-env-snorkeling_full_text-py
# ---

# # Discriminator Model Calibration

# Deep learning models can become over confident in their predictions regardless if the prediction is correct or not. Model calibration is needed to account for this problem. This notebook is designed to use Temperature Scaling to calibrate deep learning models however; it isn't designed executed locally without a gpu. This notebook holds the code in a similar format and was trained on UPenn's computing cluster.

# +
import argparse
import csv

from datasets import Dataset
import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
from sklearn.calibration import calibration_curve
import torch
from torch.utils.data import DataLoader
import tqdm
from temperature_scaling_transformers import ModelWithTemperature

# +
parser = argparse.ArgumentParser(
    description="Train BioBert as Discriminator Model. Use these two commands to run the model."
)
parser.add_argument(
    "--edge_type",
    help="The edge type to use to predict the sentences. Valid options are DaG, CbG, CtD, GiG",
)
parser.add_argument("--best_model", help="The path of the best model to calibrate")

args = parser.parse_args()

# +
if args.edge_type.lower() == "dag":
    edge_prediction = "DaG"
    curated_label = "curated_dsh"

    validation_file = "DaG/training_sen/dg_dev_test_encoded_lemmas.tsv"
    validation_labels_file = "DaG/training_sen/dg_dev_test_candidates_resampling.tsv"

    all_candidates_file = "DaG/training_sen/all_dg_abstract_encoded_lemmas.tsv"
    output_file = "DaG/all_dag_candidates.tsv"

    entity_replace_one = "DISEASE_ENTITY"
    one_replace = "@DISEASE$"
    entity_replace_two = "GENE_ENTITY"
    two_replace = "@GENE$"

if args.edge_type.lower() == "ctd":
    edge_prediction = "CtD"
    curated_label = "curated_ctd"

    validation_file = "CtD/training_sen/cd_dev_test_encoded_lemmas.tsv"
    validation_labels_file = "CtD/training_sen/cd_dev_test_candidates_resampling.tsv"

    all_candidates_file = "CtD/training_sen/all_cd_abstract_encoded_lemmas.tsv"
    output_file = "CtD/all_ctd_candidates.tsv"

    entity_replace_one = "COMPOUND_ENTITY"
    one_replace = "@CHEMICAL$"
    entity_replace_two = "DISEASE_ENTITY"
    two_replace = "@DISEASE$"

if args.edge_type.lower() == "cbg":
    edge_prediction = "CbG"
    curated_label = "curated_cbg"

    validation_file = "CbG/training_sen/cg_dev_test_encoded_lemmas.tsv"
    validation_labels_file = "CbG/training_sen/cg_dev_test_candidates_resampling.tsv"

    all_candidates_file = "CbG/training_sen/all_cg_abstract_encoded_lemmas.tsv"
    output_file = "CbG/all_cbg_candidates.tsv"

    entity_replace_one = "COMPOUND_ENTITY"
    one_replace = "@CHEMICAL$"
    entity_replace_two = "GENE_ENTITY"
    two_replace = "@GENE$"

if args.edge_type.lower() == "gig":
    edge_prediction = "GiG"
    curated_label = "curated_gig"

    validation_file = "GiG/training_sen/gg_dev_test_encoded_lemmas.tsv"
    validation_labels_file = "GiG/training_sen/gg_dev_test_candidates_resampling.tsv"

    all_candidates_file = "GiG/training_sen/all_gg_abstract_encoded_lemmas.tsv"
    output_file = "GiG/all_gig_candidates.tsv"

    entity_replace_one = "GENE_ENTITY"
    one_replace = "@GENE$"
    entity_replace_two = "GENE_ENTITY"
    two_replace = "@GENE$"
# -

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "../biobert-base-cased-v1.1", local_files_only=True
)

# +
# Set up the datasets

## Validation
validation_data = pd.read_csv(validation_file, sep="\t").rename(
    index=str, columns={"split": "dataset", curated_label: "labels"}
)
dev_split_id = validation_data.dataset.min()
test_split_id = validation_data.dataset.max()

# +
validation_dataset = Dataset.from_pandas(
    validation_data.query(f"dataset=={dev_split_id}")[["parsed_lemmas", "labels"]]
)

validation_dataset = validation_dataset.map(
    lambda x: tokenizer(
        " ".join(
            x["parsed_lemmas"]
            .replace(entity_replace_one, one_replace)
            .replace(entity_replace_two, two_replace)
            .split("|")
        ),
        padding="max_length",
        return_tensors="pt",
        max_length=100,
        truncation=True,
    ),
    remove_columns=["parsed_lemmas"],
)

validation_dataset_pt = DataLoader(validation_dataset, batch_size=10)

# +
# Set up calibration step for models

# Pre calibration
biobert = AutoModelForSequenceClassification.from_pretrained(
    args.best_model, local_files_only=True, num_labels=2
)
biobert.eval()
val_labels = []
predictions = []

# +
# Do I need cuda?
biobert = torch.nn.DataParallel(biobert)
print(biobert)

biobert = biobert.cuda()
# -

for idx, batch in tqdm.tqdm(enumerate(validation_dataset_pt)):
    attention_mask = torch.stack(batch["attention_mask"][0]).permute(1, 0).cuda()
    input_ids = torch.stack(batch["input_ids"][0]).permute(1, 0).cuda()
    labels = batch["labels"].long().cuda()
    output = biobert(attention_mask=attention_mask, input_ids=input_ids, labels=labels)

    # bring back from gpu to save memory
    attention_mask.cpu()
    input_ids.cpu()
    labels.cpu()
    output[0].cpu()

    val_labels.append(batch["labels"])
    predictions.append(
        torch.nn.functional.softmax(output[1].detach().cpu(), dim=1)[:, 1]
    )

# +
combined_labels = torch.cat(val_labels).numpy()
combined_predictions = torch.cat(predictions).detach().numpy()

prob_true, prob_pred = calibration_curve(
    combined_labels, combined_predictions, n_bins=10
)
# -

before_calibration = pd.DataFrame().from_dict(
    dict(true_proportion=prob_true, prediction_proportion=prob_pred)
)
print(before_calibration)

biobert = biobert.cpu()
biobert = ModelWithTemperature(biobert)
biobert.cuda()
biobert.set_temperature(validation_dataset_pt)

val_labels = []
predictions = []

# Post calibration
for idx, batch in tqdm.tqdm(enumerate(validation_dataset_pt)):
    output = biobert(batch)
    val_labels.append(batch["labels"])
    predictions.append(torch.nn.functional.softmax(output, dim=1)[:, 1])

combined_labels = torch.cat(val_labels).numpy()
combined_predictions = torch.cat(predictions).detach().numpy()

# +
prob_true, prob_pred = calibration_curve(
    combined_labels, combined_predictions, n_bins=10
)
after_calibration = pd.DataFrame().from_dict(
    dict(true_proportion=prob_true, prediction_proportion=prob_pred)
)
print(after_calibration)

(
    before_calibration.assign(label="before_calibration")
    .append(after_calibration.assign(label="after_calibration"))
    .to_csv(f"{edge_prediction}/{edge_prediction}_calibration.tsv", sep="\t")
)
# -

# # Use Calibrated Model to predict every Sentence

# Load the dataset for all predictions
all_data = pd.read_csv(all_candidates_file, sep="\t")
all_dataset = Dataset.from_pandas(all_data)

# +
all_dataset = all_dataset.map(
    lambda x: tokenizer(
        " ".join(
            x["parsed_lemmas"]
            .replace(entity_replace_one, one_replace)
            .replace(entity_replace_two, two_replace)
            .split("|")
        ),
        padding="max_length",
        return_tensors="pt",
        max_length=100,
        truncation=True,
    ),
    remove_columns=["parsed_lemmas"],
)

all_dataset_pt = DataLoader(all_dataset, batch_size=1000)
# -

with open(output_file, "w") as out:

    outwriter = csv.DictWriter(out, delimiter="\t", fieldnames=["pred", "candidate_id"])
    outwriter.writeheader()

    # Load the dataset for all predictions
    all_data = pd.read_csv(all_candidates_file, sep="\t", chunksize=100000)

    for df_chunk in all_data:
        all_dataset = Dataset.from_pandas(df_chunk)

        all_dataset = all_dataset.map(
            lambda x: tokenizer(
                " ".join(
                    x["parsed_lemmas"]
                    .replace(entity_replace_one, one_replace)
                    .replace(entity_replace_two, two_replace)
                    .split("|")
                ),
                padding="max_length",
                return_tensors="pt",
                max_length=100,
                truncation=True,
            ),
            remove_columns=["parsed_lemmas"],
        )

        all_dataset_pt = DataLoader(all_dataset, batch_size=10)
        biobert.cuda()

        for batch in tqdm.tqdm(all_dataset_pt):
            output = biobert(batch)
            predictions = torch.nn.functional.softmax(output, dim=1)[:, 1]
            for pred, cand_id in zip(predictions, batch["candidate_id"]):
                outwriter.writerow(
                    {"pred": pred.detach().item(), "candidate_id": cand_id.item()}
                )

        biobert.cpu()
