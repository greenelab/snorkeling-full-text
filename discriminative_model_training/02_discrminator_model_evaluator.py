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

# # Evaluate the discriminator Model

# The next step following training is to evaluate the discriminator model performance. This notebook is designed generate results from training however; it isn't designed executed locally without a gpu. This notebook holds the code in a similar format and was trained on UPenn's computing cluster.

# +
import argparse
from pathlib import Path
import re

from datasets import Dataset
import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
from sklearn.metrics import auc, precision_recall_curve, roc_curve
import torch
from torch.utils.data import DataLoader
import tqdm

# +
parser = argparse.ArgumentParser(
    description="Train BioBert as Discriminator Model. Use these two commands to run the model."
)
parser.add_argument(
    "--edge_type",
    help="The edge type to use to predict the sentences. Valid options are DaG, CbG, CtD, GiG",
)

args = parser.parse_args()

# +
if args.edge_type.lower() == "dag":
    edge_prediction = "DaG"
    curated_label = "curated_dsh"

    validation_file = "DaG/training_sen/dg_dev_test_encoded_lemmas.tsv"

    entity_replace_one = "DISEASE_ENTITY"
    one_replace = "@DISEASE$"
    entity_replace_two = "GENE_ENTITY"
    two_replace = "@GENE$"

if args.edge_type.lower() == "ctd":
    edge_prediction = "CtD"
    curated_label = "curated_ctd"

    validation_file = "CtD/training_sen/cd_dev_test_encoded_lemmas.tsv"

    entity_replace_one = "COMPOUND_ENTITY"
    one_replace = "@CHEMICAL$"
    entity_replace_two = "DISEASE_ENTITY"
    two_replace = "@DISEASE$"

if args.edge_type.lower() == "cbg":
    edge_prediction = "CbG"
    curated_label = "curated_cbg"

    validation_file = "CbG/training_sen/cg_dev_test_encoded_lemmas.tsv"

    entity_replace_one = "COMPOUND_ENTITY"
    one_replace = "@CHEMICAL$"
    entity_replace_two = "GENE_ENTITY"
    two_replace = "@GENE$"

if args.edge_type.lower() == "gig":
    edge_prediction = "GiG"
    curated_label = "curated_gig"

    validation_file = "GiG/training_sen/gg_dev_test_encoded_lemmas.tsv"

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
test_dataset = Dataset.from_pandas(
    validation_data.query(f"dataset=={test_split_id}")[["parsed_lemmas", "labels"]]
)

test_dataset = test_dataset.map(
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

test_dataset_pt = DataLoader(test_dataset, batch_size=10)
# -

data = []
for model_file in list(Path(args.edge_type).rglob("*/*model")):

    lf_num = int(re.search(r"(\d+)", str(model_file)).groups()[0])
    biobert = AutoModelForSequenceClassification.from_pretrained(
        model_file, local_files_only=True, num_labels=2
    )

    # biobert = torch.nn.DataParallel(biobert)
    # biobert = biobert.cuda()

    biobert.eval()
    validation_loss = []
    predictions = []
    val_labels = []

    for idx, batch in tqdm.tqdm(enumerate(validation_dataset_pt)):
        attention_mask = torch.stack(batch["attention_mask"][0]).permute(
            1, 0
        )  # .cuda()
        input_ids = torch.stack(batch["input_ids"][0]).permute(1, 0)  # .cuda()
        labels = batch["labels"].long()  # .cuda()
        output = biobert(
            attention_mask=attention_mask, input_ids=input_ids, labels=labels
        )

        predictions.append(
            torch.nn.functional.softmax(output[1], dim=1)[:, 1]
        )  # .cpu())
        val_labels.append(batch["labels"])  # .cpu())
        validation_loss.append(output[0].mean().item())

        combined_labels = torch.cat(val_labels).numpy()
        combined_predictions = torch.cat(predictions).detach().numpy()

        # AUROCo
        fpr, tpr, _ = roc_curve(combined_labels, combined_predictions)

        precision, recall, _ = precision_recall_curve(
            combined_labels, combined_predictions
        )
        current_model_auc = auc(recall, precision)

        data.append(
            {
                "prediction_edge": args.edge_type,
                "label_source": model_file.parents[0].stem,
                "AUPR": current_model_auc,
                "AUROC": auc(fpr, tpr),
                "dataset": "tune",
                "lf_num": lf_num,
            }
        )

    for idx, batch in tqdm.tqdm(enumerate(test_dataset_pt)):
        attention_mask = torch.stack(batch["attention_mask"][0]).permute(
            1, 0
        )  # .cuda()
        input_ids = torch.stack(batch["input_ids"][0]).permute(1, 0)  # .cuda()
        output = biobert(
            attention_mask=attention_mask,
            input_ids=input_ids,
            labels=batch["labels"].long(),  # .cuda()
        )
        predictions.append(
            torch.nn.functional.softmax(output[1], dim=1)[:, 1]
        )  # .cpu())
        val_labels.append(batch["labels"])  # .cpu())
        validation_loss.append(output[0].mean().item())

        combined_labels = torch.cat(val_labels).numpy()
        combined_predictions = torch.cat(predictions).detach().numpy()

        # AUROCo
        fpr, tpr, _ = roc_curve(combined_labels, combined_predictions)

        precision, recall, _ = precision_recall_curve(
            combined_labels, combined_predictions
        )
        current_model_auc = auc(recall, precision)

        data.append(
            {
                "prediction_edge": args.edge_type,
                "label_source": model_file.parents[0].stem,
                "AUPR": current_model_auc,
                "AUROC": auc(fpr, tpr),
                "dataset": "test",
                "lf_num": lf_num,
            }
        )

data_df = pd.DataFrame.from_records(data)
data_df.to_csv(
    f"{args.edge_type}/{args.edge_type}_total_lf_performance.tsv", sep="\t", index=False
)
