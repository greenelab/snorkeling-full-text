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

# # Train the Discriminator Model

# This notebook is not designed to be executed. Only to hold code in a similar format. Due to my lab computer not have a GPU the majority of training is on UPenn's computer cluster.

# +
import argparse
from pathlib import Path
import re

from datasets import Dataset
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
from sklearn.metrics import auc, precision_recall_curve, roc_curve
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import tqdm
# -

LF_SAMPLE_SIZE = 3
NUM_EPOCHS = 10

# +
parser = argparse.ArgumentParser(
    description="Train BioBert as Discriminator Model. Use these two commands to run the model."
)
parser.add_argument(
    "--training_marginals_folder", help="The folder to access the training models."
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

    training_file = "DaG/training_sen/train_dg_abstract_encoded_lemmas.tsv"
    training_labels_folder = args.training_marginals_folder

    entity_replace_one = "DISEASE_ENTITY"
    one_replace = "@DISEASE$"
    entity_replace_two = "GENE_ENTITY"
    two_replace = "@GENE$"

if args.edge_type.lower() == "ctd":
    edge_prediction = "CtD"
    curated_label = "curated_ctd"

    validation_file = "CtD/training_sen/cd_dev_test_encoded_lemmas.tsv"

    training_file = "CtD/training_sen/train_cd_abstract_encoded_lemmas.tsv"
    training_labels_folder = args.training_marginals_folder

    entity_replace_one = "COMPOUND_ENTITY"
    one_replace = "@CHEMICAL$"
    entity_replace_two = "DISEASE_ENTITY"
    two_replace = "@DISEASE$"

if args.edge_type.lower() == "cbg":
    edge_prediction = "CbG"
    curated_label = "curated_cbg"

    validation_file = "CbG/training_sen/cg_dev_test_encoded_lemmas.tsv"

    training_file = "CbG/training_sen/train_cg_abstract_encoded_lemmas.tsv"
    training_labels_folder = args.training_marginals_folder

    entity_replace_one = "COMPOUND_ENTITY"
    one_replace = "@CHEMICAL$"
    entity_replace_two = "GENE_ENTITY"
    two_replace = "@GENE$"

if args.edge_type.lower() == "gig":
    edge_prediction = "GiG"
    curated_label = "curated_gig"

    validation_file = "GiG/training_sen/gg_dev_test_encoded_lemmas.tsv"

    training_file = "GiG/training_sen/train_gg_abstract_encoded_lemmas.tsv"
    training_labels_folder = args.training_marginals_folder

    entity_replace_one = "GENE1_ENTITY"
    one_replace = "@GENE$"
    entity_replace_two = "GENE2_ENTITY"
    two_replace = "@GENE$"
# -

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "../biobert-base-cased-v1.1", local_files_only=True
)

## Validation
validation_data = pd.read_csv(validation_file, sep="\t").rename(
    index=str, columns={"split": "dataset", curated_label: "labels"}
)
dev_split_id = validation_data.dataset.min()

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

validation_dataset_pt = DataLoader(validation_dataset, batch_size=50)

# +
## Training
if edge_prediction == "GiG":
    training_data = pd.read_csv(training_file, sep="\t").assign(
        length=lambda x: x.parsed_lemmas.apply(lambda y: y.count("|")),
        entity_length=lambda x: x.parsed_lemmas.apply(
            lambda y: y.split("|").index(entity_replace_one)
            - y.split("|").index(entity_replace_two)
            if "GENE2_ENTITY" in y
            else 0
        ),
    )
else:
    training_data = pd.read_csv(training_file, sep="\t").assign(
        length=lambda x: x.parsed_lemmas.apply(lambda y: y.count("|"))
    )

marginal_files = sorted(list(Path(training_labels_folder).rglob("*tsv*")))
# -

for training_marginals in marginal_files:
    training_marginals_df = pd.read_csv(str(training_marginals), sep="\t")

    np.random.seed(100)
    unique_lf_num = sorted(training_marginals_df.lf_num.unique())
    lf_size = list(range(training_marginals_df.iteration.max() + 1))
    lf_sample_selection = np.random.choice(lf_size, size=LF_SAMPLE_SIZE)
    print(lf_sample_selection)

    match = re.search(
        f"(\w+)_predicts_{edge_prediction}",  # noqa: W605
        str(training_marginals),
        flags=re.I,  # noqa: W605
    )

    if match is None:
        edge_source = f"{edge_prediction}_baseline"

    else:
        edge_source = match.group(1)
        edge_source = edge_source[0:2].capitalize() + edge_source[2].upper()

    if not Path(f"{edge_prediction}/{edge_source}").exists():
        Path(f"{edge_prediction}/{edge_source}").mkdir(exist_ok=True, parents=True)

    for lf_group in unique_lf_num:
        print(lf_group)

        for accepted_iteration in lf_sample_selection:

            if Path(
                f"{edge_prediction}/{edge_source}/biobert_{lf_group}_{accepted_iteration}.model"
            ).exists():
                continue

            filtered_group_marginal_df = training_marginals_df.query(
                f"iteration=={accepted_iteration}&lf_num=={lf_group}"
            )

            # grab the marginals and set up the dataset loader
            # filtered_group_marginal_df.columns = sorted(list(filtered_group_marginal_df.columns))
            training_marginals_mapper = dict(
                zip(
                    filtered_group_marginal_df["candidate"],
                    filtered_group_marginal_df["positive_marginals"].apply(
                        lambda x: 0 if x <= 0.5 else 1
                    ),
                )
            )

            if edge_prediction == "GiG":
                training_dataset = Dataset.from_pandas(
                    training_data.query(
                        f"candidate_id in {filtered_group_marginal_df.candidate.tolist()}"
                    )
                    .query("length <= 100")
                    .query("abs(entity_length) >= 7")[["candidate_id", "parsed_lemmas"]]
                )
            else:
                training_dataset = Dataset.from_pandas(
                    training_data.query(
                        f"candidate_id in {filtered_group_marginal_df.candidate.tolist()}"
                    ).query("length <= 100")[["candidate_id", "parsed_lemmas"]]
                )

            training_dataset = training_dataset.map(
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
                remove_columns=["parsed_lemmas", "__index_level_0__"],
            )

            training_dataset = training_dataset.map(
                lambda x: {"labels": training_marginals_mapper[x["candidate_id"]]},
                remove_columns=["candidate_id"],
            )

            # here is the loader
            # 128
            training_dataset_pt = DataLoader(
                training_dataset, batch_size=256, shuffle=True
            )

            # Load the Model
            biobert = AutoModelForSequenceClassification.from_pretrained(
                "../biobert-base-cased-v1.1", local_files_only=True, num_labels=2
            )

            # Freeze the body of the model
            for param in biobert.bert.parameters():
                param.requires_grad = False

            biobert = torch.nn.DataParallel(biobert)
            biobert = biobert.cuda()
            num_epochs = NUM_EPOCHS
            learning_rate = 1e-3
            optim = AdamW(biobert.parameters(), lr=learning_rate)
            num_training_steps = num_epochs * len(training_dataset_pt)

            # Load the writer
            writer = SummaryWriter(f"{edge_prediction}/{edge_source}/{lf_group}")

            num_steps = 0

            # Run the training
            for epoch in range(num_epochs + 1):

                if epoch > 0 or True:
                    biobert.train()
                    for batch in tqdm.tqdm(training_dataset_pt):
                        # batch.cuda()
                        optim.zero_grad()
                        attention_mask = (
                            torch.stack(batch["attention_mask"][0]).permute(1, 0).cuda()
                        )
                        input_ids = (
                            torch.stack(batch["input_ids"][0], dim=0)
                            .permute(1, 0)
                            .cuda()
                        )
                        output = biobert(
                            attention_mask=attention_mask,
                            input_ids=input_ids,
                            labels=batch["labels"].cuda(),
                        )

                        loss = output[0].mean()
                        writer.add_scalar("Loss/train", loss.item(), num_steps)

                        loss.backward()

                        ## Get the gradients for each model
                        for name, param in biobert.named_parameters():
                            if (param.requires_grad) and ("bias" not in name):
                                writer.add_histogram(
                                    f"Layer/{name}",
                                    param.grad.abs().mean().item(),
                                    num_steps,
                                )

                        optim.step()
                        num_steps += 1

                biobert.eval()
                validation_loss = []
                predictions = []
                val_labels = []

                for batch in validation_dataset_pt:
                    attention_mask = (
                        torch.stack(batch["attention_mask"][0]).permute(1, 0).cuda()
                    )
                    input_ids = torch.stack(batch["input_ids"][0]).permute(1, 0).cuda()
                    output = biobert(
                        attention_mask=attention_mask,
                        input_ids=input_ids,
                        labels=batch["labels"].long().cuda(),
                    )

                    predictions.append(
                        torch.nn.functional.softmax(output[1], dim=1)[:, 1].cpu()
                    )
                    val_labels.append(batch["labels"].cpu())

                    validation_loss.append(output[0].mean().item())

                combined_labels = torch.cat(val_labels).numpy()
                combined_predictions = torch.cat(predictions).detach().numpy()

                # Tuning dataset loss
                writer.add_scalar("Loss/tune", np.mean(validation_loss), num_steps)

                # AUROCo
                fpr, tpr, _ = roc_curve(combined_labels, combined_predictions)
                writer.add_scalar("Eval/AUROC", auc(fpr, tpr), num_steps, num_steps)

                # AUPRC/PRC
                writer.add_pr_curve(
                    "Eval/PRC", combined_labels, combined_predictions, num_steps
                )

                precision, recall, _ = precision_recall_curve(
                    combined_labels, combined_predictions
                )
                current_model_auc = auc(recall, precision)
                writer.add_scalar("Eval/AUPR", current_model_auc, num_steps)

            # Training finished for model save last predictions
            biobert.module.save_pretrained(
                f"{edge_prediction}/{edge_source}/biobert_{lf_group}_{accepted_iteration}.model"
            )
