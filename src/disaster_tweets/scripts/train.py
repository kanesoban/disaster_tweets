import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import yaml
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import AdamW, BertForSequenceClassification, BertTokenizer


class TextClassificationDataset(Dataset):
    def __init__(self, text, target, tokenizer, max_length):
        self.text = text
        self.target = target
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        text_encoded = self.tokenizer.encode_plus(
            self.text[idx], max_length=self.max_length, padding="max_length", truncation=True
        )
        return {
            "input_ids": torch.tensor(text_encoded["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(text_encoded["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(self.target[idx], dtype=torch.long),
        }


class TextClassifier(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.model = BertForSequenceClassification.from_pretrained(config["bert_model"])
        self.tokenizer = BertTokenizer.from_pretrained(config["bert_model"])
        self.loss = torch.nn.CrossEntropyLoss()
        self.config = config

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return output.logits

    def training_step(self, batch, batch_idx):
        logits = self(batch["input_ids"], batch["attention_mask"])
        loss = self.loss(logits, batch["labels"])
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self(batch["input_ids"], batch["attention_mask"])
        loss = self.loss(logits, batch["labels"])
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.config["learning_rate"])


class ClassifierDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def setup(self, stage=None):
        # Load data
        train = pd.read_csv(self.config["data_path"] + "/train.csv")
        test = pd.read_csv(self.config["data_path"] + "/test.csv")

        # Split train data into train and validation sets
        train, val = train_test_split(train, test_size=self.config["validation_split"])

        # Define tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(self.config["bert_model"])

        # Create datasets
        self.train_dataset = TextClassificationDataset(
            train["text"], train["target"], self.tokenizer, self.config["max_seq_length"]
        )

        self.validation_dataset = TextClassificationDataset(
            val["text"], val["target"], self.tokenizer, self.config["max_seq_length"]
        )
        self.test_dataset = TextClassificationDataset(
            test["text"], test["target"], self.tokenizer, self.config["max_seq_length"]
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config["batch_size"],
            shuffle=True,
            num_workers=self.config["num_workers"],
        )

    def val_dataloader(self):
        return DataLoader(
            self.validation_dataset,
            batch_size=self.config["batch_size"],
            shuffle=False,
            num_workers=self.config["num_workers"],
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.config["batch_size"],
            shuffle=False,
            num_workers=self.config["num_workers"],
        )


def main(config):
    model = TextClassifier(config)
    data_module = ClassifierDataModule(config)

    trainer = pl.Trainer(gpus=config["gpus"], max_epochs=config["num_epochs"], default_root_dir=config["output_dir"])
    trainer.fit(model, data_module)

    # Predicting on test and train datasets
    test_predictions = trainer.predict(model, datamodule=data_module.test_dataloader())
    train_predictions = trainer.predict(model, datamodule=data_module.train_dataloader())

    # Confusion matrix for test dataset
    cm_test = confusion_matrix(data_module.test_dataset.target, np.argmax(test_predictions, axis=-1))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_test, display_labels=["NEG", "POS"])
    disp.plot()

    plt.title("Test Confusion Matrix")
    plt.savefig(config["output_dir"] + "/test_cm.png")

    # Plot train and validation losses
    plt.figure()
    plt.plot(trainer.logged_metrics["train_loss"], label="Train Loss")
    plt.plot(trainer.logged_metrics["val_loss"], label="Validation Loss")
    plt.title("Train vs Validation Loss")
    plt.legend()
    plt.savefig(config["output_dir"] + "/train_val_loss.png")

    # Saving misclassified samples for train dataset
    misclassified = np.argmax(train_predictions, axis=-1) != data_module.train_dataset.target
    misclassified_df = data_module.train_dataset.iloc[misclassified]
    misclassified_df.to_csv(config["output_dir"] + "/misclassified.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True, help="Path to the yml configuration file.")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    main(config)
