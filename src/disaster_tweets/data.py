"""Package for data related code."""
import pandas as pd
import pytorch_lightning as pl
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer


class ClassifierDataModule(pl.LightningDataModule):
    """Data module containing train, val and test datasets."""

    def __init__(self, config: dict):
        """
        Create train, val and test datasets.

        Parameters
        ----------
        config : dict
            Paths and settings for data loading, model selection, and training parameters.
        """
        super().__init__()
        self.config = config
        # Load data
        self.train_df = pd.read_csv(self.config["data_path"] + "/train.csv")
        self.test_df = pd.read_csv(self.config["data_path"] + "/test.csv")

        # Split train data into train and validation sets
        train, val = train_test_split(self.train_df, test_size=self.config["validation_split"])

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
            self.test_df["text"], None, self.tokenizer, self.config["max_seq_length"]
        )

    def train_dataloader(self):
        """Generate the data loader for the training dataset."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.config["batch_size"],
            shuffle=True,
            num_workers=self.config["num_workers"],
        )

    def val_dataloader(self):
        """Generate the data loader for the validation dataset."""
        return DataLoader(
            self.validation_dataset,
            batch_size=self.config["batch_size"],
            shuffle=False,
            num_workers=self.config["num_workers"],
        )

    def test_dataloader(self):
        """Generate the data loader for the test dataset."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.config["batch_size"],
            shuffle=False,
            num_workers=self.config["num_workers"],
        )


class TextClassificationDataset(Dataset):
    """Dataset module for text classification."""

    def __init__(self, text: pd.Series, target: pd.Series | None, tokenizer: BertTokenizer, max_length: int):
        """Create single dataset with targets."""
        self.text = list(text)
        self.target = list(target) if target is not None else None
        self.df_index = text.index.tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        """Get size of the dataset."""
        return len(self.text)

    def __getitem__(self, idx):
        """Fetch a single sample from the dataset."""
        text_encoded = self.tokenizer.encode_plus(
            self.text[idx], max_length=self.max_length, padding="max_length", truncation=True
        )
        item = {
            "input_ids": torch.tensor(text_encoded["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(text_encoded["attention_mask"], dtype=torch.long),
        }
        if self.target is not None:
            item["labels"] = torch.tensor(self.target[idx], dtype=torch.long)
        return item
