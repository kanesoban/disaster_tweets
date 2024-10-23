"""Defines model used for classification."""

import pytorch_lightning as pl
import torch
from transformers import AdamW, AutoModelForSequenceClassification, AutoTokenizer


class TextClassifier(pl.LightningModule):
    """Dataset module for text classification."""

    def __init__(self, config, class_weights=None):
        """Create model and tokenizer."""
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(config["model"])
        self.tokenizer = AutoTokenizer.from_pretrained(config["model"])
        self.loss = torch.nn.CrossEntropyLoss(weight=torch.tensor(class_weights))
        self.config = config
        self.train_losses = []
        self.val_losses = []
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.set_train_head_only(config.get("train_only_head", True))

    def set_train_head_only(self, train_only_head):
        """Optionally freeze most model weights before training."""
        if train_only_head:
            for name, param in self.model.named_parameters():
                if "classifier" not in name:  # Keep the classifier head trainable
                    param.requires_grad = False
        else:
            for param in self.model.parameters():
                param.requires_grad = True

    def forward(self, input_ids, attention_mask):
        """Calculate logits of the model's prediction."""
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return output.logits

    def training_step(self, batch, batch_idx):
        """Calculate loss for a single batch."""
        logits = self(batch["input_ids"], batch["attention_mask"])
        loss = self.loss(logits, batch["labels"])
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.training_step_outputs.append(loss)
        return loss

    def validation_step(self, batch, batch_idx):
        """Calculate loss for a single batch."""
        logits = self(batch["input_ids"], batch["attention_mask"])
        loss = self.loss(logits, batch["labels"])
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.validation_step_outputs.append(loss)
        return loss

    def on_train_epoch_end(self):
        """Log training loss at the end of each epoch."""
        avg_train_loss = torch.stack(self.training_step_outputs).mean().item()
        self.train_losses.append(avg_train_loss)
        self.training_step_outputs.clear()  # frees memory

    def on_validation_epoch_end(self):
        """Log validation loss at the end of each epoch."""
        avg_val_loss = torch.stack(self.validation_step_outputs).mean().item()
        self.val_losses.append(avg_val_loss)
        self.validation_step_outputs.clear()  # frees memory

    def predict_step(self, batch, batch_idx):
        """Predict on a single batch."""
        return self.forward(batch["input_ids"], batch["attention_mask"])

    def configure_optimizers(self):
        """Configure optimizer for training."""
        return AdamW(self.parameters(), lr=self.config["learning_rate"])
