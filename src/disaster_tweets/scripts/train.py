"""Defines entrypoint for training."""
import argparse
import csv
import os.path
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import yaml
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, f1_score, precision_score, recall_score

from disaster_tweets.data import ClassifierDataModule
from disaster_tweets.model import TextClassifier
from disaster_tweets.training import EarlyStoppingWithMinEpochs


def main(config):
    """Run main function."""
    data_module = ClassifierDataModule(config)
    train_df = data_module.train_df
    class_counts = train_df["target"].value_counts()
    total_samples = len(train_df)
    class_weights = {cls: total_samples / count for cls, count in class_counts.items()}
    classes = list(class_weights.keys())
    model = TextClassifier(config, class_weights=[class_weights[cls] for cls in classes])

    early_stopping_with_min_epochs = EarlyStoppingWithMinEpochs(
        min_epochs=config["min_epochs"], patience=config["patience"], monitor="val_loss", mode="min", verbose=True
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=config["max_epochs"],
        default_root_dir=config["output_dir"],
        callbacks=[early_stopping_with_min_epochs],
        logger=True,
    )
    trainer.fit(model, data_module)

    # Predicting on test and train datasets
    val_predictions = trainer.predict(model, dataloaders=data_module.val_dataloader())

    # Calculate metrics
    pred = np.concatenate([batch.argmax(axis=-1) for batch in val_predictions])
    true_labels = data_module.validation_dataset.target
    accuracy = (pred == true_labels).mean()
    precision = precision_score(true_labels, pred)
    recall = recall_score(true_labels, pred)
    f1 = f1_score(true_labels, pred)

    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
    }

    # Save metrics to a file
    with open(config["output_dir"] + "/metrics.yaml", "w") as f:
        yaml.dump(metrics, f)

    # Confusion matrix for test dataset
    cm_val = confusion_matrix(true_labels, pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_val, display_labels=["Not disaster", "Disaster"])
    disp.plot()

    plt.title("Test Confusion Matrix")
    plt.savefig(config["output_dir"] + "/val_cm.png")

    # Plot train and validation losses
    plt.figure()
    losses = np.array(model.train_losses) / model.train_losses[0]
    plt.plot(losses, label="Train Loss")
    losses = np.array(model.val_losses) / model.val_losses[0]
    plt.plot(losses, label="Validation Loss")
    plt.title("Train vs Validation Loss")
    plt.legend()
    plt.savefig(config["output_dir"] + "/train_val_loss.png")

    # Saving misclassified samples for val dataset
    train = pd.read_csv(config["data_path"] + "/train.csv")
    misclassified = np.argmax(pred, axis=-1) != true_labels
    misclassified_idx = np.array(data_module.validation_dataset.df_index)[misclassified]
    misclassified_df = train.loc[misclassified_idx]
    misclassified_df.to_csv(config["output_dir"] + "/misclassified.csv", sep=",", index=False, quoting=csv.QUOTE_ALL)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True, help="Path to the yml configuration file.")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    os.makedirs(config["output_dir"], exist_ok=True)
    shutil.copy(args.config, os.path.join(config["output_dir"], "config.yml"))
    main(config)
