
import evaluate
import pytorch_lightning as pl
import torch.nn.functional as F


import logging

from transformers import AdamW

from lilt_fine_tune.prediction_utility import get_labels
from lilt_fine_tune.ser_classification import SERClassification

# configure logging at the root level of Lightning
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

# # configure logging on module level, redirect to file
# logger = logging.getLogger("pytorch_lightning.core")
# logger.addHandler(logging.FileHandler("core.log"))
class LitModule(pl.LightningModule):

    def __init__(self, labels, lr=5e-5):

        super(LitModule, self).__init__()
        self.save_hyperparameters()

        self.model = SERClassification(num_classes=len(labels))

        ## Metrics
        self.train_metric = evaluate.load("seqeval")
        self.val_metric = evaluate.load("seqeval")
        self.labels = labels
        ## Parameters
        self.lr = lr

    def forward(self, batch):
        return self.model(batch)

    def configure_optimizers(self):
        return AdamW(self.model.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        ## Forward Propagatipn
        outputs = self.forward(batch)

        ## Predictions and adding the metrics
        predictions = outputs['logits'].argmax(-1)
        true_predictions, true_labels = get_labels(self.labels, predictions, batch["labels"])
        self.train_metric.add_batch(references=true_labels, predictions=true_predictions)

        ## Logging Purpose
        results = self.train_metric.compute()
        loss = F.cross_entropy(outputs['logits'].view(-1, len(self.labels)), batch["labels"].view(-1))
        logging.debug(f"train_loss{loss.item()}")
        self.log("train_loss", loss.item(), prog_bar=True)
        self.log("train_overall_fl", results["overall_f1"], prog_bar=True)
        self.log("train_overall_recall", results["overall_recall"], prog_bar=True)
        self.log("train_overall_precision", results["overall_precision"], prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self.forward(batch)
        predictions = outputs['logits'].argmax(-1)
        true_predictions, true_labels = get_labels(self.labels, predictions, batch["labels"])
        self.val_metric.add_batch(references=true_labels, predictions=true_predictions)

        ## Logging Purpose
        results = self.val_metric.compute()
        loss = F.cross_entropy(outputs['logits'].view(-1, len(self.labels)), batch["labels"].view(-1))

        self.log("val_loss", loss.item(), prog_bar=True)
        self.log("val_overall_fl", results["overall_f1"], prog_bar=True)
        self.log("val_overall_recall", results["overall_recall"], prog_bar=True)
        self.log("val_overall_precision", results["overall_precision"], prog_bar=True)

        return loss