from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import pytorch_lightning as pl
from torch.fx.experimental.unification import variables

from lilt_fine_tune.data_module import DataModule
from lilt_fine_tune.Lit_module import LitModule
import torch
torch.cuda.empty_cache()
import gc
del variables

def build_trainer(labels, train_dataset, eval_dataset, checkpoint_filename):

    checkpoint_callback = ModelCheckpoint(
        dirpath="../../data/checkpoint", monitor="val_loss", mode="min", filename=checkpoint_filename)
    early_stop_callback = EarlyStopping(monitor="val_loss",
                                        patience=8,
                                        mode="min",
                                        verbose=False,
                                        strict=True)
    max_epochs = 16
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        default_root_dir="../../data/logs",
        accelerator="auto",
        devices="auto",
        logger=True,
        callbacks=[checkpoint_callback]
    )

    pl_model = LitModule(labels)
    pl_dl = DataModule(train_dataset=train_dataset, eval_dataset=eval_dataset)

    trainer.fit(pl_model, pl_dl)
    print(torch.cuda.memory_summary())
    return pl_model, pl_dl
