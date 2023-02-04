import pytorch_lightning as pl
from torch.utils.data import DataLoader


class DataModule(pl.LightningDataModule):

    def __init__(self, train_dataset, eval_dataset):
        super(DataModule, self).__init__()
        self.eval_dataset = eval_dataset
        self.train_dataset = train_dataset
        self.batch_size = 6

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                          shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.eval_dataset, batch_size=self.batch_size,
                          shuffle=False)
