from typing import Optional, Any
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.callbacks import RichProgressBar
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBarTheme
from models import LeNetModule
from datasets import GtsrbMinimalisticModule  # PyTorch-Lightning DataModule


def main():
    # General Settings:
    random_seed_everything = 10
    rich_progress_bar = True
    dataset_path = "/your/path/to/dataset"
    epochs = 40  # number of epochs for training
    
    # seed everything:
    pl.seed_everything(random_seed_everything)
    datamodule_seed = 100
    
    # Calbacks:
    checkpoint_callback = ModelCheckpoint(monitor="Validation accuracy",
                                          mode='max',
                                          every_n_epochs=1,
                                          save_top_k=2,
                                          save_last=True,
                                          save_on_train_epoch_end=False)
    
    if rich_progress_bar:  # fancy aesthetic progress bar
        progress_bar = RichProgressBar(theme=RichProgressBarTheme(description="green_yellow",
                                                            progress_bar="green1",
                                                            progress_bar_finished="green1",
                                                            batch_progress="green_yellow",
                                                            time="grey82",
                                                            processing_speed="grey82",
                                                            metrics="grey82"))
    else:  # normal aesthetic progress bar
        progress_bar = TQDMProgressBar(refresh_rate=10)     
        
    
    # Data Module
    data_module = GtsrbMinimalisticModule(data_path=dataset_path,
                                          batch_size=32,
                                          shuffle=True)
    data_module.setup(stage="fit")
    data_module.setup(stage="validate")
    # data_module.setup(stage="test")
    
    # Neural Network Module
    model_module = LeNetModule(num_classes=18,
                               loss_fn='nll',
                               optimizer_lr=1e-3,
                               optimizer_weight_decay=1e-4,
                               max_nro_epochs=epochs)
    
    # PyTorch-Lightning Trainer
    trainer = pl.Trainer(accelerator='gpu',  # specify your hardware device for training
                         devices=-1,  # choose the number of hardware devices, -1 -> all available
                         max_epochs=epochs,  # specify the number of epochs
                         callbacks=[progress_bar, checkpoint_callback])  #  add the callbacks
    # Fit Trainer
    trainer.fit(model=model_module, datamodule=data_module)  # fit a model!


if __name__ == "__main__":
    main()

