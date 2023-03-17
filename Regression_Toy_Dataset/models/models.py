from typing import Tuple, List, Any, Optional
import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT, EPOCH_OUTPUT
import torchmetrics


class SimpleFeedForwardNN(nn.Module):
    """Simple Feed-Forward Neural Network Class

    """
    def __init__(self, input_dim: int = 1, output_val_dim: int = 1):
        """Simple Feed-Forward Neural Network Class Constructor

        :param input_dim: Neural network input shape (dim-size), defaults to 1
        :type input_dim: int, optional
        :param output_val_dim: Neural netwrok output shape (dim-size), defaults to 1
        :type output_val_dim: int, optional
        """
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, 16)
        self.mean = nn.Linear(16, output_val_dim)

    def forward(self, x):
        
        x = self.fc1(x)
        x = F.relu(x)
        
        x = self.fc2(x)
        x = F.relu(x)
        
        output_mean = self.mean(x)

        return output_mean


class SimpleFeedForwardNNModule(pl.LightningModule):
    """Simple Feed-Forward Neural Network Lightning-Module
    
    :param optimizer_lr: _description_, defaults to 1e-4
    :type optimizer_lr: float, optional
    :param optimizer_weight_decay: _description_, defaults to 1e-5
    :type optimizer_weight_decay: float, optional
    :param max_nro_epochs: _description_, defaults to None
    :type max_nro_epochs: int, optional
    """     
    def __init__(self,
                 optimizer_lr: float = 1e-4,
                 optimizer_weight_decay: float = 1e-5,
                 max_nro_epochs: int = None) -> None:
        """Simple Feed-Forward Neural Network Lightning-Module Constructor

        :param optimizer_lr: _description_, defaults to 1e-4
        :type optimizer_lr: float, optional
        :param optimizer_weight_decay: _description_, defaults to 1e-5
        :type optimizer_weight_decay: float, optional
        :param max_nro_epochs: _description_, defaults to None
        :type max_nro_epochs: int, optional
        """
        super().__init__()
        self.optimizer_lr = optimizer_lr
        self.optimizer_weight_decay = optimizer_weight_decay
        self.max_nro_epochs = max_nro_epochs
        self.model = SimpleFeedForwardNN()
        self.loss_fn = self.get_loss_fn()
        self.save_hyperparameters()  # save model hyperparameters!
        
    def get_loss_fn(self) -> Any:
        """Get Loss Function

        :return: Loss function for Neural Network Training
        :rtype: Any
        """
        loss_fn = nn.MSELoss(reduction='mean')
        return loss_fn
    
    def configure_optimizers(self) -> Any:
        """Configures the PyTorch Optimizer

        :return: Optimizer
        :rtype: Any
        """
        # ToDo: Try different optimizers! optim={SGD, RMSProp, ...}
        optimizer = torch.optim.Adam(self.model.parameters(),  # specify your model (neural network) parameters (weights)
                                     lr=self.optimizer_lr,  # learning rate
                                     weight_decay=self.optimizer_weight_decay,  # L2 penalty regularizer
                                     eps=1e-7)  # adds numerical numerical stability (avoids division by 0)
        return [optimizer]
    
    def training_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
        sample, label = batch
        sample = torch.unsqueeze(sample, 1)  # expand dimensions
        label = torch.unsqueeze(label, 1)  # expand dimensions
        
        y_pred = self.model(sample)  # make a prediction (forward-pass)
        train_loss = self.loss_fn(y_pred, label)  # compute loss
        
        self.log_dict({"Train loss": train_loss}, on_step=False, on_epoch=True, prog_bar=True)  # write log
        
        return train_loss
    
    def validation_step(self, batch: Any, batch_idx: int) -> Optional[STEP_OUTPUT]:
        sample, label = batch
        sample = torch.unsqueeze(sample, 1)
        label = torch.unsqueeze(label, 1)
        
        y_pred = self.model(sample)
        valid_loss = self.loss_fn(y_pred, label)
        
        self.log_dict({"Validation loss": valid_loss}, on_step=False, on_epoch=True, prog_bar=True)
        return valid_loss
    
    def test_step(self, batch: Any, batch_idx: int) -> Optional[STEP_OUTPUT]:
        # ToDo: if required
        pass
    
    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        # ToDo: if required
        pass

