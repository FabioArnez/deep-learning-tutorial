from typing import Tuple, List, Any, Optional
import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torchmetrics
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
import numpy as np
import matplotlib.pyplot as plt
import random
from icecream import ic


class ToyDataset1(Dataset):
    """Toy Dataset 1
    """
    def __init__(self,
                 num_samples=6001,
                 xt=None,
                 xt_range=[-1 , 1],
                 noise_coef=0.3):
        """Toy Dataset 1 Constructor

        :param num_samples: number of samples, defaults to 6001
        :type num_samples: int, optional
        :param xt: predifined x_t values, defaults to None
        :type xt: numpy.ndarray, optional
        :param xt_range: _description_, defaults to [-1 , 1]
        :type xt_range: list, optional
        :param noise_coef: _description_, defaults to 0.3
        :type noise_coef: float, optional
        """
        self.num_samples = num_samples
        self.xt_range = xt_range
        self.noise_coef = noise_coef
        if xt is None:
            self.xt = np.linspace(self.xt_range[0], self.xt_range[1], num_samples)
        else:
            self.xt = xt
        
        self.yt = np.power(self.xt, 3) + (self.noise_coef * np.random.uniform(0, 1, self.xt.shape[0]))
        
        self.xt = np.expand_dims(self.xt, axis=1)
        self.yt = np.expand_dims(self.yt, axis=1)

    def __len__(self):
        return len(self.xt)

    def __getitem__(self, index):
        x_val = self.xt[index]
        y_val = self.yt[index]
        # sample = torch.from_numpy(x_val)
        # label = torch.from_numpy(y_val)
        sample = torch.Tensor(x_val)
        label = torch.Tensor(y_val)
        return sample, label

    def plot_dataset(self):
        plt.scatter(self.xt, self.yt, s=2, label=r"$y = f(x^{3} + 0.3 * \mathbf{U}(0, 1))$")
        plt.ylabel("$y$")
        plt.xlabel("$x$")
        plt.legend()
        plt.show()


class ToyDataset2(Dataset):
    def __init__(self, num_samples=6001, xt=None):
        self.num_samples = num_samples
        if xt is None:
            self.xt = np.linspace(-6, 6, num_samples)
        else:
            self.xt = xt
        # mix prob values
        b = np.random.binomial(1, 0.5, num_samples)
        self.y_mu = (+0.1) * (self.xt ** 2) + (0.0 * self.xt) + 0
        self.y_sigma = b * (0.15 * np.power((1 + np.exp(-0.4 * self.xt)), -1)) + \
                       (1 - b) * (0.15 * np.power((1 + np.exp(0.4 * self.xt)), -1))
        self.yt = np.random.normal(self.y_mu, self.y_sigma)

    def __len__(self):
        return len(self.xt)

    def __getitem__(self, index):
        x_val = self.xt[index]
        y_val = self.yt[index]
        sample = torch.Tensor(x_val)
        label = torch.Tensor(y_val)
        return sample, label

    def plot_dataset(self):
        plt.scatter(self.xt, self.yt, s=1)
        plt.show()


class ToyDataset3(Dataset):
    def __init__(self, num_samples=6001, xt=None):
        self.num_samples = num_samples
        if xt is None:
            self.xt = np.linspace(-3, 3, num_samples)
        else:
            self.xt = xt
        # mix prob values
        # b = np.random.binomial(1, 0.5, num_samples)
        self.y_mu = 10 * np.sin(self.xt)
        self.y_sigma = 3.0 * np.power((1 + np.exp(-self.xt)), -1)
        self.yt = np.random.normal(self.y_mu, self.y_sigma)

    def __len__(self):
        return len(self.xt)

    def __getitem__(self, index):
        x_val = self.xt[index]
        y_val = self.yt[index]
        sample = torch.Tensor(x_val)
        label = torch.Tensor(y_val)
        return sample, label

    def plot_dataset(self):
        plt.scatter(self.xt, self.yt, s=1)
        plt.show()


class ToyDataset4(Dataset):
    def __init__(self, num_samples=6001, xt=None):
        self.num_samples = num_samples
        if xt is None:
            self.xt = np.linspace(-3, 3, num_samples)
        else:
            self.xt = xt
        # mix prob values
        b = np.random.binomial(1, 0.5, num_samples)
        self.y_mu = b * (7 * np.sin(self.xt)) + (1 - b) * (6 * np.cos(self.xt * np.pi / 2))
        self.y_sigma = 1.5 * np.power((1 + np.exp(-self.xt)), -1)
        self.yt = np.random.normal(self.y_mu, self.y_sigma)

    def __len__(self):
        return len(self.xt)

    def __getitem__(self, index):
        x_val = self.xt[index]
        y_val = self.yt[index]
        x_val = np.array(x_val)
        y_val = np.array(y_val)
        sample = torch.Tensor(x_val)
        label = torch.Tensor(y_val)

        return sample, label

    def plot_dataset(self):
        plt.scatter(self.xt, self.yt, s=1)
        plt.show()


class ToyDatasetModule(object):
    """Toy Datasets Module"""
    def __init__(self,
                 dataset_name: str,
                 num_samples: int = 6001,
                 xt_range: List = [-1 , 1],
                 noise_coef: float = 0.3,
                 batch_size: int = 32,
                 valid_size: float = 0.3,
                 num_workers: int = 10,
                 seed: int = 10):
        """
        Toy Dataset Module Constructor

        :param dataset_name: dataset, values = {"ToyDataset1", ..., "ToyDatasetx"}
        :type dataset_name: str
        :param num_samples: number of samples in the dataset, defaults to 6001
        :type num_samples: int, optional
        :param xt_range: x interval for the generating function f(x) , defaults to [-1 , 1]
        :type xt_range: List, optional
        :param noise_coef: Noise coefficient in generating Function f(x), defaults to 0.3
        :type noise_coef: float, optional
        :param batch_size: number of samples per batch, defaults to 32
        :type batch_size: int, optional
        :param valid_size:  size (%) of the validation set, defaults to 0.3
        :type valid_size: float, optional
        :param num_workers: number of workers, defaults to 10
        :type num_workers: int, optional
        :param seed: random seed for dataset shuffling, defaults to 10
        :type seed: int, optional
        :raises ValueError: Dataset name error
        """
        self.dataset_name = dataset_name
        self.num_samples = num_samples
        self.xt_range = xt_range
        self.noise_coef = noise_coef
        self.batch_size = batch_size
        self.valid_size = valid_size
        self.num_workers = num_workers
        self.seed = seed

        if self.dataset_name == "ToyDataset1":  # dataset for train&val
            self.dataset = ToyDataset1(num_samples=self.num_samples, xt_range=self.xt_range, noise_coef=self.noise_coef)
        else:
            raise ValueError("Not a valid dataset name value!")

        self.dataset_len = len(self.dataset)
        self.train_sampler = None
        self.valid_sampler = None
        self.train_dataloader = None
        self.valid_dataloader = None
        print("Dataset name: ", self.dataset_name)
        print("Dataset length: ", self.dataset_len)
        print("Dataset random seed: ", self.seed)

    def setup(self) -> None:
        """
        Dataset Module Setup
        """
        indices = list(range(self.dataset_len))
        random.seed(self.seed)
        random.shuffle(indices)
        split = int(np.floor(self.valid_size * self.dataset_len))
        train_idx, valid_idx = indices[split:], indices[:split]
        self.train_sampler = SubsetRandomSampler(train_idx)
        self.valid_sampler = SubsetRandomSampler(valid_idx)
        pass

    def get_train_dataloader(self) -> DataLoader:
        """
        Return  train dataloader
        """
        self.train_dataloader = DataLoader(self.dataset,
                                           batch_size=self.batch_size,
                                           sampler=self.train_sampler,
                                           num_workers=self.num_workers)
        return self.train_dataloader

    def get_valid_dataloader(self) -> DataLoader:
        """
        Return valid dataloader
        """
        valid_dataloader = DataLoader(self.dataset,
                                           batch_size=self.batch_size,
                                           sampler=self.valid_sampler,
                                           num_workers=self.num_workers)
        return valid_dataloader


class ToyDatasetPlModule(LightningDataModule):
    """Toy Datasets PyTorch-Lightning Module"""
    def __init__(self,
                 dataset_name: str,
                 num_samples: int = 6001,
                 xt_range: List = [-1 , 1],
                 noise_coef: float = 0.3,
                 batch_size: int = 32,
                 valid_size: float = 0.3,
                 num_workers: int = 10,
                 seed: int = 10):
        """
        Toy Dataset Module Constructor

        :param dataset_name: dataset, values = {"ToyDataset1", ..., "ToyDatasetx"}
        :type dataset_name: str
        :param num_samples: number of samples in the dataset, defaults to 6001
        :type num_samples: int, optional
        :param xt_range: x interval for the generating function f(x) , defaults to [-1 , 1]
        :type xt_range: List, optional
        :param noise_coef: Noise coefficient in generating Function f(x), defaults to 0.3
        :type noise_coef: float, optional
        :param batch_size: number of samples per batch, defaults to 32
        :type batch_size: int, optional
        :param valid_size:  size (%) of the validation set, defaults to 0.3
        :type valid_size: float, optional
        :param num_workers: number of workers, defaults to 10
        :type num_workers: int, optional
        :param seed: random seed for dataset shuffling, defaults to 10
        :type seed: int, optional
        :raises ValueError: Dataset name error
        """
        super().__init__()
        self.dataset_name = dataset_name
        self.num_samples = num_samples
        self.xt_range = xt_range
        self.noise_coef = noise_coef
        self.batch_size = batch_size
        self.valid_size = valid_size
        self.num_workers = num_workers
        self.seed = seed

        if self.dataset_name == "ToyDataset1":  # dataset for train&val
            self.dataset = ToyDataset1(num_samples=self.num_samples, xt_range=self.xt_range, noise_coef=self.noise_coef)
        else:
            raise ValueError("Not a valid dataset name value!")

        self.dataset_len = len(self.dataset)
        self.train_sampler = None
        self.valid_sampler = None

        print("Dataset name: ", self.dataset_name)
        print("Dataset length: ", self.dataset_len)
        print("Dataset random seed: ", self.seed)
        self.save_hyperparameters()
        
    def setup(self, stage: Optional[str] = None) -> None:
        """
        Dataset Module Setup
        """
        if stage == 'fit':
            indices = list(range(self.dataset_len))
            random.seed(self.seed)
            random.shuffle(indices)
            split = int(np.floor(self.valid_size * self.dataset_len))
            train_idx, valid_idx = indices[split:], indices[:split]
            self.train_sampler = SubsetRandomSampler(train_idx)
            self.valid_sampler = SubsetRandomSampler(valid_idx)
            print("samplers ready!")

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        """
        Get train dataloader
        """
        train_dataloader = DataLoader(self.dataset,
                                      batch_size=self.batch_size,
                                      sampler=self.train_sampler,
                                      num_workers=self.num_workers)
        return train_dataloader

    def val_dataloader(self) -> EVAL_DATALOADERS:
        """
        get valid dataloader
        """
        valid_dataloader = DataLoader(self.dataset,
                                      batch_size=self.batch_size,
                                      sampler=self.valid_sampler,
                                      num_workers=self.num_workers)
        return valid_dataloader
