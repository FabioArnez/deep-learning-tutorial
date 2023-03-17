import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT, EPOCH_OUTPUT
import torchmetrics



class SimpleProbNN(nn.Module):
    def __init__(self, input_dim: int = 1, output_val_dim: int = 1):
        """
        Simple Probabilistic Neural Network
        :param input_dim:
        :param output_val_dim:
        """
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3_mean = nn.Linear(32, 16)
        self.fc3_logvar = nn.Linear(32, 16)
        self.mean = nn.Linear(16, output_val_dim)
        self.logvar = nn.Linear(16, output_val_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x_mean = self.fc3_mean(x)
        x_mean = F.relu(x_mean)
        x_logvar = self.fc3_logvar(x)
        x_logvar = F.relu(x_logvar)
        output_mean = self.mean(x_mean)
        output_logvar = self.logvar(x_logvar)

        return output_mean, output_logvar


class SimpleMdnNN(nn.Module):

    def __init__(self, input_dim: int = 1, k_gaussians: int = 3):
        super(SimpleMdnNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.pi = nn.Linear(16, k_gaussians)
        self.mu = nn.Linear(16, k_gaussians)
        self.sigma = nn.Linear(16, k_gaussians)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = torch.tanh(x)

        pi_logits = self.pi(x)
        sigma_logits = self.sigma(x)
        mu = self.mu(x)

        pi = F.softmax(pi_logits, dim=1)
        sigma = torch.exp(sigma_logits)

        return pi, mu, sigma


class SimpleRVInputProbNN(nn.Module):
    def __init__(self, input_dim: int = 2, output_val_dim: int = 1):
        """
        Simple Random Variable Input Probabilistic Neural Network
        :param input_dim: input value dimensions
        :param output_val_dim: output value dimensions
        """
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3_mean = nn.Linear(32, 16)
        self.fc3_logvar = nn.Linear(32, 16)
        self.mean = nn.Linear(16, output_val_dim)
        self.logvar = nn.Linear(16, output_val_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x_mean = self.fc3_mean(x)
        x_mean = F.relu(x_mean)
        x_logvar = self.fc3_logvar(x)
        x_logvar = F.relu(x_logvar)
        output_mean = self.mean(x_mean)
        output_logvar = self.logvar(x_logvar)

        return output_mean, output_logvar


class EnsembleProbNN(nn.Module):
    def __init__(self, num_ensemble=5):
        super().__init__()
        self.ProbNN_0 = SimpleProbNN()
        self.ProbNN_1 = SimpleProbNN()
        self.ProbNN_2 = SimpleProbNN()
        self.ProbNN_3 = SimpleProbNN()
        self.ProbNN_4 = SimpleProbNN()

    def forward(self, x):
        mean0, logvar0 = self.ProbNN_0(x)
        mean1, logvar1 = self.ProbNN_1(x)
        mean2, logvar2 = self.ProbNN_2(x)
        mean3, logvar3 = self.ProbNN_3(x)
        mean4, logvar4 = self.ProbNN_4(x)

        ensemble_mean_samples = torch.cat((mean0,
                                           mean1,
                                           mean2,
                                           mean3,
                                           mean4))

        ensemble_mean = torch.mean(ensemble_mean_samples, 0, keepdim=True)

        ensemble_variance = torch.mean(torch.cat((torch.exp(logvar0) + torch.pow(mean0, 2),
                                                  torch.exp(logvar1) + torch.pow(mean1, 2),
                                                  torch.exp(logvar2) + torch.pow(mean2, 2),
                                                  torch.exp(logvar3) + torch.pow(mean3, 2),
                                                  torch.exp(logvar4) + torch.pow(mean4, 2))),
                                       0, keepdim=True) - torch.pow(ensemble_mean, 2)

        return ensemble_mean, ensemble_variance, ensemble_mean_samples


class EnsembleLVInputProbNN(nn.Module):
    def __init__(self, num_ensemble=5):
        super().__init__()
        self.ProbNN_0 = SimpleRVInputProbNN()
        self.ProbNN_1 = SimpleRVInputProbNN()
        self.ProbNN_2 = SimpleRVInputProbNN()
        self.ProbNN_3 = SimpleRVInputProbNN()
        self.ProbNN_4 = SimpleRVInputProbNN()

    def forward(self, x):
        mean0, logvar0 = self.ProbNN_0(x)
        mean1, logvar1 = self.ProbNN_1(x)
        mean2, logvar2 = self.ProbNN_2(x)
        mean3, logvar3 = self.ProbNN_3(x)
        mean4, logvar4 = self.ProbNN_4(x)

        ensemble_mean_samples = torch.cat((mean0,
                                           mean1,
                                           mean2,
                                           mean3,
                                           mean4))

        ensemble_mean = torch.mean(ensemble_mean_samples, 0, keepdim=True)

        ensemble_variance = torch.mean(torch.cat((torch.exp(logvar0) + torch.pow(mean0, 2),
                                                  torch.exp(logvar1) + torch.pow(mean1, 2),
                                                  torch.exp(logvar2) + torch.pow(mean2, 2),
                                                  torch.exp(logvar3) + torch.pow(mean3, 2),
                                                  torch.exp(logvar4) + torch.pow(mean4, 2))),
                                       0, keepdim=True) - torch.pow(ensemble_mean, 2)

        return ensemble_mean, ensemble_variance, ensemble_mean_samples