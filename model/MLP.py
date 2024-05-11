import torch
import torch.nn as nn

class MLP(torch.nn.Sequential):
    """From https://github.com/clabrugere/pytorch-scarf/blob/master/scarf/model.py"""

    def __init__(self, input_dim, hidden_dim, n_layers, dropout=0.0):
        layers = []
        in_dim = input_dim
        for _ in range(n_layers - 1):
            layers.append(torch.nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(torch.nn.Dropout(dropout))
            in_dim = hidden_dim

        layers.append(torch.nn.Linear(in_dim, hidden_dim))

        super().__init__(*layers)

class MLP2(torch.nn.Sequential):
    def __init__(self, input_dim, output_dim, n_layers, dropout=0.0):
        layers = []
        in_dim = input_dim
        for _ in range(n_layers - 1):
            hidden_dim = int(in_dim/4)
            if hidden_dim > output_dim:
                layers.append(torch.nn.Linear(in_dim, hidden_dim))
                layers.append(nn.ReLU(inplace=True))
                layers.append(torch.nn.Dropout(dropout))
                in_dim = hidden_dim
            else:
                n_layers = 1

        layers.append(torch.nn.Linear(in_dim, output_dim))

        super().__init__(*layers)