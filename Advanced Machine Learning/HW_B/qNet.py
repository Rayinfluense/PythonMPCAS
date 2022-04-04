import torch
import torch.nn as nn
import torch.nn.functional as F

class QNet(nn.Module):
    def __init__(self, gameboard, hidden_layer_width):
        super(QNet, self).__init__()

        # n_cols*n_rows number of binary inputs, representing the game board state, times number of tiles.
        self.fc1 = nn.Linear(gameboard.N_row * gameboard.N_col + 1, hidden_layer_width)

        self.fc2 = nn.Linear(hidden_layer_width, hidden_layer_width)

        self.fc3 = nn.Linear(hidden_layer_width, hidden_layer_width)

        self.fc4 = nn.Linear(hidden_layer_width,
                             gameboard.N_col * 4)  # Number of positions and rotations to put the tiles.

    def forward(self, x):
        # Defines how to feed input through the network, including activation functions.
        #x = x.to(torch.float32)
        x = F.relu(self.fc1(x))

        x = F.relu(self.fc2(x))

        x = F.relu(self.fc3(x))

        x = self.fc4(x)

        return x
