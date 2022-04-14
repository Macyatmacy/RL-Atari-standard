import torch
import torch.nn as nn

import numpy as np


class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)


class DQN1(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)


class QNetworkAtari(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(QNetworkAtari, self).__init__()
        # CNN: 4 * 84 * 84 -> 32 * 20 * 20 -> 64 * 9 * 9 -> 64 * 7 * 7
        self.conv_1 = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4),
            nn.ReLU(True),
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(True),
        )
        self.conv_3 = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(True),
        )

        # Fully connected: 64 * 7 * 7 = 3136 -> 512 -> num_action
        self.fc_1 = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(True)
        )
        self.fc_2 = nn.Linear(512, 3)

    def forward(self, state_in):
        q_out = self.conv_1(state_in)
        q_out = self.conv_2(q_out)
        q_out = self.conv_3(q_out)
        q_out = torch.flatten(q_out, start_dim=1)
        q_out = self.fc_1(q_out)
        q_out = self.fc_2(q_out)
        return q_out