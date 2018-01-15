from torch import nn
from torch.nn import functional as F


class A2C(nn.Module):

    def __init__(self, n_obs, n_act):
        super(A2C, self).__init__()

        self.input_layer = nn.Linear(n_obs, 128)
        self.hidden_layer1 = nn.Linear(128, 128)

        self.v = nn.Linear(128, 1)
        self.a = nn.Linear(128, n_act)

        self._initialize()

    def forward(self, x):
        x = F.relu(self.input_layer(x))
        x = F.relu(self.hidden_layer1(x))

        v = self.v(x)
        a = F.softmax(self.a(x))

        return v, a

    def _initialize(self):
        self.a.weight.data.zero_()
        self.a.bias.data.zero_()


class RecurrentA2C(nn.Module):

    def __init__(self, n_obs, n_act):
        super(RecurrentA2C, self).__init__()

        self.input_layer = nn.Linear(n_obs, 128)
        self.lstm1 = nn.LSTM(input_size=128,
                             hidden_size=128,
                             num_layers=1,
                             batch_first=True)

        self.v = nn.Linear(128, 1)
        self.a = nn.Linear(128, n_act)

        self._initialize()

    def forward(self, x, tuple):
        x = F.relu(self.input_layer(x)).unsqueeze(0)
        x, tuple = self.lstm1(x, tuple)
        x = x.squeeze(0)

        v = self.v(x)
        a = F.softmax(self.a(x))

        return v, a, tuple

    def _initialize(self):
        self.a.weight.data.zero_()
        self.a.bias.data.zero_()
