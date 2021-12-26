import torch
import torch.nn as nn
import torch.nn.functional as F

class ActorNetwork(nn.Module):
    def __init__(self, input_dims, layer1_dims, layer2_dims, action_dim):
        super(ActorNetwork, self).__init__()

        self.fc1 = nn.Linear(input_dims, layer1_dims)
        self.fc2 = nn.Linear(layer1_dims, layer2_dims)

        self.act_lay = nn.Linear(layer2_dims, action_dim)
        self.act_lay.weight.data.uniform_(-0.003, 0.003)

    def forward(self, obs):

        out = F.relu(self.fc1(obs))
        out = F.relu(self.fc2(out))
        out = torch.tanh(self.act_lay(out))

        return out


class CriticNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(CriticNetwork, self).__init__()
        hidden_dim = 300
        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.l5 = nn.Linear(hidden_dim, hidden_dim)
        self.l6 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1
