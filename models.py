import torch
import torch.nn as nn
import torch.nn.functional as F
from  torch.distributions.categorical import Categorical


class SACActor(nn.Module):
    def __init__(self, state_size, action_size, seed, fc_1=64, fc_2=64):
        super(SACActor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.leak = 0.01
        self.state_size = state_size
        self.action_size = action_size
        self.layer1 = nn.Linear(state_size, fc_1)
        self.layer2 = nn.Linear(fc_1, fc_2)
        self.layer3 = nn.Linear(fc_2, action_size)
        self.softmax = F.softmax
        self.reset_parameters()

    def reset_parameters(self):
        """ Initilaize the weights using He et al (2015) weights """
        torch.nn.init.kaiming_normal_(self.layer1.weight.data, a=self.leak, mode='fan_in')
        torch.nn.init.kaiming_normal_(self.layer2.weight.data, a=self.leak, mode='fan_in')
        torch.nn.init.uniform_(self.layer3.weight.data, -3e-4, 3e-4)


    def forward(self, state):
        x1 = F.relu(self.layer1(state))
        x2 = F.relu(self.layer2(x1))
        x3 = self.layer3(x2)
        action_prob = self.softmax(x3, dim=1)
        log_prob = action_prob + torch.finfo(torch.float32).eps
        log_prob = torch.log(log_prob)
        return action_prob, log_prob


    def sample(self, state):
        action_prob, log_prob = self.forward(state)
        m = Categorical(action_prob)
        action = m.sample()
        return action

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed, fc_1, fc_2):
        super(QNetwork, self).__init__()
        self.leak = 0.01
        self.seed = torch.manual_seed(seed)
        self.state_size = state_size
        self.action_size = action_size
        self.layer1 = nn.Linear(state_size, fc_1)
        self.layer2 = nn.Linear(fc_1, fc_2)
        self.layer3 = nn.Linear(fc_2, action_size)
        self.layer4 = nn.Linear(state_size, fc_1)
        self.layer5 = nn.Linear(fc_1, fc_2)
        self.layer6 = nn.Linear(fc_2, action_size)
        self.reset_parameters()
    
    def forward(self, state):
        x1 = F.relu(self.layer1(state))
        x2 = F.relu(self.layer2(x1))
        x3 = self.layer3(x2)

        x4 = F.relu(self.layer4(state))
        x5 = F.relu(self.layer5(x4))
        x6 = self.layer6(x5)
        return x3, x6
    
    def Q1(self, state):
        x1 = F.relu(self.layer1(state))
        x2 = F.relu(self.layer2(x1))
        x3 = self.layer3(x2)
        return x3

    def reset_parameters(self):
        """ Initilaize the weights using He et al (2015) weights """
        torch.nn.init.kaiming_normal_(self.layer1.weight.data, a=self.leak, mode='fan_in')
        torch.nn.init.kaiming_normal_(self.layer2.weight.data, a=self.leak, mode='fan_in')
        torch.nn.init.uniform_(self.layer3.weight.data, -3e-4, 3e-4)
        torch.nn.init.kaiming_normal_(self.layer4.weight.data, a=self.leak, mode='fan_in')
        torch.nn.init.kaiming_normal_(self.layer5.weight.data, a=self.leak, mode='fan_in')
        torch.nn.init.uniform_(self.layer6.weight.data, -3e-4, 3e-4)


class SQNetwork(nn.Module):
    """Actor (Policy) Model."""
    def __init__(self, state_size, action_size, seed, fc1_units=64,fc2_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(SQNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class Classifier(nn.Module):
    """ Classifier Model."""

    def __init__(self, state_size, action_dim, seed, fc1_units=64,fc2_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Classifier, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_dim)
        self.bn1 = nn.BatchNorm1d(fc1_units)
        self.bn2 = nn.BatchNorm1d(fc2_units)

    def forward(self, state, train=False, stats=False):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = self.bn1(x)
        x = F.dropout(x,training=train)
        x = F.relu(self.fc2(x))
        x = self.bn2(x)
        x = F.dropout(x,training=train)
        output = self.fc3(x)
        return output
