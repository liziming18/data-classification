import torch
from torch import nn
import torch.nn.functional as F


class RNNCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.input_layer = nn.Linear(input_size, hidden_size)
        self.hidden_layer = nn.Linear(hidden_size, hidden_size, bias=False)

    def init(self, batch_size, device):
        # return the initial state
        return torch.zeros(batch_size, self.hidden_size, device=device)

    def forward(self, incoming, state):
        output = (self.input_layer(incoming) + self.hidden_layer(state)).tanh()
        new_state = output  # stored for next step
        return output, new_state


class GRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # intialize weights and layers
        self.input_layer_z = nn.Linear(input_size, hidden_size) # input to update
        self.input_layer_r = nn.Linear(input_size, hidden_size) # input to reset
        self.input_layer_h = nn.Linear(input_size, hidden_size) # input to hidden
        self.hidden_layer_z = nn.Linear(hidden_size, hidden_size, bias=False) # hidden to update
        self.hidden_layer_r = nn.Linear(hidden_size, hidden_size, bias=False) # hidden to reset
        self.hidden_layer_h = nn.Linear(hidden_size, hidden_size, bias=False) # hidden to hidden

    def init(self, batch_size, device):
        # return the initial state
        return torch.rand(batch_size, self.hidden_size, device=device)

    def forward(self, incoming, state):
        z_t = torch.sigmoid(self.input_layer_z(incoming) + self.hidden_layer_z(state))
        r_t = torch.sigmoid(self.input_layer_r(incoming) + self.hidden_layer_r(state))
        h_tilduh = (self.input_layer_h(incoming) + self.hidden_layer_h(torch.mul(r_t, state))).tanh() # output candidate
        output = torch.mul(1 - z_t, state) + torch.mul(z_t, h_tilduh)
        new_state = output
        return output, new_state


class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # intialize weights and layers
        self.input_layer_i = nn.Linear(input_size, hidden_size)
        self.input_layer_o = nn.Linear(input_size, hidden_size)
        self.input_layer_f = nn.Linear(input_size, hidden_size)
        self.input_layer_c = nn.Linear(input_size, hidden_size)
        self.hidden_layer_i = nn.Linear(hidden_size, hidden_size, bias=False)
        self.hidden_layer_o = nn.Linear(hidden_size, hidden_size, bias=False)
        self.hidden_layer_f = nn.Linear(hidden_size, hidden_size, bias=False)
        self.hidden_layer_c = nn.Linear(hidden_size, hidden_size, bias=False)

    def init(self, batch_size, device):
        # return the initial state (which can be a tuple)
        return torch.rand(batch_size, self.hidden_size, device=device), torch.rand(batch_size, self.hidden_size, device=device)

    def forward(self, incoming, state):
        # calculate output and new_state
        current_h, current_c = state[0], state[1]
        i_t = torch.sigmoid(self.input_layer_i(incoming) + self.hidden_layer_i(current_h))
        o_t = torch.sigmoid(self.input_layer_o(incoming) + self.hidden_layer_o(current_h))
        f_t = torch.sigmoid(self.input_layer_f(incoming) + self.hidden_layer_f(current_h))
        c_tilduh = (self.input_layer_c(incoming) + self.hidden_layer_c(current_h)).tanh()
        new_c = torch.mul(f_t, current_c) + torch.mul(i_t, c_tilduh)
        new_h = torch.mul(o_t, new_c.tanh())
        output = new_h
        return output, (new_h, new_c)
