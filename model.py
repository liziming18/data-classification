import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from rnn_cell import RNNCell, GRUCell, LSTMCell


class RNN(nn.Module):
    def __init__(self,
            num_embed_units,  # pretrained wordvec size: size of input wordvec
            num_units,        # RNN units size : size of hidden units
            num_classes):     # number of different classes

        super().__init__()

        self.cell = GRUCell(num_embed_units, num_units)  # fill the initialization of cells
        self.num_embed_units = num_embed_units
        self.num_units = num_units
        self.num_classes = num_classes

        # initialize other layers
        self.linear = nn.Linear(num_units, num_classes)

    def forward(self, batched_data, batched_label, device):
        sent_data = batched_data  # shape: (batch_size, length, num_embed_units)
        sent_label = batched_label  # shape: (batch_size)

        batch_size, seqlen, _ = sent_data.shape

        # randomly initialize the hidden state
        now_state = torch.rand(batch_size, self.num_units)

        loss = 0
        probability_distribution = []
        for i in range(seqlen):
            incoming = sent_data[:, i]  # (batch_size, num_embed_units)
            incoming, now_state = self.cell(incoming, now_state)  # shape: (batch_size, num_units)

            logits = self.linear(incoming)  # shape: (batch_size, num_classes)
            if i == seqlen - 1:
                probability_distribution = logits  # shape: (batch_size, num_classes)

        # calculate loss
        loss = F.cross_entropy(probability_distribution, sent_label, ignore_index=0, reduction='mean')
        
        ans = torch.argmax(probability_distribution, dim=1)
        correct_percentage = torch.sum(ans == sent_label).item() / batch_size

        return loss, correct_percentage
    
    def predict(self, batched_data, device):
        sent_data = batched_data

        batch_size, seqlen, _ = sent_data.shape

        now_state = torch.rand(batch_size, self.num_units)

        probability_distribution = []
        for i in range(seqlen):
            incoming = sent_data[:, i]
            incoming, now_state = self.cell(incoming, now_state)

            logits = self.linear(incoming)
            if i == seqlen - 1:
                probability_distribution = logits

        ans = torch.argmax(probability_distribution, dim=1)

        return ans
    