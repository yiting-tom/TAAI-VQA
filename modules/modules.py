import math
import numpy as np
from tqdm import tqdm

import torch
import torch.optim
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm


class FCNet(nn.Module):
    """
    Non-linear fully-connected network.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        mid_dim: int = 0,
        layer: int = 1,
        dropout: float = 0,
    ):
        """Input:
        in_dim: input dimension
        out_dim: output dimension
        mid_dim: dimension of the layers in the middle of the network (default = 0)
        layer: number of layers (default = 1)
        dropout: dropout (default = 0, i.e. no use of dropout)
        """

        super().__init__()
        layers = []

        if layer == 1 or mid_dim == 0:
            # If 1-layer:
            # Layer 1: in_dim -> out_dim
            layers.append(weight_norm(nn.Linear(in_dim, out_dim), dim=None))
            layers.append(nn.ReLU())
        else:
            # Else:
            # Suppose there are N layers
            # Layer 1: in_dim -> mid_dim
            layers.append(weight_norm(nn.Linear(in_dim, mid_dim), dim=None))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout, inplace=True))

            # Layer 2 ~ N-1: mid_dim -> mid_dim
            for _ in range(layer - 2):
                layers.append(weight_norm(nn.Linear(mid_dim, mid_dim), dim=None))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout, inplace=True))

            # Layer n: mid_dim -> out_dim
            layers.append(weight_norm(nn.Linear(mid_dim, out_dim), dim=None))

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class DotProduct(nn.Module):
    def __init__(self, a_dim, b_dim, out_dim):
        super().__init__()
        self.wa = nn.Linear(a_dim, out_dim)
        self.wb = nn.Linear(b_dim, out_dim)

    def forward(self, a, b):
        """
        a: [batch, a_len, a_dim]
        b: [batch, b_len, b_dim]
        output: [batch, a_len, b_len]
        """
        a = self.wa(a)
        b = self.wb(b)
        b = torch.transpose(b, 1, 2)
        return torch.bmm(a, b)


class SentenceEmbedding(nn.Module):
    """
    Sentence embedding module.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        device: str,
        rnn_layer: int = 1,
        dropout: float = 0.0,
        rnn_type: str = "LSTM",
        bidirect: bool = False,
    ):
        """Input:
        in_dim: input dimension (i.e. dimension of word embedding)
        hidden_dim: dimension of the hidden state
        rnn_layer: number of RNN layers (default = 1)
        dropout: dropout (default = 0.5)
        rnn_type: choose the type of RNN (default =LSTM)
        bidirect: if True, use a bidirectional RNN (default = False)
        """
        super().__init__()
        assert rnn_type == "LSTM" or rnn_type == "GRU"
        rnn_cls = nn.LSTM if rnn_type == "LSTM" else nn.GRU

        self.rnn = rnn_cls(
            input_size=in_dim,
            hidden_size=hidden_dim,
            num_layers=rnn_layer,
            dropout=dropout,
            bidirectional=bidirect,
            batch_first=True,
        )

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.rnn_layer = rnn_layer
        self.rnn_type = rnn_type
        self.ndirections = 1 + int(bidirect)
        self.device = device

    def init_hidden(self, batch):
        """Initialize hidden states."""
        shape = (self.rnn_layer * self.ndirections, batch, self.hidden_dim)
        if self.rnn_type == "LSTM":
            return (
                torch.zeros(shape).to(self.device),
                torch.zeros(shape).to(self.device),
            )
        else:
            return torch.zeros(shape).to(self.device)

    def forward_all(self, batch):
        """Return the whole results."""
        hidden = self.init_hidden(batch.size(0))
        self.rnn.flatten_parameters()
        output, hidden = self.rnn(batch, hidden)
        return output

    def forward(self, batch):
        """Return the result of the last time step."""
        output = self.forward_all(batch)
        if self.ndirections == 1:
            return output[:, -1]

        forward = output[:, -1, : self.hidden_dim]
        backward = output[:, 0, self.hidden_dim :]
        return torch.cat((forward, backward), dim=1)


class PretrainedWordEmbedding(nn.Module):
    """
    Pre-trained word embedding module.
    """

    def __init__(self, vocab_path: str, device: str):
        """
        vocab_path: path for loading pre-trained word vectors
        device: device
        """
        super().__init__()
        with open(vocab_path) as f:
            lines = f.readlines()

        self.device = device
        self.vocab_dim = len(lines[0].split()) - 1
        self.vocab_len = (
            len(lines) + 4
        )  # vocabulary size = GloVe vocabulary + <oov> + <start> + <end> + <pad>
        vocab = np.zeros((self.vocab_len, self.vocab_dim))
        for i, line in enumerate(lines):
            vocab[i, :] = np.asarray(
                line.split()[1:], "float32"
            )  # save pre-trained vectors
        self.vocab = torch.Tensor(vocab)

    def forward(self, s):
        """
        Input:
            s: [batch, s_len]

        Output:[batch, s_len, vocab_dim]
        """
        batch, s_len = s.size()
        output = torch.zeros(batch, s_len, self.vocab_dim)
        for i in range(batch):
            output[i, :, :] = self.vocab[s[i, :]]

        return output.to(self.device)
