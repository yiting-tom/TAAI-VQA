import torch
import torch.optim
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm
from .modules import FCNet


def set_att(att_type):
    return {
        "base": ConcatAttention,
        "new": MultiplyAttention,
    }[att_type]


class ConcatAttention(nn.Module):
    """
    Concat attention module.
    Given v = visual features, q = query embedding,
    output = Softmax([v:q])
    """

    def __init__(self, v_dim, q_dim, hidden_dim):
        super().__init__()
        self.sequence = nn.Sequential(
            weight_norm(nn.Linear(v_dim + q_dim, hidden_dim), dim=None),
            nn.ReLU(),
            weight_norm(nn.Linear(hidden_dim, 1), dim=None),
        )

    def logits(self, v, q):
        num_objs = v.size(1)
        q = q.unsqueeze(1).repeat(1, num_objs, 1)  # [batch, num_objs, q_dim]
        # concat each visual features with question features
        vq = torch.cat((v, q), 2)  # [batch, num_objs, v_dim+q_dim]
        # non-linear layers
        vq = self.sequence(vq)  # [batch, num_objs, 1]
        return vq

    def forward(self, v, q):
        """
        Input:
        v: [batch, num_objs, v_dim]
        q: [batch, q_dim]

        Output:[batch, num_objs, 1]
        """
        logits = self.logits(v, q)
        return nn.functional.softmax(logits, 1)


class MultiplyAttention(nn.Module):
    """
    Element-wise multiplication attention module.
    Given v = visual features, q = query embedding,
    output = Softmax(W(Wv(v) * Wq(q)))
    """

    def __init__(self, v_dim, q_dim, hidden_dim, dropout=0.2):
        super().__init__()
        self.W_v = FCNet(v_dim, hidden_dim)
        self.W_q = FCNet(q_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear = weight_norm(nn.Linear(hidden_dim, 1), dim=None)

    def logits(self, v, q):
        batch, num_objs, _ = v.size()
        v = self.W_v(v)  # [batch, num_objs, hidden_dim]
        q = (
            self.W_q(q).unsqueeze(1).repeat(1, num_objs, 1)
        )  # [batch, num_objs, hidden_dim]
        # element-wise multiply each visual features with question features
        joint = v * q
        joint = self.dropout(joint)
        joint = self.linear(joint)
        return joint  # [batch, num_objs, 1]

    def forward(self, v, q):
        """Input:
        v: [batch, v_len, v_dim]
        q: [batch, q_dim]

        Output:[batch, num_obj, 1]
        """
        logits = self.logits(v, q)
        return nn.functional.softmax(logits, 1)
