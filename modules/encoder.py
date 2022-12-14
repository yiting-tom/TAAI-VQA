import torch
import torch.nn as nn
from .modules import FCNet, SentenceEmbedding, PretrainedWordEmbedding
from .attention import set_att
from .graph_att_layer import GAttNet


def set_encoder(
    encoder_type: str,
    ntoken: int,
    v_dim: int,
    embed_dim: int,
    hidden_dim: int,
    device: str,
    dropout: float,
    rnn_type: str,
    rnn_layer: int,
    att_type: str,
    vocab_path: str = "",
):
    if encoder_type == "none":
        return
    if encoder_type == "base":
        model = BaseEncoder(
            ntoken=ntoken,
            v_dim=v_dim,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            device=device,
            dropout=dropout,
            rnn_type=rnn_type,
            rnn_layer=rnn_layer,
            att_type=att_type,
        )
    if encoder_type == "relation":
        model = RelationEncoder(
            ntoken=ntoken,
            v_dim=v_dim,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            device=device,
            dropout=dropout,
            rnn_type=rnn_type,
            rnn_layer=rnn_layer,
            att_type=att_type,
        )

    if vocab_path != "":
        model.embedding = PretrainedWordEmbedding(vocab_path=vocab_path, device=device)
    return model.to(device)


# This model is based on the winning entry of the 2017 VQA Challenge, following the system described in
# 'Bottom-Up ad Top-Down Attention for Image Captioning and Visual Question Answering' (https://arxiv.org/abs/1707.07998) and
# 'Tips and Tricks for Visual Question Answering: Learning from teh 2017 Challenge' (https://arxiv.org/abs/1708.02711)
#
# Code reference: https://github.com/hengyuan-hu/bottom-up-attention-vqa


class BaseEncoder(nn.Module):
    """
    This is for the winning entry of the 2017 VQA Challenge.
    """

    def __init__(
        self,
        ntoken: int,
        embed_dim: int,
        hidden_dim: int,
        rnn_layer: int,
        v_dim: int,
        device: str,
        dropout: float = 0.5,
        rnn_type: str = "GRU",
        att_type: str = "base",
    ):
        """Input:
        For question embedding:
            ntoken: number of tokens (i.e. size of vocabulary)
            embed_dim: dimension of question embedding
            hidden_dim: dimension of hidden layers
            rnn_layer: number of RNN layers
        For attention:
            v_dim: dimension of image features
        Others:
            device: device
            dropout: dropout (default = 0.5)
        """

        super().__init__()
        self.device = device

        # Word embedding for question
        self.embedding = nn.Embedding(ntoken + 1, embed_dim, padding_idx=ntoken)

        # RNN for question
        self.q_rnn = SentenceEmbedding(
            in_dim=embed_dim,
            hidden_dim=hidden_dim,
            rnn_layer=rnn_layer,
            dropout=0.0,
            device=device,
            rnn_type=rnn_type,
        )

        # Attention layer for image features based on questions
        self.attention = set_att(att_type)(
            v_dim=v_dim, q_dim=hidden_dim, hidden_dim=hidden_dim
        )

        # Non-linear layers for image features
        self.q_net = FCNet(hidden_dim, hidden_dim)

    def base_forward(self, batch):
        """
        Input:
            v: [batch, num_objs, v_dim]
            q: [batch, q_len]
        """
        # Setup inputs
        v = batch["v"].to(self.device)
        q = batch["q"].to(self.device)

        # Embed words in question and take the last output of RNN layer as the question embedding
        q = self.embedding(q)  # [batch, q_len, q_embed_dim]
        q = self.q_rnn(q)  # [batch, hidden_dim]
        output = {
            "v": v,  # [batch, num_objs, v_dim]
            "q": q,  # [batch, hidden_dim]
            "a": batch["a"],  # [batch, ans_dim]
        }

        # Caption embedding
        if "c" in batch:
            output["c_target"] = batch["c"].to(self.device)
            output["c_len"] = batch["c_len"].to(self.device)
            output["c"] = self.embedding(
                output["c_target"]
            )  # [batch, c_len, embed_dim]
        return output

    def forward(self, batch):
        output = self.base_forward(batch)

        # Get the attention of visual features based on question embedding
        output["v_att"] = self.attention(
            output["v"], output["q"]
        )  # [batch, num_objs, 1]
        output["v"] = output["v_att"] * output["v"]  # [batch, num_objs, v_dim]

        # Question embedding
        output["q"] = self.q_net(output["q"])  # [batch, hidden_dim]
        return output


class RelationEncoder(BaseEncoder):
    """
    This is for 'Relation-Aware Graph Network for Visual Question Answering'
    """

    def __init__(
        self,
        ntoken: int,
        embed_dim: int,
        hidden_dim: int,
        rnn_layer: int,
        v_dim: int,
        device: str,
        dropout: float = 0.5,
        rnn_type: str = "GRU",
        att_type: str = "base",
        num_objs: int = 36,
        dir_num: int = 2,
        label_num: int = 11,
    ):
        super().__init__(
            ntoken,
            embed_dim,
            hidden_dim,
            rnn_layer,
            v_dim,
            device,
            dropout,
            rnn_type,
            att_type,
        )
        self.label_num = label_num
        self.spatial_encoder = GAttNet(
            dir_num=dir_num,
            label_num=label_num,
            in_feat_dim=v_dim,
            out_feat_dim=v_dim,
            nongt_dim=num_objs,
            dropout=dropout,
        )

    def forward(self, batch, show_att=False):
        """
        Input:
            v: [batch, num_objs, v_dim]
            q: [batch, q_len]
            graph: [batch, num_objs, num_objs]
        Output:
            v: [batch, num_objs, v_dim]
            q: [batch, hidden_dim]
        """
        output = self.base_forward(batch)

        # Transfer graph structure
        # [batch, num_objs, num_objs, label_num]
        graph = (
            torch.zeros_like(batch["graph"])
            .unsqueeze(3)
            .repeat(1, 1, 1, self.label_num)
        )
        for i in range(self.label_num):
            graph[:, :, :, i][batch["graph"] == i + 1] = 1
        graph = graph.to(self.device)

        # Get the attention of visual features based on question embedding
        output["v_att"] = self.attention(
            output["v"], output["q"]
        )  # [batch, num_objs, 1]
        output["v"] = output["v_att"] * output["v"]  # [batch, num_objs, v_dim]

        # Question embedding
        output["q"] = self.q_net(output["q"])  # [batch, hidden_dim]

        # Relation encoder
        output_v = self.spatial_encoder(
            output["v"], graph, show_att=show_att
        )  # [batch, num_objs, v_dim]
        if show_att:
            return output_v
        output["v"] = output_v
        return output
