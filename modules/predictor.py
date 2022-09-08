import torch.nn as nn
from .modules import FCNet, SentenceEmbedding
from .attention import set_att

def set_predictor(predictor_type: str,
                  v_dim: int,
                  embed_dim: int,
                  hidden_dim: int,
                  ans_dim: int,
                  device: str,
                  cls_layer: int,
                  dropout: float,
                  c_len: int,
                  neg_slope: float,
    ):
    if predictor_type == 'base':
        return BasePredictor(
            v_dim=v_dim,
            hidden_dim=hidden_dim,
            ans_dim=ans_dim,
            device=device,
            cls_layer=cls_layer,
            dropout=dropout
        ).to(device)

    if predictor_type == 'base-cap':
        return BaseCaptionPredictor(
            v_dim=v_dim,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            ans_dim=ans_dim,
            device=device,
            cls_layer=cls_layer,
            dropout=dropout
        ).to(device)

    if predictor_type == 'att-cap':
        return AttCaptionPredictor(
            v_dim=v_dim,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            ans_dim=ans_dim,
            device=device,
            cls_layer=cls_layer,
            dropout=dropout
        ).to(device)

class BasePredictor(nn.Module):
    """
    This is for the winning entry of the 2017 VQA Challenge.
    """
    def __init__(self,
                 v_dim: int,
                 hidden_dim: int,
                 ans_dim: int,
                 device: str,
                 cls_layer: int = 2,
                 dropout: float = 0.5,
    ):
        super().__init__()
        self.device = device

        # Non-linear layers
        self.v_net = FCNet(v_dim, hidden_dim)

        # Classifier
        self.classifier = FCNet(
            in_dim=hidden_dim,
            mid_dim=2*hidden_dim,
            out_dim=ans_dim,
            layer=cls_layer,
            dropout=dropout
        )

    def forward(self, batch):
        v = batch['v'].to(self.device)
        q = batch['q'].to(self.device)

        v = v.sum(1) # [batch, v_dim]

        # FC layers
        v = self.v_net(v) # [batch, hidden_dim]
        
        # Joint question features (multiply)
        joint = q * v # [batch, hidden_dim]
        
        return self.classifier(joint)


class AttCaptionPredictor(BasePredictor):
    def __init__(self,
                 v_dim: int,
                 embed_dim: int,
                 hidden_dim: int,
                 ans_dim: int,
                 device: str,
                 cls_layer: int = 2,
                 dropout: float = 0.5,
    ):
        super().__init__(v_dim, hidden_dim, ans_dim, device, cls_layer, dropout)
        self.c_rnn = SentenceEmbedding(
            in_dim=embed_dim,
            hidden_dim=hidden_dim,
            rnn_layer=1,
            device=device,
            rnn_type='GRU'
        )
        self.c_net = FCNet(hidden_dim, hidden_dim, dropout=dropout)
        self.attention = set_att('new')(v_dim, hidden_dim, hidden_dim)
    
    def forward(self, batch, show_att=False):
        v = batch['v'].to(self.device)
        q = batch['q'].to(self.device)
        c = batch['c'].to(self.device)

        # caption embedding
        c = self.c_net(self.c_rnn(c))

        # att_v
        v_att = self.attention(v, c) # [batch, num_objs, 1]
        if show_att: return v_att

        v = (v_att * v).sum(1) # [batch, v_dim]

        # FC layers
        v = self.v_net(v) # [batch, hidden_dim]

        # Joint visual and caption embedding (add)
        joint = c + v
        
        # Joint question features (multiply)
        joint = q * joint # [batch, hidden_dim]
        self.logit_grad = joint
        # self.logit_grad.retain_grad()
        
        return self.classifier(joint)

class BaseCaptionPredictor(BasePredictor):
    def __init__(self,
                 v_dim: int,
                 embed_dim: int,
                 hidden_dim: int,
                 ans_dim: int,
                 device: str,
                 cls_layer: int = 2,
                 dropout: float = 0.5,
    ):
        super().__init__(v_dim, hidden_dim, ans_dim, device, cls_layer, dropout)
        self.c_rnn = SentenceEmbedding(
            in_dim=embed_dim,
            hidden_dim=hidden_dim,
            rnn_layer=1,
            device=device,
            rnn_type='GRU'
        )
        self.c_net = FCNet(hidden_dim, hidden_dim, dropout=dropout)
    
    def forward(self, batch):
        v = batch['v'].to(self.device)
        q = batch['q'].to(self.device)
        c = batch['c'].to(self.device)

        # v_mean
        v = v.sum(1) # [batch, v_dim]

        # caption embedding
        c = self.c_net(self.c_rnn(c))
        self.c_grad = c
        # self.c_grad.retain_grad()

        # FC layers
        v = self.v_net(v) # [batch, hidden_dim]

        # Joint visual and caption embedding (add)
        joint = c + v
        
        # Joint question features (multiply)
        joint = q * joint # [batch, hidden_dim]
        self.logit_grad = joint
        # self.logit_grad.retain_grad()
        
        return self.classifier(joint)
