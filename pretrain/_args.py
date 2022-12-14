#%%
from typing import Any
from dataclasses import dataclass
import argparse

#%%


@dataclass
class Argument:
    type: type
    default: Any
    help: str


def parse_args():
    # set parameters
    p = argparse.ArgumentParser()

    def a(**kwargs):
        p.add_argument(**kwargs)

    # save settings
    a("--exp_name", type=str, default="exp1", help="comment for Tensorboard")
    a(
        "--load_setting",
        type=bool,
        default=False,
        help="if true, load param.pkl as model setting (default=False)",
    )
    a(
        "--device",
        type=str,
        default="",
        help="set device (automatically select if not assign)",
    )
    a("--seed", type=int, default=9527, help="random seed")

    # path settings
    a(
        "--vocab_path",
        type=str,
        default="../data/vocab_list.txt",
        help="path for vocabulary list",
    )
    a(
        "--load_path",
        type=str,
        default="../annot/VQA-E",
        help="path for loading dataset",
    )
    a(
        "--feature_path",
        type=str,
        default="../COCO_feature_36",
        help="path for COCO image features",
    )
    a(
        "--graph_path",
        type=str,
        default="../COCO_graph_36",
        help="path for COCO spatial relation graphs",
    )
    a(
        "--index_path",
        type=str,
        default="index.pkl",
        help="path for index of different answer types",
    )
    a("--dataset_type", type=str, default="train2014")
    a("--save_path", type=str, default="")

    # dataset and dataloader settings
    a("--batch_size", type=int, default=128, help="batch size")
    a("--shuffle", type=bool, default=True, help="shuffle dataloader or not")
    a("--c_len", type=int, default=20)

    # encoder settings
    a(
        "--encoder_type",
        type=str,
        default="base",
        help="encoder type (base/relation, default = base)",
    )
    a(
        "--rnn_type",
        type=str,
        default="GRU",
        help="RNN layer type (GRU/LSTM, default = GRU)",
    )
    a(
        "--att_type",
        type=str,
        default="new",
        help="attention layer type (base/new, default = base)",
    )
    a("--embed_dim", type=int, default=300, help="the dimension of embedding")
    a(
        "--hidden_dim",
        type=int,
        default=1024,
        help="the dimension of hidden layers (default = 512)",
    )
    a("--v_dim", type=int, default=2048, help="the dimension of visual embedding")
    a("--dropout", type=float, default=0.2, help="dropout")
    a(
        "--rnn_layer",
        type=int,
        default=1,
        help="the number of RNN layers for question embedding",
    )

    # predictor settings
    a(
        "--predictor_type",
        type=str,
        default="base",
        help="predictor type (none/base/q-cap, default=base)",
    )
    a(
        "--cls_layer",
        type=int,
        default=2,
        help="the number of non-linear layers in the classifier",
    )

    # use pre-trained word embedding
    a(
        "--pretrained_embed_path",
        type=str,
        default="../data/glove.6B/glove.6B.300d.txt",
        help="path for pre-trained word embedding (default = '' means using embedding layer)",
    )

    # decoder settings
    a(
        "--decoder_type",
        type=str,
        default="base",
        help="decoder type (none/base/butd, default = base)",
    )
    a(
        "--decoder_hidden_dim",
        type=int,
        default=512,
        help="the dimension of hidden layers in decoder (default = 512)",
    )
    a(
        "--decoder_device",
        type=str,
        default="",
        help="device for decoder (model parallel)",
    )

    # learning rate scheduler settings
    a("--lr", type=float, default=0.002, help="general learning rate")
    a(
        "--lr_vqa",
        type=float,
        default=0,
        help="learning rate for VQA (if = 0 i.e. use the general lr)",
    )
    a(
        "--lr_cap",
        type=float,
        default=0,
        help="learning rate for captioning (if = 0 i.e. use the general lr)",
    )
    a("--warm_up", type=int, default=0, help="wram-up epoch number")
    a("--step_size", type=int, default=0, help="step size for learning rate scheduler")
    a("--gamma", type=float, default=0.5, help="gamma for learning rate scheduler")
    a(
        "--use_mtl",
        type=bool,
        default=True,
        help="use weighted loss or not (default = True)",
    )

    # training/validating process settings
    a("--mode", type=str, default="train", help="mode: train/val/decode")
    a(
        "--load_model",
        type=str,
        default="",
        help="path for the trained model to evaluate",
    )
    a(
        "--load_finetune_model",
        type=str,
        default="",
        help="path for the fine-tuned model to evaluate",
    )
    a("--load_caption", type=str, default="", help="path for the generated captions")
    a("--epoches", type=int, default=15, help="the number of epoches")
    a(
        "--batches",
        type=int,
        default=0,
        help="the number of batches we want to run (default = 0 means to run the whole epoch)",
    )
    a(
        "--start_epoch",
        type=int,
        default=0,
        help="the previous epoch number if need to train continuosly",
    )
    a(
        "--start_batch",
        type=int,
        default=0,
        help="the previous batch number if need to train continuosly",
    )

    return p.parse_args()
