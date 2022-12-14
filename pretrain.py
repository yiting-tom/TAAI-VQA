import os
import argparse
import json
import pickle
import time
import traceback
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import VQAEDataset, VQADataset
from util.train import train, evaluate
from modules.wrapper import set_model
from util.utils import *
from tools.caption import decode_one_batch


class Argument:
    def __init__(self, load_path):
        with open(os.path.join(load_path, "param.pkl"), "rb") as f:
            inputs = pickle.load(f)

        for key, value in inputs.items():
            setattr(self, key, value)

    def __repr__(self):
        output = ""
        for k, v in self.__dict__.items():
            output = output + f"{k}: {v}" + "\n"
        return output

    def save(self, load_path):
        with open(os.path.join(load_path, "param.pkl"), "wb") as f:
            pickle.dump(self.__dict__, f)


def parse_args():
    # set parameters
    parser = argparse.ArgumentParser()

    # save settings
    parser.add_argument(
        "--comment", type=str, default="exp1", help="comment for Tensorboard"
    )
    parser.add_argument(
        "--load_setting",
        type=bool,
        default=False,
        help="if true, load param.pkl as model setting (default=False)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="set device (automatically select if not assign)",
    )
    parser.add_argument("--seed", type=int, default=1111, help="random seed")

    # path settings
    parser.add_argument(
        "--vocab_path",
        type=str,
        default="../data/vocab_list.txt",
        help="path for vocabulary list",
    )
    parser.add_argument(
        "--ans_path",
        type=str,
        default="../data/answer_candidate.txt",
        help="path for answer candidate list",
    )
    parser.add_argument(
        "--load_path",
        type=str,
        default="../annot/VQA-E",
        help="path for loading dataset",
    )
    parser.add_argument(
        "--feature_path",
        type=str,
        default="../COCO_feature_36",
        help="path for COCO image features",
    )
    parser.add_argument(
        "--graph_path",
        type=str,
        default="../COCO_graph_36",
        help="path for COCO spatial relation graphs",
    )
    parser.add_argument(
        "--index_path",
        type=str,
        default="index.pkl",
        help="path for index of different answer types",
    )
    parser.add_argument("--dataset_type", type=str, default="train2014")
    parser.add_argument("--save_path", type=str, default="")

    # dataset and dataloader settings
    parser.add_argument("--batch_size", type=int, default=128, help="batch size")
    parser.add_argument(
        "--shuffle", type=bool, default=True, help="shuffle dataloader or not"
    )
    parser.add_argument("--c_len", type=int, default=20)

    # encoder settings
    parser.add_argument(
        "--encoder_type",
        type=str,
        default="base",
        help="encoder type (base/relation, default = base)",
    )
    parser.add_argument(
        "--rnn_type",
        type=str,
        default="GRU",
        help="RNN layer type (GRU/LSTM, default = GRU)",
    )
    parser.add_argument(
        "--att_type",
        type=str,
        default="new",
        help="attention layer type (base/new, default = base)",
    )
    parser.add_argument(
        "--embed_dim", type=int, default=300, help="the dimension of embedding"
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=1024,
        help="the dimension of hidden layers (default = 512)",
    )
    parser.add_argument(
        "--v_dim", type=int, default=2048, help="the dimension of visual embedding"
    )
    parser.add_argument("--dropout", type=float, default=0.2, help="dropout")
    parser.add_argument(
        "--rnn_layer",
        type=int,
        default=1,
        help="the number of RNN layers for question embedding",
    )

    # predictor settings
    parser.add_argument(
        "--predictor_type",
        type=str,
        default="base",
        help="predictor type (none/base/q-cap, default=base)",
    )
    parser.add_argument(
        "--cls_layer",
        type=int,
        default=2,
        help="the number of non-linear layers in the classifier",
    )

    # use pre-trained word embedding
    parser.add_argument(
        "--pretrained_embed_path",
        type=str,
        default="../data/glove.6B/glove.6B.300d.txt",
        help="path for pre-trained word embedding (default = '' means using embedding layer)",
    )

    # decoder settings
    parser.add_argument(
        "--decoder_type",
        type=str,
        default="base",
        help="decoder type (none/base/butd, default = base)",
    )
    parser.add_argument(
        "--decoder_hidden_dim",
        type=int,
        default=512,
        help="the dimension of hidden layers in decoder (default = 512)",
    )
    parser.add_argument(
        "--decoder_device",
        type=str,
        default="",
        help="device for decoder (model parallel)",
    )

    # learning rate scheduler settings
    parser.add_argument("--lr", type=float, default=0.002, help="general learning rate")
    parser.add_argument(
        "--lr_vqa",
        type=float,
        default=0,
        help="learning rate for VQA (if = 0 i.e. use the general lr)",
    )
    parser.add_argument(
        "--lr_cap",
        type=float,
        default=0,
        help="learning rate for captioning (if = 0 i.e. use the general lr)",
    )
    parser.add_argument("--warm_up", type=int, default=0, help="wram-up epoch number")
    parser.add_argument(
        "--step_size", type=int, default=0, help="step size for learning rate scheduler"
    )
    parser.add_argument(
        "--gamma", type=float, default=0.5, help="gamma for learning rate scheduler"
    )
    parser.add_argument(
        "--use_mtl",
        type=bool,
        default=True,
        help="use weighted loss or not (default = True)",
    )

    # training/validating process settings
    parser.add_argument(
        "--mode", type=str, default="train", help="mode: train/val/decode"
    )
    parser.add_argument(
        "--load_model",
        type=str,
        default="",
        help="path for the trained model to evaluate",
    )
    parser.add_argument(
        "--load_finetune_model",
        type=str,
        default="",
        help="path for the fine-tuned model to evaluate",
    )
    parser.add_argument(
        "--load_caption", type=str, default="", help="path for the generated captions"
    )
    parser.add_argument("--epoches", type=int, default=15, help="the number of epoches")
    parser.add_argument(
        "--batches",
        type=int,
        default=0,
        help="the number of batches we want to run (default = 0 means to run the whole epoch)",
    )
    parser.add_argument(
        "--start_epoch",
        type=int,
        default=0,
        help="the previous epoch number if need to train continuosly",
    )
    parser.add_argument(
        "--start_batch",
        type=int,
        default=0,
        help="the previous batch number if need to train continuosly",
    )

    args = parser.parse_args()
    return args


def main():
    # get parameters
    args = parse_args()
    if args.load_setting:
        args = Argument(os.path.join("checkpoint", args.comment, "param.pkl"))

    ###### settings ######
    # prepare logger
    logger = Logger(args.comment)
    # set random seed
    random_seed(args.seed)
    # set device
    args.device = args.device if args.device != "" else set_device()
    # prepare vocabulary list
    vocab_list = get_vocab_list(args.vocab_path)
    # answer candidate list
    ans_list = get_vocab_list(args.ans_path)
    # save the settings
    save_path = os.path.join("checkpoint", args.comment)
    with open(os.path.join(save_path, "param.pkl"), "wb") as f:
        pickle.dump(args.__dict__, f)
    with open(os.path.join(save_path, "param.txt"), "w") as f:
        s = ""
        for key, value in args.__dict__.items():
            f.write(f"{key}: {value}\n")
            s = s + f"{key}: {value}\n"
    logger.write(s)

    # setup model
    model = set_model(
        encoder_type=args.encoder_type,
        predictor_type=args.predictor_type,
        decoder_type=args.decoder_type,
        ntoken=len(vocab_list),
        v_dim=args.v_dim,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        decoder_hidden_dim=args.decoder_hidden_dim,
        rnn_layer=args.rnn_layer,
        ans_dim=len(ans_list),
        cls_layer=args.cls_layer,
        c_len=args.c_len,
        dropout=args.dropout,
        device=args.device,
        rnn_type=args.rnn_type,
        att_type=args.att_type,
        decoder_device=args.decoder_device,
        pretrained_embed_path=args.pretrained_embed_path,
        use_mtl=args.use_mtl,
    )
    print("model ready.")

    if args.mode == "train":
        # setup training and validation datasets
        train_data = VQAEDataset(
            dataset_type="train",
            load_path=args.load_path,
            feature_path=args.feature_path,
            graph_path=args.graph_path,
            ans_num=len(ans_list),
        )
        train_loader = DataLoader(
            train_data,
            batch_size=args.batch_size,
            shuffle=args.shuffle,
            num_workers=4,
            pin_memory=True,
        )

        val_data = VQAEDataset(
            dataset_type="val",
            load_path=args.load_path,
            feature_path=args.feature_path,
            graph_path=args.graph_path,
            ans_num=len(ans_list),
        )
        val_loader = DataLoader(
            val_data,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        # if need to train continously, load the previous status of model
        if args.start_epoch != 0:
            path = os.path.join(save_path, f"epoch_{args.start_epoch-1}.pt")
            model.load_state_dict(torch.load(path))
            print("load parameters:", path)

        print("start training.")
        train(
            model=model,
            lr=args.lr,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epoches=args.epoches,
            save_path=save_path,
            logger=logger,
            checkpoint=10000,
            max_norm=0.25,
            comment=args.comment + "_train",
            start_epoch=args.start_epoch,
            batches=args.batches,
            warm_up=args.warm_up,
            step_size=args.step_size,
            gamma=args.gamma,
            lr_vqa=args.lr_vqa,
            lr_cap=args.lr_cap,
        )

    # Evaluate
    if args.mode == "val" or args.mode == "vqa-val":
        # load model: if not specified, load the best model
        if args.load_model == "":
            args.load_model = "best_model.pt"
        model.load_state_dict(
            torch.load(os.path.join("checkpoint", args.comment, args.load_model))
        )
        print("load parameters: ", args.load_model)
        if args.load_finetune_model != "":
            model.load_state_dict(
                torch.load(
                    os.path.join("checkpoint", args.comment, args.load_finetune_model),
                    map_location=args.device,
                ),
                strict=False,
            )
        if args.load_caption != "":
            args.load_caption = os.path.join(
                "checkpoint", args.comment, args.load_caption
            )

        # load index of different answer types
        with open(os.path.join(args.load_path, "index.pkl"), "rb") as f:
            ans_index = pickle.load(f)

        # setup validation dataset
        if args.mode == "val":
            val_data = VQAEDataset(
                dataset_type="val",
                load_path=args.load_path,
                feature_path=args.feature_path,
                graph_path=args.graph_path,
                ans_num=len(ans_list),
                caption_path=args.load_caption,
            )
        elif args.mode == "vqa-val":
            val_data = VQADataset(
                load_path=args.load_path,
                feature_path=args.feature_path,
                dataset_name=args.dataset_type,
                vocab_list=vocab_list,
                ans_list=ans_list,
                graph_path=args.graph_path,
            )
        val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)
        writer = SummaryWriter(comment=args.comment + "_val")

        # Evaluate
        metric = evaluate(
            model=model,
            dataloader=val_loader,
            logger=logger,
            writer=writer,
            ans_index=ans_index,
            save_path=os.path.join("checkpoint", args.comment, "valid_vqa-e"),
        )

        # Show results
        for i in metric:
            print(f"{i}\t {metric[i] * 100:.4f} %")

        # Write the results to Tensorboard
        writer.add_hparams(
            hparam_dict={
                "exp_name": args.comment,
                "hidden_dim": args.hidden_dim,
                "rnn_layer": args.rnn_layer,
                "cls_layer": args.cls_layer,
                "encoder_type": args.encoder_type,
                "decoder_type": args.decoder_type,
                "predictor_type": args.predictor_type,
            },
            metric_dict=metric,
        )

    if "decode" in args.mode:
        if args.load_model == "":
            args.load_model = "best_model"
        if not os.path.exists(os.path.join(save_path, "decode")):
            os.makedirs(os.path.join(save_path, "decode"))
        path = os.path.join(save_path, f"{args.load_model}.pt")
        print("load parameters: ", path)
        model.load_state_dict(torch.load(path, map_location=args.device), strict=False)

        print(args.mode)
        if args.mode == "decode":
            dataset = VQAEDataset(
                dataset_type=args.dataset_type,
                load_path=args.load_path,
                feature_path=args.feature_path,
                graph_path=args.graph_path,
                ans_num=len(ans_list),
                caption_path=args.load_caption,
            )
        elif args.mode == "vqa-decode":
            dataset = VQADataset(
                load_path=args.load_path,
                feature_path=args.feature_path,
                graph_path=args.graph_path,
                vocab_list=vocab_list,
                ans_list=ans_list,
                dataset_name=args.dataset_type,
            )

        dataloader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        if args.save_path == "":
            args.save_path = f"{args.load_model}.txt"
        with open(os.path.join(save_path, "decode", args.save_path), "w") as f:
            for i, batch in enumerate(tqdm(dataloader)):
                if args.start_batch > i:
                    continue
                if args.batches == i and i != 0:
                    break
                result = decode_one_batch(
                    model=model,
                    batch=batch,
                    vocab_list=vocab_list,
                    c_len=args.c_len,
                    k=3,
                )

                result = result.replace("<start> ", "")

                f.write(result)
                f.write("\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        error = traceback.format_exc()
        print(error)
        with open("checkpoint/error.txt", "w") as f:
            f.write(time.ctime())
            f.write("\n")
            f.write(error)
