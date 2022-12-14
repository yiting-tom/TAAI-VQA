import os
import pickle
import argparse
import time
import traceback

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from modules.wrapper import set_model
from dataset import VQACaptionDataset
from util.utils import *
from util.train import fine_tune, fine_tune_evaluate


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


def parse_args():
    parser = argparse.ArgumentParser()

    # save settings
    parser.add_argument("--load_model", type=str, default="exp1", help="model name")
    parser.add_argument(
        "--load_epoch",
        type=int,
        default=-1,
        help="load certain epoch (default = -1 means select the best epoch)",
    )
    parser.add_argument("--load_predictor", type=int, default=-1)
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="set device (automatically select if not assign)",
    )
    parser.add_argument("--seed", type=int, default=1111, help="random seed")
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--save_path", type=str, default="fine-tune")

    # dataset settings
    parser.add_argument("--load_path", type=str, default="../annot")
    parser.add_argument("--dataset_name", type=str, default="train2014")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--c_len", type=int, default=15)

    # learning rate scheduler settings
    parser.add_argument("--lr", type=float, default=0.002, help="general learning rate")

    # training/validating process settings
    parser.add_argument("--epoches", type=int, default=30, help="the number of epoches")
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

    args = parser.parse_args()
    return args


def main():
    # get parameters
    main_args = parse_args()
    name = (
        f"epoch_{main_args.load_epoch}.pt"
        if main_args.load_epoch != -1
        else "best_model.pt"
    )
    load_path = os.path.join("checkpoint", main_args.load_model)
    args = Argument(os.path.join(load_path))

    ###### settings ######
    # prepare logger
    logger = Logger(os.path.join(args.comment, "fine-tune"))
    # set random seed
    random_seed(main_args.seed)
    # set device
    main_args.device = main_args.device if main_args.device != "" else set_device()
    # prepare vocabulary list
    vocab_list = get_vocab_list(args.vocab_path)
    # answer candidate list
    ans_list = get_vocab_list(args.ans_path)
    # save the settings
    save_path = os.path.join(load_path, main_args.save_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with open(os.path.join(save_path, "param.pkl"), "wb") as f:
        pickle.dump(main_args.__dict__, f)
    with open(os.path.join(save_path, "param.txt"), "w") as f:
        s = ""
        for key, value in main_args.__dict__.items():
            f.write(f"{key}: {value}\n")
            s = s + f"{key}: {value}\n"
    logger.write(s)

    print("load parameters")
    param = torch.load(os.path.join(load_path, name), map_location=main_args.device)
    print("prepare encoder")
    encoder = set_model(
        encoder_type=args.encoder_type,
        decoder_type="none",
        predictor_type="none",
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
        device=main_args.device,
        rnn_type=args.rnn_type,
        att_type=args.att_type,
        decoder_device=args.decoder_device,
        pretrained_embed_path=args.pretrained_embed_path,
        use_mtl=args.use_mtl,
    )
    encoder.load_state_dict(param, strict=False)

    print("prepare predictor")
    predictor = set_model(
        predictor_type=args.predictor_type,
        decoder_type="none",
        encoder_type="none",
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
        device=main_args.device,
        rnn_type=args.rnn_type,
        att_type=args.att_type,
        decoder_device=args.decoder_device,
        pretrained_embed_path=args.pretrained_embed_path,
        use_mtl=args.use_mtl,
    )
    print("model ready.")
    if args.predictor_type != "att-cap":
        caption_path = ""
    else:
        if main_args.mode in ["metric"]:
            caption_path = os.path.join(load_path, "json", "vqa_val.json")
        else:
            caption_path = os.path.join(load_path, "json", f"vqa_{main_args.mode}.json")

    if main_args.mode == "train":
        # load the pre-trained parameters
        predictor.load_state_dict(param, strict=False)

        train_data = VQACaptionDataset(
            load_path=main_args.load_path,
            feature_path=args.feature_path,
            graph_path=args.graph_path,
            dataset_name="train2014",
            vocab_list=vocab_list,
            ans_list=ans_list,
            caption_path=caption_path,
        )
        train_loader = DataLoader(
            train_data,
            batch_size=main_args.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )

        score = 0.0
        print("start training")
        fine_tune(
            encoder=encoder,
            predictor=predictor,
            lr=main_args.lr,
            train_loader=train_loader,
            num_epoches=main_args.epoches,
            save_path=save_path,
            logger=logger,
            checkpoint=10000,
            max_norm=0.25,
            comment=args.comment + "_fine-tune",
            start_epoch=args.start_epoch,
            batches=args.batches,
            warm_up=args.warm_up,
            step_size=args.step_size,
            gamma=args.gamma,
        )

    val_data = VQACaptionDataset(
        load_path=main_args.load_path,
        feature_path=args.feature_path,
        graph_path=args.graph_path,
        dataset_name="val2014",
        vocab_list=vocab_list,
        ans_list=ans_list,
        caption_path=caption_path,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=main_args.batch_size,
        shuffle=False,
        num_workers=4,
    )

    if main_args.mode in ["train", "val"]:
        writer = SummaryWriter(comment=args.comment + "_val" + main_args.save_path)
        best_score = 0
        best_epoch = 0
        for epoch in range(main_args.epoches):
            # load fine-tuned parameters
            param = torch.load(
                os.path.join(load_path, main_args.save_path, f"epoch_{epoch}.pt"),
                map_location=main_args.device,
            )
            predictor.load_state_dict(param)

            score, target_score = fine_tune_evaluate(
                encoder=encoder,
                predictor=predictor,
                dataloader=val_loader,
                logger=logger,
            )
            logger.show(
                f"[Epoch {epoch}] score: {score:.10f} / bound: {target_score:.10f}"
            )
            writer.add_scalar("val/vqa/score", score, epoch)

            if best_score < score:
                best_score = score
                best_epoch = epoch
                torch.save(
                    predictor.state_dict(), os.path.join(save_path, "best_epoch.pt")
                )
            logger.show(f"[BEST] epoch: {best_epoch}, score: {best_score:.10f}")
    else:
        # load index of different answer types
        with open(os.path.join(args.load_path, "index.pkl"), "rb") as f:
            ans_index = pickle.load(f)

        name = (
            "best_epoch.pt"
            if main_args.load_predictor == -1
            else f"epoch_{main_args.load_predictor}.pt"
        )
        param = torch.load(
            os.path.join(load_path, main_args.save_path, name),
            map_location=main_args.device,
        )
        predictor.load_state_dict(param)
        metric = fine_tune_evaluate(
            encoder=encoder,
            predictor=predictor,
            dataloader=val_loader,
            logger=logger,
            ans_index=ans_index,
            save_path=os.path.join(load_path, "vqa"),
        )
        msg = ""
        for i, j in metric.items():
            msg = msg + f"{i}: {j:.6f}\n"
        logger.show(msg)


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
