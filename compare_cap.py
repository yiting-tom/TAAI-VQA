import os
import json
import pickle
import argparse
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from dataset import VQAEDataset
from modules.wrapper import set_model
from util.utils import get_vocab_list, set_device, Logger
from tools.cap_eval import get_score
from tools.caption import decode_one_batch


def parse_args():
    # set parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_ref', type=str, default='../annot/vqa-e_val.json')
    parser.add_argument('--exp_name', type=str, default='code_review')
    parser.add_argument('--len', type=int, default=150)
    parser.add_argument('--device', type=str, default='')
    args = parser.parse_args()
    return args

def token_to_word(tokens, vocab_list):
    words = []
    for i in tokens:
        if vocab_list[i] == '<pad>': break
        words.append(vocab_list[i])
    return ' '.join(words)

class Argument():
    def __init__(self, load_path):
        with open(os.path.join('checkpoint', load_path, 'param.pkl'), 'rb') as f:
            inputs = pickle.load(f)
            
        for key, value in inputs.items():
            setattr(self, key, value)
    
    def __repr__(self):
        output = ''
        for k, v in self.__dict__.items():
            output = output + f'{k}: {v}' + '\n'
        return output
    
    def save(self, load_path):
        with open(os.path.join(load_path, 'param.pkl'), 'wb') as f:
            pickle.dump(self.__dict__, f)

def main(my_args):
    save_path = os.path.join('checkpoint', my_args.exp_name, 'decode')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    logger = Logger(my_args.exp_name, log_name='cap_log.txt')
    
    args = Argument(my_args.exp_name)
    args.device = set_device() if my_args.device == '' else my_args.device
    vocab_list = get_vocab_list(args.vocab_path)
    ans_list = get_vocab_list(args.ans_path)
    
    print('Load target captions:', my_args.load_ref)
    references = {}
    with open(my_args.load_ref) as f:
        target = json.load(f)
        for i in range(my_args.len):
            references[i] = [target['data'][i]['c_word']]

    print('Prepare dataset.')
    dataset = VQAEDataset(
        dataset_type='val',
        load_path=args.load_path,
        feature_path=args.feature_path,
        graph_path=args.graph_path,
        ans_num=len(ans_list)
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=4,
        pin_memory=True
    )

    print('Set model.')
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
    model = model.to(my_args.device)

    print('Start decode:')
    best_score = 0
    files = sorted([file for file in os.listdir(os.path.join('checkpoint', my_args.exp_name)) if file.endswith('.pt') and file != 'best_model.pt'])
    for file in files:
        model.load_state_dict(torch.load(os.path.join('checkpoint', my_args.exp_name, file), map_location=my_args.device))
        hypotheses = {}
        with open(os.path.join(save_path, file[:-3] + '.txt'), 'w') as f:
            for i, batch in enumerate(tqdm(dataloader, desc=file)):
                if i == my_args.len: break
                result = decode_one_batch(
                    model=model,
                    batch=batch,
                    vocab_list=vocab_list,
                    c_len=args.c_len,
                    k=3,
                )
                result = result.replace('<start> ', '')
                f.write(result + '\n')
                hypotheses[i] = [result]
            
        bleu = list(get_score(references, hypotheses, meteor=False, cider=False, rouge=False).values())
        score = sum(bleu) / len(bleu)
        str_bleu = ''
        for i in bleu: str_bleu = str_bleu + f'{i:.4f}/'
        msg = f'{file.ljust(11)}: {score:.4f}\t({str_bleu[:-1]})'
        if best_score < score:
            best_score = score
            best_model = file
            msg = msg + '\t*NEW BEST'
            torch.save(model.state_dict(), os.path.join('checkpoint', my_args.exp_name, 'best_epoch.pt'))
        logger.write(msg)
    logger.write(f'Best model: {best_model}, avg_bleu: {best_score}')

if __name__ == '__main__':
    args = parse_args()
    main(args)