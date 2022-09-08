import os
import json
import argparse
from tqdm import tqdm
from utils import get_vocab_list, get_tokens, padding

def parse_args():
    # set parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_path', type=str)
    parser.add_argument('--data_name', type=str, default='')
    parser.add_argument('--split', type=int, default=2)
    parser.add_argument('--vocab_path', type=str, default='../data/vocab_list.txt')
    parser.add_argument('--max_l', type=int, default=15)
    args = parser.parse_args()
    return args

def main(args):
    vocab_list = get_vocab_list(args.vocab_path)
    data = []
    for index in range(args.split):
        with open(os.path.join('checkpoint', args.load_path, 'decode', f'{args.data_name}_{index}.txt')) as f:
            load = [i for i in f.read().split('\n') if len(i) != 0]
            for i in tqdm(range(len(load)), desc=f'{index+1}/{args.split}'):
                tokens, _ = get_tokens(load[i], vocab_list)
                tokens, l = padding(tokens, args.max_l, vocab_list)
                data.append({
                    'c': tokens,
                    'c_len': l
                })

    path = os.path.join('checkpoint', args.load_path, 'json')
    if not os.path.exists(path): os.makedirs(path)
    with open(os.path.join(path, f'{args.data_name}.json'), 'w') as f:
        json.dump(data, f)

if __name__ == '__main__':
    args = parse_args()
    main(args)