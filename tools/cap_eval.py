"""
This is for evaluating the performance of caption generation.
"""

import os
import json
import argparse
from tqdm import tqdm

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider


def parse_args():
    """Set parameters"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_ref', type=str, default='../annot/VQA-E_15/val2014_captions.json')
    parser.add_argument('--load_path', type=str)
    parser.add_argument('--len', type=int, default=0)
    args = parser.parse_args()
    return args


def get_score(ref, sample, meteor=False, cider=True, rouge=True):
    """Compute the evaluation result"""
    # ref and sample are both dict
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
    ]
    # if meteor: scorers.append((Meteor(),"METEOR")) # need Java
    if cider: scorers.append((Cider(), "CIDEr"))
    if rouge: scorers.append((Rouge(), "ROUGE_L"))
    final_scores = {}
    for scorer, method in scorers:
        print('computing %s score with COCO-EVAL...' % (scorer.method()))
        score, scores = scorer.compute_score(ref, sample)
        if type(score) == list:
            for m, s in zip(method, score):
                final_scores[m] = s * 100
        else:
            final_scores[method] = score * 100
    return final_scores


if __name__ == '__main__':
    args = parse_args()
    hypotheses = {}
    index = 0
    print('Load predicted captions:', args.load_path)
    with open(os.path.join('checkpoint', args.load_path)) as f:
        predict = f.read().split('\n')

    references = {}
    index = 0
    print('Load target captions:', args.load_ref)
    with open(args.load_ref) as f:
        target = json.load(f)
    
    if args.len == 0: args.len = len(predict)
    for i in tqdm(range(args.len)):
        if len(predict[i]) == 0: break
        hypotheses[i] = [predict[i]]
        references[i] = [target['data'][i]['c_word']]

    result = get_score(references, hypotheses)
    print('================================================')
    print(args.load_path)
    path = os.path.dirname(args.load_path)
    with open(os.path.join('checkpoint', path, 'eval_result.txt'), 'w') as f:
        for k, v in result.items():
            print(f'{k}: {v:.4f} %')
            f.write(f'{k}: {v:.8f} %')
            f.write('\n')
    print()