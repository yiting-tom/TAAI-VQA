from pathlib import Path
from typing import List
from tqdm import tqdm
from configs import pathes
from configs.logger import l

def load_glove_vocabulary(glove_file: Path) -> List:
    """Load GloVe vocabulary"""
    l.info('Loading GloVe vocabulary ...')
    vocab = []
    with open(glove_file) as f:
        load = f.read().split('\n')
        for i in tqdm(load, total=len(load)):
            temp = i.split(' ')[0]
            if temp != '':
                vocab.append(temp)
    return vocab

def save_glove_vocabulary(vocab: List, save_path: Path) -> None:
    """Save GloVe vocabulary to file"""
    l.info('Saving GloVe vocabulary')
    with open(save_path, 'w') as f:
        f.write('\n'.join(vocab))

def main():
    l.info("Generating GloVe vocabulary ...")
    vocab = load_glove_vocabulary(
        glove_file=pathes.f_GLOVE/'glove.6B.300d.txt',
    )

    vocab += ['<start>', '<end>', '<pad>', '<oov>']

    save_glove_vocabulary(
        vocab=vocab,
        save_path=pathes.f_GOLVE_VOCABULARIES,
    )

if __name__ == '__main__':
    main()