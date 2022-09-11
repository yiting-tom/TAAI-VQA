from pathlib import Path
from typing import List
from tqdm import tqdm
import logging

logging.basicConfig(format="[%(levelname)s] %(message)s", level=logging.DEBUG)

def load_glove_vocabulary(glove_file: Path) -> List:
    """Load GloVe vocabulary"""
    logging.info('Loading GloVe vocabulary ...')
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
    logging.info('Saving GloVe vocabulary')
    with open(save_path, 'w') as f:
        f.write('\n'.join(vocab))

def main():
    logging.info("Generating GloVe vocabulary ...")
    vocab = load_glove_vocabulary(
        glove_file=Path('/home/P76104419/VQA/Thesis-VQA/data/glove.6B/glove.6B.300d.txt')
    )

    vocab += ['<start>', '<end>', '<pad>', '<oov>']

    save_glove_vocabulary(
        vocab=vocab,
        save_path=Path('/home/P76104419/VQA/Thesis-VQA/data/glove_vocabularies.txt')
    )
    logging.info("Done")

if __name__ == '__main__':
    main()