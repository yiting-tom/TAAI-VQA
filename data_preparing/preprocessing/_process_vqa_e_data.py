from collections import Counter
import json
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm
from util import utils
from configs.logger import l

def process_vqa_e_data(
    dataset_type: str,
    vocab_dict: Dict[str, int],
    ans_dict: Dict[str, int],
    vqa_e_dir: str,
    save_path: str,
    q_len: int,
    c_len: int,
) -> None:
    """process_vqa_e_data

    Process the VQA-E dataset.
    Would save the processed data into the `save_path`.

    Args:
        dataset_type (str): The dataset type
        vocab_dict (Dict[str, int]): The vocabulary dictionary
        ans_dict (Dict[str, int]): The answer dictionary
        vqa_e_dir (str): The directory of the VQA-E dataset
        save_path (str): The json file to save the processed data
        q_len (int): The maximum length of the question
        c_len (int): The maximum length of the context
    """
    # Record
    result: List[dict] = []

    l.info(f"Reading VQA-E data from {vqa_e_dir}")
    # Read-out VQA-E data
    with open(f'{vqa_e_dir}/{dataset_type}_set.json') as f:
        data = json.load(f)

    # Iterate over all data
    for index in tqdm(range(len(data)), desc=f'VQA-E {dataset_type}'):
        temp = {}
        temp['img_id'] = data[index]['img_id']
        # ============== Process question ===============
        _, ids = utils.get_tokens_and_ids(
            sentence=data[index]['question'],
            vocab_dict=vocab_dict,
        )
        ids, _ = utils.padding_ids(
            ids=ids,
            max_len=q_len,
            vocab_dict=vocab_dict,
        )
        temp['q'] = ids.copy()
        # ============= Process explanation =============
        tokens, ids = utils.get_tokens_and_ids(
            sentence=data[index]['explanation'][0],
            vocab_dict=vocab_dict,
        )
        ids, pad_len = utils.padding_ids(
            ids=ids,
            max_len=c_len,
            vocab_dict=vocab_dict,
        )
        temp['c'] = ids.copy()
        temp['c_tokens'] = ' '.join(tokens)
        temp['c_len'] = pad_len
        # =============== Process answer ================
        cur_answer_list: List[str] = data[index]['answers']
        cnt = Counter(cur_answer_list)
        temp['a'] = dict(zip(
            # Change the key from token to id
            utils.tokens_to_ids(
                token_list=cnt.keys(),
                vocab_dict=ans_dict,
            ),
            cnt.values(),
        ))
        # ===============================================
        result.append(temp.copy())

    save_path: Path = Path(f'{save_path}/vqa-e/{dataset_type}.json')
    # Ensure the directory exists
    save_path.parent.mkdir(parents=True, exist_ok=True)
    # Save the processed data
    l.info(f"Saving VQA-E {dataset_type} data to {save_path}")
    with open(save_path, 'w') as f:
        json.dump({
            'q_len': q_len,
            'c_len': c_len,
            'data': result
        }, f)