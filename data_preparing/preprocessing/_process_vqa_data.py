"""
This file defines 2 preprocessing functions for VQA-E dataset
for questions and answers.
"""
import json
import pickle
from typing import Counter, Dict, List
from pathlib import Path
from tqdm import tqdm
from util import utils
from configs.logger import l

def process_vqa_questions(
    dataset_type: str,
    vocab_dict: Dict[str, int],
    q_len: int,
    vqa_dir: str,
    save_path: str,
) -> None:
    """process_vqa_questions

    Args:
        dataset_type (str): The dataset type
        vocab_dict (Dict[str, int]): The vocabulary dictionary
        q_len (int): The maximum length of the question
        vqa_dir (str): The directory of the VQA dataset
        save_path (str): The json file to save the processed data
    """
    l.info(f"Reading VQA questions data from {vqa_dir}")
    with open(f'{vqa_dir}/v2_OpenEnded_mscoco_{dataset_type}2014_questions.json') as f:
        q_json = json.load(f)['questions']

    q_data = []
    # Iterate over the questions
    # q is a dict like this: {
    #     'image_id': 524291,
    #     'question': "What is in the person's hand?"
    #     'question_id': 524291000,
    # }
    for q in tqdm(q_json, desc=f'VQA {dataset_type} question'):
        image_id: int = q['image_id']
        _, ids = utils.get_tokens_and_ids(
            sentence=q['question'],
            vocab_dict=vocab_dict,
        )
        ids, _ = utils.padding_ids(
            ids=ids,
            max_len=q_len,
            vocab_dict=vocab_dict,
        )
        q_data.append({
            'img_file': f'COCO_{dataset_type}2014_{str(image_id).zfill(12)}.npz',
            'q': ids,
            'q_word': q['question'],
        })
    
    result_save_path: Path = Path(f'{save_path}/vqa/{dataset_type}2014_questions.json')
    # Ensure the directory exists
    result_save_path.parent.mkdir(parents=True, exist_ok=True)
    # Save the processed data
    l.info(f"Saving VQA {dataset_type} question data to {result_save_path}")
    with open(result_save_path, 'w') as f:
        f.write(
            json.dumps({
                'description': 'This is VQA v2.0 question dataset.', 
                'data_type': dataset_type,
                'data': q_data,
            })
        )
    
def process_vqa_answers(
    dataset_type: str,
    vocab_dict: Dict[str, int],
    ans_dict: Dict[str, int],
    vqa_dir: str,
    save_path: str,
) -> None:
    """process_vqa_answers
    
    Args:
        dataset_type (str): The dataset type
        vocab_dict (Dict[str, int]): The vocabulary dictionary
        ans_dict (Dict[str, int]): The answer dictionary
        vqa_dir (str): The directory of the VQA dataset
        save_path (str): The json file to save the processed data
    """
    l.info(f"Reading VQA annotations data from {vqa_dir}")
    with open(f'{vqa_dir}/v2_mscoco_{dataset_type}2014_annotations.json') as f:
        a_json = json.load(f)['annotations']

    a_data = []
    answer_types = {'yes/no':[], 'number':[], 'other':[]}
    # a is a dict like this: {
    #    'question_type': 'what is this',
    #    'multiple_choice_answer': 'net',
    #    'answers': [
    #        {'answer': 'net', 'answer_confidence': 'maybe', 'answer_id': 1},
    #        {'answer': 'net', 'answer_confidence': 'yes', 'answer_id': 2},
    #        {'answer': 'net', 'answer_confidence': 'yes', 'answer_id': 3},
    #        {'answer': 'netting', 'answer_confidence': 'yes', 'answer_id': 4},
    #        {'answer': 'net', 'answer_confidence': 'yes', 'answer_id': 5},
    #        {'answer': 'net', 'answer_confidence': 'yes', 'answer_id': 6},
    #        {'answer': 'mesh', 'answer_confidence': 'maybe', 'answer_id': 7},
    #        {'answer': 'net', 'answer_confidence': 'yes', 'answer_id': 8},
    #        {'answer': 'net', 'answer_confidence': 'yes', 'answer_id': 9},
    #        {'answer': 'net', 'answer_confidence': 'yes', 'answer_id': 10}
    #     ],
    #    'image_id': 458752,
    #    'answer_type': 'other',
    #    'question_id': 458752000,
    # }
    for idx, row in tqdm(enumerate(a_json), desc=f'VQA {dataset_type} answer'):
        answers: List[str] = []

        # get all answers from answer dictionary
        for answer_dict in row['answers']:
            answers.append(answer_dict['answer'])

        # count the number of each answer
        cnt = Counter(answers)

        # filter out the answer which appears in the answer dictionary
        filtered_answers = dict(filter(
            lambda x: x[0] in ans_dict,
            cnt.items(),
        ))

        # encode the tokens
        ans_cnt_dict = dict(zip(
            utils.tokens_to_ids(
                token_list=filtered_answers.keys(),
                vocab_dict=vocab_dict,
            ),
            filtered_answers.values(),
        ))

        # record the answer counter dictionary
        a_data.append(ans_cnt_dict)

        # record the answer counter dictionary
        answer_types[row['answer_type']].append(idx)

    result_save_path: Path = Path(f'{save_path}/vqa/{dataset_type}2014_answers.json')
    # Ensure the directory exists
    result_save_path.parent.mkdir(parents=True, exist_ok=True)
    # Save the processed data
    l.info(f"Saving VQA {dataset_type} question data to {result_save_path}")
    with open(result_save_path, 'w') as f:
        f.write(
            json.dumps({
            'description': 'This is VQA v2.0 answer dataset.', 
            'data_type': dataset_type,
            'data': a_data
            })
        )

    index_save_path: Path = Path(f'{save_path}/index.pkl')
    # Save the answer type
    l.info(f"Saving VQA answer type to {result_save_path}")
    if dataset_type == 'val':
        with open(index_save_path, 'wb') as f:
            pickle.dump(answer_types, f)