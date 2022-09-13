from pathlib import Path
from typing import List
from data_preparing import preprocessing
from configs.logger import l

def process_all():
    args = preprocessing.parse_default()
    l.info(f"preprocess all with args: {args.__dict__}")

    # Ensure save path exists
    Path(args.save_path).mkdir(parents=True, exist_ok=True)

    # Load vocabulary and answer dictionaries
    vocab_dict, ans_dict = preprocessing.load_vocab_ans_dicts(
        vocab_file=args.vocab_file,
        ans_file=args.ans_file,
    )

    # Specify the dataset type
    dataset_types: List = ['train', 'val'] if args.dataset_type == 'all' else [args.dataset_type]
    
    # Iterate over the dataset types
    for dataset_type in dataset_types:
        # VQA-E preprocessing
        preprocessing.process_vqa_e_data(
            dataset_type=dataset_type,
            vocab_dict=vocab_dict,
            ans_dict=ans_dict,
            vqa_e_dir=args.vqa_e_dir,
            save_path=args.save_path,
            q_len=args.q_len,
            c_len=args.c_len,
        )
        # VQA preprocessing (Questions)
        preprocessing.process_vqa_questions(
            dataset_type=dataset_type,
            vocab_dict=vocab_dict,
            q_len=args.q_len,
            vqa_dir=args.vqa_dir,
            save_path=args.save_path,
        )
        # VQA preprocessing (Answers)
        preprocessing.process_vqa_answers(
            dataset_type=dataset_type,
            vocab_dict=vocab_dict,
            ans_dict=ans_dict,
            vqa_dir=args.vqa_dir,
            save_path=args.save_path,
        )
        # relationship graph preprocessing
        preprocessing.process_relationship(
            dataset_type=dataset_type,
            feature_dir=args.feature_dir,
            graph_dir=args.graph_dir,
        )