import argparse
from configs import pathes

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    # Read
    p.add_argument('--vqa_e_dir',      type=str, default=pathes.d_VQAE)
    p.add_argument('--vqa_dir',        type=str, default=pathes.d_VQA)
    p.add_argument('--feature_dir',    type=str, default=pathes.d_COCO_FEATURE)
    p.add_argument('--ans_file',    type=str, default=pathes.f_CANDIDATE_ANSWERS)
    p.add_argument('--vocab_file',     type=str, default=pathes.f_GOLVE_VOCABULARIES)

    # Write
    p.add_argument('--save_path',      type=str, default=pathes.d_ANNOTATIONS)
    p.add_argument('--graph_dir',      type=str, default=pathes.d_COCO_GRAPH)

    # Parameters
    p.add_argument('--dataset_type',   type=str, default='all')
    p.add_argument('--q_len',          type=int, default=10)
    p.add_argument('--c_len',          type=int, default=15)

    return p.parse_args()