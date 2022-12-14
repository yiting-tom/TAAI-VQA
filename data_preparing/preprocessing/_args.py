import argparse
from configs import paths


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    def a(*args, **kwargs):
        p.add_argument(*args, **kwargs)

    # Read
    a("--vqa_e_dir", type=str, default=paths.d_VQAE)
    a("--vqa_dir", type=str, default=paths.d_VQA)
    a("--feature_dir", type=str, default=paths.d_COCO_FEATURE)
    a("--ans_file", type=str, default=paths.f_CANDIDATE_ANSWERS)
    a("--vocab_file", type=str, default=paths.f_GOLVE_VOCABULARIES)

    # Write
    a("--save_path", type=str, default=paths.d_ANNOTATIONS)
    a("--graph_dir", type=str, default=paths.d_COCO_GRAPH)

    # Parameters
    a("--dataset_type", type=str, default="all", choices=["all", "train", "val"])
    a("--q_len", type=int, default=10)
    a("--c_len", type=int, default=15)

    return p.parse_args()
