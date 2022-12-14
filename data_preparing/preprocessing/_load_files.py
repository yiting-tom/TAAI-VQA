from typing import Any, Dict, List, Tuple


def load_vocab_ans_dicts(
    vocab_file: str,
    ans_file: str,
) -> Tuple[Dict[str, int], Dict[str, int]]:
    """load_vocab_ans_dicts

    Returns:
        Tuple[Dict[str, int], Dict[str, int]]: The vocab and ans dicts
    """
    # Read-out vocabulary list and candidate answers
    vocab_list: List[str] = __read_file_line_by_line(vocab_file)
    # Transform vocabulary list to token -> id dictionary
    vocab_dict: Dict[str, int] = __to_idx_dict(vocab_list)

    ans_list: List[str] = __read_file_line_by_line(ans_file)
    ans_dict: Dict[str, int] = __to_idx_dict(ans_list)

    return vocab_dict, ans_dict


def __to_idx_dict(target: List[Any]) -> Dict[Any, int]:
    """to_idx_dict

    Args:
        target (List[Any]): The target list to be converted.

    Returns:
        Dict[Any, int]: The converted dictionary.
    """
    return {t: i for i, t in enumerate(target)}


def __read_file_line_by_line(vocab_path: str) -> List[str]:
    """__get_vocab_list

    Args:
        vocab_path (str): The path to the vocabulary file.

    Returns:
        List[str]: The vocabulary list.
    """
    with open(vocab_path, encoding="utf-8") as f:
        vocab_list = f.read().split("\n")
    return vocab_list
