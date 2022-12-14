import random
from typing import Dict, List, Optional, Tuple
import re
import numpy as np
import torch


def random_seed(seed=10):
    """
    set random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    torch.backends.cudnn.deterministic = True


def set_device():
    """
    set device as 'cuda' if available, otherwise 'cpu'
    """
    # Use cuda if available
    if torch.cuda.is_available():
        device = "cuda"
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
    else:
        device = "cpu"
    return device


def get_tokens_and_ids(
    sentence: str, vocab_dict: Dict[str, int], is_cap: Optional[bool] = False
) -> Tuple[List[int], List[str]]:
    """get_tokens_and_ids

    Args:
        sentence (str): The sentence to tokenize.
        vocab_dict (Dict[str, int]): The vocabulary dictionary.
        is_cap (Optional[bool], optional): Adding <start> and <end> tokens into head and tail. Defaults to False.

    Returns:
        Tuple[List[int], List[str]]: The tokens and ids in list format.
    """
    # to lower case
    sentence = sentence.lower()
    # substitute to space
    sentence = re.sub(r"( ')|(' )|(\" )|( \")|(\n)", " ", sentence)
    # remove special characters
    sentence = re.sub(r"\.|,|\?", "", sentence)
    # reformat the `'s` to ` 's`
    sentence = re.sub(r"'s", " 's", sentence)

    # get all tokens which are not space
    tokens: List[str] = list(filter(lambda t: len(t) > 0, sentence.split()))

    # insert the <start> and <end> tags if is caption
    if is_cap:
        tokens = ["<start>"] + tokens + ["<end>"]

    # get token ids
    ids: List[int] = [vocab_dict.get(t, vocab_dict["<oov>"]) for t in tokens]

    return tokens, ids


def tokens_to_ids(
    token_list: List[str],
    vocab_dict: Dict[str, int],
) -> List[int]:
    """tokens_to_ids

    Args:
        token_list (List[str]): The list of tokens to be converted.
        vocab_dict (Dict[str, int]): The vocabulary dictionary.

    Returns:
        List[int]: The list of ids.

    Raises:
        ValueError: If the token is not in the vocabulary.
    """
    return [int(vocab_dict.get(t, -1)) for t in token_list]


def padding_ids(
    ids: List[int], max_len: int, vocab_dict: Dict[str, int]
) -> Tuple[np.ndarray, int]:
    """padding_ids

    Args:
        ids (List[int]): The list of id to be padded.
        max_len (int): The max length to padding.
        vocab_dict (Dict[str, int]): The vocabulary dictionary.

    Returns:
        Tuple[List[int], int]: The padded ids and the length of the ids.
    """
    # get the padding length
    pad_len = min(len(ids), max_len)

    if pad_len < max_len:
        # if the length is less than max_len, padding with <pad>
        extend_len = max_len - pad_len
        extend_list = [vocab_dict["<pad>"]] * extend_len
        ids.extend(extend_list)
    else:
        # if the length is more than max_len, truncate the ids
        ids = ids[:pad_len]

    return np.array(ids), pad_len
