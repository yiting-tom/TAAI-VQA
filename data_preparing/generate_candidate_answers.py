import json
from pathlib import Path
import re
import string
from typing import List, Dict
from configs import paths
from configs.logger import l

contractions = {
    "aint": "ain't", "arent": "aren't", "cant": "can't", "couldve":
    "could've", "couldnt": "couldn't", "couldn'tve": "couldn't've",
    "couldnt've": "couldn't've", "didnt": "didn't", "doesnt":
    "doesn't", "dont": "don't", "hadnt": "hadn't", "hadnt've":
    "hadn't've", "hadn'tve": "hadn't've", "hasnt": "hasn't", "havent":
    "haven't", "hed": "he'd", "hed've": "he'd've", "he'dve":
    "he'd've", "hes": "he's", "howd": "how'd", "howll": "how'll",
    "hows": "how's", "Id've": "I'd've", "I'dve": "I'd've", "Im":
    "I'm", "Ive": "I've", "isnt": "isn't", "itd": "it'd", "itd've":
    "it'd've", "it'dve": "it'd've", "itll": "it'll", "let's": "let's",
    "maam": "ma'am", "mightnt": "mightn't", "mightnt've":
    "mightn't've", "mightn'tve": "mightn't've", "mightve": "might've",
    "mustnt": "mustn't", "mustve": "must've", "neednt": "needn't",
    "notve": "not've", "oclock": "o'clock", "oughtnt": "oughtn't",
    "ow's'at": "'ow's'at", "'ows'at": "'ow's'at", "'ow'sat":
    "'ow's'at", "shant": "shan't", "shed've": "she'd've", "she'dve":
    "she'd've", "she's": "she's", "shouldve": "should've", "shouldnt":
    "shouldn't", "shouldnt've": "shouldn't've", "shouldn'tve":
    "shouldn't've", "somebody'd": "somebodyd", "somebodyd've":
    "somebody'd've", "somebody'dve": "somebody'd've", "somebodyll":
    "somebody'll", "somebodys": "somebody's", "someoned": "someone'd",
    "someoned've": "someone'd've", "someone'dve": "someone'd've",
    "someonell": "someone'll", "someones": "someone's", "somethingd":
    "something'd", "somethingd've": "something'd've", "something'dve":
    "something'd've", "somethingll": "something'll", "thats":
    "that's", "thered": "there'd", "thered've": "there'd've",
    "there'dve": "there'd've", "therere": "there're", "theres":
    "there's", "theyd": "they'd", "theyd've": "they'd've", "they'dve":
    "they'd've", "theyll": "they'll", "theyre": "they're", "theyve":
    "they've", "twas": "'twas", "wasnt": "wasn't", "wed've":
    "we'd've", "we'dve": "we'd've", "weve": "we've", "werent":
    "weren't", "whatll": "what'll", "whatre": "what're", "whats":
    "what's", "whatve": "what've", "whens": "when's", "whered":
    "where'd", "wheres": "where's", "whereve": "where've", "whod":
    "who'd", "whod've": "who'd've", "who'dve": "who'd've", "wholl":
    "who'll", "whos": "who's", "whove": "who've", "whyll": "why'll",
    "whyre": "why're", "whys": "why's", "wont": "won't", "wouldve":
    "would've", "wouldnt": "wouldn't", "wouldnt've": "wouldn't've",
    "wouldn'tve": "wouldn't've", "yall": "y'all", "yall'll":
    "y'all'll", "y'allll": "y'all'll", "yall'd've": "y'all'd've",
    "y'alld've": "y'all'd've", "y'all'dve": "y'all'd've", "youd":
    "you'd", "youd've": "you'd've", "you'dve": "you'd've", "youll":
    "you'll", "youre": "you're", "youve": "you've"
}
manual_map = {
    'none': '0',
    'zero': '0',
    'one': '1',
    'two': '2',
    'three': '3',
    'four': '4',
    'five': '5',
    'six': '6',
    'seven': '7',
    'eight': '8',
    'nine': '9',
    'ten': '10'
}
articles = ['a', 'an', 'the']
period_strip = re.compile("(?<!\d)(\.)(?!\d)")
comma_strip = re.compile("(\d)(\,)(\d)")
PUNCTUATIONS = list(string.punctuation)

def process_punctuation(inText: str) -> str:
    outText = inText
    for p in PUNCTUATIONS:
        if (
            p + ' ' in inText or ' ' + p in inText
        ) or (
            re.search(comma_strip, inText) != None
        ):
            outText = outText.replace(p, '')
        else:
            outText = outText.replace(p, ' ')
    outText = period_strip.sub("", outText, re.UNICODE)
    return outText


def process_digit_article(inText: str) -> str:
    outText = []
    tempText = inText.lower().split()
    for word in tempText:
        word = manual_map.setdefault(word, word)
        if word not in articles:
            outText.append(word)
        else:
            pass
    for wordId, word in enumerate(outText):
        if word in contractions:
            outText[wordId] = contractions[word]
    outText = ' '.join(outText)
    return outText


def multiple_replace(text: str, wordDict: Dict[str, str]) -> str:
    for key in wordDict:
        text = text.replace(key, wordDict[key])
    return text


def preprocess_answer(answer: str) -> str:
    answer = process_digit_article(process_punctuation(answer))
    answer = answer.replace(',', '')
    return answer


def filter_answers(answers_dset, min_occurence) -> Dict:
    """This will change the answer to preprocessed version"""
    occurence = {}

    for ans_entry in answers_dset:
        gtruth = ans_entry['multiple_choice_answer']
        gtruth = preprocess_answer(gtruth)
        if gtruth not in occurence:
            occurence[gtruth] = set()
        occurence[gtruth].add(ans_entry['question_id'])
    temp = occurence.copy()
    for answer in temp.keys():
        if len(occurence[answer]) < min_occurence:
            occurence.pop(answer)

    l.info('Num of answers that appear >= %d times: %d' % (
        min_occurence, len(occurence)))
    return occurence

def load_datasets_answers(train_file: Path, val_file: Path) -> List:
    """Load answers from train and validation dataset"""
    l.info('Loading training dataset ...')
    train_answer_file = (train_file).absolute()
    train_answers = json.load(open(train_answer_file))

    l.info('Loading validation dataset ...')
    val_answer_file = (val_file).absolute()
    val_answers = json.load(open(val_answer_file))

    return train_answers['annotations'] + val_answers['annotations']

def save_candidate_answer(occurence: List, save_path: Path) -> None:
    """Save candidate answers to file"""
    l.info('Saving candidate answers')
    with open(save_path, 'w') as f:
        f.write('\n'.join(occurence))

def main():
    l.info('Generating candidate answers ...')
    answers: Dict = load_datasets_answers(
        train_file=paths.d_VQA / 'v2_mscoco_train2014_annotations.json',
        val_file=paths.d_VQA / 'v2_mscoco_val2014_annotations.json',
    )

    occurence: Dict = filter_answers(
        answers_dset=answers,
        min_occurence=9,
    )

    save_candidate_answer(
        occurence=occurence.keys(),
        sagve_path=paths.f_CANDIDATE_ANSWERS,
    )

if __name__ == '__main__':
    main()