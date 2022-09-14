#%%
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
from configs import pathes
import json
import numpy as np
#%%
dataset_name = 'train2014'
candidate_answers = pathes.f_CANDIDATE_ANSWERS.read_text().splitlines()
glove_vocabularies = pathes.f_GOLVE_VOCABULARIES.read_text().splitlines()

questions_path = pathes.d_ANNOTATIONS / 'vqa' / f'{dataset_name}_questions-chunk_0.npz'
questions = np.load(questions_path, allow_pickle=True)['arr_0']
#%%
questions[0].keys()

#%%
def load_answer(index):
    output = torch.zeros(len(candidate_answers))

    for key, value in answers[index].items():
        output[int(key)] = min(value, 3)

    return np.divide(output, 3)

la = load_answer(0)

# output = torch.zeros(len(candidate_answers))
#%%
answers[0]

#%%
class VQADataset(Dataset):
    """ VQA dataset"""
    def __init__(
        self,
        dataset_name: str,
    ):
        self.setup(dataset_name)

        
    def setup(self, dataset_name: str):
        """setup

        Args:
            dataset_name (str): train2014 or val2014
        """
        self.questions_path = pathes.d_ANNOTATIONS / 'vqa' / f'{dataset_name}_questions.json'
        self.questions = json.load(open(self.questions_path, 'r'))['data']

        self.answers_path = pathes.d_ANNOTATIONS / 'vqa' / f'{dataset_name}_answers.json'
        self.answers = json.load(open(self.answers_path, 'r'))['data']

        self.feature_path = pathes.d_COCO_FEATURE / dataset_name
        self.graph_path = pathes.d_COCO_GRAPH / dataset_name
            
        self.candidate_answers = pathes.f_CANDIDATE_ANSWERS.read_text().splitlines()
        self.glove_vocabularies = pathes.f_GOLVE_VOCABULARIES.read_text().splitlines()

        assert len(questions) == len(answers)
        
    def __len__(self):
        return len(self.questions)
    
    def load_answer(self, index):
        output = torch.zeros(len(self.candidate_answers))

        for key, value in self.answers[index].items():
            output[int(key)] = min(value, 3)

        return np.divide(output, 3)
        
    def get_vqa(self, index):
        filename = self.questions[index]['img_file']

        img_path = self.feature_path / filename
        image = np.load(img_path)['x']

        graph_path = self.graph_path / filename
        graph = np.load(graph_path)['graph']

        q = torch.tensor(self.questions[index]['q'], dtype=torch.long)

        return {
            'id': index,
            'v': image,
            'q': q,
            'a': self.load_answer(index),
            'graph': graph
        }
    
    def make_cache(self):
        cache_list = []
        for i in tqdm(range(len(self))):
            cache_list.append(self.get_vqa(i))
    
    def __getitem__(self, index):
        return self.get_vqa(index)