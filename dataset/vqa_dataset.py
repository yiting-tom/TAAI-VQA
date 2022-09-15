#%%
import torch
from torch.utils.data import Dataset
from configs import paths
import numpy as np
from configs import consts
#%%
chunk_questions_path = paths.d_ANNOTATIONS / 'vqa' / 'train2014_questions-chunk_1.npz'
q1 = np.load(chunk_questions_path, allow_pickle=True)['arr_0']
len(q1)
#%%
indexes = np.linspace(
    start=0,
    stop=consts.COCO_TRAIN2014_QA_SIZE-1,
    num=consts.FILE_CHUNK_NUM+1,
    dtype=int,
)[1:].tolist()

chunk_size = (consts.COCO_TRAIN2014_QA_SIZE-1) / (consts.FILE_CHUNK_NUM+1)
index = 266253
(index // chunk_size - 1), (index % chunk_size)
chunk_size


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
        self.chunk_indexes = np.linspace(
            start=0,
            stop=consts.COCO_TRAIN2014_DATA_SIZE,
            num=consts.FILE_CHUNK_NUM+1,
            dtype=int,
        )[1:].tolist()
        self.current_chunk_index = 0
        self.chunk_data_num = consts.COCO_TRAIN2014_DATA_SIZE // consts.FILE_CHUNK_NUM

        self.answers_path = paths.d_ANNOTATIONS / 'vqa' / f'{dataset_name}_answers.npz'
        self.answers = np.load(self.answers_path, allow_pickle=True)['arr_0']

        self.feature_path = paths.d_COCO_FEATURE / dataset_name
        self.graph_path = paths.d_COCO_GRAPH / dataset_name
            
        self.candidate_answers = paths.f_CANDIDATE_ANSWERS.read_text().splitlines()
        self.glove_vocabularies = paths.f_GOLVE_VOCABULARIES.read_text().splitlines()

    def __load_questions_data(self, index: int):
        if index > self.chunk_indexes[self.current_chunk_index]:
            self.chunk_questions_path = paths.d_ANNOTATIONS / 'vqa' / f'{self.dataset_name}_questions-chunk_{self.current_chunk_index}.npz'
            self.chunk_questions = np.load(self.chunk_questions_path, allow_pickle=True)['arr_0']
            self.current_chunk_index += 1

        if index < self.chunk_indexes[0]:
            return self.chunk_questions[index]

        shifted_index = index - self.chunk_indexes[self.current_chunk_index-1]

        return self.chunk_questions[shifted_index]
    
    def __load_answer_data(self, index: int):
        top_k = 3
        output = torch.zeros(len(self.candidate_answers))

        for key, value in self.answers[index].items():
            output[key] = min(value, top_k)

        return output / top_k

    def __len__(self):
        return len(self.answers)
        
    def get_vqa(self, index):
        a = self.__load_answer_data(index)
        q, image, graph = self.__load_questions_data(index)

        return {
            'id': index,
            'v': image,
            'q': q,
            'a': a,
            'graph': graph,
        }
    
    def __getitem__(self, index):
        return self.get_vqa(index)