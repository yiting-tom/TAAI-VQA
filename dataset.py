import os
import json
import numpy as np
from torch.utils.data import Dataset


class VQADataset(Dataset):
    """ VQA dataset"""
    def __init__(self,
                 load_path: str,
                 feature_path: str,
                 dataset_name: str,
                 vocab_list: list,
                 ans_list: list,
                 graph_path: str='',
    ):
        """
        load_path: path for VQA annotations
        feature_path: path for COCO image features
        dataset_name: train2014 / val2014
        vocab_list: GloVe vocabulary list
        ans_list: answer candidate list
        graph_path: path for COCO graph (default = '' i.e. don't use graph)
        caption_id_path: no use
        """
        
        print(f'load {dataset_name} dataset')
        with open(os.path.join(load_path, f'{dataset_name}_questions.json')) as f:
            self.questions = json.load(f)['data']
        with open(os.path.join(load_path, f'{dataset_name}_answers.json')) as f:
            self.answers = json.load(f)['data']
            
        self.feature_path = os.path.join(feature_path, dataset_name)
        self.graph_path = os.path.join(graph_path, dataset_name)
        self.vocab_list = vocab_list
        self.ans_list = ans_list
        
        
    def __len__(self): return len(self.questions)
    
    def load_answer(self, index):
        output = np.array([0]*len(self.ans_list))
        for key, value in self.answers[index].items():
            output[int(key)] = min(value, 3)
        return np.divide(output, 3)
        
    def get_vqa(self, index):
        img_file = self.questions[index]['img_file']
        img = np.load(os.path.join(self.feature_path, img_file))
        output = {
            'id': index,
            'v': img['x'],
            'q': np.array(self.questions[index]['q']),
            'a': self.load_answer(index),
        }
        if self.graph_path != '':
            output['graph'] = np.load(os.path.join(self.graph_path, img_file))['graph']
        return output
    
    def __getitem__(self, index):
        return self.get_vqa(index)

class VQAEDataset(Dataset):
    def __init__(
        self,
        dataset_type: str,
        load_path: str,
        feature_path: str,
        graph_path: str = '',
        ans_num: int = 3129,
        caption_path: str = ''
    ) -> None:
        super().__init__()
        with open(os.path.join(load_path, f'vqa-e_{dataset_type}.json')) as f:
            load = json.load(f)
        self.dataset_type = {
            'train': 'train2014',
            'val': 'val2014',
            'test': 'test2015'
        }[dataset_type]
        self.c_len = load['c_len']
        self.data = load['data']
        self.feature_path = feature_path
        self.graph_path = graph_path
        self.ans_num = ans_num

        # if use costumized captions:
        if caption_path != '':
            print('load captions')
            with open(caption_path) as f:
                self.captions = json.load(f)
        else: self.captions = None


    def __len__(self) -> int:
        return len(self.data)

    def _load_feature(self, path, img_id):
        name = f'COCO_{self.dataset_type}_{str(img_id).zfill(12)}.npz'
        return np.load(os.path.join(path, self.dataset_type, name))

    def _load_answer(self, ans):
        output = np.array([0]*self.ans_num)
        for k, v in ans.items():
            output[int(k)] = min(v, 3)
        return np.divide(output, 3)

    def __getitem__(self, index):
        output = self.data[index].copy()
        output['id'] = index
        output['q'] = np.array(output['q'])
        if self.captions is not None: output['c'] = np.array(self.captions[index]['c'])
        else: output['c'] = np.array(output['c'])
        output['a'] = self._load_answer(output['a'])
        output['v'] = self._load_feature(self.feature_path, output['img_id'])['x']
        if self.graph_path != '':
            output['graph'] = self._load_feature(self.graph_path, output['img_id'])['graph']
        return output
        

class VQACaptionDataset(VQADataset):
    """VQA + COCO caption datset which use one caption for each Q-A pair."""
    def __init__(self,
                 load_path: str,
                 feature_path: str,
                 dataset_name: str,
                 vocab_list: list,
                 ans_list: list,
                 graph_path: str='',
                 caption_path: str='',
                 c_len: int = 15
    ):
        """
        load_path: path for VQA annotations
        feature_path: path for COCO image features
        dataset_name: train2014 / val2014
        vocab_list: GloVe vocabulary list
        ans_list: answer candidate list
        caption_path: path for captions
        graph_path: path for COCO graph (default = '' i.e. don't use graph)
        """
        super().__init__(load_path, feature_path, dataset_name, vocab_list, ans_list, graph_path)
        self.vocab_list = vocab_list
        if caption_path != '':
            print('load captions')
            with open(caption_path) as f:
                self.captions = json.load(f)
            self.c_len = c_len
        else: self.captions = None

    def __len__(self): return len(self.questions)

    def __getitem__(self, index):
        img_id = str(int(self.questions[index]['img_file'][-16:-4]))
        output = self.get_vqa(index)
        if self.captions is not None:
            output['c'] = np.array(self.captions[index]['c'])
            output['c_len'] = self.captions[index]['c_len']
        return output

