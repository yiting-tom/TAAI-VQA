import os
import json
import pickle
import argparse
from tqdm import tqdm
import numpy as np
from util.utils import get_vocab_list, get_tokens, padding
from util.relation import spatial_relation

def parse_args():
    # set parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_path', type=str, default='../data')
    parser.add_argument('--feature_path', type=str, default='../COCO_feature_36')
    parser.add_argument('--vocab_path', type=str, default='../data/vocab_list.txt')
    parser.add_argument('--dataset_type', type=str, default='all')
    parser.add_argument('--ans_path', type=str, default='../data/answer_candidate.txt')
    parser.add_argument('--save_path', type=str, default='../annot')
    parser.add_argument('--graph_path', type=str, default='../COCO_graph_36')
    parser.add_argument('--q_len', type=int, default=10)
    parser.add_argument('--c_len', type=int, default=15)
    args = parser.parse_args()
    return args


def main():
    # Prepare save path
    args = parse_args()
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    vocab_list = get_vocab_list(args.vocab_path)
    ans_list = get_vocab_list(args.ans_path)

    dataset_types = ['train', 'val'] if args.dataset_type == 'all' else [args.dataset_type]
    
    for dataset_type in dataset_types:
        # VQA-E preprocessing
        with open(os.path.join(args.load_path, f'VQA-E_{dataset_type}_set.json')) as f:
            data = json.load(f)
        
        save = []
        for index in tqdm(range(len(data)), desc=f'VQA-E {dataset_type}'):
            temp = {}
            temp['img_id'] = data[index]['img_id']

            # for question
            tokens, _ = get_tokens(data[index]['question'], vocab_list)
            tokens, _ = padding(tokens, args.q_len, vocab_list)
            temp['q'] = tokens.copy()

            # for caption
            tokens, words = get_tokens(data[index]['explanation'][0], vocab_list)
            tokens, l = padding(tokens, args.c_len, vocab_list)
            temp['c'] = tokens.copy()
            temp['c_word'] = ' '.join(words)
            temp['c_len'] = l

            # for answer
            temp['a'] = {}
            for ans in set(data[index]['answers']):
                if ans in ans_list:
                    temp['a'][ans_list.index(ans)] = data[index]['answers'].count(ans)
            save.append(temp.copy())

        with open(os.path.join(args.save_path, f'vqa-e_{dataset_type}.json'), 'w') as f:
            json.dump({
                'q_len': args.q_len,
                'c_len': args.c_len,
                'data': save
            }, f)


        # VQA preprocessing
        # Questions
        with open(os.path.join(args.load_path, f'v2_OpenEnded_mscoco_{dataset_type}2014_questions.json')) as f:
            q_json = json.load(f)['questions']
            print('Load question json file.')

        q_data = []
        for i in tqdm(range(len(q_json)), desc=f'VQA {dataset_type} question'):
            image_id = q_json[i]['image_id']
            tokens, _ = get_tokens(q_json[i]['question'], vocab_list)
            tokens, _ = padding(tokens, args.q_len, vocab_list)
            q_data.append({
                'img_file': f'COCO_{dataset_type}2014_{str(image_id).zfill(12)}.npz',
                'q': tokens,
                'q_word': q_json[i]['question'],
            })

        with open(os.path.join(args.save_path, f'{dataset_type}2014_questions.json'), 'w') as f:
            f.write(json.dumps({'description': 'This is VQA v2.0 question dataset.', 
                                'data_type': dataset_type,
                                'data': q_data}))
        
        # Answers
        with open(os.path.join(args.load_path, f'v2_mscoco_{dataset_type}2014_annotations.json')) as f:
            a_json = json.load(f)['annotations']
            print('Load answer json file.')

        a_data = []
        ans_type = {'yes/no':[], 'number':[], 'other':[]}
        for i in tqdm(range(len(a_json)), desc=f'VQA {dataset_type} answer'):
            image_id = a_json[i]['image_id']
            answers = []
            for a in a_json[i]['answers']:
                answers.append(a['answer'])
            ans_dict = {}
            for a in set(answers):
                if a in ans_list: ans_dict[ans_list.index(a)] = answers.count(a)
            a_data.append(ans_dict)

            ans_type[a_json[i]['answer_type']].append(i)

        # Save answer dataset
        with open(os.path.join(args.save_path, f'{dataset_type}2014_answers.json'), 'w') as f:
            f.write(json.dumps({'description': 'This is VQA v2.0 answer dataset.', 
                                'data_type': dataset_type,
                                'data': a_data}))
        print('answer dataset saved.')
        if dataset_type == 'val':
            with open(os.path.join(args.save_path, 'index.pkl'), 'wb') as f:
                pickle.dump(ans_type, f)

        # Spatial relations
        graph_path = os.path.join(args.graph_path, dataset_type+'2014')
        if not os.path.exists(graph_path):
                os.makedirs(graph_path)

        feature_path = os.path.join(args.feature_path, dataset_type+'2014')
        files = os.listdir(feature_path)
        for file in tqdm(files, desc='graph '+dataset_type):
            feature = np.load(os.path.join(feature_path, file))
            graph = np.zeros([36, 36])
            for i in range(36):
                for j in range(i+1, 36):
                    graph[i,j], graph[j,i] = spatial_relation(feature['bbox'][i,:], feature['bbox'][j,:], h=feature['image_h'], w=feature['image_w'])
            np.savez(os.path.join(graph_path, file), graph=graph)

if __name__ =='__main__':
    main()