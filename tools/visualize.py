import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import matplotlib.pyplot as plt
import seaborn as sns


def show_att(att, img, bbox, k=3, output=None):
    """Given the attention scores, show the top-k relevant regions."""
    # Find top-k relevant regions
    value, index = att.topk(k)
    value = value.tolist()
    index = index.tolist()
    
    if output is None:
        output = img.copy()
        output.putalpha(30)

    # Show the regions
    regions = {}
    for i in range(1,1+k):
        b = bbox[index[-i]]
        region = img.crop([b[0], b[1], b[2], b[3]])
        regions[index[-i]] = region
        if value[-i] < max(value):
            region.putalpha(128)
        output.paste(region, (int(b[0]), int(b[1])))

    draw = ImageDraw.Draw(output)
    font = ImageFont.load_default()
    
    # Draw rectangles and texts
    color = 'red'
    for i in range(k):
        b = bbox[index[i]]
        draw.rectangle([(b[0], b[1]), (b[2], b[3])], fill=None, outline=color, width=2)
        text = f'{index[i]} ({value[i]:.2f})'
        w, h = font.getsize(text)
        draw.rectangle([(b[0], b[1]), (b[0]+w+1, b[1]+h+1)], fill=color)
        draw.text([b[0], b[1]], text)
        color = 'lightcoral'
    return output, regions


def print_result(batch, predict, ans_list):
    print('Q:', batch['q_word'])
    print('C:', batch['c_word'])
    print('target:')
    for i, j in batch['target'].items():
        print(f'{min(j,3)/3:.2f}', ans_list[int(i)])
    print('\npredict: ', ans_list[torch.argmax(predict).item()])


def show_top_k_regions(model, dataset, ans_list, batch, img_path='../COCO', dataset_type='val2014', k=3):
    """Given a question, show the top-k relevant regions."""

    # Get prediction and attention map
    model.eval()
    with torch.no_grad():
        predict, att = model.get_att(batch)
        att = att.squeeze()

    # Prepare image and bbox
    img_file = batch['feature'][:-3] + 'jpg'
    img = Image.open(os.path.join(img_path, dataset_type, img_file))
    bbox = np.load(os.path.join(dataset.feature_path, batch['feature']))['bbox']

    # Show the top-k relevant regions
    output, regions = show_att(att, img, bbox, k=k)
    
    # Print results
    print_result(batch, predict, ans_list)
    return output, regions


def show_graph_att(model, batch, img_path='../COCO', feature_path='../COCO_feature_36', dataset_type='val2014', k=3, layer=-1):
    """Given a question, show the most important region and the top-k relevant regions according to the correlation scores."""
    # Get prediction and graph attentions
    model.eval()
    with torch.no_grad():
        predict, att = model.get_att(batch)
        index = att.argmax().item()
        g_att = model.encoder(batch, True)[layer].squeeze().mean(dim=1)[index,:]
        g_att[index] = 1
    
    # Prepare image and bbox
    img_file = batch['feature'][:-3] + 'jpg'
    img = Image.open(os.path.join(img_path, dataset_type, img_file))
    bbox = np.load(os.path.join(feature_path, dataset_type, batch['feature']))['bbox']
    
    # Show top-k relevant regions
    output = show_att(g_att, img, bbox, k=k+1)
    
    # Print results
    # print_result(batch, predict, ans_list)
    return output


def heat_map(model, batch, direction=0):
    a = model.encoder(batch, show_att=True)
    map = a[direction].squeeze().mean(dim=1).T
    map = map.cpu().detach()
    plt.figure(figsize=(10,8))
    ax = sns.heatmap(map, vmax=0.215)
    plt.xticks(rotation=0)
    plt.xlabel('region no. X')
    plt.ylabel('region no. Y')
    plt.title('direction =', direction)
    return ax