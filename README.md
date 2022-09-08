# master_thesis_ChingShan_Tseng_P7608601

Relation-Aware Image Captioning for Explainable Visual Question Answering
- [oral ppt](http://redmine.ikmlab.csie.ncku.edu.tw/attachments/download/11041/20210916-oral.pdf)
- [thesis](http://redmine.ikmlab.csie.ncku.edu.tw/attachments/download/11053/20210928_ChingShan_thesis.pdf)


## Requirements
- torch==1.8.1+cu111
- torchvison=0.9.1+cu111
- numpy
- matplotlib

## Data
- [VQA v2 dataset](https://visualqa.org/download.html)
    - Questions
    - Annotations
    - Images
- [VQA-E dataset](https://github.com/liqing-ustc/VQA-E)
    - Questions
    - Annotations
- [Visual features](https://github.com/MILVLG/bottom-up-attention.pytorch)
    - 36 features per image (fixed)
    - Follow the instruction of feature extraction.

## Setup

```
bash scripts/setup.sh
```
This script will:
- Setup folders
- Download data
- Extract features following this [repo](https://github.com/MILVLG/bottom-up-attention.pytorch)

## Preprocessing

```
bash scripts/preprocessing.sh
```

This script will:
- Tokenize questions and captions of VQA-E and VQA
- Construct relation graph for each image

## Pre-train Stage

```
bash scripts/pre-train.sh
```

## Decode Captions

To speed up the process, we seperate the whole decoding process into two scripts and execute them parallelly.
Afterward, the decoded text files are merged as a json file.

```
bash scripts/decode.sh
```
```
bash scripts/decode2.sh
```

## Fine-tune Stage
```
bash scripts/fine-tune.sh
```

## Analysis

`analysis.ipynb` shows the figures and `sample.ipynb` shows the attention map of given image.