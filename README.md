# VQA Research

Relation-Aware Image Captioning for Explainable Visual Question Answering
- [oral ppt](http://redmine.ikmlab.csie.ncku.edu.tw/attachments/download/11041/20210916-oral.pdf)
- [thesis](http://redmine.ikmlab.csie.ncku.edu.tw/attachments/download/11053/20210928_ChingShan_thesis.pdf)

## Requirements
- torch==1.10.1+cu111
- torchvison==0.11.2+cu111
- ray==1.13.1
- opencv-python==4.6.0.66

## Datasets
- [VQA v2 dataset](https://visualqa.org/download.html)
    - Questions
    - Annotations
    - Images
- [VQA-E dataset](https://github.com/liqing-ustc/VQA-E)
    - Questions
    - Annotations
- [Visual features](https://github.com/MILVLG/bottom-up-attention.pytorch)
    - 36 features per image (fixed)
    - Follow the instruction to extract features.

## Setup
1. Download the datasets.
```bash
make download_datasets
```
2. Download and set up the feature extraction module repository.
```bash
# Clone the repository
make download_feature_extraction
# Set up the repository
make setup_feature_extraction
```
3. Download the pre-trained model.
```bash
make download_pretrained_model
```
4. Extract bbox.
```bash
make extract_train_bbox
make extract_val_bbox
```
5. Extract features by bbox.
```bash
make extract_train_feat
make extract_val_feat
```

## Docker
Please make sure that the docker and docker-compose are installed.
1. Build the docker image.
```bash
make docker_build
```
2. Run the docker container.
```bash
make docker_run
```