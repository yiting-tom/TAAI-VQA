#############################################################################
# Setup environments
#############################################################################
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install ray[default]

#############################################################################
# Prepare Data
#############################################################################
# Setup folders
mkdir ../COCO
mkdir ../annot
mkdir ../data

# GLoVe
curl https://nlp.stanford.edu/data/glove.6B.zip > glove.6B.zip
unzip glove.6B.zip -d ../data/glove.6B

# Images
wget http://images.cocodataset.org/zips/train2014.zip
unzip train2014.zip -d ../COCO/train2014

wget http://images.cocodataset.org/zips/val2014.zip
unzip val2014.zip -d ../COCO/val2014

# VQA Dataset
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip
unzip v2_Annotations_Train_mscoco.zip -d ../data/
rm v2_Annotations_Train_mscoco.zip

wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip
unzip v2_Annotations_Val_mscoco.zip -d ../data/
rm v2_Annotations_Val_mscoco.zip

wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip
unzip v2_Questions_Train_mscoco.zip -d ../data/
rm v2_Questions_Train_mscoco.zip

wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip
unzip v2_Questions_Val_mscoco.zip -d ../data/
rm v2_Questions_Val_mscoco.zip

# VQA-E Dataset
wget -O ../data/VQA-E_train_set.json "https://drive.google.com/u/0/uc?id=1CXogPObRixI1iR51T2px-Q75jdnhByCX&export=download"
wget -O ../data/VQA-E_val_set.json "https://drive.google.com/u/0/uc?id=12e8Px79J4lOT0NBUe2JVzTjbgfRy06qY&export=download"

#############################################################################
# Feature Extraction
#############################################################################
# Setup tools
git clone --recursive https://github.com/MILVLG/bottom-up-attention.pytorch
# Install Detectron2
cd detectron2
pip install -e .
cd ..

# Install apex
git clone https://github.com/NVIDIA/apex.git
cd apex
python setup.py install
cd ..

# install the rest modules
python setup.py build develop
pip install ray

# Download pre-trained feature model (Faster R-CNN, 36 regions per image)
wget -O bua-caffe-frcn-r101_with_attributes_fix36.pth https://awma1-my.sharepoint.com/:u:/g/personal/yuz_l0_tn/EUKhQ3hSRv9JrrW64qpNLSIBGoOjEGCkF8zvgBP9gKax-w?download=1

# Extract features
# Training
python extract_features.py  --mode caffe \
                            --gpu '2' \
                            --extract-mode bbox_feats \
                            --min-max-boxes 36,36 \
                            --config-file configs/bua-caffe/extract-bua-caffe-r101-fix36.yaml \
                            --image-dir ../../COCO/train2014 \
                            --bbox-dir ../../COCO/bbox \
                            --out-dir ../../COCO_feature_36/train2014
# Validation
python extract_features.py  --mode caffe \
                            --gpu '2' \
                            --extract-mode bbox_feats \
                            --min-max-boxes 36,36 \
                            --config-file configs/bua-caffe/extract-bua-caffe-r101-fix36.yaml \
                            --image-dir ../../COCO/val2014 \
                            --bbox-dir ../../COCO/bbox \
                            --out-dir ../../COCO_feature_36/val2014