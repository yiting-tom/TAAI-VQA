ROOT=$(shell pwd)
DATA=$(ROOT)/data
BOTTOM_UP_DIR=$(ROOT)/bottom-up-attention.pytorch

docker_build:
	docker-compose build

docker_run:
	docker run --gpus all -it --rm --name vqa -v ${ROOT}:/app vqa/core:latest

decompress_dataset:
	@echo "Unzipping GLoVe..."
	@unzip ${DATA}/zips/glove.6B.zip -d ${DATA}/glove.6B

	@echo "Unzipping COCO train 2014..."
	@unzip ${DATA}/zips/train2014.zip -d ${DATA}/COCO/images/

	@echo "Unzipping COCO val 2014..."
	@unzip ${DATA}/zips/val2014.zip -d ${DATA}/COCO/images/

	@echo "Unzipping VQA v2..."
	@unzip ${DATA}/zips/v2_Annotations_Train_mscoco.zip -d ${DATA}
	@unzip ${DATA}/zips/v2_Annotations_Val_mscoco.zip -d ${DATA}
	@unzip ${DATA}/zips/v2_Questions_Train_mscoco.zip -d ${DATA}
	@unzip ${DATA}/zips/v2_Questions_Val_mscoco.zip -d ${DATA}

	@echo "Unzipping VQA v2.0..."

download_datasets:
	@echo "Make directories 'data/' and 'zips/'"
	@mkdir -p ${DATA}/zips

	@echo "Downloading GLoVe..."
	@curl https://downloads.cs.stanford.edu/nlp/data/glove.6B.zip > ${DATA}/zips/glove.6B.zip

	@echo "Downloading COCO train 2014..."
	@curl http://images.cocodataset.org/zips/train2014.zip > ${DATA}/zips/train2014.zip

	@echo "Downloading COCO val 2014..."
	@curl http://images.cocodataset.org/zips/val2014.zip > ${DATA}/zips/val2014.zip

	@echo "Downloading VQA v2(4 files)..."
	@curl "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip" > ${DATA}/zips/v2_Annotations_Train_mscoco.zip
	@curl "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip" > ${DATA}/zips/v2_Annotations_Val_mscoco.zip
	@curl "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip" > ${DATA}/zips/v2_Questions_Train_mscoco.zip
	@curl "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip" > ${DATA}/zips/v2_Questions_Val_mscoco.zip

	@echo "Downloading VQA-E Dataset..."

download_vqae:
	@wget "https://drive.google.com/u/0/uc?id=1CXogPObRixI1iR51T2px-Q75jdnhByCX&export=download" -O ${DATA}/vqa-e/train_set.json
	@wget "https://drive.google.com/u/0/uc?id=12e8Px79J4lOT0NBUe2JVzTjbgfRy06qY&export=download" -O ${DATA}/vqa-e/val_set.json

	@echo "Downloading Datasets DONE!"

download_feature_extraction:
ifeq (,$(wildcard ${BOTTOM_UP_DIR}))
	@echo "Clone bottom-up-attention repo..."
	@git clone --recursive https://github.com/MILVLG/bottom-up-attention.pytorch ${BOTTOM_UP_DIR}
endif
ifeq (,$(wildcard ${BOTTOM_UP_DIR}/apex))
	@echo "Clone apex repo..."
	@git clone https://github.com/NVIDIA/apex.git ${BOTTOM_UP_DIR}/apex
endif
setup_feature_extraction:
	@cd ${BOTTOM_UP_DIR}/apex && python setup.py install
	@cd ${BOTTOM_UP_DIR}/detectron2 && pip install -e .
	@cd ${BOTTOM_UP_DIR} && python setup.py build develop

# Download pre-trained feature model (Faster R-CNN, 36 regions per image)
download_pretrained_model:
ifeq (,$(wildcard ${BOTTOM_UP_DIR}/bua-caffe-frcn-r101-k36.pth))
	wget https://awma1-my.sharepoint.com/:u:/g/personal/yuz_l0_tn/EUKhQ3hSRv9JrrW64qpNLSIBGoOjEGCkF8zvgBP9gKax-w?download=1 -O \
	${BOTTOM_UP_DIR}/bua-caffe-frcn-r101-k36.pth
endif

extract_train_bbox: download_pretrained_model
	@mkdir -p ${DATA}/COCO/bbox_36/train2014
	cd ${BOTTOM_UP_DIR} && python extract_features.py  --mode 'caffe' \
		--num-cpus 0 \
		--gpus '0,1' \
		--extract-mode 'bboxes' \
		--min-max-boxes '36,36' \
		--config-file ${BOTTOM_UP_DIR}/configs/caffe/test-caffe-r101-fix36.yaml \
		--image-dir ${DATA}/COCO/images/train2014 \
		--out-dir ${DATA}/COCO/bbox_36/train2014 \
		--fastmode

extract_val_bbox: download_pretrained_model
	@echo "Make dir for features..."
	@mkdir -p ${DATA}/COCO/bbox_36/val2014
	cd ${BOTTOM_UP_DIR} && python extract_features.py  --mode 'caffe' \
		--num-cpus 0 \
		--gpus '0,1' \
		--extract-mode 'bboxes' \
		--min-max-boxes 36,36 \
		--config-file ${BOTTOM_UP_DIR}/configs/caffe/test-caffe-r101-fix36.yaml \
		--image-dir ${DATA}/COCO/images/val2014 \
		--out-dir ${DATA}/COCO/bbox_36/val2014 \
		--fastmode

extract_train_feat: download_pretrained_model
	@echo "Make dir for features..."
	@mkdir -p ${DATA}/COCO/feature_36/train2014
	cd ${BOTTOM_UP_DIR} && python extract_features.py  --mode 'caffe' \
		--num-cpus 0 \
		--gpus '0,1' \
		--extract-mode 'bbox_feats' \
		--min-max-boxes '36,36' \
		--config-file ${BOTTOM_UP_DIR}/configs/caffe/test-caffe-r101-fix36.yaml \
		--image-dir ${DATA}/COCO/images/train2014 \
		--bbox-dir ${DATA}/COCO/bbox_36/train2014 \
		--out-dir ${DATA}/COCO/feature_36/train2014 \
		--fastmode

extract_val_feat: download_pretrained_model
	@echo "Make dir for features..."
	@mkdir -p ${DATA}/COCO/feature_36/val2014
	cd ${BOTTOM_UP_DIR} && python extract_features.py  --mode 'caffe' \
		--num-cpus 0 \
		--gpus '0,1' \
		--extract-mode 'bbox_feats' \
		--min-max-boxes 36,36 \
		--config-file ${BOTTOM_UP_DIR}/configs/caffe/test-caffe-r101-fix36.yaml \
		--image-dir ${DATA}/COCO/images/val2014 \
		--bbox-dir ${DATA}/COCO/bbox_36/val2014 \
		--out-dir ${DATA}/COCO/feature_36/val2014 \
		--fastmode

extract_all:
	@echo "Extracting features..."
	@make extract_train_bbox
	@make extract_train_feat
	@make extract_val_bbox
	@make extract_val_feat
	@echo "Extracting features DONE!"

preprocessing:
	@python3 ${ROOT}/data-preparing/generate_candidate_answers.py
	@python3 ${ROOT}/data-preparing/generate_glove_vocabularies.py
	@python3 ${ROOT}/data-preparing/preprocessing/preprocessing.py