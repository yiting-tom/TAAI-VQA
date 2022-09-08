python pretrain.py  --comment base+gat+cap-pred \
                --load_path ../annot \
                --feature_path ../COCO_feature_36 \
                --decoder_type base \
                --encoder_type relation \
                --predictor_type att-cap \
                --decoder_hidden_dim 1024 \
                --batch_size 64 \
                --device cuda:0 \
                --rnn_type LSTM \
                --c_len 15 \
                --mode decode \
                --load_model epoch_3 \
                --dataset_type val \
                --save_path vqa-e_val_0.txt \
                --batches 44244

python pretrain.py  --comment base+gat+cap-pred \
                --load_path ../annot \
                --feature_path ../COCO_feature_36 \
                --decoder_type base \
                --encoder_type relation \
                --predictor_type att-cap \
                --decoder_hidden_dim 1024 \
                --batch_size 64 \
                --device cuda:0 \
                --rnn_type LSTM \
                --c_len 15 \
                --mode vqa-decode \
                --load_model epoch_3 \
                --dataset_type train2014 \
                --save_path vqa_train_0.txt \
                --batches 221878

python pretrain.py  --comment base+gat+cap-pred \
                --load_path ../annot \
                --feature_path ../COCO_feature_36 \
                --decoder_type base \
                --encoder_type relation \
                --predictor_type att-cap \
                --decoder_hidden_dim 1024 \
                --batch_size 64 \
                --device cuda:0 \
                --rnn_type LSTM \
                --c_len 15 \
                --mode vqa-decode \
                --load_model epoch_3 \
                --dataset_type val2014 \
                --save_path vqa_val_0.txt \
                --batches 107177