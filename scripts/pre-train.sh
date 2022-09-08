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
                --c_len 15

python compare_cap.py --device cuda:0 \
                      --exp_name base+gat+cap-pred