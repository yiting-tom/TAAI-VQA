python finetune.py  --load_model base+gat+cap-pred \
                --load_epoch 3 \
                --batch_size 64 \
                --device cuda:1 \
                --lr 0.008 \
                --save_path fine-tune

python finetune.py  --load_model base+gat+cap-pred \
                --load_epoch 3 \
                --batch_size 64 \
                --device cuda:1 \
                --save_path fine-tune \
                --mode metric