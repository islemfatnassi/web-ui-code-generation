#!/bin/bash

python train.py  --config_file 'config-web-vitcnnattn.yaml' \
--model_arch 'vitcnn-attn' --dataset "web" \
--checkpoint_dir '/home/sana/TheseIslem/code/code-generation/checkpoints/'

#python eval.py --config_file 'config-flickr-vitcnnattn.yaml' --model_arch 'vitcnn-attn' --dataset "flickr" --checkpoint_dir '/home/sana/TheseIslem/code/image-captioning-main/checkpoints/vitcnn-attn/flickr/bs64_lr0.001_es256/checkpoint_epoch_50.pth.tar'