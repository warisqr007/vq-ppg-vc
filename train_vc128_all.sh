#!/bin/bash

. ./path.sh || exit 1;
export CUDA_VISIBLE_DEVICES=2


python main.py  --config /mnt/data1/waris/repo/vc-vq-subset/conf/transformer_vc_vq128_all.yaml \
                --name=transformer-vc-vq128-all \
                --ckpdir=/mnt/nvme-data1/waris/model_checkpoints/vc-vq \
                --seed=2 \
                --transvcsplinpconc



# python main.py  --config /mnt/data1/waris/repo/vc-vq-prosody/conf/transformer_vc_vq_prosody_v2.yaml \
#                 --name=transformer-vc-vq-dr-v2 \
#                 --seed=2 \
#                 --transvcsplinpconc