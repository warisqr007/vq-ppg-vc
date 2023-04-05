#!/bin/bash

. ./path.sh || exit 1;
export CUDA_VISIBLE_DEVICES=2




# python main.py  --config /mnt/data1/waris/repo/vc-vq-subset/conf/transformer_vc_vq256_prosody_ecapa_esd.yaml \
#                 --name=transformer-vc-vq256-all-prosody-ecapa-esd \
#                 --ckpdir=/mnt/nvme-data1/waris/model_checkpoints/vc-vq \
#                 --seed=2 \
#                 --transvcprosodyecapa256

python main.py  --config /mnt/data1/waris/repo/vc-vq-subset/conf/transformer_vc_vq256_prosody_ecapa.yaml \
                --name=transformer-vc-vq256-all-prosody-ecapa-rd \
                --ckpdir=/mnt/nvme-data1/waris/model_checkpoints/vc-vq \
                --seed=2 \
                --transvcprosodyecapa256
