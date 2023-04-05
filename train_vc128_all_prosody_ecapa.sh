#!/bin/bash

. ./path.sh || exit 1;
export CUDA_VISIBLE_DEVICES=3


# python main.py  --config /mnt/data1/waris/repo/vc-vq-subset/conf/transformer_vc_vq64_all_spk.yaml \
#                 --name=transformer-vc-vq64-all-spk-bout \
#                 --seed=2 \
#                 --transvcspk

python main.py  --config /mnt/data1/waris/repo/vc-vq-subset/conf/transformer_vc_vq128_prosody_ecapa.yaml \
                --name=transformer-vc-vq128-all-prosody-ecapa \
                --ckpdir=/mnt/nvme-data1/waris/model_checkpoints/vc-vq \
                --seed=2 \
                --transvcprosodyecapa



# python main.py  --config /mnt/data1/waris/repo/vc-vq-prosody/conf/transformer_vc_vq_prosody_v2.yaml \
#                 --name=transformer-vc-vq-dr-v2 \
#                 --seed=2 \
#                 --transvcsplinpconc