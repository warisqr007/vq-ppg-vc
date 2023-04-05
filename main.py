#!/usr/bin/env python
# coding: utf-8
import sys
import torch
import argparse
import numpy as np
from utils.load_yaml import HpsYaml

# For reproducibility, comment these may speed up training
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Arguments
parser = argparse.ArgumentParser(description=
        'Training PPG2Mel VC model.')
parser.add_argument('--config', type=str, 
                    help='Path to experiment config, e.g., config/vc.yaml')
parser.add_argument('--name', default=None, type=str, help='Name for logging.')
parser.add_argument('--logdir', default='log/', type=str,
                    help='Logging path.', required=False)
parser.add_argument('--ckpdir', default='ckpt/', type=str,
                    help='Checkpoint path.', required=False)
parser.add_argument('--outdir', default='result/', type=str,
                    help='Decode output path.', required=False)
parser.add_argument('--load', default=None, type=str,
                    help='Load pre-trained model (for training only)', required=False)
parser.add_argument('--warm_start', action='store_true',
                    help='Load model weights only, ignore specified layers.')
parser.add_argument('--seed', default=0, type=int,
                    help='Random seed for reproducable results.', required=False)
parser.add_argument('--njobs', default=8, type=int,
                    help='Number of threads for dataloader/decoding.', required=False)
parser.add_argument('--cpu', action='store_true', help='Disable GPU training.')
parser.add_argument('--no-pin', action='store_true',
                    help='Disable pin-memory for dataloader')
parser.add_argument('--test', action='store_true', help='Test the model.')
parser.add_argument('--no-msg', action='store_true', help='Hide all messages.')
parser.add_argument('--finetune', action='store_true', help='Finetune model')
parser.add_argument('--oneshotvc', action='store_true', help='Oneshot VC model')
parser.add_argument('--ownarc', action='store_true', help='Oneshot VC model')
parser.add_argument('--bilstm', action='store_true', help='BiLSTM VC model')
parser.add_argument('--lsa', action='store_true', help='Use location-sensitive attention (LSA)')
parser.add_argument('--transvc', action='store_true', help='BiLSTM VC model')
parser.add_argument('--transvcspdloss', action='store_true', help='BiLSTM VC model')
parser.add_argument('--transvcspdlossmel', action='store_true', help='BiLSTM VC model')
parser.add_argument('--transvcsplinpconc', action='store_true', help='Transformer VC model')
parser.add_argument('--transvcspk', action='store_true', help='Transformer VC model - SPK')
parser.add_argument('--transvcprosody', action='store_true', help='Transformer VC model - Prosody')
parser.add_argument('--transvcprosodyecapa', action='store_true', help='Transformer VC model - Prosody ECAPA')
parser.add_argument('--transvcprosodyecapa256', action='store_true', help='Transformer VC model - Prosody ECAPA 256')
parser.add_argument('--transvcprosodyecapanodr', action='store_true', help='Transformer VC model - Prosody ECAPA NO DR')
###

paras = parser.parse_args()
setattr(paras, 'gpu', not paras.cpu)
setattr(paras, 'pin_memory', not paras.no_pin)
setattr(paras, 'verbose', not paras.no_msg)
# Make the config dict dot visitable
config = HpsYaml(paras.config) # yaml.load(open(paras.config, 'r'), Loader=yaml.FullLoader)

np.random.seed(paras.seed)
torch.manual_seed(paras.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(paras.seed)
# For debug use
# torch.autograd.set_detect_anomaly(True)

# Hack to preserve GPU ram just in case OOM later on server
# if paras.gpu and paras.reserve_gpu > 0:
    # buff = torch.randn(int(paras.reserve_gpu*1e9//4)).cuda()
    # del buff
if paras.transvcprosodyecapanodr:
    print(">>> Transformer VC (Prosody ECAPA) training ...")
    from bin.train_ppg2mel_transformer_prosody_ecapa_nodr import Solver
    mode = "train"
    solver = Solver(config, paras, mode)
    solver.load_data()
    solver.set_model()
    solver.exec()
    print(">>> Transformer VC (Prosody) train finished!")
    sys.exit(0)
elif paras.transvcprosodyecapa:
    print(">>> Transformer VC (Prosody ECAPA) training ...")
    from bin.train_ppg2mel_transformer_prosody_ecapa import Solver
    mode = "train"
    solver = Solver(config, paras, mode)
    solver.load_data()
    solver.set_model()
    solver.exec()
    print(">>> Transformer VC (Prosody) train finished!")
    sys.exit(0)
elif paras.transvcprosodyecapa256:
    print(">>> Transformer VC (Prosody ECAPA) training ...")
    from bin.train_ppg2mel_transformer_prosody_ecapa_256 import Solver
    mode = "train"
    solver = Solver(config, paras, mode)
    solver.load_data()
    solver.set_model()
    solver.exec()
    print(">>> Transformer VC (Prosody) train finished!")
    sys.exit(0)
elif paras.transvcprosody:
    print(">>> Transformer VC (Prosody) training ...")
    from bin.train_ppg2mel_transformer_prosody import Solver
    mode = "train"
    solver = Solver(config, paras, mode)
    solver.load_data()
    solver.set_model()
    solver.exec()
    print(">>> Transformer VC (Prosody) train finished!")
    sys.exit(0)
elif paras.transvcspk:
    print(">>> Transformer VC (SPK) training ...")
    from bin.train_ppg2mel_transformer_speaker_loss import Solver
    mode = "train"
    solver = Solver(config, paras, mode)
    solver.load_data()
    solver.set_model()
    solver.exec()
    print(">>> Transformer VC (SPK) train finished!")
    sys.exit(0)
elif paras.transvcsplinpconc:
    print(">>> VCT - Speaker loss input conc ...")
    from bin.train_ppg2mel_own_arctic_transformer_speaker_loss_from_inpconct import Solver
    mode = "train"
    solver = Solver(config, paras, mode)
    solver.load_data()
    solver.set_model()
    solver.exec()
    print(">>> Oneshot VC train finished!")
    sys.exit(0)
elif paras.ownarc:
    print(">>> Own VC training ...")
    if paras.bilstm:
        from bin.train_ppg2mel_own_arctic import Solver
    else:
        from bin.train_linglf02mel_seq2seq_oneshotvc import Solver
    mode = "train"
    solver = Solver(config, paras, mode)
    solver.load_data()
    solver.set_model()
    solver.exec()
    print(">>> Oneshot VC train finished!")
    sys.exit(0)
elif paras.transvc:
    print(">>> Own VC training ...")
    from bin.train_ppg2mel_own_arctic_transformer import Solver
    mode = "train"
    solver = Solver(config, paras, mode)
    solver.load_data()
    solver.set_model()
    solver.exec()
    print(">>> Oneshot VC train finished!")
    sys.exit(0)
elif paras.transvcspdloss:
    print(">>> VCT - Speaker loss ...")
    from bin.train_ppg2mel_own_arctic_transformer_speaker_loss import Solver
    mode = "train"
    solver = Solver(config, paras, mode)
    solver.load_data()
    solver.set_model()
    solver.exec()
    print(">>> Oneshot VC train finished!")
    sys.exit(0)

elif paras.transvcspdlossmel:
    print(">>> VCT - Speaker loss ...")
    from bin.train_ppg2mel_own_arctic_transformer_speaker_loss_from_mel import Solver
    mode = "train"
    solver = Solver(config, paras, mode)
    solver.load_data()
    solver.set_model()
    solver.exec()
    print(">>> Oneshot VC train finished!")
    sys.exit(0)
