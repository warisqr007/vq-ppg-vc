from .mel_decoder_mol_encAddlf0 import MelDecoderMOL
from .mel_decoder_mol_v2 import MelDecoderMOLv2
from .rnn_ppg2mel import BiRnnPpg2MelModel
from .mel_decoder_lsa import MelDecoderLSA
from .transformer_bnftomel import Transformer
from .transformer_bnftomel_prosody import Transformer as TransformerProsody
from .transformer_bnftomel_prosody_ecapa import Transformer as TransformerProsodyEcapa
from .transformer_bnftomel_prosody_ecapa_256 import Transformer as TransformerProsodyEcapa256

def build_model(model_name: str):
    if model_name == "seq2seqmol":
        return MelDecoderMOL
    elif model_name == "seq2seqmolv2":
        return MelDecoderMOLv2
    elif model_name == "bilstm":
        return BiRnnPpg2MelModel
    elif model_name == "seq2seqlsa":
        return MelDecoderLSA
    elif model_name == "transformer-vc":
        return Transformer
    elif model_name == "transformer-vc-prosody":
        return TransformerProsody
    elif model_name == "transformer-vc-prosody-et":
        return TransformerProsodyEcapa
    elif model_name == "transformer-vc-prosody-et-256":
        return TransformerProsodyEcapa256
    else:
        raise ValueError(f"Unknown model name: {model_name}.")
