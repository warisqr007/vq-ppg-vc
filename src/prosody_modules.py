
"""Prosody Network related modules."""

'''
This is the ECAPA-TDNN model.
This model is modified and combined based on the following three projects:
  1. https://github.com/clovaai/voxceleb_trainer/issues/86
  2. https://github.com/lawlict/ECAPA-TDNN/blob/master/ecapa_tdnn.py
  3. https://github.com/speechbrain/speechbrain/blob/96077e9a1afff89d3f5ff47cab4bca0202770e4f/speechbrain/lobes/models/ECAPA_TDNN.py
'''

import math, torch
import torch.nn as nn
import torch.nn.functional as F

"""Prosody Network related modules."""

import warnings
warnings.filterwarnings("ignore")
import numpy as np


class SEModule(nn.Module):
    def __init__(self, channels, bottleneck=128):
        super(SEModule, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, bottleneck, kernel_size=1, padding=0),
            nn.ReLU(),
            # nn.BatchNorm1d(bottleneck), # I remove this layer
            nn.Conv1d(bottleneck, channels, kernel_size=1, padding=0),
            nn.Sigmoid(),
            )

    def forward(self, input):
        x = self.se(input)
        return input * x

class Bottle2neck(nn.Module):

    def __init__(self, inplanes, planes, kernel_size=None, dilation=None, scale = 8):
        super(Bottle2neck, self).__init__()
        width       = int(math.floor(planes / scale))
        self.conv1  = nn.Conv1d(inplanes, width*scale, kernel_size=1)
        self.bn1    = nn.BatchNorm1d(width*scale)
        self.nums   = scale -1
        convs       = []
        bns         = []
        num_pad = math.floor(kernel_size/2)*dilation
        for i in range(self.nums):
            convs.append(nn.Conv1d(width, width, kernel_size=kernel_size, dilation=dilation, padding=num_pad))
            bns.append(nn.BatchNorm1d(width))
        self.convs  = nn.ModuleList(convs)
        self.bns    = nn.ModuleList(bns)
        self.conv3  = nn.Conv1d(width*scale, planes, kernel_size=1)
        self.bn3    = nn.BatchNorm1d(planes)
        self.relu   = nn.ReLU()
        self.width  = width
        self.se     = SEModule(planes)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
          if i==0:
            sp = spx[i]
          else:
            sp = sp + spx[i]
          sp = self.convs[i](sp)
          sp = self.relu(sp)
          sp = self.bns[i](sp)
          if i==0:
            out = sp
          else:
            out = torch.cat((out, sp), 1)
        out = torch.cat((out, spx[self.nums]),1)

        out = self.conv3(out)
        out = self.relu(out)
        out = self.bn3(out)
        
        out = self.se(out)
        out += residual
        return out 

class PreEmphasis(torch.nn.Module):

    def __init__(self, coef: float = 0.97):
        super().__init__()
        self.coef = coef
        self.register_buffer(
            'flipped_filter', torch.FloatTensor([-self.coef, 1.]).unsqueeze(0).unsqueeze(0)
        )

    def forward(self, input: torch.tensor) -> torch.tensor:
        input = input.unsqueeze(1)
        input = F.pad(input, (1, 0), 'reflect')
        return F.conv1d(input, self.flipped_filter).squeeze(1)

class FbankAug(nn.Module):

    def __init__(self, freq_mask_width = (0, 8), time_mask_width = (0, 10)):
        self.time_mask_width = time_mask_width
        self.freq_mask_width = freq_mask_width
        super().__init__()

    def mask_along_axis(self, x, dim):
        original_size = x.shape
        batch, fea, time = x.shape
        if dim == 1:
            D = fea
            width_range = self.freq_mask_width
        else:
            D = time
            width_range = self.time_mask_width

        mask_len = torch.randint(width_range[0], width_range[1], (batch, 1), device=x.device).unsqueeze(2)
        mask_pos = torch.randint(0, max(1, D - mask_len.max()), (batch, 1), device=x.device).unsqueeze(2)
        arange = torch.arange(D, device=x.device).view(1, 1, -1)
        mask = (mask_pos <= arange) * (arange < (mask_pos + mask_len))
        mask = mask.any(dim=1)

        if dim == 1:
            mask = mask.unsqueeze(2)
        else:
            mask = mask.unsqueeze(1)
            
        x = x.masked_fill_(mask, 0.0)
        return x.view(*original_size)

    def forward(self, x):    
        x = self.mask_along_axis(x, dim=2)
        x = self.mask_along_axis(x, dim=1)
        return x

class ECAPA_TDNN(nn.Module):

    def __init__(self, C=1024):

        super(ECAPA_TDNN, self).__init__()

        # self.torchfbank = torch.nn.Sequential(
        #     PreEmphasis(),            
        #     torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=512, win_length=400, hop_length=160, \
        #                                          f_min = 20, f_max = 7600, window_fn=torch.hamming_window, n_mels=80),
        #     )

        self.specaug = FbankAug() # Spec augmentation

        self.conv1  = nn.Conv1d(80, C, kernel_size=5, stride=1, padding=2)
        self.relu   = nn.ReLU()
        self.bn1    = nn.BatchNorm1d(C)
        self.layer1 = Bottle2neck(C, C, kernel_size=3, dilation=2, scale=8)
        self.layer2 = Bottle2neck(C, C, kernel_size=3, dilation=3, scale=8)
        self.layer3 = Bottle2neck(C, C, kernel_size=3, dilation=4, scale=8)
        # I fixed the shape of the output from MFA layer, that is close to the setting from ECAPA paper.
        self.layer4 = nn.Conv1d(3*C, 1536, kernel_size=1)
        # self.layer4 = nn.Conv1d(2*C, 1536, kernel_size=1)
        self.attention = nn.Sequential(
            nn.Conv1d(4608, 256, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Tanh(), # I add this layer
            nn.Conv1d(256, 1536, kernel_size=1),
            nn.Softmax(dim=2),
            )
        self.bn5 = nn.BatchNorm1d(3072)
        self.fc6 = nn.Linear(3072, 192)
        self.bn6 = nn.BatchNorm1d(192)


    def forward(self, x, aug=True):
        # with torch.no_grad():
        #     # x = self.torchfbank(x)+1e-6
        #     # x = x.log()   
        #     x = x - torch.mean(x, dim=-1, keepdim=True)
        #     if aug == True and self.training:
        #         x = self.specaug(x)

        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x+x1)
        x3 = self.layer3(x+x1+x2)

        x = self.layer4(torch.cat((x1,x2,x3),dim=1))
        # x = self.layer4(torch.cat((x1,x2),dim=1))
        x = self.relu(x)

        t = x.size()[-1]

        global_x = torch.cat((x,torch.mean(x,dim=2,keepdim=True).repeat(1,1,t), torch.sqrt(torch.var(x,dim=2,keepdim=True).clamp(min=1e-4)).repeat(1,1,t)), dim=1)
        
        w = self.attention(global_x)

        mu = torch.sum(x * w, dim=2)
        sg = torch.sqrt( ( torch.sum((x**2) * w, dim=2) - mu**2 ).clamp(min=1e-4) )

        x = torch.cat((mu,sg),1)
        x = self.bn5(x)
        x = self.fc6(x)
        x = self.bn6(x)

        return x


####################################################################


class ProsodyEncoder(torch.nn.Module):
    """ Mel-Style Encoder """

    def __init__(self, input_channels, output_channels, n_spectral_layer=2, 
                n_temporal_layer=2, n_slf_attn_layer=1, n_slf_attn_head=4):
        super(ProsodyEncoder, self).__init__()
        # n_position = model_config["max_seq_len"] + 1
        # melencoder:
        # encoder_hidden: 128
        # spectral_layer: 2
        # temporal_layer: 2
        # slf_attn_layer: 1
        # slf_attn_head: 2
        # conv_kernel_size: 5
        # encoder_dropout: 0.1
        # add_llayer_for_adv: True
        self.n_mel_channels = input_channels #model_config["odim"]
        self.d_melencoder = output_channels #model_config["melencoder"]["encoder_hidden"]
        self.n_spectral_layer = n_spectral_layer #model_config["melencoder"]["spectral_layer"]
        self.n_temporal_layer = n_temporal_layer #model_config["melencoder"]["temporal_layer"]
        self.n_slf_attn_layer = n_slf_attn_layer #model_config["melencoder"]["slf_attn_layer"]
        self.n_slf_attn_head = n_slf_attn_head #model_config["melencoder"]["slf_attn_head"]
        d_k = d_v = (
            128 #model_config["melencoder"]["encoder_hidden"]
            // self.n_slf_attn_head #model_config["melencoder"]["slf_attn_head"]
        )
        kernel_size = 5 #model_config["melencoder"]["conv_kernel_size"]
        dropout = 0.2 #model_config["melencoder"]["encoder_dropout"]

        self.add_extra_linear = True #model_config["melencoder"]["add_llayer_for_adv"]

        # self.max_seq_len = model_config["max_seq_len"]

        self.fc_1 = FCBlock(self.n_mel_channels, self.d_melencoder)

        self.spectral_stack = torch.nn.ModuleList(
            [
                FCBlock(
                    self.d_melencoder, self.d_melencoder, activation=Mish()
                )
                for _ in range(n_spectral_layer)
            ]
        )

        self.temporal_stack = torch.nn.ModuleList(
            [
                nn.Sequential(
                    Conv1DBlock(
                        self.d_melencoder, 2 * self.d_melencoder, kernel_size, activation=Mish(), dropout=dropout
                    ),
                    nn.GLU(),
                )
                for _ in range(n_temporal_layer)
            ]
        )

        self.bottleneck_stack = torch.nn.ModuleList(
            [
                nn.Sequential(
                    torch.nn.Conv1d(self.d_melencoder, self.d_melencoder//8, kernel_size=1),
                    torch.nn.LeakyReLU(0.1),

                    torch.nn.InstanceNorm1d(32, affine=False),
                    torch.nn.Conv1d(
                        self.d_melencoder//8, self.d_melencoder//8, 
                        kernel_size=3*3, 
                        stride=1, 
                        padding=4,
                    ),
                    torch.nn.LeakyReLU(0.1),
            
                    torch.nn.InstanceNorm1d(32, affine=False),
                    torch.nn.Conv1d(
                        self.d_melencoder//8, self.d_melencoder, 
                        kernel_size=1,
                        stride=1
                    ),
                    torch.nn.LeakyReLU(0.1),
                    torch.nn.InstanceNorm1d(self.d_melencoder, affine=False),
                )
                for _ in range(n_temporal_layer)
            ]
        )

        self.slf_attn_stack = torch.nn.ModuleList(
            [
                MultiHeadAttention(
                    self.n_slf_attn_head, self.d_melencoder, d_k, d_v, dropout=dropout, layer_norm=True
                )
                for _ in range(n_slf_attn_layer)
            ]
        )

        self.fc_2 = FCBlock(self.d_melencoder, self.d_melencoder)

        if self.add_extra_linear:
            self.fc_3 = FCBlock(self.d_melencoder, self.d_melencoder)

    def forward(self, mel, mask=None):

        max_len = mel.shape[1]
        if mask is not None:
            slf_attn_mask = None#mask.expand(-1, max_len, -1)
        else:
            slf_attn_mask = None

        enc_output = self.fc_1(mel)

        # Spectral Processing
        for _, layer in enumerate(self.spectral_stack):
            enc_output = layer(enc_output)

        # Temporal Processing
        for _, layer in enumerate(self.temporal_stack):
            residual = enc_output
            enc_output = layer(enc_output)
            enc_output = residual + enc_output

        for _, layer in enumerate(self.bottleneck_stack):
            residual = enc_output
            # print(f'enc before: {enc_output.shape}')
            enc_output = layer(enc_output.transpose(1,2))
            # print(f'enc: {enc_output.shape}')
            # print(f'res: {residual.shape}')
            enc_output = residual + enc_output.transpose(1,2)

        # print(enc_output.shape)
        # Multi-head self-attention
        for _, layer in enumerate(self.slf_attn_stack):
            residual = enc_output
            enc_output, _ = layer(
                enc_output, enc_output, enc_output, mask=slf_attn_mask
            )
            enc_output = residual + enc_output

        # Final Layer
        enc_output = self.fc_2(enc_output) # [B, T, H]

        residual = enc_output
        if self.add_extra_linear:
            enc_output = self.fc_3(enc_output)

        # Temporal Average Pooling
        enc_output = torch.mean(enc_output, dim=1, keepdim=True) # [B, 1, H]

        return enc_output.squeeze(1)


class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class StyleAdaptiveLayerNorm(nn.Module):
    """ Style-Adaptive Layer Norm (SALN) """

    def __init__(self, w_size, hidden_size, bias=False):
        super(StyleAdaptiveLayerNorm, self).__init__()
        self.hidden_size = hidden_size
        self.affine_layer = LinearNorm(
            w_size,
            2 * hidden_size, # For both b (bias) g (gain) 
            bias,
        )

    def forward(self, h, w):
        """
        h --- [B, T, H_m]
        w --- [B, 1, H_w]
        o --- [B, T, H_m]
        """

        # Normalize Input Features
        mu, sigma = torch.mean(h, dim=-1, keepdim=True), torch.std(h, dim=-1, keepdim=True)
        y = (h - mu) / sigma # [B, T, H_m]

        # Get Bias and Gain
        b, g = torch.split(self.affine_layer(w), self.hidden_size, dim=-1)  # [B, 1, 2 * H_m] --> 2 * [B, 1, H_m]

        # Perform Scailing and Shifting
        o = g * y + b # [B, T, H_m]

        return o


class FCBlock(nn.Module):
    """ Fully Connected Block """

    def __init__(self, in_features, out_features, activation=None, bias=False, dropout=None, spectral_norm=False):
        super(FCBlock, self).__init__()
        self.fc_layer = nn.Sequential()
        self.fc_layer.add_module(
            "fc_layer",
            LinearNorm(
                in_features,
                out_features,
                bias,
                spectral_norm,
            ),
        )
        if activation is not None:
            self.fc_layer.add_module("activ", activation)
        self.dropout = dropout

    def forward(self, x):
        x = self.fc_layer(x)
        if self.dropout is not None:
            x = F.dropout(x, self.dropout, self.training)
        return x


class LinearNorm(nn.Module):
    """ LinearNorm Projection """

    def __init__(self, in_features, out_features, bias=False, spectral_norm=False):
        super(LinearNorm, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias)

        nn.init.xavier_uniform_(self.linear.weight)
        if bias:
            nn.init.constant_(self.linear.bias, 0.0)
        if spectral_norm:
            self.linear = nn.utils.spectral_norm(self.linear)

    def forward(self, x):
        x = self.linear(x)
        return x


class Conv1DBlock(nn.Module):
    """ 1D Convolutional Block """

    def __init__(self, in_channels, out_channels, kernel_size, activation=None, dropout=None, spectral_norm=False):
        super(Conv1DBlock, self).__init__()

        self.conv_layer = nn.Sequential()
        self.conv_layer.add_module(
            "conv_layer",
            ConvNorm(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=int((kernel_size - 1) / 2),
                dilation=1,
                w_init_gain="tanh",
                spectral_norm=spectral_norm,
            ),
        )
        if activation is not None:
            self.conv_layer.add_module("activ", activation)
        self.dropout = dropout

    def forward(self, x, mask=None):
        x = x.contiguous().transpose(1, 2)
        x = self.conv_layer(x)

        if self.dropout is not None:
            x = F.dropout(x, self.dropout, self.training)

        x = x.contiguous().transpose(1, 2)
        if mask is not None:
            x = x.masked_fill(mask.unsqueeze(-1), 0)

        return x


class ConvNorm(nn.Module):
    """ 1D Convolution """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=None,
        dilation=1,
        bias=True,
        w_init_gain="linear",
        spectral_norm=False,
    ):
        super(ConvNorm, self).__init__()

        if padding is None:
            assert kernel_size % 2 == 1
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )
        if spectral_norm:
            self.conv = nn.utils.spectral_norm(self.conv)

    def forward(self, signal):
        conv_signal = self.conv(signal)

        return conv_signal



class MultiHeadAttention(nn.Module):
    """ Multi-Head Attention """

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1, layer_norm=False, spectral_norm=False):
        super(MultiHeadAttention, self).__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = LinearNorm(d_model, n_head * d_k, spectral_norm=spectral_norm)
        self.w_ks = LinearNorm(d_model, n_head * d_k, spectral_norm=spectral_norm)
        self.w_vs = LinearNorm(d_model, n_head * d_v, spectral_norm=spectral_norm)

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model) if layer_norm else None

        self.fc = LinearNorm(n_head * d_v, d_model, spectral_norm=spectral_norm)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)
        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)  # (n*b) x lv x dv

        if mask is not None:
            mask = mask.repeat(n_head, 1, 1)  # (n*b) x .. x ..
        output, attn = self.attention(q, k, v, mask=mask)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = (
            output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)
        )  # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = output + residual
        if self.layer_norm is not None:
            output = self.layer_norm(output)

        return output, attn


class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention """

    def __init__(self, temperature):
        super(ScaledDotProductAttention, self).__init__()
        self.temperature = temperature
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)

        attn = self.softmax(attn)
        output = torch.bmm(attn, v)

        return output, attn
