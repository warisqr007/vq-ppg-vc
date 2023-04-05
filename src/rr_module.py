import torch
import torch.nn as nn
import torch.nn.functional as F

class InterpLnr(nn.Module):
    
    def __init__(self, max_len_seq=128, max_len_pad=192, min_len_seg=19, max_len_seg=32):
        super().__init__()
        self.max_len_seq = max_len_seq
        self.max_len_pad = max_len_pad
        
        self.min_len_seg = min_len_seg
        self.max_len_seg = max_len_seg
        
        self.max_num_seg = self.max_len_seq // self.min_len_seg + 1
        #self.training = config.train
        
        
    def pad_sequences(self, sequences, max_ilen):
        channel_dim = sequences[0].size()[-1]
        #out_dims = (len(sequences), self.max_len_pad, channel_dim)
        out_dims = (len(sequences), max_ilen, channel_dim)
        out_tensor = sequences[0].data.new(*out_dims).fill_(0)
        
        for i, tensor in enumerate(sequences):
            #print(f'in rr before pad seq: {tensor.shape}')
            length = tensor.size(0)
            #out_tensor[i, :length, :] = tensor[:self.max_len_pad]
            out_tensor[i, :length, :] = tensor[:max_ilen]
            
        return out_tensor 
    

    def forward(self, x, len_seq):  
        
        # if not self.training:
        #     #print("Not Training! (from RR)")
        #     return x
        
        #print(f'Training: {self.training}')
        
        device = x.device
        batch_size = x.size(0)
        max_ilen = max(len_seq)

        # indices of each sub segment
        indices = torch.arange(self.max_len_seg*2, device=device)\
                  .unsqueeze(0).expand(batch_size*self.max_num_seg, -1)
        # scales of each sub segment
        scales = torch.rand(batch_size*self.max_num_seg, 
                            device=device) + 0.5
        
        idx_scaled = indices / scales.unsqueeze(-1)
        idx_scaled_fl = torch.floor(idx_scaled)
        lambda_ = idx_scaled - idx_scaled_fl
        
        len_seg = torch.randint(low=self.min_len_seg, 
                                high=self.max_len_seg, 
                                size=(batch_size*self.max_num_seg,1),
                                device=device)
        
        # end point of each segment
        idx_mask = idx_scaled_fl < (len_seg - 1)
       
        offset = len_seg.view(batch_size, -1).cumsum(dim=-1)
        # offset starts from the 2nd segment
        offset = F.pad(offset[:, :-1], (1,0), value=0).view(-1, 1)
        
        idx_scaled_org = idx_scaled_fl + offset
        
        len_seq_rp = torch.repeat_interleave(len_seq, self.max_num_seg)
        idx_mask_org = idx_scaled_org < (len_seq_rp - 1).unsqueeze(-1)
        
        idx_mask_final = idx_mask & idx_mask_org
        
        counts = idx_mask_final.sum(dim=-1).view(batch_size, -1).sum(dim=-1)
        
        index_1 = torch.repeat_interleave(torch.arange(batch_size, 
                                            device=device), counts)
        
        index_2_fl = idx_scaled_org[idx_mask_final].long()
        index_2_cl = index_2_fl + 1
        
        y_fl = x[index_1, index_2_fl, :]
        y_cl = x[index_1, index_2_cl, :]
        lambda_f = lambda_[idx_mask_final].unsqueeze(-1)
        
        y = (1-lambda_f)*y_fl + lambda_f*y_cl
        
        sequences = torch.split(y, counts.tolist(), dim=0)
       
        seq_padded = self.pad_sequences(sequences, max_ilen)
        
        return seq_padded 