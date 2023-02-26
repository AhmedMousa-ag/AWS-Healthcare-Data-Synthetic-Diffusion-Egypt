import torch
from torch.nn import Linear
import torch.nn as nn
import torch.nn.functional as F


class EMA_model:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model, step_start_ema=2000):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())


class Generator_conditional(nn.Module):
    def __init__(self, c_in=15, c_out=15, time_dim=1, num_classes=2):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.time_dim = time_dim
        self.inc = Linear(c_in, 15)
        self.flatten = nn.Flatten()
        self.s1 = Linear(15,64)
        self.outc = Linear(64, c_out)
        if num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_dim)
    
    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc.type(torch.FloatTensor).to(self.device)
    
    def forward(self, x, t, y):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)
        
        x1 = self.inc(x)
        x1 = F.relu(x1)
        if y is not None:
            t += self.label_emb(y)
        
        if len(x1.shape) < 2:
            x1 = torch.unsqueeze(x1,1)
        x2 = torch.cat((x1,t),1)
        x2 = torch.squeeze(x2)
        x2 = self.s1(x2)
        x2 = F.relu(x2)
        output = self.outc(x2)
        output = F.relu(output)
        return output