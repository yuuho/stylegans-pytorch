import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# mapping前に潜在変数を超球面上に正規化
class PixelwiseNormalization(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x / torch.sqrt((x**2).mean(1,keepdim=True) + 1e-8)


# 移動平均を用いて潜在変数を正規化する．
class TruncationTrick(nn.Module):
    def __init__(self, num_target, threshold, output_num, style_dim):
        super().__init__()
        self.num_target = num_target
        self.threshold = threshold
        self.output_num = output_num
        self.register_buffer('avg_style', torch.zeros((style_dim,)))

    def forward(self, x):
        # in:(N,D) -> out:(N,O,D)
        N,D = x.shape
        O = self.output_num
        x = x.view(N,1,D).expand(N,O,D)
        rate = torch.cat([  torch.ones((N, self.num_target,   D)) *self.threshold,
                            torch.ones((N, O-self.num_target, D)) *1.0              ],1).to(x.device)
        avg = self.avg_style.view(1,1,D).expand(N,O,D)
        return avg + (x-avg)*rate


# 特徴マップ信号を増幅する
class Amplify(nn.Module):
    def __init__(self, rate):
        super().__init__()
        self.rate = rate
    def forward(self,x):
        return x * self.rate


# チャンネルごとに畳み込みのバイアス項を足す
class AddBias(nn.Module):
    def __init__(self, out_channels, lr):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(out_channels))
        torch.nn.init.zeros_(self.bias.data)
        self.bias_scaler = lr
    def forward(self, x):
        oC,*_ = self.bias.shape
        y = x + self.bias.view(1,oC,1,1)*self.bias_scaler
        return y


# 学習率を調整したFC層
class EqualizedFullyConnect(nn.Module):
    def __init__(self, in_dim, out_dim, lr):
        super().__init__()
        #gain, lr = 2**0.5, 0.01
        
        self.weight = nn.Parameter(torch.randn((out_dim,in_dim)))
        torch.nn.init.normal_(self.weight.data, mean=0.0, std=1.0/lr)
        self.weight_scaler = 1/(in_dim**0.5)*lr
        
        self.bias = nn.Parameter(torch.randn((out_dim,)))
        torch.nn.init.zeros_(self.bias.data)
        self.bias_scaler = lr

    def forward(self, x):
        # x (N,D)
        return F.linear(x, self.weight*self.weight_scaler, self.bias*self.bias_scaler)