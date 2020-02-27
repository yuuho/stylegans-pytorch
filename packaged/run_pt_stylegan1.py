import argparse
from pathlib import Path
import pickle

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F


# コマンドライン引数の取得
def parse_args():
    parser = argparse.ArgumentParser(description='著者実装を動かしたり重みを抜き出したり')
    parser.add_argument('-w','--weight_dir',type=str,default='/tmp/stylegans-pytorch',
                            help='学習済みのモデルを保存する場所')
    parser.add_argument('-o','--output_dir',type=str,default='/tmp/stylegans-pytorch',
                            help='生成された画像を保存する場所')
    parser.add_argument('--batch_size',type=int,default=1,
                            help='バッチサイズ')
    parser.add_argument('--device',type=str,default='gpu',choices=['gpu','cpu'],
                            help='デバイス')
    args = parser.parse_args()
    args.resolution = 1024
    return args


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


# チャンネルごとにバイアス項を足す
class AddChannelwiseBias(nn.Module):
    def __init__(self, out_channels, lr):
        super().__init__()
        # lr = 1.0 (conv,mod,AdaIN), 0.01 (mapping)

        self.bias = nn.Parameter(torch.zeros(out_channels))
        torch.nn.init.zeros_(self.bias.data)
        self.bias_scaler = lr

    def forward(self, x):
        oC,*_ = self.bias.shape
        shape = (1,oC) if x.ndim==2 else (1,oC,1,1)
        y = x + self.bias.view(*shape)*self.bias_scaler
        return y


# 学習率を調整したFC層
class EqualizedFullyConnect(nn.Module):
    def __init__(self, in_dim, out_dim, lr):
        super().__init__()
        # lr = 0.01 (mapping), 1.0 (mod,AdaIN)

        self.weight = nn.Parameter(torch.randn((out_dim,in_dim)))
        torch.nn.init.normal_(self.weight.data, mean=0.0, std=1.0/lr)
        self.weight_scaler = 1/(in_dim**0.5)*lr
        
    def forward(self, x):
        # x (N,D)
        return F.linear(x, self.weight*self.weight_scaler, None)


# 固定ノイズ
class ElementwiseNoise(nn.Module):
    def __init__(self, ch, size_hw):
        super().__init__()
        self.register_buffer("const_noise", torch.randn((1, 1, size_hw, size_hw)))
        self.noise_scaler = nn.Parameter(torch.zeros((ch,)))

    def forward(self, x):
        N,C,H,W = x.shape
        noise = self.const_noise.expand(N,C,H,W)
        scaler = self.noise_scaler.view(1,C,1,1)
        return x + noise * scaler


# ブラー : 解像度を上げる畳み込みの後に使う
class Blur3x3(nn.Module):
    def __init__(self):
        super().__init__()
        f = np.array( [ [1/16, 2/16, 1/16],
                        [2/16, 4/16, 2/16],
                        [1/16, 2/16, 1/16]], dtype=np.float32).reshape([1, 1, 3, 3])
        self.filter = torch.from_numpy(f)

    def forward(self, x):
        _N,C,_H,_W = x.shape
        return F.conv2d(x, self.filter.to(x.device).expand(C,1,3,3), padding=1, groups=C)


# 学習率を調整した転置畳み込み (ブラーのための拡張あり)
class EqualizedFusedConvTransposed2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, lr):
        super().__init__()
        # lr = 1.0

        self.stride, self.padding = stride, padding

        self.weight = nn.Parameter(torch.empty(in_channels, out_channels, kernel_size, kernel_size))
        torch.nn.init.normal_(self.weight.data, mean=0.0, std=1.0/lr)
        self.weight_scaler = 1 / ((in_channels * (kernel_size ** 2) )**0.5) * lr

    def forward(self, x):
        # 3x3 conv を 4x4 transposed conv として使う
        i_ch, o_ch, _kh, _kw = self.weight.shape
        # Padding (L,R,T,B) で4x4の四隅に3x3フィルタを寄せて和で合成
        weight_4x4 = torch.cat([F.pad(self.weight, pad).view(1,i_ch,o_ch,4,4)
                for pad in [(0,1,0,1),(1,0,0,1),(0,1,1,0),(1,0,1,0)]]).sum(dim=0)
        return F.conv_transpose2d(x, weight_4x4*self.weight_scaler, stride=2, padding=1)
        # 3x3でconvしてからpadで4隅に寄せて計算しても同じでは？
        # padding0にしてStyleGAN2のBlurを使っても同じでは？


# 学習率を調整した畳込み
class EqualizedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, lr):
        super().__init__()
        # lr = 1.0

        self.stride, self.padding = stride, padding

        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        torch.nn.init.normal_(self.weight.data, mean=0.0, std=1.0/lr)
        self.weight_scaler = 1 / ((in_channels * (kernel_size ** 2) )**0.5) * lr

    def forward(self, x):
        N,C,H,W = x.shape
        return F.conv2d(x, self.weight*self.weight_scaler, None,
                    self.stride, self.padding)


# 学習率を調整したAdaIN
class EqualizedAdaIN(nn.Module):
    def __init__(self, fmap_ch, style_ch, lr):
        super().__init__()
        # lr = 1.0
        self.fc = EqualizedFullyConnect(style_ch, fmap_ch*2, lr)
        self.bias = AddChannelwiseBias(fmap_ch*2,lr)

    def forward(self, pack):
        x, style = pack
        #N,D = w.shape
        N,C,H,W = x.shape

        _vec = self.bias( self.fc(style) ).view(N,2*C,1,1) # (N,2C,1,1)
        scale, shift = _vec[:,:C,:,:], _vec[:,C:,:,:] # (N,C,1,1), (N,C,1,1)
        return (scale+1) * F.instance_norm(x, eps=1e-8) + shift


class Generator(nn.Module):

    structure = {
        'mapping': [['pixel_norm'], ['fc',512,512],['amp'],['b',512],['Lrelu'],['fc',512,512],['amp'],['b',512],['Lrelu'],
                                    ['fc',512,512],['amp'],['b',512],['Lrelu'],['fc',512,512],['amp'],['b',512],['Lrelu'],
                                    ['fc',512,512],['amp'],['b',512],['Lrelu'],['fc',512,512],['amp'],['b',512],['Lrelu'],
                                    ['fc',512,512],['amp'],['b',512],['Lrelu'],['fc',512,512],['amp'],['b',512],['Lrelu'],['truncation']],
        'START'    : [                                                      ['noiseE',512,    4], ['bias',512], ['Lrelu'] ],  'adain4a'   : [['adain',512]],
        'Fconv4'   : [        ['EqConv3x3',512, 512],              ['amp'], ['noiseE',512,    4], ['bias',512], ['Lrelu'] ],  'adain4b'   : [['adain',512]],  'toRGB_4'   : [['EqConv1x1',512, 3], ['bias',3]],
        'Uconv8'   : [['up'], ['EqConv3x3',512, 512], ['blur3x3'], ['amp'], ['noiseE',512,    8], ['bias',512], ['Lrelu'] ],  'adain8a'   : [['adain',512]],
        'Fconv8'   : [        ['EqConv3x3',512, 512],              ['amp'], ['noiseE',512,    8], ['bias',512], ['Lrelu'] ],  'adain8b'   : [['adain',512]],  'toRGB_8'   : [['EqConv1x1',512, 3], ['bias',3]],
        'Uconv16'  : [['up'], ['EqConv3x3',512, 512], ['blur3x3'], ['amp'], ['noiseE',512,   16], ['bias',512], ['Lrelu'] ],  'adain16a'  : [['adain',512]],
        'Fconv16'  : [        ['EqConv3x3',512, 512],              ['amp'], ['noiseE',512,   16], ['bias',512], ['Lrelu'] ],  'adain16b'  : [['adain',512]],  'toRGB_16'  : [['EqConv1x1',512, 3], ['bias',3]],
        'Uconv32'  : [['up'], ['EqConv3x3',512, 512], ['blur3x3'], ['amp'], ['noiseE',512,   32], ['bias',512], ['Lrelu'] ],  'adain32a'  : [['adain',512]],
        'Fconv32'  : [        ['EqConv3x3',512, 512],              ['amp'], ['noiseE',512,   32], ['bias',512], ['Lrelu'] ],  'adain32b'  : [['adain',512]],  'toRGB_32'  : [['EqConv1x1',512, 3], ['bias',3]],
        'Uconv64'  : [['up'], ['EqConv3x3',512, 256], ['blur3x3'], ['amp'], ['noiseE',256,   64], ['bias',256], ['Lrelu'] ],  'adain64a'  : [['adain',256]],
        'Fconv64'  : [        ['EqConv3x3',256, 256],              ['amp'], ['noiseE',256,   64], ['bias',256], ['Lrelu'] ],  'adain64b'  : [['adain',256]],  'toRGB_64'  : [['EqConv1x1',256, 3], ['bias',3]],
        'Uconv128' : [     ['EqConvT3x3EX',256, 128], ['blur3x3'], ['amp'], ['noiseE',128,  128], ['bias',128], ['Lrelu'] ],  'adain128a' : [['adain',128]],
        'Fconv128' : [        ['EqConv3x3',128, 128],              ['amp'], ['noiseE',128,  128], ['bias',128], ['Lrelu'] ],  'adain128b' : [['adain',128]],  'toRGB_128' : [['EqConv1x1',128, 3], ['bias',3]],
        'Uconv256' : [     ['EqConvT3x3EX',128,  64], ['blur3x3'], ['amp'], ['noiseE', 64,  256], ['bias', 64], ['Lrelu'] ],  'adain256a' : [['adain', 64]],
        'Fconv256' : [        ['EqConv3x3', 64,  64],              ['amp'], ['noiseE', 64,  256], ['bias', 64], ['Lrelu'] ],  'adain256b' : [['adain', 64]],  'toRGB_256' : [['EqConv1x1', 64, 3], ['bias',3]],
        'Uconv512' : [     ['EqConvT3x3EX', 64,  32], ['blur3x3'], ['amp'], ['noiseE', 32,  512], ['bias', 32], ['Lrelu'] ],  'adain512a' : [['adain', 32]],
        'Fconv512' : [        ['EqConv3x3', 32,  32],              ['amp'], ['noiseE', 32,  512], ['bias', 32], ['Lrelu'] ],  'adain512b' : [['adain', 32]],  'toRGB_512' : [['EqConv1x1', 32, 3], ['bias',3]],
        'Uconv1024': [     ['EqConvT3x3EX', 32,  16], ['blur3x3'], ['amp'], ['noiseE', 16, 1024], ['bias', 16], ['Lrelu'] ],  'adain1024a': [['adain', 16]],
        'Fconv1024': [        ['EqConv3x3', 16,  16],              ['amp'], ['noiseE', 16, 1024], ['bias', 16], ['Lrelu'] ],  'adain1024b': [['adain', 16]],  'toRGB_1024': [['EqConv1x1', 16, 3], ['bias',3]],
    }

    def _make_sequential(self,key):
        definition = {
            'pixel_norm'   :    lambda *config: PixelwiseNormalization(),
            'truncation'   :    lambda *config: TruncationTrick(
                                    num_target=8, threshold=0.7, output_num=18, style_dim=512 ),
            'fc'           :    lambda *config: EqualizedFullyConnect(
                                    in_dim=config[0],out_dim=config[1], lr=0.01),
            'b'            :    lambda *config: AddChannelwiseBias(out_channels=config[0], lr=0.01),
            'bias'         :    lambda *config: AddChannelwiseBias(out_channels=config[0], lr=1.0),
            'amp'          :    lambda *config: Amplify(2**0.5),
            'Lrelu'        :    lambda *config: nn.LeakyReLU(negative_slope=0.2),
            'EqConvT3x3EX' :    lambda *config: EqualizedFusedConvTransposed2d(
                                    in_channels=config[0], out_channels=config[1],
                                    kernel_size=3, stride=1, padding=1, lr=1.0),
            'EqConv3x3'    :    lambda *config: EqualizedConv2d(
                                    in_channels=config[0], out_channels=config[1],
                                    kernel_size=3, stride=1, padding=1, lr=1.0),
            'EqConv1x1'    :    lambda *config: EqualizedConv2d(
                                    in_channels=config[0], out_channels=config[1],
                                    kernel_size=1, stride=1, padding=0, lr=1.0),
            'noiseE'       :    lambda *config: ElementwiseNoise(ch=config[0], size_hw=config[1]),
            'blur3x3'      :    lambda *config: Blur3x3(),
            'up'           :    lambda *config: nn.Upsample(
                                    scale_factor=2,mode='nearest'),
            'adain'        :    lambda *config: EqualizedAdaIN(
                                    fmap_ch=config[0], style_ch=512, lr=1.0),
        }
        return nn.Sequential(*[ definition[k](*cfg) for k,*cfg in self.structure[key]])

    def __init__(self):
        super().__init__()

        # 固定入力値
        self.register_buffer('const',torch.ones((1, 512, 4, 4),dtype=torch.float32))

        # 今回は使わない
        self.register_buffer('image_mixing_rate',torch.zeros((1,))) # 複数のtoRGBの合成比率
        self.register_buffer('style_mixing_rate',torch.zeros((1,))) # スタイルの合成比率

        # 潜在変数のマッピングネットワーク
        self.mapping = self._make_sequential('mapping')
        self.blocks = nn.ModuleList([self._make_sequential(k) for k in [
                'START',    'Fconv4',   'Uconv8',   'Fconv8',   'Uconv16',  'Fconv16',
                'Uconv32',  'Fconv32',  'Uconv64',  'Fconv64',  'Uconv128', 'Fconv128',
                'Uconv256', 'Fconv256', 'Uconv512', 'Fconv512', 'Uconv1024','Fconv1024'
            ] ])
        self.adains = nn.ModuleList([self._make_sequential(k) for k in [
                'adain4a',    'adain4b',   'adain8a',   'adain8b',
                'adain16a',   'adain16b',  'adain32a',  'adain32b',
                'adain64a',   'adain64b',  'adain128a', 'adain128b',
                'adain256a',  'adain256b', 'adain512a', 'adain512b',
                'adain1024a', 'adain1024b'
            ] ])
        self.toRGBs = nn.ModuleList([self._make_sequential(k) for k in [
                'toRGB_4',   'toRGB_8',   'toRGB_16',  'toRGB_32',
                'toRGB_64',  'toRGB_128', 'toRGB_256', 'toRGB_512',
                'toRGB_1024'
            ] ])

    def forward(self, z):
        '''
        input:  z   : (N,D) D=512
        output: img : (N,3,1024,1024)
        '''
        N,D = z.shape

        styles = self.mapping(z) # (N,18,D)
        tmp = self.const.expand(N,512,4,4)
        for i, (adain, conv) in enumerate(zip(self.adains, self.blocks)):
            tmp = conv(tmp)
            tmp = adain( (tmp, styles[:,i,:]) )
        img = self.toRGBs[-1](tmp)

        return img


########## 以下，重み変換 ########

name_trans_dict = {
    'const'                    : ['any', 'G_synthesis/4x4/Const/const'                       ],
    'image_mixing_rate'        : ['uns', 'G_synthesis/lod'                                   ],
    'style_mixing_rate'        : ['uns', 'lod'                                               ],
    'mapping.1.weight'         : ['fc_', 'G_mapping/Dense0/weight'                           ],
    'mapping.3.bias'           : ['any', 'G_mapping/Dense0/bias'                             ],
    'mapping.5.weight'         : ['fc_', 'G_mapping/Dense1/weight'                           ],
    'mapping.7.bias'           : ['any', 'G_mapping/Dense1/bias'                             ],
    'mapping.9.weight'         : ['fc_', 'G_mapping/Dense2/weight'                           ],
    'mapping.11.bias'          : ['any', 'G_mapping/Dense2/bias'                             ],
    'mapping.13.weight'        : ['fc_', 'G_mapping/Dense3/weight'                           ],
    'mapping.15.bias'          : ['any', 'G_mapping/Dense3/bias'                             ],
    'mapping.17.weight'        : ['fc_', 'G_mapping/Dense4/weight'                           ],
    'mapping.19.bias'          : ['any', 'G_mapping/Dense4/bias'                             ],
    'mapping.21.weight'        : ['fc_', 'G_mapping/Dense5/weight'                           ],
    'mapping.23.bias'          : ['any', 'G_mapping/Dense5/bias'                             ],
    'mapping.25.weight'        : ['fc_', 'G_mapping/Dense6/weight'                           ],
    'mapping.27.bias'          : ['any', 'G_mapping/Dense6/bias'                             ],
    'mapping.29.weight'        : ['fc_', 'G_mapping/Dense7/weight'                           ],
    'mapping.31.bias'          : ['any', 'G_mapping/Dense7/bias'                             ],
    'mapping.33.avg_style'     : ['any', 'dlatent_avg'                                       ],
    'blocks.0.0.noise_scaler'  : ['any', 'G_synthesis/4x4/Const/Noise/weight'                ],
    'blocks.0.0.const_noise'   : ['any', 'G_synthesis/noise0'                                ],
    'blocks.0.1.bias'          : ['any', 'G_synthesis/4x4/Const/bias'                        ],
    'blocks.1.0.weight'        : ['con', 'G_synthesis/4x4/Conv/weight'                       ],
    'blocks.1.2.noise_scaler'  : ['any', 'G_synthesis/4x4/Conv/Noise/weight'                 ],
    'blocks.1.2.const_noise'   : ['any', 'G_synthesis/noise1'                                ],
    'blocks.1.3.bias'          : ['any', 'G_synthesis/4x4/Conv/bias'                         ],
    'blocks.2.1.weight'        : ['con', 'G_synthesis/8x8/Conv0_up/weight'                   ],
    'blocks.2.4.noise_scaler'  : ['any', 'G_synthesis/8x8/Conv0_up/Noise/weight'             ],
    'blocks.2.4.const_noise'   : ['any', 'G_synthesis/noise2'                                ],
    'blocks.2.5.bias'          : ['any', 'G_synthesis/8x8/Conv0_up/bias'                     ],
    'blocks.3.0.weight'        : ['con', 'G_synthesis/8x8/Conv1/weight'                      ],
    'blocks.3.2.noise_scaler'  : ['any', 'G_synthesis/8x8/Conv1/Noise/weight'                ],
    'blocks.3.2.const_noise'   : ['any', 'G_synthesis/noise3'                                ],
    'blocks.3.3.bias'          : ['any', 'G_synthesis/8x8/Conv1/bias'                        ],
    'blocks.4.1.weight'        : ['con', 'G_synthesis/16x16/Conv0_up/weight'                 ],
    'blocks.4.4.noise_scaler'  : ['any', 'G_synthesis/16x16/Conv0_up/Noise/weight'           ],
    'blocks.4.4.const_noise'   : ['any', 'G_synthesis/noise4'                                ],
    'blocks.4.5.bias'          : ['any', 'G_synthesis/16x16/Conv0_up/bias'                   ],
    'blocks.5.0.weight'        : ['con', 'G_synthesis/16x16/Conv1/weight'                    ],
    'blocks.5.2.noise_scaler'  : ['any', 'G_synthesis/16x16/Conv1/Noise/weight'              ],
    'blocks.5.2.const_noise'   : ['any', 'G_synthesis/noise5'                                ],
    'blocks.5.3.bias'          : ['any', 'G_synthesis/16x16/Conv1/bias'                      ],
    'blocks.6.1.weight'        : ['con', 'G_synthesis/32x32/Conv0_up/weight'                 ],
    'blocks.6.4.noise_scaler'  : ['any', 'G_synthesis/32x32/Conv0_up/Noise/weight'           ],
    'blocks.6.4.const_noise'   : ['any', 'G_synthesis/noise6'                                ],
    'blocks.6.5.bias'          : ['any', 'G_synthesis/32x32/Conv0_up/bias'                   ],
    'blocks.7.0.weight'        : ['con', 'G_synthesis/32x32/Conv1/weight'                    ],
    'blocks.7.2.noise_scaler'  : ['any', 'G_synthesis/32x32/Conv1/Noise/weight'              ],
    'blocks.7.2.const_noise'   : ['any', 'G_synthesis/noise7'                                ],
    'blocks.7.3.bias'          : ['any', 'G_synthesis/32x32/Conv1/bias'                      ],
    'blocks.8.1.weight'        : ['con', 'G_synthesis/64x64/Conv0_up/weight'                 ],
    'blocks.8.4.noise_scaler'  : ['any', 'G_synthesis/64x64/Conv0_up/Noise/weight'           ],
    'blocks.8.4.const_noise'   : ['any', 'G_synthesis/noise8'                                ],
    'blocks.8.5.bias'          : ['any', 'G_synthesis/64x64/Conv0_up/bias'                   ],
    'blocks.9.0.weight'        : ['con', 'G_synthesis/64x64/Conv1/weight'                    ],
    'blocks.9.2.noise_scaler'  : ['any', 'G_synthesis/64x64/Conv1/Noise/weight'              ],
    'blocks.9.2.const_noise'   : ['any', 'G_synthesis/noise9'                                ],
    'blocks.9.3.bias'          : ['any', 'G_synthesis/64x64/Conv1/bias'                      ],
    'blocks.10.0.weight'       : ['Tco', 'G_synthesis/128x128/Conv0_up/weight'               ],
    'blocks.10.3.noise_scaler' : ['any', 'G_synthesis/128x128/Conv0_up/Noise/weight'         ],
    'blocks.10.3.const_noise'  : ['any', 'G_synthesis/noise10'                               ],
    'blocks.10.4.bias'         : ['any', 'G_synthesis/128x128/Conv0_up/bias'                 ],
    'blocks.11.0.weight'       : ['con', 'G_synthesis/128x128/Conv1/weight'                  ],
    'blocks.11.2.noise_scaler' : ['any', 'G_synthesis/128x128/Conv1/Noise/weight'            ],
    'blocks.11.2.const_noise'  : ['any', 'G_synthesis/noise11'                               ],
    'blocks.11.3.bias'         : ['any', 'G_synthesis/128x128/Conv1/bias'                    ],
    'blocks.12.0.weight'       : ['Tco', 'G_synthesis/256x256/Conv0_up/weight'               ],
    'blocks.12.3.noise_scaler' : ['any', 'G_synthesis/256x256/Conv0_up/Noise/weight'         ],
    'blocks.12.3.const_noise'  : ['any', 'G_synthesis/noise12'                               ],
    'blocks.12.4.bias'         : ['any', 'G_synthesis/256x256/Conv0_up/bias'                 ],
    'blocks.13.0.weight'       : ['con', 'G_synthesis/256x256/Conv1/weight'                  ],
    'blocks.13.2.noise_scaler' : ['any', 'G_synthesis/256x256/Conv1/Noise/weight'            ],
    'blocks.13.2.const_noise'  : ['any', 'G_synthesis/noise13'                               ],
    'blocks.13.3.bias'         : ['any', 'G_synthesis/256x256/Conv1/bias'                    ],
    'blocks.14.0.weight'       : ['Tco', 'G_synthesis/512x512/Conv0_up/weight'               ],
    'blocks.14.3.noise_scaler' : ['any', 'G_synthesis/512x512/Conv0_up/Noise/weight'         ],
    'blocks.14.3.const_noise'  : ['any', 'G_synthesis/noise14'                               ],
    'blocks.14.4.bias'         : ['any', 'G_synthesis/512x512/Conv0_up/bias'                 ],
    'blocks.15.0.weight'       : ['con', 'G_synthesis/512x512/Conv1/weight'                  ],
    'blocks.15.2.noise_scaler' : ['any', 'G_synthesis/512x512/Conv1/Noise/weight'            ],
    'blocks.15.2.const_noise'  : ['any', 'G_synthesis/noise15'                               ],
    'blocks.15.3.bias'         : ['any', 'G_synthesis/512x512/Conv1/bias'                    ],
    'blocks.16.0.weight'       : ['Tco', 'G_synthesis/1024x1024/Conv0_up/weight'             ],
    'blocks.16.3.noise_scaler' : ['any', 'G_synthesis/1024x1024/Conv0_up/Noise/weight'       ],
    'blocks.16.3.const_noise'  : ['any', 'G_synthesis/noise16'                               ],
    'blocks.16.4.bias'         : ['any', 'G_synthesis/1024x1024/Conv0_up/bias'               ],
    'blocks.17.0.weight'       : ['con', 'G_synthesis/1024x1024/Conv1/weight'                ],
    'blocks.17.2.noise_scaler' : ['any', 'G_synthesis/1024x1024/Conv1/Noise/weight'          ],
    'blocks.17.2.const_noise'  : ['any', 'G_synthesis/noise17'                               ],
    'blocks.17.3.bias'         : ['any', 'G_synthesis/1024x1024/Conv1/bias'                  ],
    'adains.0.0.fc.weight'     : ['fc_', 'G_synthesis/4x4/Const/StyleMod/weight'             ],
    'adains.0.0.bias.bias'     : ['any', 'G_synthesis/4x4/Const/StyleMod/bias'               ],
    'adains.1.0.fc.weight'     : ['fc_', 'G_synthesis/4x4/Conv/StyleMod/weight'              ],
    'adains.1.0.bias.bias'     : ['any', 'G_synthesis/4x4/Conv/StyleMod/bias'                ],
    'adains.2.0.fc.weight'     : ['fc_', 'G_synthesis/8x8/Conv0_up/StyleMod/weight'          ],
    'adains.2.0.bias.bias'     : ['any', 'G_synthesis/8x8/Conv0_up/StyleMod/bias'            ],
    'adains.3.0.fc.weight'     : ['fc_', 'G_synthesis/8x8/Conv1/StyleMod/weight'             ],
    'adains.3.0.bias.bias'     : ['any', 'G_synthesis/8x8/Conv1/StyleMod/bias'               ],
    'adains.4.0.fc.weight'     : ['fc_', 'G_synthesis/16x16/Conv0_up/StyleMod/weight'        ],
    'adains.4.0.bias.bias'     : ['any', 'G_synthesis/16x16/Conv0_up/StyleMod/bias'          ],
    'adains.5.0.fc.weight'     : ['fc_', 'G_synthesis/16x16/Conv1/StyleMod/weight'           ],
    'adains.5.0.bias.bias'     : ['any', 'G_synthesis/16x16/Conv1/StyleMod/bias'             ],
    'adains.6.0.fc.weight'     : ['fc_', 'G_synthesis/32x32/Conv0_up/StyleMod/weight'        ],
    'adains.6.0.bias.bias'     : ['any', 'G_synthesis/32x32/Conv0_up/StyleMod/bias'          ],
    'adains.7.0.fc.weight'     : ['fc_', 'G_synthesis/32x32/Conv1/StyleMod/weight'           ],
    'adains.7.0.bias.bias'     : ['any', 'G_synthesis/32x32/Conv1/StyleMod/bias'             ],
    'adains.8.0.fc.weight'     : ['fc_', 'G_synthesis/64x64/Conv0_up/StyleMod/weight'        ],
    'adains.8.0.bias.bias'     : ['any', 'G_synthesis/64x64/Conv0_up/StyleMod/bias'          ],
    'adains.9.0.fc.weight'     : ['fc_', 'G_synthesis/64x64/Conv1/StyleMod/weight'           ],
    'adains.9.0.bias.bias'     : ['any', 'G_synthesis/64x64/Conv1/StyleMod/bias'             ],
    'adains.10.0.fc.weight'    : ['fc_', 'G_synthesis/128x128/Conv0_up/StyleMod/weight'      ],
    'adains.10.0.bias.bias'    : ['any', 'G_synthesis/128x128/Conv0_up/StyleMod/bias'        ],
    'adains.11.0.fc.weight'    : ['fc_', 'G_synthesis/128x128/Conv1/StyleMod/weight'         ],
    'adains.11.0.bias.bias'    : ['any', 'G_synthesis/128x128/Conv1/StyleMod/bias'           ],
    'adains.12.0.fc.weight'    : ['fc_', 'G_synthesis/256x256/Conv0_up/StyleMod/weight'      ],
    'adains.12.0.bias.bias'    : ['any', 'G_synthesis/256x256/Conv0_up/StyleMod/bias'        ],
    'adains.13.0.fc.weight'    : ['fc_', 'G_synthesis/256x256/Conv1/StyleMod/weight'         ],
    'adains.13.0.bias.bias'    : ['any', 'G_synthesis/256x256/Conv1/StyleMod/bias'           ],
    'adains.14.0.fc.weight'    : ['fc_', 'G_synthesis/512x512/Conv0_up/StyleMod/weight'      ],
    'adains.14.0.bias.bias'    : ['any', 'G_synthesis/512x512/Conv0_up/StyleMod/bias'        ],
    'adains.15.0.fc.weight'    : ['fc_', 'G_synthesis/512x512/Conv1/StyleMod/weight'         ],
    'adains.15.0.bias.bias'    : ['any', 'G_synthesis/512x512/Conv1/StyleMod/bias'           ],
    'adains.16.0.fc.weight'    : ['fc_', 'G_synthesis/1024x1024/Conv0_up/StyleMod/weight'    ],
    'adains.16.0.bias.bias'    : ['any', 'G_synthesis/1024x1024/Conv0_up/StyleMod/bias'      ],
    'adains.17.0.fc.weight'    : ['fc_', 'G_synthesis/1024x1024/Conv1/StyleMod/weight'       ],
    'adains.17.0.bias.bias'    : ['any', 'G_synthesis/1024x1024/Conv1/StyleMod/bias'         ],
    'toRGBs.0.0.weight'        : ['con', 'G_synthesis/ToRGB_lod8/weight'                     ],
    'toRGBs.0.1.bias'          : ['any', 'G_synthesis/ToRGB_lod8/bias'                       ],
    'toRGBs.1.0.weight'        : ['con', 'G_synthesis/ToRGB_lod7/weight'                     ],
    'toRGBs.1.1.bias'          : ['any', 'G_synthesis/ToRGB_lod7/bias'                       ],
    'toRGBs.2.0.weight'        : ['con', 'G_synthesis/ToRGB_lod6/weight'                     ],
    'toRGBs.2.1.bias'          : ['any', 'G_synthesis/ToRGB_lod6/bias'                       ],
    'toRGBs.3.0.weight'        : ['con', 'G_synthesis/ToRGB_lod5/weight'                     ],
    'toRGBs.3.1.bias'          : ['any', 'G_synthesis/ToRGB_lod5/bias'                       ],
    'toRGBs.4.0.weight'        : ['con', 'G_synthesis/ToRGB_lod4/weight'                     ],
    'toRGBs.4.1.bias'          : ['any', 'G_synthesis/ToRGB_lod4/bias'                       ],
    'toRGBs.5.0.weight'        : ['con', 'G_synthesis/ToRGB_lod3/weight'                     ],
    'toRGBs.5.1.bias'          : ['any', 'G_synthesis/ToRGB_lod3/bias'                       ],
    'toRGBs.6.0.weight'        : ['con', 'G_synthesis/ToRGB_lod2/weight'                     ],
    'toRGBs.6.1.bias'          : ['any', 'G_synthesis/ToRGB_lod2/bias'                       ],
    'toRGBs.7.0.weight'        : ['con', 'G_synthesis/ToRGB_lod1/weight'                     ],
    'toRGBs.7.1.bias'          : ['any', 'G_synthesis/ToRGB_lod1/bias'                       ],
    'toRGBs.8.0.weight'        : ['con', 'G_synthesis/ToRGB_lod0/weight'                     ],
    'toRGBs.8.1.bias'          : ['any', 'G_synthesis/ToRGB_lod0/bias'                       ],
}



























# 変換関数
ops_dict = {
    # 変調転置畳み込みの重み (iC,oC,kH,kW)
    'mTc' : lambda weight: torch.flip(torch.from_numpy(weight.transpose((2,3,0,1))), [2, 3]),
    # 転置畳み込みの重み (iC,oC,kH,kW)
    'Tco' : lambda weight: torch.from_numpy(weight.transpose((2,3,0,1))), 
    # 畳み込みの重み (oC,iC,kH,kW)
    'con' : lambda weight: torch.from_numpy(weight.transpose((3,2,0,1))),
    # 全結合層の重み (oD, iD)
    'fc_' : lambda weight: torch.from_numpy(weight.transpose((1, 0))),
    # 全結合層のバイアス項, 固定入力, 固定ノイズ, v1ノイズの重み (無変換)
    'any' : lambda weight: torch.from_numpy(weight),
    # Style-Mixingの値, v2ノイズの重み (scalar)
    'uns' : lambda weight: torch.from_numpy(np.array(weight).reshape(1)),
}


if __name__ == '__main__':
    # コマンドライン引数の取得
    args = parse_args()

    cfg = {
        'src_weight': 'stylegan1_ndarray.pkl',
        'src_latent': 'latents1.pkl',
        'dst_image' : 'stylegan1_pt.png',
        'dst_weight': 'stylegan1_state_dict.pth'
    }

    print('model construction...')
    generator = Generator()
    base_dict = generator.state_dict()

    print('model weights load...')
    with (Path(args.weight_dir)/cfg['src_weight']).open('rb') as f:
        src_dict = pickle.load(f)

    print('set state_dict...')
    new_dict = { k : ops_dict[v[0]](src_dict[v[1]]) for k,v in name_trans_dict.items()}
    generator.load_state_dict(new_dict)

    print('load latents...')
    with (Path(args.output_dir)/cfg['src_latent']).open('rb') as f:
        latents = pickle.load(f)
    latents = torch.from_numpy(latents.astype(np.float32))

    print('network forward...')
    device = torch.device('cuda') if torch.cuda.is_available() and args.device=='gpu' else torch.device('cpu')
    with torch.no_grad():
        N,_ = latents.shape
        generator.to(device)
        images = np.empty((N,args.resolution,args.resolution,3),dtype=np.uint8)

        for i in range(0,N,args.batch_size):
            j = min(i+args.batch_size,N)
            z = latents[i:j].to(device)
            img = generator(z)
            normalized = (img.clamp(-1,1)+1)/2*255
            images[i:j] = normalized.permute(0,2,3,1).cpu().numpy().astype(np.uint8)
            del z, img, normalized

    # 出力を並べる関数
    def make_table(imgs):
        # 出力する個数，解像度
        num_H, num_W = 4,4
        H = W = args.resolution
        num_images = num_H*num_W

        canvas = np.zeros((H*num_H,W*num_W,3),dtype=np.uint8)
        for i,p in enumerate(imgs[:num_images]):
            h,w = i//num_W, i%num_W
            canvas[H*h:H*-~h,W*w:W*-~w,:] = p[:,:,::-1]
        return canvas
 
    print('image output...')
    cv2.imwrite(str(Path(args.output_dir)/cfg['dst_image']), make_table(images))
    
    print('weight save...')
    torch.save(generator.state_dict(),str(Path(args.weight_dir)/cfg['dst_weight']))
    
    print('all done')
