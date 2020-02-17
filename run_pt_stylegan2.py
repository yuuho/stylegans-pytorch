import argparse
import pickle
from pathlib import Path

import numpy as np
import cv2
import torch
from torch import nn
from torch.nn import functional as F


# コマンドライン引数の取得
def parse_args():
    parser = argparse.ArgumentParser(description='著者実装を動かしたり重みを抜き出したり')
    parser.add_argument('-w','--weight_dir',type=str,default='/tmp/stylegans-pytorch',
                            help='学習済みのモデルを保存する場所')
    parser.add_argument('-o','--output_dir',type=str,default='/tmp/stylegans-pytorch',
                            help='生成された画像を保存する場所')
    return parser.parse_args()


# mapping前に潜在変数を超球面上に正規化
class PixelwiseNormalization(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x / torch.sqrt((x**2).mean(1,keepdim=True) + 1e-8)


# 移動平均を用いて潜在変数を正規化する．
class TruncationTrick(nn.Module):
    def __init__(self):
        super().__init__()
        self.max_layer = 10
        self.threshold = 0.7
        self.register_buffer('avg_style', torch.zeros((512,)))

    def forward(self, x):
        # in:(N,D) -> out:(N,18,D)
        N,D = x.shape
        x = x.view(N,1,D).expand(N,18,D)
        rate = torch.cat([  torch.ones((N, self.max_layer,    D)) *self.threshold,
                            torch.ones((N, 18-self.max_layer, D)) *1.0              ],1).to(x.device)
        avg = self.avg_style.view(1,1,D).expand(N,18,D)
        return avg + (x-avg)*rate


# 解像度上げるときのModConvのみ
class Blur(nn.Module):
    def __init__(self):
        super().__init__()
        kernel = np.array([ [1/16, 2/16, 1/16],
                            [2/16, 4/16, 2/16],
                            [1/16, 2/16, 1/16]],dtype=np.float32)
        pads = [[(0,1),(0,1)],[(0,1),(1,0)],[(1,0),(0,1)],[(1,0),(1,0)]]
        kernel = np.stack( [np.pad(kernel,pad,'constant') for pad in pads] ).sum(0)
        #kernel [ [1/16, 3/16, 3/16, 1/16,],
        #         [3/16, 9/16, 9/16, 3/16,],
        #         [3/16, 9/16, 9/16, 3/16,],
        #         [1/16, 3/16, 3/16, 1/16,] ]
        self.kernel = torch.from_numpy(kernel)

    def forward(self, feature):
        # featureは(N,C,H+1,W+1)
        kernel = self.kernel.clone().to(feature.device)
        _N,C,_Hp1,_Wp1 = feature.shape
        return F.conv2d(feature, kernel.expand(C,1,4,4), padding=1, groups=C)


class Amplify(nn.Module):
    def __init__(self, rate):
        super().__init__()
        self.rate = rate
    def forward(self,x):
        return x * self.rate


class NoiseInjection(nn.Module):
    def __init__(self, resolution):
        super().__init__()
        self.noise_scaler = nn.Parameter(torch.zeros(1))
        self.register_buffer("const_noise", torch.randn((1, 1, resolution, resolution)))

    def forward(self, x):
        N,C,H,W = x.shape
        noise = self.const_noise.expand(N,C,H,W)
        return x + self.noise_scaler * noise


class AddBias(nn.Module):
    def __init__(self, out_channels):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(out_channels))
    def forward(self, x):
        oC,*_ = self.bias.shape
        y = x + self.bias.view(1,oC,1,1)
        return y


# MappingもModulationもAffine(=biasあり)
class EqualizedFullyConnect(nn.Module):
    def __init__(self, in_dim, out_dim, lr=1):
        super().__init__()

        self.weight = nn.Parameter(torch.empty(out_dim, in_dim))
        torch.nn.init.normal_(self.weight.data, mean=0.0, std=1.0/lr)
        self.weight_scaler = 1 / in_dim**0.5 * lr

        self.bias = nn.Parameter(torch.zeros(out_dim))
        self.bias_scaler = lr

    def forward(self, feature):
        out = F.linear(feature, self.weight*self.weight_scaler, bias=self.bias*self.bias_scaler)
        return out


class EqualizedModulatedConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, style_dim, padding, stride, demodulate=True, lr=1):
        super().__init__()

        self.padding, self.stride = padding, stride
        self.demodulate = demodulate

        self.weight = nn.Parameter( torch.randn(in_channels, out_channels, kernel_size, kernel_size))
        torch.nn.init.normal_(self.weight.data, mean=0.0, std=1.0/lr)
        self.weight_scaler = 1 / (in_channels * kernel_size*kernel_size)**0.5 * lr

        self.fc = EqualizedFullyConnect(style_dim, in_channels, lr=1)

    def forward(self, pack):
        x, style = pack
        N, iC, H, W = x.shape
        iC, oC, kH, kW = self.weight.shape

        mod_rates = self.fc(style)+1
        modulated_weight = self.weight_scaler*self.weight.view(1,iC,oC,kH,kW) * mod_rates.view(N,iC,1,1,1)

        if self.demodulate:
            demod_norm = 1 / ((modulated_weight**2).sum([1,3,4]) + 1e-8)**0.5 # (N, oC)
            weight = modulated_weight * demod_norm.view(N, 1, oC, 1, 1) # (N,oC,iC,kH,kW)
        else:
            weight = modulated_weight

        x = x.view(1, N*iC, H, W)
        weight = weight.view(N*iC,oC,kH,kW)
        out = F.conv_transpose2d(x, weight, padding=self.padding, stride=self.stride, groups=N)

        _, _, Hp1, Wp1 = out.shape
        out = out.view(N, oC, Hp1, Wp1)

        return out


class EqualizedModulatedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, style_dim, padding, stride, demodulate=True, lr=1):
        super().__init__()

        self.padding, self.stride = padding, stride
        self.demodulate = demodulate

        self.weight = nn.Parameter( torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        torch.nn.init.normal_(self.weight.data, mean=0.0, std=1.0/lr)
        self.weight_scaler = 1 / (in_channels*kernel_size*kernel_size)**0.5 * lr
        
        self.fc = EqualizedFullyConnect(style_dim, in_channels, lr=1)

    def forward(self, pack):
        x, style = pack
        N, iC, H, W = x.shape
        oC, iC, kH, kW = self.weight.shape

        mod_rates = self.fc(style)+1 # (N, iC)
        modulated_weight = self.weight_scaler*self.weight.view(1,oC,iC,kH,kW) * mod_rates.view(N,1,iC,1,1) # (N,oC,iC,kH,kW)

        if self.demodulate:
            demod_norm = 1 / ((modulated_weight**2).sum([2,3,4]) + 1e-8)**0.5 # (N, oC)
            weight = modulated_weight * demod_norm.view(N, oC, 1, 1, 1) # (N,oC,iC,kH,kW)
        else: # ToRGB
            weight = modulated_weight

        out = F.conv2d(x.view(1,N*iC,H,W), weight.view(N*oC,iC,kH,kW),padding=self.padding, stride=self.stride, groups=N).view(N,oC,H,W)
        return out


class Generator(nn.Module):

    structure = {
        'mapping': [['pixel_norm'],['fc',512,512],['amp'],['Lrelu'],['fc',512,512],['amp'],['Lrelu'],['fc',512,512],['amp'],['Lrelu'],
                                    ['fc',512,512],['amp'],['Lrelu'],['fc',512,512],['amp'],['Lrelu'],['fc',512,512],['amp'],['Lrelu'],
                                    ['fc',512,512],['amp'],['Lrelu'],['fc',512,512],['amp'],['Lrelu'],['truncation']],
        'Fconv4'   : [['EqModConv3x3',  512, 512],           ['noise',   4], ['bias',512], ['amp'], ['Lrelu'] ], 'toRGB_4'    : [['EqModConv1x1', 512,   3], ['bias',3]],
        'Uconv8'   : [['EqModConvT3x3', 512, 512], ['blur'], ['noise',   8], ['bias',512], ['amp'], ['Lrelu'] ],
        'Fconv8'   : [['EqModConv3x3',  512, 512],           ['noise',   8], ['bias',512], ['amp'], ['Lrelu'] ], 'toRGB_8'    : [['EqModConv1x1', 512,   3], ['bias',3]],
        'Uconv16'  : [['EqModConvT3x3', 512, 512], ['blur'], ['noise',  16], ['bias',512], ['amp'], ['Lrelu'] ],
        'Fconv16'  : [['EqModConv3x3',  512, 512],           ['noise',  16], ['bias',512], ['amp'], ['Lrelu'] ], 'toRGB_16'   : [['EqModConv1x1', 512,   3], ['bias',3]],
        'Uconv32'  : [['EqModConvT3x3', 512, 512], ['blur'], ['noise',  32], ['bias',512], ['amp'], ['Lrelu'] ],
        'Fconv32'  : [['EqModConv3x3',  512, 512],           ['noise',  32], ['bias',512], ['amp'], ['Lrelu'] ], 'toRGB_32'   : [['EqModConv1x1', 512,   3], ['bias',3]],
        'Uconv64'  : [['EqModConvT3x3', 512, 512], ['blur'], ['noise',  64], ['bias',512], ['amp'], ['Lrelu'] ],
        'Fconv64'  : [['EqModConv3x3',  512, 512],           ['noise',  64], ['bias',512], ['amp'], ['Lrelu'] ], 'toRGB_64'   : [['EqModConv1x1', 512,   3], ['bias',3]],
        'Uconv128' : [['EqModConvT3x3', 512, 256], ['blur'], ['noise', 128], ['bias',256], ['amp'], ['Lrelu'] ],
        'Fconv128' : [['EqModConv3x3',  256, 256],           ['noise', 128], ['bias',256], ['amp'], ['Lrelu'] ], 'toRGB_128'  : [['EqModConv1x1', 256,   3], ['bias',3]],
        'Uconv256' : [['EqModConvT3x3', 256, 128], ['blur'], ['noise', 256], ['bias',128], ['amp'], ['Lrelu'] ],
        'Fconv256' : [['EqModConv3x3',  128, 128],           ['noise', 256], ['bias',128], ['amp'], ['Lrelu'] ], 'toRGB_256'  : [['EqModConv1x1', 128,   3], ['bias',3]],
        'Uconv512' : [['EqModConvT3x3', 128,  64], ['blur'], ['noise', 512], ['bias', 64], ['amp'], ['Lrelu'] ],
        'Fconv512' : [['EqModConv3x3',   64,  64],           ['noise', 512], ['bias', 64], ['amp'], ['Lrelu'] ], 'toRGB_512'  : [['EqModConv1x1',  64,   3], ['bias',3]],
        'Uconv1024': [['EqModConvT3x3',  64,  32], ['blur'], ['noise',1024], ['bias', 32], ['amp'], ['Lrelu'] ],
        'Fconv1024': [['EqModConv3x3',   32,  32],           ['noise',1024], ['bias', 32], ['amp'], ['Lrelu'] ], 'toRGB_1024' : [['EqModConv1x1',  32,   3], ['bias',3]],
    }

    def _make_sequential(self,key):
        definition = {
            'pixel_norm'   :    lambda *config: PixelwiseNormalization(),
            'truncation'   :    lambda *config: TruncationTrick(),
            'fc'           :    lambda *config: EqualizedFullyConnect(
                                    in_dim=config[0], out_dim=config[1], lr=0.01),
            'EqModConvT3x3':    lambda *config: EqualizedModulatedConvTranspose2d(
                                    in_channels=config[0], out_channels=config[1],
                                    kernel_size=3, stride=2, padding=0,
                                    demodulate=True, lr=1.0, style_dim=512),
            'EqModConv3x3' :    lambda *config: EqualizedModulatedConv2d(
                                    in_channels=config[0], out_channels=config[1],
                                    kernel_size=3, stride=1, padding=1,
                                    demodulate=True, lr=1.0, style_dim=512),
            'EqModConv1x1' :    lambda *config: EqualizedModulatedConv2d(
                                    in_channels=config[0], out_channels=config[1],
                                    kernel_size=1, stride=1, padding=0,
                                    demodulate=False, lr=1.0, style_dim=512),
            'bias'         :    lambda *config: AddBias(out_channels=config[0]),
            'amp'          :    lambda *config: Amplify(2**0.5),
            'Lrelu'        :    lambda *config: nn.LeakyReLU(negative_slope=0.2),
            'noise'        :    lambda *config: NoiseInjection(resolution=config[0]),
            'blur'         :    lambda *config: Blur(),
        }
        return nn.Sequential(*[ definition[k](*cfg) for k,*cfg in self.structure[key]])


    def __init__(self):
        super().__init__()
        style_dim = 512

        self.const_input = nn.Parameter(torch.randn(1, 512, 4, 4))
        self.register_buffer('style_mixing_rate',torch.zeros((1,))) # スタイルの合成比率，今回は使わない

        self.mapping = self._make_sequential('mapping')
        self.blocks = nn.ModuleList([self._make_sequential(k) for k in [
                            'Fconv4',   'Uconv8',   'Fconv8',   'Uconv16',  'Fconv16',
                'Uconv32',  'Fconv32',  'Uconv64',  'Fconv64',  'Uconv128', 'Fconv128',
                'Uconv256', 'Fconv256', 'Uconv512', 'Fconv512', 'Uconv1024','Fconv1024'
            ] ])
        self.toRGBs = nn.ModuleList([self._make_sequential(k) for k in [
                'toRGB_4',   'toRGB_8',   'toRGB_16',  'toRGB_32',
                'toRGB_64',  'toRGB_128', 'toRGB_256', 'toRGB_512',
                'toRGB_1024'
            ] ])
        

    def forward(self, z, truncation_rate=0.5):
        N,D = z.shape

        # 潜在変数からスタイルへ変換
        styles = self.mapping(z)
        styles = [styles[:,i] for i in range(18)]

        out = self.const_input.repeat(N, 1, 1, 1)
        out = self.blocks[0]( (out,styles[0]) )
        skip = self.toRGBs[0]( (out,styles[1]) )

        for convU, convF, toRGB, styU,styF,styT in zip( \
                self.blocks[1::2], self.blocks[2::2], self.toRGBs[1:],
                styles[1::2], styles[2::2], styles[3::2]):
            out = convU( (out,styU) )
            out = convF( (out,styF) )
            skip = toRGB( (out,styT) ) + F.interpolate(skip,scale_factor=2,mode='bilinear',align_corners=False)
        
        return skip


########## 以下，重み変換 ########

ops_dict = {
    'Tco' : lambda weight: torch.flip(torch.from_numpy(weight.transpose((2,3,0,1))), [2, 3]),   # transposed conv weight (iC,oC,kH,kW)
    'con' : lambda weight: torch.from_numpy(weight.transpose((3,2,0,1))),                       # conv weight
    'fc_' : lambda weight: torch.from_numpy(weight.transpose((1, 0))),                          # fc weight
    'any' : lambda weight: torch.from_numpy(weight),                                            # fc bias, const input, const noise
    'uns' : lambda weight: torch.from_numpy(np.array(weight).reshape(1)),                       # style mixing rate, noise scaler
}

name_trans_dict = {
    'const_input'              : ['any', 'G_synthesis/4x4/Const/const'                  ],
    'style_mixing_rate'        : ['uns', 'lod'                                          ],
    'mapping.1.weight'         : ['fc_', 'G_mapping/Dense0/weight'                      ],
    'mapping.1.bias'           : ['any', 'G_mapping/Dense0/bias'                        ],
    'mapping.4.weight'         : ['fc_', 'G_mapping/Dense1/weight'                      ],
    'mapping.4.bias'           : ['any', 'G_mapping/Dense1/bias'                        ],
    'mapping.7.weight'         : ['fc_', 'G_mapping/Dense2/weight'                      ],
    'mapping.7.bias'           : ['any', 'G_mapping/Dense2/bias'                        ],
    'mapping.10.weight'        : ['fc_', 'G_mapping/Dense3/weight'                      ],
    'mapping.10.bias'          : ['any', 'G_mapping/Dense3/bias'                        ],
    'mapping.13.weight'        : ['fc_', 'G_mapping/Dense4/weight'                      ],
    'mapping.13.bias'          : ['any', 'G_mapping/Dense4/bias'                        ],
    'mapping.16.weight'        : ['fc_', 'G_mapping/Dense5/weight'                      ],
    'mapping.16.bias'          : ['any', 'G_mapping/Dense5/bias'                        ],
    'mapping.19.weight'        : ['fc_', 'G_mapping/Dense6/weight'                      ],
    'mapping.19.bias'          : ['any', 'G_mapping/Dense6/bias'                        ],
    'mapping.22.weight'        : ['fc_', 'G_mapping/Dense7/weight'                      ],
    'mapping.22.bias'          : ['any', 'G_mapping/Dense7/bias'                        ],
    'mapping.25.avg_style'     : ['any', 'dlatent_avg'                                  ],
    'blocks.0.0.weight'        : ['con', 'G_synthesis/4x4/Conv/weight'                  ],
    'blocks.0.0.fc.weight'     : ['fc_', 'G_synthesis/4x4/Conv/mod_weight'              ],
    'blocks.0.0.fc.bias'       : ['any', 'G_synthesis/4x4/Conv/mod_bias'                ],
    'blocks.0.1.noise_scaler'  : ['uns', 'G_synthesis/4x4/Conv/noise_strength'          ],
    'blocks.0.1.const_noise'   : ['any', 'G_synthesis/noise0'                           ],
    'blocks.0.2.bias'          : ['any', 'G_synthesis/4x4/Conv/bias'                    ],
    'blocks.1.0.weight'        : ['Tco', 'G_synthesis/8x8/Conv0_up/weight'              ],
    'blocks.1.0.fc.weight'     : ['fc_', 'G_synthesis/8x8/Conv0_up/mod_weight'          ],
    'blocks.1.0.fc.bias'       : ['any', 'G_synthesis/8x8/Conv0_up/mod_bias'            ],
    'blocks.1.2.noise_scaler'  : ['uns', 'G_synthesis/8x8/Conv0_up/noise_strength'      ],
    'blocks.1.2.const_noise'   : ['any', 'G_synthesis/noise1'                           ],
    'blocks.1.3.bias'          : ['any', 'G_synthesis/8x8/Conv0_up/bias'                ],
    'blocks.2.0.weight'        : ['con', 'G_synthesis/8x8/Conv1/weight'                 ],
    'blocks.2.0.fc.weight'     : ['fc_', 'G_synthesis/8x8/Conv1/mod_weight'             ],
    'blocks.2.0.fc.bias'       : ['any', 'G_synthesis/8x8/Conv1/mod_bias'               ],
    'blocks.2.1.noise_scaler'  : ['uns', 'G_synthesis/8x8/Conv1/noise_strength'         ],
    'blocks.2.1.const_noise'   : ['any', 'G_synthesis/noise2'                           ],
    'blocks.2.2.bias'          : ['any', 'G_synthesis/8x8/Conv1/bias'                   ],
    'blocks.3.0.weight'        : ['Tco', 'G_synthesis/16x16/Conv0_up/weight'            ],
    'blocks.3.0.fc.weight'     : ['fc_', 'G_synthesis/16x16/Conv0_up/mod_weight'        ],
    'blocks.3.0.fc.bias'       : ['any', 'G_synthesis/16x16/Conv0_up/mod_bias'          ],
    'blocks.3.2.noise_scaler'  : ['uns', 'G_synthesis/16x16/Conv0_up/noise_strength'    ],
    'blocks.3.2.const_noise'   : ['any', 'G_synthesis/noise3'                           ],
    'blocks.3.3.bias'          : ['any', 'G_synthesis/16x16/Conv0_up/bias'              ],
    'blocks.4.0.weight'        : ['con', 'G_synthesis/16x16/Conv1/weight'               ],
    'blocks.4.0.fc.weight'     : ['fc_', 'G_synthesis/16x16/Conv1/mod_weight'           ],
    'blocks.4.0.fc.bias'       : ['any', 'G_synthesis/16x16/Conv1/mod_bias'             ],
    'blocks.4.1.noise_scaler'  : ['uns', 'G_synthesis/16x16/Conv1/noise_strength'       ],
    'blocks.4.1.const_noise'   : ['any', 'G_synthesis/noise4'                           ],
    'blocks.4.2.bias'          : ['any', 'G_synthesis/16x16/Conv1/bias'                 ],
    'blocks.5.0.weight'        : ['Tco', 'G_synthesis/32x32/Conv0_up/weight'            ],
    'blocks.5.0.fc.weight'     : ['fc_', 'G_synthesis/32x32/Conv0_up/mod_weight'        ],
    'blocks.5.0.fc.bias'       : ['any', 'G_synthesis/32x32/Conv0_up/mod_bias'          ],
    'blocks.5.2.noise_scaler'  : ['uns', 'G_synthesis/32x32/Conv0_up/noise_strength'    ],
    'blocks.5.2.const_noise'   : ['any', 'G_synthesis/noise5'                           ],
    'blocks.5.3.bias'          : ['any', 'G_synthesis/32x32/Conv0_up/bias'              ],
    'blocks.6.0.weight'        : ['con', 'G_synthesis/32x32/Conv1/weight'               ],
    'blocks.6.0.fc.weight'     : ['fc_', 'G_synthesis/32x32/Conv1/mod_weight'           ],
    'blocks.6.0.fc.bias'       : ['any', 'G_synthesis/32x32/Conv1/mod_bias'             ],
    'blocks.6.1.noise_scaler'  : ['uns', 'G_synthesis/32x32/Conv1/noise_strength'       ],
    'blocks.6.1.const_noise'   : ['any', 'G_synthesis/noise6'                           ],
    'blocks.6.2.bias'          : ['any', 'G_synthesis/32x32/Conv1/bias'                 ],
    'blocks.7.0.weight'        : ['Tco', 'G_synthesis/64x64/Conv0_up/weight'            ],
    'blocks.7.0.fc.weight'     : ['fc_', 'G_synthesis/64x64/Conv0_up/mod_weight'        ],
    'blocks.7.0.fc.bias'       : ['any', 'G_synthesis/64x64/Conv0_up/mod_bias'          ],
    'blocks.7.2.noise_scaler'  : ['uns', 'G_synthesis/64x64/Conv0_up/noise_strength'    ],
    'blocks.7.2.const_noise'   : ['any', 'G_synthesis/noise7'                           ],
    'blocks.7.3.bias'          : ['any', 'G_synthesis/64x64/Conv0_up/bias'              ],
    'blocks.8.0.weight'        : ['con', 'G_synthesis/64x64/Conv1/weight'               ],
    'blocks.8.0.fc.weight'     : ['fc_', 'G_synthesis/64x64/Conv1/mod_weight'           ],
    'blocks.8.0.fc.bias'       : ['any', 'G_synthesis/64x64/Conv1/mod_bias'             ],
    'blocks.8.1.noise_scaler'  : ['uns', 'G_synthesis/64x64/Conv1/noise_strength'       ],
    'blocks.8.1.const_noise'   : ['any', 'G_synthesis/noise8'                           ],
    'blocks.8.2.bias'          : ['any', 'G_synthesis/64x64/Conv1/bias'                 ],
    'blocks.9.0.weight'        : ['Tco', 'G_synthesis/128x128/Conv0_up/weight'          ],
    'blocks.9.0.fc.weight'     : ['fc_', 'G_synthesis/128x128/Conv0_up/mod_weight'      ],
    'blocks.9.0.fc.bias'       : ['any', 'G_synthesis/128x128/Conv0_up/mod_bias'        ],
    'blocks.9.2.noise_scaler'  : ['uns', 'G_synthesis/128x128/Conv0_up/noise_strength'  ],
    'blocks.9.2.const_noise'   : ['any', 'G_synthesis/noise9'                           ],
    'blocks.9.3.bias'          : ['any', 'G_synthesis/128x128/Conv0_up/bias'            ],
    'blocks.10.0.weight'       : ['con', 'G_synthesis/128x128/Conv1/weight'             ],
    'blocks.10.0.fc.weight'    : ['fc_', 'G_synthesis/128x128/Conv1/mod_weight'         ],
    'blocks.10.0.fc.bias'      : ['any', 'G_synthesis/128x128/Conv1/mod_bias'           ],
    'blocks.10.1.noise_scaler' : ['uns', 'G_synthesis/128x128/Conv1/noise_strength'     ],
    'blocks.10.1.const_noise'  : ['any', 'G_synthesis/noise10'                          ],
    'blocks.10.2.bias'         : ['any', 'G_synthesis/128x128/Conv1/bias'               ],
    'blocks.11.0.weight'       : ['Tco', 'G_synthesis/256x256/Conv0_up/weight'          ],
    'blocks.11.0.fc.weight'    : ['fc_', 'G_synthesis/256x256/Conv0_up/mod_weight'      ],
    'blocks.11.0.fc.bias'      : ['any', 'G_synthesis/256x256/Conv0_up/mod_bias'        ],
    'blocks.11.2.noise_scaler' : ['uns', 'G_synthesis/256x256/Conv0_up/noise_strength'  ],
    'blocks.11.2.const_noise'  : ['any', 'G_synthesis/noise11'                          ],
    'blocks.11.3.bias'         : ['any', 'G_synthesis/256x256/Conv0_up/bias'            ],
    'blocks.12.0.weight'       : ['con', 'G_synthesis/256x256/Conv1/weight'             ],
    'blocks.12.0.fc.weight'    : ['fc_', 'G_synthesis/256x256/Conv1/mod_weight'         ],
    'blocks.12.0.fc.bias'      : ['any', 'G_synthesis/256x256/Conv1/mod_bias'           ],
    'blocks.12.1.noise_scaler' : ['uns', 'G_synthesis/256x256/Conv1/noise_strength'     ],
    'blocks.12.1.const_noise'  : ['any', 'G_synthesis/noise12'                          ],
    'blocks.12.2.bias'         : ['any', 'G_synthesis/256x256/Conv1/bias'               ],
    'blocks.13.0.weight'       : ['Tco', 'G_synthesis/512x512/Conv0_up/weight'          ],
    'blocks.13.0.fc.weight'    : ['fc_', 'G_synthesis/512x512/Conv0_up/mod_weight'      ],
    'blocks.13.0.fc.bias'      : ['any', 'G_synthesis/512x512/Conv0_up/mod_bias'        ],
    'blocks.13.2.noise_scaler' : ['uns', 'G_synthesis/512x512/Conv0_up/noise_strength'  ],
    'blocks.13.2.const_noise'  : ['any', 'G_synthesis/noise13'                          ],
    'blocks.13.3.bias'         : ['any', 'G_synthesis/512x512/Conv0_up/bias'            ],
    'blocks.14.0.weight'       : ['con', 'G_synthesis/512x512/Conv1/weight'             ],
    'blocks.14.0.fc.weight'    : ['fc_', 'G_synthesis/512x512/Conv1/mod_weight'         ],
    'blocks.14.0.fc.bias'      : ['any', 'G_synthesis/512x512/Conv1/mod_bias'           ],
    'blocks.14.1.noise_scaler' : ['uns', 'G_synthesis/512x512/Conv1/noise_strength'     ],
    'blocks.14.1.const_noise'  : ['any', 'G_synthesis/noise14'                          ],
    'blocks.14.2.bias'         : ['any', 'G_synthesis/512x512/Conv1/bias'               ],
    'blocks.15.0.weight'       : ['Tco', 'G_synthesis/1024x1024/Conv0_up/weight'        ],
    'blocks.15.0.fc.weight'    : ['fc_', 'G_synthesis/1024x1024/Conv0_up/mod_weight'    ],
    'blocks.15.0.fc.bias'      : ['any', 'G_synthesis/1024x1024/Conv0_up/mod_bias'      ],
    'blocks.15.2.noise_scaler' : ['uns', 'G_synthesis/1024x1024/Conv0_up/noise_strength'],
    'blocks.15.2.const_noise'  : ['any', 'G_synthesis/noise15'                          ],
    'blocks.15.3.bias'         : ['any', 'G_synthesis/1024x1024/Conv0_up/bias'          ],
    'blocks.16.0.weight'       : ['con', 'G_synthesis/1024x1024/Conv1/weight'           ],
    'blocks.16.0.fc.weight'    : ['fc_', 'G_synthesis/1024x1024/Conv1/mod_weight'       ],
    'blocks.16.0.fc.bias'      : ['any', 'G_synthesis/1024x1024/Conv1/mod_bias'         ],
    'blocks.16.1.noise_scaler' : ['uns', 'G_synthesis/1024x1024/Conv1/noise_strength'   ],
    'blocks.16.1.const_noise'  : ['any', 'G_synthesis/noise16'                          ],
    'blocks.16.2.bias'         : ['any', 'G_synthesis/1024x1024/Conv1/bias'             ],
    'toRGBs.0.0.weight'        : ['con', 'G_synthesis/4x4/ToRGB/weight'                 ],
    'toRGBs.0.0.fc.weight'     : ['fc_', 'G_synthesis/4x4/ToRGB/mod_weight'             ],
    'toRGBs.0.0.fc.bias'       : ['any', 'G_synthesis/4x4/ToRGB/mod_bias'               ],
    'toRGBs.0.1.bias'          : ['any', 'G_synthesis/4x4/ToRGB/bias'                   ],
    'toRGBs.1.0.weight'        : ['con', 'G_synthesis/8x8/ToRGB/weight'                 ],
    'toRGBs.1.0.fc.weight'     : ['fc_', 'G_synthesis/8x8/ToRGB/mod_weight'             ],
    'toRGBs.1.0.fc.bias'       : ['any', 'G_synthesis/8x8/ToRGB/mod_bias'               ],
    'toRGBs.1.1.bias'          : ['any', 'G_synthesis/8x8/ToRGB/bias'                   ],
    'toRGBs.2.0.weight'        : ['con', 'G_synthesis/16x16/ToRGB/weight'               ],
    'toRGBs.2.0.fc.weight'     : ['fc_', 'G_synthesis/16x16/ToRGB/mod_weight'           ],
    'toRGBs.2.0.fc.bias'       : ['any', 'G_synthesis/16x16/ToRGB/mod_bias'             ],
    'toRGBs.2.1.bias'          : ['any', 'G_synthesis/16x16/ToRGB/bias'                 ],
    'toRGBs.3.0.weight'        : ['con', 'G_synthesis/32x32/ToRGB/weight'               ],
    'toRGBs.3.0.fc.weight'     : ['fc_', 'G_synthesis/32x32/ToRGB/mod_weight'           ],
    'toRGBs.3.0.fc.bias'       : ['any', 'G_synthesis/32x32/ToRGB/mod_bias'             ],
    'toRGBs.3.1.bias'          : ['any', 'G_synthesis/32x32/ToRGB/bias'                 ],
    'toRGBs.4.0.weight'        : ['con', 'G_synthesis/64x64/ToRGB/weight'               ],
    'toRGBs.4.0.fc.weight'     : ['fc_', 'G_synthesis/64x64/ToRGB/mod_weight'           ],
    'toRGBs.4.0.fc.bias'       : ['any', 'G_synthesis/64x64/ToRGB/mod_bias'             ],
    'toRGBs.4.1.bias'          : ['any', 'G_synthesis/64x64/ToRGB/bias'                 ],
    'toRGBs.5.0.weight'        : ['con', 'G_synthesis/128x128/ToRGB/weight'             ],
    'toRGBs.5.0.fc.weight'     : ['fc_', 'G_synthesis/128x128/ToRGB/mod_weight'         ],
    'toRGBs.5.0.fc.bias'       : ['any', 'G_synthesis/128x128/ToRGB/mod_bias'           ],
    'toRGBs.5.1.bias'          : ['any', 'G_synthesis/128x128/ToRGB/bias'               ],
    'toRGBs.6.0.weight'        : ['con', 'G_synthesis/256x256/ToRGB/weight'             ],
    'toRGBs.6.0.fc.weight'     : ['fc_', 'G_synthesis/256x256/ToRGB/mod_weight'         ],
    'toRGBs.6.0.fc.bias'       : ['any', 'G_synthesis/256x256/ToRGB/mod_bias'           ],
    'toRGBs.6.1.bias'          : ['any', 'G_synthesis/256x256/ToRGB/bias'               ],
    'toRGBs.7.0.weight'        : ['con', 'G_synthesis/512x512/ToRGB/weight'             ],
    'toRGBs.7.0.fc.weight'     : ['fc_', 'G_synthesis/512x512/ToRGB/mod_weight'         ],
    'toRGBs.7.0.fc.bias'       : ['any', 'G_synthesis/512x512/ToRGB/mod_bias'           ],
    'toRGBs.7.1.bias'          : ['any', 'G_synthesis/512x512/ToRGB/bias'               ],
    'toRGBs.8.0.weight'        : ['con', 'G_synthesis/1024x1024/ToRGB/weight'           ],
    'toRGBs.8.0.fc.weight'     : ['fc_', 'G_synthesis/1024x1024/ToRGB/mod_weight'       ],
    'toRGBs.8.0.fc.bias'       : ['any', 'G_synthesis/1024x1024/ToRGB/mod_bias'         ],
    'toRGBs.8.1.bias'          : ['any', 'G_synthesis/1024x1024/ToRGB/bias'             ],
}


if __name__ == '__main__':
    # コマンドライン引数の取得
    args = parse_args()

    print('model construction...')
    generator = Generator()
    base_dict = generator.state_dict()

    print('model weights load...')
    with (Path(args.weight_dir)/'stylegan2_ndarray.pkl').open('rb') as f:
        src_dict = pickle.load(f)

    print('set state_dict...')
    new_dict = { k : ops_dict[v[0]](src_dict[v[1]]) for k,v in name_trans_dict.items()}
    generator.load_state_dict(new_dict)

    print('load latents...')
    with (Path(args.output_dir)/'latents2.pkl').open('rb') as f:
        latents = pickle.load(f)
    latents = torch.from_numpy(latents.astype(np.float32))

    print('network forward...')
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    with torch.no_grad():
        z = latents.to(device)
        generator.to(device)
        img = generator(z)
        normalized = (img.clamp(-1,1)+1)/2*255
        images = normalized.permute(0,2,3,1).cpu().numpy().astype(np.uint8)

    print(img.shape, img.dtype)

    # 出力する個数，解像度
    num_H, num_W = 4,4
    num_images = num_H*num_W
    H = W = 1024

    print('image output...')
    # 出力を並べる関数
    def make_table(imgs):
        canvas = np.zeros((H*num_H,W*num_W,3),dtype=np.uint8)
        for i,p in enumerate(imgs):
            h,w = i//num_W, i%num_W
            canvas[H*h:H*-~h,W*w:W*-~w,:] = p[:,:,::-1]
        return canvas

    cv2.imwrite(str(Path(args.output_dir)/'stylegan2_pt.png'), make_table(images))

    print('weight save...')
    torch.save(generator.state_dict(),str(Path(args.weight_dir)/'stylegan2_state_dict.pth'))

    print('all done')
