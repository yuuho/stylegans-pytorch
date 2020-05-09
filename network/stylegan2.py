import numpy as np
import cv2
import torch
from torch import nn
from torch.nn import functional as F

from common import PixelwiseNormalization, Amplify, AddChannelwiseBias, EqualizedFullyConnect, TruncationTrick


# 固定ノイズ
class PixelwiseNoise(nn.Module):
    def __init__(self, resolution):
        super().__init__()
        self.register_buffer("const_noise", torch.randn((1, 1, resolution, resolution)))
        self.noise_scaler = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        N,C,H,W = x.shape
        noise = self.const_noise.expand(N,C,H,W)
        return x + noise * self.noise_scaler


# 解像度上げるときのModConvのみ
class FusedBlur3x3(nn.Module):
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


# 学習率を調整した変調転置畳み込み
class EqualizedModulatedConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, style_dim, padding, stride, demodulate=True, lr=1):
        super().__init__()

        self.padding, self.stride = padding, stride
        self.demodulate = demodulate

        self.weight = nn.Parameter( torch.randn(in_channels, out_channels, kernel_size, kernel_size))
        torch.nn.init.normal_(self.weight.data, mean=0.0, std=1.0/lr)
        self.weight_scaler = 1 / (in_channels * kernel_size*kernel_size)**0.5 * lr

        self.fc = EqualizedFullyConnect(style_dim, in_channels, lr)
        self.bias = AddChannelwiseBias(in_channels, lr)

    def forward(self, pack):
        x, style = pack
        N, iC, H, W = x.shape
        iC, oC, kH, kW = self.weight.shape

        mod_rates = self.bias(self.fc(style))+1 # (N, iC)
        modulated_weight = self.weight_scaler*self.weight.view(1,iC,oC,kH,kW) \
                                * mod_rates.view(N,iC,1,1,1) # (N,iC,oC,kH,kW)

        if self.demodulate:
            demod_norm = 1 / ((modulated_weight**2).sum([1,3,4]) + 1e-8)**0.5 # (N, oC)
            weight = modulated_weight * demod_norm.view(N, 1, oC, 1, 1) # (N,iC,oC,kH,kW)
        else:
            weight = modulated_weight

        x = x.view(1, N*iC, H, W)
        weight = weight.view(N*iC,oC,kH,kW)
        out = F.conv_transpose2d(x, weight, padding=self.padding, stride=self.stride, groups=N)

        _, _, Hp1, Wp1 = out.shape
        out = out.view(N, oC, Hp1, Wp1)

        return out


# 学習率を調整した変調畳み込み
class EqualizedModulatedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, style_dim, padding, stride, demodulate=True, lr=1):
        super().__init__()

        self.padding, self.stride = padding, stride
        self.demodulate = demodulate

        self.weight = nn.Parameter( torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        torch.nn.init.normal_(self.weight.data, mean=0.0, std=1.0/lr)
        self.weight_scaler = 1 / (in_channels*kernel_size*kernel_size)**0.5 * lr
        
        self.fc = EqualizedFullyConnect(style_dim, in_channels, lr)
        self.bias = AddChannelwiseBias(in_channels, lr)

    def forward(self, pack):
        x, style = pack
        N, iC, H, W = x.shape
        oC, iC, kH, kW = self.weight.shape

        mod_rates = self.bias(self.fc(style))+1 # (N, iC)
        modulated_weight = self.weight_scaler*self.weight.view(1,oC,iC,kH,kW) \
                                * mod_rates.view(N,1,iC,1,1) # (N,oC,iC,kH,kW)

        if self.demodulate:
            demod_norm = 1 / ((modulated_weight**2).sum([2,3,4]) + 1e-8)**0.5 # (N, oC)
            weight = modulated_weight * demod_norm.view(N, oC, 1, 1, 1) # (N,oC,iC,kH,kW)
        else: # ToRGB
            weight = modulated_weight

        out = F.conv2d(x.view(1,N*iC,H,W), weight.view(N*oC,iC,kH,kW),
                    padding=self.padding, stride=self.stride, groups=N).view(N,oC,H,W)
        return out


class Generator(nn.Module):

    structure = {
        'mapping': [['pixel_norm'], ['fc',512,512],['b',512],['amp'],['Lrelu'],['fc',512,512],['b',512],['amp'],['Lrelu'],
                                    ['fc',512,512],['b',512],['amp'],['Lrelu'],['fc',512,512],['b',512],['amp'],['Lrelu'],
                                    ['fc',512,512],['b',512],['amp'],['Lrelu'],['fc',512,512],['b',512],['amp'],['Lrelu'],
                                    ['fc',512,512],['b',512],['amp'],['Lrelu'],['fc',512,512],['b',512],['amp'],['Lrelu'],['truncation']],
        'Fconv4'   : [['EqModConv3x3',  512, 512],             ['noiseP',   4], ['bias',512], ['amp'], ['Lrelu'] ],  'toRGB_4'   : [['EqModConv1x1',512, 3], ['bias',3]],
        'Uconv8'   : [['EqModConvT3x3', 512, 512], ['blurEX'], ['noiseP',   8], ['bias',512], ['amp'], ['Lrelu'] ],
        'Fconv8'   : [['EqModConv3x3',  512, 512],             ['noiseP',   8], ['bias',512], ['amp'], ['Lrelu'] ],  'toRGB_8'   : [['EqModConv1x1',512, 3], ['bias',3]],
        'Uconv16'  : [['EqModConvT3x3', 512, 512], ['blurEX'], ['noiseP',  16], ['bias',512], ['amp'], ['Lrelu'] ],
        'Fconv16'  : [['EqModConv3x3',  512, 512],             ['noiseP',  16], ['bias',512], ['amp'], ['Lrelu'] ],  'toRGB_16'  : [['EqModConv1x1',512, 3], ['bias',3]],
        'Uconv32'  : [['EqModConvT3x3', 512, 512], ['blurEX'], ['noiseP',  32], ['bias',512], ['amp'], ['Lrelu'] ],
        'Fconv32'  : [['EqModConv3x3',  512, 512],             ['noiseP',  32], ['bias',512], ['amp'], ['Lrelu'] ],  'toRGB_32'  : [['EqModConv1x1',512, 3], ['bias',3]],
        'Uconv64'  : [['EqModConvT3x3', 512, 512], ['blurEX'], ['noiseP',  64], ['bias',512], ['amp'], ['Lrelu'] ],
        'Fconv64'  : [['EqModConv3x3',  512, 512],             ['noiseP',  64], ['bias',512], ['amp'], ['Lrelu'] ],  'toRGB_64'  : [['EqModConv1x1',512, 3], ['bias',3]],
        'Uconv128' : [['EqModConvT3x3', 512, 256], ['blurEX'], ['noiseP', 128], ['bias',256], ['amp'], ['Lrelu'] ],
        'Fconv128' : [['EqModConv3x3',  256, 256],             ['noiseP', 128], ['bias',256], ['amp'], ['Lrelu'] ],  'toRGB_128' : [['EqModConv1x1',256, 3], ['bias',3]],
        'Uconv256' : [['EqModConvT3x3', 256, 128], ['blurEX'], ['noiseP', 256], ['bias',128], ['amp'], ['Lrelu'] ],
        'Fconv256' : [['EqModConv3x3',  128, 128],             ['noiseP', 256], ['bias',128], ['amp'], ['Lrelu'] ],  'toRGB_256' : [['EqModConv1x1',128, 3], ['bias',3]],
        'Uconv512' : [['EqModConvT3x3', 128,  64], ['blurEX'], ['noiseP', 512], ['bias', 64], ['amp'], ['Lrelu'] ],
        'Fconv512' : [['EqModConv3x3',   64,  64],             ['noiseP', 512], ['bias', 64], ['amp'], ['Lrelu'] ],  'toRGB_512' : [['EqModConv1x1', 64, 3], ['bias',3]],
        'Uconv1024': [['EqModConvT3x3',  64,  32], ['blurEX'], ['noiseP',1024], ['bias', 32], ['amp'], ['Lrelu'] ],
        'Fconv1024': [['EqModConv3x3',   32,  32],             ['noiseP',1024], ['bias', 32], ['amp'], ['Lrelu'] ],  'toRGB_1024': [['EqModConv1x1', 32, 3], ['bias',3]],
    }

    def _make_sequential(self,key):
        definition = {
            'pixel_norm'   :    lambda *config: PixelwiseNormalization(),
            'truncation'   :    lambda *config: TruncationTrick(
                                    num_target=10, threshold=0.7, output_num=18, style_dim=512),
            'fc'           :    lambda *config: EqualizedFullyConnect(
                                    in_dim=config[0], out_dim=config[1], lr=0.01),
            'b'            :    lambda *config: AddChannelwiseBias(out_channels=config[0], lr=0.01),
            'bias'         :    lambda *config: AddChannelwiseBias(out_channels=config[0], lr=1.0),
            'amp'          :    lambda *config: Amplify(2**0.5),
            'Lrelu'        :    lambda *config: nn.LeakyReLU(negative_slope=0.2),
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
            'noiseP'       :    lambda *config: PixelwiseNoise(resolution=config[0]),
            'blurEX'       :    lambda *config: FusedBlur3x3(),
        }
        return nn.Sequential(*[ definition[k](*cfg) for k,*cfg in self.structure[key]])


    def __init__(self):
        super().__init__()

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
        

    def forward(self, z):
        N,D = z.shape

        # 潜在変数からスタイルへ変換
        styles = self.mapping(z) # (N,18,D)
        styles = [styles[:,i] for i in range(18)] # list[(N,D),]x18

        tmp = self.const_input.repeat(N, 1, 1, 1)
        tmp = self.blocks[0]( (tmp,styles[0]) )
        skip = self.toRGBs[0]( (tmp,styles[1]) )

        for convU, convF, toRGB, styU,styF,styT in zip( \
                self.blocks[1::2], self.blocks[2::2], self.toRGBs[1:],
                styles[1::2], styles[2::2], styles[3::2]):
            tmp = convU( (tmp,styU) )
            tmp = convF( (tmp,styF) )
            skip = toRGB( (tmp,styT) ) + F.interpolate(skip,scale_factor=2,mode='bilinear',align_corners=False)
        
        return skip


# { pytorchでの名前 : [変換関数, tensorflowでの名前] }
name_trans_dict = {
    'const_input'              : ['any', 'G_synthesis/4x4/Const/const'                  ],
    'style_mixing_rate'        : ['uns', 'lod'                                          ],
    'mapping.1.weight'         : ['fc_', 'G_mapping/Dense0/weight'                      ],
    'mapping.2.bias'           : ['any', 'G_mapping/Dense0/bias'                        ],
    'mapping.5.weight'         : ['fc_', 'G_mapping/Dense1/weight'                      ],
    'mapping.6.bias'           : ['any', 'G_mapping/Dense1/bias'                        ],
    'mapping.9.weight'         : ['fc_', 'G_mapping/Dense2/weight'                      ],
    'mapping.10.bias'          : ['any', 'G_mapping/Dense2/bias'                        ],
    'mapping.13.weight'        : ['fc_', 'G_mapping/Dense3/weight'                      ],
    'mapping.14.bias'          : ['any', 'G_mapping/Dense3/bias'                        ],
    'mapping.17.weight'        : ['fc_', 'G_mapping/Dense4/weight'                      ],
    'mapping.18.bias'          : ['any', 'G_mapping/Dense4/bias'                        ],
    'mapping.21.weight'        : ['fc_', 'G_mapping/Dense5/weight'                      ],
    'mapping.22.bias'          : ['any', 'G_mapping/Dense5/bias'                        ],
    'mapping.25.weight'        : ['fc_', 'G_mapping/Dense6/weight'                      ],
    'mapping.26.bias'          : ['any', 'G_mapping/Dense6/bias'                        ],
    'mapping.29.weight'        : ['fc_', 'G_mapping/Dense7/weight'                      ],
    'mapping.30.bias'          : ['any', 'G_mapping/Dense7/bias'                        ],
    'mapping.33.avg_style'     : ['any', 'dlatent_avg'                                  ],
    'blocks.0.0.weight'        : ['con', 'G_synthesis/4x4/Conv/weight'                  ],
    'blocks.0.0.fc.weight'     : ['fc_', 'G_synthesis/4x4/Conv/mod_weight'              ],
    'blocks.0.0.bias.bias'     : ['any', 'G_synthesis/4x4/Conv/mod_bias'                ],
    'blocks.0.1.noise_scaler'  : ['uns', 'G_synthesis/4x4/Conv/noise_strength'          ],
    'blocks.0.1.const_noise'   : ['any', 'G_synthesis/noise0'                           ],
    'blocks.0.2.bias'          : ['any', 'G_synthesis/4x4/Conv/bias'                    ],
    'blocks.1.0.weight'        : ['mTc', 'G_synthesis/8x8/Conv0_up/weight'              ],
    'blocks.1.0.fc.weight'     : ['fc_', 'G_synthesis/8x8/Conv0_up/mod_weight'          ],
    'blocks.1.0.bias.bias'     : ['any', 'G_synthesis/8x8/Conv0_up/mod_bias'            ],
    'blocks.1.2.noise_scaler'  : ['uns', 'G_synthesis/8x8/Conv0_up/noise_strength'      ],
    'blocks.1.2.const_noise'   : ['any', 'G_synthesis/noise1'                           ],
    'blocks.1.3.bias'          : ['any', 'G_synthesis/8x8/Conv0_up/bias'                ],
    'blocks.2.0.weight'        : ['con', 'G_synthesis/8x8/Conv1/weight'                 ],
    'blocks.2.0.fc.weight'     : ['fc_', 'G_synthesis/8x8/Conv1/mod_weight'             ],
    'blocks.2.0.bias.bias'     : ['any', 'G_synthesis/8x8/Conv1/mod_bias'               ],
    'blocks.2.1.noise_scaler'  : ['uns', 'G_synthesis/8x8/Conv1/noise_strength'         ],
    'blocks.2.1.const_noise'   : ['any', 'G_synthesis/noise2'                           ],
    'blocks.2.2.bias'          : ['any', 'G_synthesis/8x8/Conv1/bias'                   ],
    'blocks.3.0.weight'        : ['mTc', 'G_synthesis/16x16/Conv0_up/weight'            ],
    'blocks.3.0.fc.weight'     : ['fc_', 'G_synthesis/16x16/Conv0_up/mod_weight'        ],
    'blocks.3.0.bias.bias'     : ['any', 'G_synthesis/16x16/Conv0_up/mod_bias'          ],
    'blocks.3.2.noise_scaler'  : ['uns', 'G_synthesis/16x16/Conv0_up/noise_strength'    ],
    'blocks.3.2.const_noise'   : ['any', 'G_synthesis/noise3'                           ],
    'blocks.3.3.bias'          : ['any', 'G_synthesis/16x16/Conv0_up/bias'              ],
    'blocks.4.0.weight'        : ['con', 'G_synthesis/16x16/Conv1/weight'               ],
    'blocks.4.0.fc.weight'     : ['fc_', 'G_synthesis/16x16/Conv1/mod_weight'           ],
    'blocks.4.0.bias.bias'     : ['any', 'G_synthesis/16x16/Conv1/mod_bias'             ],
    'blocks.4.1.noise_scaler'  : ['uns', 'G_synthesis/16x16/Conv1/noise_strength'       ],
    'blocks.4.1.const_noise'   : ['any', 'G_synthesis/noise4'                           ],
    'blocks.4.2.bias'          : ['any', 'G_synthesis/16x16/Conv1/bias'                 ],
    'blocks.5.0.weight'        : ['mTc', 'G_synthesis/32x32/Conv0_up/weight'            ],
    'blocks.5.0.fc.weight'     : ['fc_', 'G_synthesis/32x32/Conv0_up/mod_weight'        ],
    'blocks.5.0.bias.bias'     : ['any', 'G_synthesis/32x32/Conv0_up/mod_bias'          ],
    'blocks.5.2.noise_scaler'  : ['uns', 'G_synthesis/32x32/Conv0_up/noise_strength'    ],
    'blocks.5.2.const_noise'   : ['any', 'G_synthesis/noise5'                           ],
    'blocks.5.3.bias'          : ['any', 'G_synthesis/32x32/Conv0_up/bias'              ],
    'blocks.6.0.weight'        : ['con', 'G_synthesis/32x32/Conv1/weight'               ],
    'blocks.6.0.fc.weight'     : ['fc_', 'G_synthesis/32x32/Conv1/mod_weight'           ],
    'blocks.6.0.bias.bias'     : ['any', 'G_synthesis/32x32/Conv1/mod_bias'             ],
    'blocks.6.1.noise_scaler'  : ['uns', 'G_synthesis/32x32/Conv1/noise_strength'       ],
    'blocks.6.1.const_noise'   : ['any', 'G_synthesis/noise6'                           ],
    'blocks.6.2.bias'          : ['any', 'G_synthesis/32x32/Conv1/bias'                 ],
    'blocks.7.0.weight'        : ['mTc', 'G_synthesis/64x64/Conv0_up/weight'            ],
    'blocks.7.0.fc.weight'     : ['fc_', 'G_synthesis/64x64/Conv0_up/mod_weight'        ],
    'blocks.7.0.bias.bias'     : ['any', 'G_synthesis/64x64/Conv0_up/mod_bias'          ],
    'blocks.7.2.noise_scaler'  : ['uns', 'G_synthesis/64x64/Conv0_up/noise_strength'    ],
    'blocks.7.2.const_noise'   : ['any', 'G_synthesis/noise7'                           ],
    'blocks.7.3.bias'          : ['any', 'G_synthesis/64x64/Conv0_up/bias'              ],
    'blocks.8.0.weight'        : ['con', 'G_synthesis/64x64/Conv1/weight'               ],
    'blocks.8.0.fc.weight'     : ['fc_', 'G_synthesis/64x64/Conv1/mod_weight'           ],
    'blocks.8.0.bias.bias'     : ['any', 'G_synthesis/64x64/Conv1/mod_bias'             ],
    'blocks.8.1.noise_scaler'  : ['uns', 'G_synthesis/64x64/Conv1/noise_strength'       ],
    'blocks.8.1.const_noise'   : ['any', 'G_synthesis/noise8'                           ],
    'blocks.8.2.bias'          : ['any', 'G_synthesis/64x64/Conv1/bias'                 ],
    'blocks.9.0.weight'        : ['mTc', 'G_synthesis/128x128/Conv0_up/weight'          ],
    'blocks.9.0.fc.weight'     : ['fc_', 'G_synthesis/128x128/Conv0_up/mod_weight'      ],
    'blocks.9.0.bias.bias'     : ['any', 'G_synthesis/128x128/Conv0_up/mod_bias'        ],
    'blocks.9.2.noise_scaler'  : ['uns', 'G_synthesis/128x128/Conv0_up/noise_strength'  ],
    'blocks.9.2.const_noise'   : ['any', 'G_synthesis/noise9'                           ],
    'blocks.9.3.bias'          : ['any', 'G_synthesis/128x128/Conv0_up/bias'            ],
    'blocks.10.0.weight'       : ['con', 'G_synthesis/128x128/Conv1/weight'             ],
    'blocks.10.0.fc.weight'    : ['fc_', 'G_synthesis/128x128/Conv1/mod_weight'         ],
    'blocks.10.0.bias.bias'    : ['any', 'G_synthesis/128x128/Conv1/mod_bias'           ],
    'blocks.10.1.noise_scaler' : ['uns', 'G_synthesis/128x128/Conv1/noise_strength'     ],
    'blocks.10.1.const_noise'  : ['any', 'G_synthesis/noise10'                          ],
    'blocks.10.2.bias'         : ['any', 'G_synthesis/128x128/Conv1/bias'               ],
    'blocks.11.0.weight'       : ['mTc', 'G_synthesis/256x256/Conv0_up/weight'          ],
    'blocks.11.0.fc.weight'    : ['fc_', 'G_synthesis/256x256/Conv0_up/mod_weight'      ],
    'blocks.11.0.bias.bias'    : ['any', 'G_synthesis/256x256/Conv0_up/mod_bias'        ],
    'blocks.11.2.noise_scaler' : ['uns', 'G_synthesis/256x256/Conv0_up/noise_strength'  ],
    'blocks.11.2.const_noise'  : ['any', 'G_synthesis/noise11'                          ],
    'blocks.11.3.bias'         : ['any', 'G_synthesis/256x256/Conv0_up/bias'            ],
    'blocks.12.0.weight'       : ['con', 'G_synthesis/256x256/Conv1/weight'             ],
    'blocks.12.0.fc.weight'    : ['fc_', 'G_synthesis/256x256/Conv1/mod_weight'         ],
    'blocks.12.0.bias.bias'    : ['any', 'G_synthesis/256x256/Conv1/mod_bias'           ],
    'blocks.12.1.noise_scaler' : ['uns', 'G_synthesis/256x256/Conv1/noise_strength'     ],
    'blocks.12.1.const_noise'  : ['any', 'G_synthesis/noise12'                          ],
    'blocks.12.2.bias'         : ['any', 'G_synthesis/256x256/Conv1/bias'               ],
    'blocks.13.0.weight'       : ['mTc', 'G_synthesis/512x512/Conv0_up/weight'          ],
    'blocks.13.0.fc.weight'    : ['fc_', 'G_synthesis/512x512/Conv0_up/mod_weight'      ],
    'blocks.13.0.bias.bias'    : ['any', 'G_synthesis/512x512/Conv0_up/mod_bias'        ],
    'blocks.13.2.noise_scaler' : ['uns', 'G_synthesis/512x512/Conv0_up/noise_strength'  ],
    'blocks.13.2.const_noise'  : ['any', 'G_synthesis/noise13'                          ],
    'blocks.13.3.bias'         : ['any', 'G_synthesis/512x512/Conv0_up/bias'            ],
    'blocks.14.0.weight'       : ['con', 'G_synthesis/512x512/Conv1/weight'             ],
    'blocks.14.0.fc.weight'    : ['fc_', 'G_synthesis/512x512/Conv1/mod_weight'         ],
    'blocks.14.0.bias.bias'    : ['any', 'G_synthesis/512x512/Conv1/mod_bias'           ],
    'blocks.14.1.noise_scaler' : ['uns', 'G_synthesis/512x512/Conv1/noise_strength'     ],
    'blocks.14.1.const_noise'  : ['any', 'G_synthesis/noise14'                          ],
    'blocks.14.2.bias'         : ['any', 'G_synthesis/512x512/Conv1/bias'               ],
    'blocks.15.0.weight'       : ['mTc', 'G_synthesis/1024x1024/Conv0_up/weight'        ],
    'blocks.15.0.fc.weight'    : ['fc_', 'G_synthesis/1024x1024/Conv0_up/mod_weight'    ],
    'blocks.15.0.bias.bias'    : ['any', 'G_synthesis/1024x1024/Conv0_up/mod_bias'      ],
    'blocks.15.2.noise_scaler' : ['uns', 'G_synthesis/1024x1024/Conv0_up/noise_strength'],
    'blocks.15.2.const_noise'  : ['any', 'G_synthesis/noise15'                          ],
    'blocks.15.3.bias'         : ['any', 'G_synthesis/1024x1024/Conv0_up/bias'          ],
    'blocks.16.0.weight'       : ['con', 'G_synthesis/1024x1024/Conv1/weight'           ],
    'blocks.16.0.fc.weight'    : ['fc_', 'G_synthesis/1024x1024/Conv1/mod_weight'       ],
    'blocks.16.0.bias.bias'    : ['any', 'G_synthesis/1024x1024/Conv1/mod_bias'         ],
    'blocks.16.1.noise_scaler' : ['uns', 'G_synthesis/1024x1024/Conv1/noise_strength'   ],
    'blocks.16.1.const_noise'  : ['any', 'G_synthesis/noise16'                          ],
    'blocks.16.2.bias'         : ['any', 'G_synthesis/1024x1024/Conv1/bias'             ],
    'toRGBs.0.0.weight'        : ['con', 'G_synthesis/4x4/ToRGB/weight'                 ],
    'toRGBs.0.0.fc.weight'     : ['fc_', 'G_synthesis/4x4/ToRGB/mod_weight'             ],
    'toRGBs.0.0.bias.bias'     : ['any', 'G_synthesis/4x4/ToRGB/mod_bias'               ],
    'toRGBs.0.1.bias'          : ['any', 'G_synthesis/4x4/ToRGB/bias'                   ],
    'toRGBs.1.0.weight'        : ['con', 'G_synthesis/8x8/ToRGB/weight'                 ],
    'toRGBs.1.0.fc.weight'     : ['fc_', 'G_synthesis/8x8/ToRGB/mod_weight'             ],
    'toRGBs.1.0.bias.bias'     : ['any', 'G_synthesis/8x8/ToRGB/mod_bias'               ],
    'toRGBs.1.1.bias'          : ['any', 'G_synthesis/8x8/ToRGB/bias'                   ],
    'toRGBs.2.0.weight'        : ['con', 'G_synthesis/16x16/ToRGB/weight'               ],
    'toRGBs.2.0.fc.weight'     : ['fc_', 'G_synthesis/16x16/ToRGB/mod_weight'           ],
    'toRGBs.2.0.bias.bias'     : ['any', 'G_synthesis/16x16/ToRGB/mod_bias'             ],
    'toRGBs.2.1.bias'          : ['any', 'G_synthesis/16x16/ToRGB/bias'                 ],
    'toRGBs.3.0.weight'        : ['con', 'G_synthesis/32x32/ToRGB/weight'               ],
    'toRGBs.3.0.fc.weight'     : ['fc_', 'G_synthesis/32x32/ToRGB/mod_weight'           ],
    'toRGBs.3.0.bias.bias'     : ['any', 'G_synthesis/32x32/ToRGB/mod_bias'             ],
    'toRGBs.3.1.bias'          : ['any', 'G_synthesis/32x32/ToRGB/bias'                 ],
    'toRGBs.4.0.weight'        : ['con', 'G_synthesis/64x64/ToRGB/weight'               ],
    'toRGBs.4.0.fc.weight'     : ['fc_', 'G_synthesis/64x64/ToRGB/mod_weight'           ],
    'toRGBs.4.0.bias.bias'     : ['any', 'G_synthesis/64x64/ToRGB/mod_bias'             ],
    'toRGBs.4.1.bias'          : ['any', 'G_synthesis/64x64/ToRGB/bias'                 ],
    'toRGBs.5.0.weight'        : ['con', 'G_synthesis/128x128/ToRGB/weight'             ],
    'toRGBs.5.0.fc.weight'     : ['fc_', 'G_synthesis/128x128/ToRGB/mod_weight'         ],
    'toRGBs.5.0.bias.bias'     : ['any', 'G_synthesis/128x128/ToRGB/mod_bias'           ],
    'toRGBs.5.1.bias'          : ['any', 'G_synthesis/128x128/ToRGB/bias'               ],
    'toRGBs.6.0.weight'        : ['con', 'G_synthesis/256x256/ToRGB/weight'             ],
    'toRGBs.6.0.fc.weight'     : ['fc_', 'G_synthesis/256x256/ToRGB/mod_weight'         ],
    'toRGBs.6.0.bias.bias'     : ['any', 'G_synthesis/256x256/ToRGB/mod_bias'           ],
    'toRGBs.6.1.bias'          : ['any', 'G_synthesis/256x256/ToRGB/bias'               ],
    'toRGBs.7.0.weight'        : ['con', 'G_synthesis/512x512/ToRGB/weight'             ],
    'toRGBs.7.0.fc.weight'     : ['fc_', 'G_synthesis/512x512/ToRGB/mod_weight'         ],
    'toRGBs.7.0.bias.bias'     : ['any', 'G_synthesis/512x512/ToRGB/mod_bias'           ],
    'toRGBs.7.1.bias'          : ['any', 'G_synthesis/512x512/ToRGB/bias'               ],
    'toRGBs.8.0.weight'        : ['con', 'G_synthesis/1024x1024/ToRGB/weight'           ],
    'toRGBs.8.0.fc.weight'     : ['fc_', 'G_synthesis/1024x1024/ToRGB/mod_weight'       ],
    'toRGBs.8.0.bias.bias'     : ['any', 'G_synthesis/1024x1024/ToRGB/mod_bias'         ],
    'toRGBs.8.1.bias'          : ['any', 'G_synthesis/1024x1024/ToRGB/bias'             ],
}
