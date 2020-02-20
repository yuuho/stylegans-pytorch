
import argparse
from pathlib import Path
import pickle

import numpy as np
import cv2
import torch


# コマンドライン引数の取得
def parse_args():
    parser = argparse.ArgumentParser(description='著者実装を動かしたり重みを抜き出したり')
    parser.add_argument('version',type=int,
                            help='version name')
    parser.add_argument('-w','--weight_dir',type=str,default='/tmp/stylegans-pytorch',
                            help='学習済みのモデルを保存する場所')
    parser.add_argument('-o','--output_dir',type=str,default='/tmp/stylegans-pytorch',
                            help='生成された画像を保存する場所')
    return parser.parse_args()


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


setting = {
    1: {
        'src_weight': 'stylegan1_ndarray.pkl',
        'src_latent': 'latents1.pkl',
        'dst_image' : 'stylegan1_pt.png',
        'dst_weight': 'stylegan1_state_dict.pth'    },
    2: {
        'src_weight': 'stylegan2_ndarray.pkl',
        'src_latent': 'latents2.pkl',
        'dst_image' : 'stylegan2_pt.png',
        'dst_weight': 'stylegan2_state_dict.pth'    },
}


if __name__ == '__main__':
    # コマンドライン引数の取得
    args = parse_args()

    # バージョンによって切り替え
    cfg = setting[args.version]
    if args.version==1:
        from stylegan1 import Generator, name_trans_dict
    elif args.version==2:
        from stylegan2 import Generator, name_trans_dict


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
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    with torch.no_grad():
        z = latents.to(device)
        generator.to(device)
        img = generator(z)
        normalized = (img.clamp(-1,1)+1)/2*255
        images = normalized.permute(0,2,3,1).cpu().numpy().astype(np.uint8)

    # 出力を並べる関数
    def make_table(imgs):
        # 出力する個数，解像度
        num_H, num_W = 4,4
        H = W = 1024

        canvas = np.zeros((H*num_H,W*num_W,3),dtype=np.uint8)
        for i,p in enumerate(imgs):
            h,w = i//num_W, i%num_W
            canvas[H*h:H*-~h,W*w:W*-~w,:] = p[:,:,::-1]
        return canvas
 
    print('image output...')
    cv2.imwrite(str(Path(args.output_dir)/cfg['dst_image']), make_table(images))
    
    print('weight save...')
    torch.save(generator.state_dict(),str(Path(args.weight_dir)/cfg['dst_weight']))
    
    print('all done')
