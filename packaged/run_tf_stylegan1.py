
from pathlib import Path
import argparse
import pickle
# from tempfile import TemporaryDirectory

import numpy as np
import PIL.Image
import tensorflow as tf

# import dnnlib


# コマンドライン引数の取得
def parse_args():
    parser = argparse.ArgumentParser(description='著者実装を動かしたり重みを抜き出したり')
    parser.add_argument('-w','--weight_dir',type=str,default='/tmp/stylegans-pytorch',
                            help='学習済みのモデルを保存する場所')
    parser.add_argument('-o','--output_dir',type=str,default='/tmp/stylegans-pytorch',
                            help='生成された画像を保存する場所')
    parser.add_argument('--batch_size',type=int,default=1,
                            help='バッチサイズ')
    args = parser.parse_args()
    args.resolution = 1024
    return args


# tensorflowの初期化
def init_tf():
    tf_random_seed = np.random.randint(1 << 31)
    tf.set_random_seed(tf_random_seed)

    config_proto = tf.ConfigProto()
    config_proto.graph_options.place_pruned_graph = True
    config_proto.gpu_options.allow_growth = True

    session = tf.Session(config=config_proto)
    session._default_session = session.as_default()
    session._default_session.enforce_nesting = False
    session._default_session.__enter__()

    return session


# tensorflowの出力を正規化
def convert_images_to_uint8(images):
    images = tf.cast(images, tf.float32)
    images = tf.transpose(images, [0, 2, 3, 1])
    images = (images+1.0) / 2.0 * 255.0
    images = tf.saturate_cast(images, tf.uint8)
    return images


# メイン関数
def generate_images(args):
    file_names = {
        'input_weight'  : 'karras2019stylegan-ffhq-1024x1024.pkl',
        'output_weight' : 'stylegan1_ndarray.pkl',
        'used_latents'  : 'latents1.pkl',
        'output_image'  : 'stylegan1_tf.png',
    }
    
    init_tf()

    # 配布されている重みの読み込み
    with (Path(args.weight_dir)/file_names['input_weight']).open('rb') as f:
        *_, Gs = pickle.load(f)
    
    # 重みをnumpy形式に変換
    ndarrays = {k:v.eval() for k,v in Gs.vars.items()}
    [print(k,v.shape) for k,v in ndarrays.items()]

    # 重みをnumpy形式で保存
    print('weight save...')
    with (Path(args.weight_dir)/file_names['output_weight']).open('wb') as f:
        pickle.dump(ndarrays,f)


    # 画像を出力してみる
    print('run network')
    
    # 出力する個数，解像度
    num_H, num_W = 4,4
    N = num_images = num_H*num_W
    H = W = args.resolution
    
    # 出力を並べる関数
    def make_table(imgs):
        canvas = np.zeros((H*num_H,W*num_W,3),dtype=np.uint8)
        for i,p in enumerate(imgs):
            h,w = i//num_W, i%num_W
            canvas[H*h:H*-~h,W*w:W*-~w,:] = p
        return canvas

    # 乱数シードを固定，潜在変数の取得・保存
    latents = np.random.RandomState(5).randn(N, 512)
    with (Path(args.output_dir)/file_names['used_latents']).open('wb') as f:
        pickle.dump(latents, f)

    images = np.empty((N,args.resolution,args.resolution,3),dtype=np.uint8)
    for i in range(0, N, args.batch_size):
        j = min(i+args.batch_size, N)
        z = latents[i:j]
        images[i:j] = Gs.run(z, None, truncation_psi=0.7, randomize_noise=False,
                            output_transform= {'func': convert_images_to_uint8})

    # 画像の保存
    PIL.Image.fromarray(make_table(images)).save(Path(args.output_dir)/file_names['output_image'])


if __name__ == '__main__':
    args = parse_args()

    generate_images(args)
