
from pathlib import Path
import argparse
import pickle
from tempfile import TemporaryDirectory

import numpy as np
import PIL.Image
import tensorflow as tf

import dnnlib


# コマンドライン引数の取得
def parse_args():
    parser = argparse.ArgumentParser(description='著者実装を動かしたり重みを抜き出したり')
    parser.add_argument('-w','--weight_dir',type=str,default='/tmp/stylegans-pytorch',
                            help='学習済みのモデルを保存する場所')
    parser.add_argument('-o','--output_dir',type=str,default='/tmp/stylegans-pytorch',
                            help='生成された画像を保存する場所')
    return parser.parse_args()


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


# 画像生成関数(dnnlib.submit_runに渡すコールバック関数)
def generate_images(args):
    file_names = {
        'input_weight'  : 'stylegan2-ffhq-config-f.pkl',
        'output_weight' : 'stylegan2_ndarray.pkl',
        'used_latents'  : 'latents2.pkl',
        'output_image'  : 'stylegan2_tf.png',
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
    num_images = num_H*num_W
    H = W = 1024
    
    # 出力を並べる関数
    def make_table(imgs):
        canvas = np.zeros((H*num_H,W*num_W,3),dtype=np.uint8)
        for i,p in enumerate(imgs):
            h,w = i//num_W, i%num_W
            canvas[H*h:H*-~h,W*w:W*-~w,:] = p
        return canvas

    # 乱数シードを固定，潜在変数の取得・保存
    latents = np.random.RandomState(5).randn(num_images, 512)
    with (Path(args.output_dir)/file_names['used_latents']).open('wb') as f:
        pickle.dump(latents, f)

    images = Gs.run(latents, None, truncation_psi=0.7, randomize_noise=False,
                        output_transform= {'func': convert_images_to_uint8})

    # 画像の保存
    PIL.Image.fromarray(make_table(images)).save(Path(args.output_dir)/file_names['output_image'])


if __name__ == '__main__':
    args = parse_args()

    with TemporaryDirectory() as dir_name:
        # 設定に必要な情報を書き込んで実行
        config = dnnlib.SubmitConfig()
        config.local.do_not_copy_source_files   = True
        config.run_dir_root                     = dir_name  # 変なファイルが自動生成される場所

        dnnlib.submit_run(config,
            'run_tf_stylegan2.generate_images', # コールバック関数
            args=args)                          # コールバック関数の引数
