# rootless docker

最新のDockerではsudo権限がなくても各ユーザーが利用できるような仕組みが導入された．

インターネットで拾ってきたイメージを各ユーザーがroot権限で実行できるようになっていては危険なので
共有サーバーではrootless Dockerを利用する必要がある．

# 概要
rootless dockerは各ユーザーの ``~/bin/`` にインストールされ，
各ユーザーが ``systemctl --user start docker`` として自分の権限でdockerデーモンを起動して使う．

gpuを使うためにはシステム権限でnvidia-container-runtimeを入れて，
各ユーザーがアクセスできるように設定しておく必要がある．

# インストール方法
## sudo権限が必要なこと
root権限で実行しなければならないのはnvidia-container-runtimeのインストールと設定のみである．

### nvidia-container-runtime用のGPG鍵とPPAの追加
```
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
```
nvidiaのGPG鍵は定期的にアップデートしないとapt updateが通らなくなるので注意．上記コマンドを再度行えばOK

### nvidia-container-runtimeのインストール
```
sudo apt install nvidia-container-runtime
```

### ユーザーがGPUを使えるように設定
``sudo vim /etc/nvidia-container-runtime/config.toml``
一行だけ書き換える
```
no-cgroups = true
```

再起動
```
sudo reboot
```

## 各ユーザーがやること
  * rootless dockerのインストール
  * dockerデーモンの起動

### rootless dockerのインストール
これだけ
```
curl -sSL https://get.docker.com/rootless | sh
```

``~/bin/``にdockerデーモンがインストールされるのでパスを通す必要がある．
dockerはどこに接続すればいいか知りたいので ``DOCKER_HOST`` が設定されている必要がある．

### dockerデーモンの起動
ちゃんとパスが通せていたら動くはず
``systemctl --user start docker``
動かなかったらエラーメッセージをよく読んでパスを通す．
LDAPなどユーザーの管理が各サーバー上に無い場合はできません...


### 実行できるかテスト
```
docker run --gpus all -it --rm nvidia/cuda:8.0-cudnn6-devel-ubuntu16.04
nvidia-smi
nvcc -V
```

