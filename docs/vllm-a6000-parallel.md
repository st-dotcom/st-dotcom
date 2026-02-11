# A6000 2枚を使ってローカルLLMをKVキャッシュを効かせつつ並列実行する方法

NVIDIA RTX A6000 2枚を用いた環境で、vLLMを使用して `openai/gpt-oss-20b` を並列推論（Tensor Parallelism）させ、かつKVキャッシュ（Prefix Caching）を有効化して高速に動作させる手順をまとめます。

## 1. 環境構築（Minicondaのインストール）

Linux個人環境（一般ユーザ権限）でPython環境を構築するためにMinicondaを利用します。
すでにConda環境がある場合は、このステップをスキップしてください。

### インストーラのダウンロード
```bash
cd ~/Downloads
# x86_64向け (ARM系の場合はURLを変更してください)
wget [https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh](https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh)
セットアップ実行
Bash
bash Miniconda3-latest-Linux-x86_64.sh
対話モードで以下のように回答します：

利用規約: yes

インストール場所: ~/miniconda3 （デフォルト推奨）

conda initを実行: yes

シェルの再読み込みと確認
Bash
# Bashの場合
source ~/.bashrc

# Zshの場合
source ~/.zshrc

# バージョン確認
conda --version
初期設定（推奨）
conda-forge を優先設定にしておくと、パッケージ依存解決がスムーズになります。

Bash
conda config --add channels conda-forge
conda config --set channel_priority strict
### 2. vLLM実行用環境の作成
vLLM用の仮想環境を作成し、必要なライブラリをインストールします。

仮想環境の作成 (Python 3.11)
Bash
conda create -n vllm python=3.11 -y
conda activate vllm
必要なライブラリのインストール
vLLM本体と、Hugging Face Hubへのログインツールを入れます。

Bash
pip install vllm huggingface_hub
### 3. モデルの実行
Hugging Face トークンの設定
モデルのダウンロード権限があるトークンを環境変数に設定します。

Bash
export HUGGINGFACE_HUB_TOKEN=あなたのトークン文字列
vLLMサーバーの起動
以下のコマンドでサーバーを立ち上げます。

GPU数: A6000 × 2枚 (--tensor-parallel-size 2)

機能: KVキャッシュ有効化 (--enable-prefix-caching)

Bash
python -m vllm.entrypoints.openai.api_server \
  --model openai/gpt-oss-20b \
  --tensor-parallel-size 2 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.95 \
  --dtype bfloat16 \
  --trust-remote-code \
  --enforce-eager \
  --enable-prefix-caching
パラメータ解説
--tensor-parallel-size 2: GPUを2枚使用してモデルを分割ロードします。

--enable-prefix-caching: プロンプトのKVキャッシュを再利用し、処理を高速化します。

--gpu-memory-utilization 0.95: GPUメモリの95%をvLLMに使用させます。

--enforce-eager: CUDAグラフ関連のエラーが出る場合にEagerモードで実行させます（安定性重視）。
