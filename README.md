# competition2018 Tensorflow
Tensorflowによるテンプレートです．  
全結合3層(Fully connected), CNNを実装しています．

## 実行方法
`python classification.py`
以下のようにフォルダを配置してください．  
images  
└ target  
└ others

## オプション
### `--cnn`
指定すると，cnnモデルを使用します．
### `--gpu`
指定すると，gpuを使用します．

### 実行例
`python classification.py --cnn --gpu`
