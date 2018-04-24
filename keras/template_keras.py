import os
import sys
from argparse import ArgumentParser

from keras import Sequential, Input, Model
from keras.callbacks import TensorBoard
from keras.layers import Flatten, Dense, Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import plot_model, multi_gpu_model

sys.path.append('../')

from competition_utility import dafaset_loader as dl


LR = 0.01
BATCH_SIZE = 20
EPOCHS = 500


def build_sequential_model():
    """
    Sequentialモデルを利用した全結合層のみからなるモデル
    Functional APIほど複雑なことはできない
    """

    # SequentialオブジェクトにLayerを追加していくことで，モデルを組み立てる
    model = Sequential()
    model.add(Flatten(input_shape=(64, 64, 3)))     # 入力を平滑化して全結合層に入力可能な形に変更
    model.add(Dense(2000, activation="sigmoid"))    # 全結合層．ユニット数と活性化関数を指定
    model.add(Dense(2, activation="sigmoid"))       # 2クラス分類の最終層はsoftmaxよりもsigmoidが良いらしい
    return model


def build_functional_api_model():
    """
    Functional APIを利用した全結合層飲みからなるモデル
    分岐や統合などがあるモデルを構築することが可能
    """

    # Layerに対して__call__メソッドを利用することで，モデルを組み立てる
    inputs = Input(shape=(64, 64, 3))
    x = Flatten()(inputs)                       # 入力を平滑化して全結合層に入力可能な形に変更
    x = Dense(2000, activation="sigmoid")(x)    # 全結合層．ユニット数と活性化関数を指定
    x = Dense(2, activation="sigmoid")(x)       # 2クラス分類の最終層はsoftmaxよりもsigmoidが良いらしい
    return Model(inputs=inputs, outputs=x)      # 最終的にできたモデルの入力と出力を指定．複数可


def build_cnn_model():
    """
    3層のCNN(Convolutional Neural Network)と2層の全結合層(FC)からなるモデル
    """
    model = Sequential()
    # CNN．ConvolutionとPoolingを交互に繰り返す
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same", input_shape=(64, 64, 3)))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), activation="relu", padding="same"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # FC
    model.add(Flatten())
    model.add(Dense(1024, activation="sigmoid"))
    model.add(Dense(2, activation="sigmoid"))
    return model


def main(path):
    # データセット読み込み．こちらで用意したクラスを利用しているので，調べても情報は出てこないことに注意
    dirs = [os.path.join(path, "target"), os.path.join(path, "others")]
    labels = [dl.DataSet.TARGET, dl.DataSet.OTHERS]
    loader = dl.DatasetLoader(dirs, labels)
    train, test = loader.load_train_test()
    train.print_information()
    test.print_information()

    # モデルの構築
    # model = build_sequential_model()
    # model = build_functional_api_model()
    model = build_cnn_model()

    # 複数のGPUを利用して学習する際に指定
    # model = multi_gpu_model(model, gpus=2)

    # 作成したモデルを画像として可視化する．pipでgraphvizとpydot-ngのインストールが必要
    plot_model(model, show_shapes=True)

    # どのような学習処理を行なうかを設定．
    model.compile(optimizer=SGD(lr=LR),         # Stochastic Gradient Descent(確率的勾配降下法)
                  loss="binary_crossentropy",   # 交差エントロピー
                  metrics=["accuracy"])         # モデルの評価に使用する関数を指定．学習には利用されない

    # コールバックの作成．エポックやバッチの終了時などに行いたい処理のコールバック
    # TensorBoardに出力をすることで，学習過程を可視化できる
    # 他には過学習を防止するためのEarlyStoppingや，モデルを保存するModelCheckpointなどが存在する
    # また，自身でコールバックを作成することも可能
    # >tensorboard --logdir=<logが存在するディレクトリのパス>
    # 表示されるURLをブラウザで開くことで，確認ができる
    tb = TensorBoard(log_dir="./logs")

    # 学習の開始．学習での精度やロスの遷移などは，Historyオブジェクトとして返される
    history = model.fit(train.images, train.labels,                  # 学習データ
                        batch_size=BATCH_SIZE,                       # バッチサイズ．一般的にはGPUのメモリに乗るだけ大きくしたほうが良い
                        epochs=EPOCHS,                               # エポック数．指定した数だけ学習を繰り返す．1epochで学習データを全て一回ずつ使う
                        validation_data=(test.images, test.labels),  # テストデータ．学習データと異なり，入力とラベルをタプルで渡す必要がある
                        callbacks=[tb])                              # コールバック．リストで全てを一度に渡す


if __name__ == "__main__":
    parser = ArgumentParser(description="Kerasを利用したテンプレート")
    parser.add_argument("images_path", type=str,
                        help="画像を保存しているディレクトリへのパス")
    args = parser.parse_args()
    main(args.images_path)
