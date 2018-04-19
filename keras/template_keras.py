import os
import sys
from argparse import ArgumentParser

from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.python.keras.layers import Dense, Flatten
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.optimizers import SGD

sys.path.append('../')

from competition_utility import dafaset_loader as dl


LR = 0.01
BATCH_SIZE = 20
EPOCHS = 100


def build_sequential_model():
    model = Sequential()
    model.add(Flatten(input_shape=(64, 64, 3)))
    model.add(Dense(2000, activation="sigmoid"))
    model.add(Dense(2, activation="sigmoid"))
    return model


def build_functional_api_model():
    inputs = Input(shape=(64, 64, 3))
    x = Flatten()(inputs)
    x = Dense(2000, activation="sigmoid")(x)
    x = Dense(2, activation="softmax")(x)
    return Model(inputs=inputs, outputs=x)


def main(path):
    dirs = [os.path.join(path, "target"), os.path.join(path, "others")]
    labels = [dl.DataSet.TARGET, dl.DataSet.OTHERS]
    loader = dl.DatasetLoader(dirs, labels)
    train, test = loader.load_train_test()
    train.print_information()
    test.print_information()

    # model = build_sequential_model()
    model = build_functional_api_model()
    model.compile(optimizer=SGD(lr=LR),
                  loss="binary_crossentropy",
                  metrics=["accuracy"])

    tb_cp = TensorBoard()
    model.fit(train.images, train.labels,
              batch_size=BATCH_SIZE,
              epochs=EPOCHS,
              validation_data=(test.images, test.labels),
              callbacks=[tb_cp])


if __name__ == "__main__":
    parser = ArgumentParser(description="Kerasを利用したテンプレート")
    parser.add_argument("images_path", type=str,
                        help="画像を保存しているディレクトリへのパス")
    args = parser.parse_args()
    main(args.images_path)
