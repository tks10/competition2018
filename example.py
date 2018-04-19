from competition_utility import dafaset_loader as dl

if __name__ == "__main__":
    loader = dl.DatasetLoader(["Images", "Images"], [dl.DataSet.TARGET, dl.DataSet.OTHERS])

    # [load_train_test]
    # trainとtestがDatasetオブジェクトで返ってきます
    train, test = loader.load_train_test()
    train.print_information()
    test.print_information()

    # [load_raw_dataset]
    # データ全体がDatasetオブジェクトで返ってきます
    dataset = loader.load_raw_dataset()
    dataset.print_information()

    # [load_train_test_batch]
    # trainとtestが[batch_count][dict][width][height][channel]で返ってきます
    # Keyは"data"と"label"
    train, test = loader.load_train_test_batch()
    for batch in train:
        print(batch["data"].shape, batch["label"].shape)

