from competition_utility import dafaset_loader as dl

if __name__ == "__main__":
    loader = dl.DatasetLoader(["Images", "OTHERS"], [dl.DataSet.TARGET, dl.DataSet.OTHERS])

    # [load_train_test]
    # trainとtestがDatasetオブジェクトで返ってきます
    # Datasetオブジェクトの__call__はイテレータブルになっています
    train, test = loader.load_train_test(train_rate=0.75)
    train.print_information()
    test.print_information()
    for batch in train(batch_size=30):
        print(batch.images.shape, batch.labels.shape)
        print(batch.images[0])

    # [load_raw_dataset]
    # データ全体がDatasetオブジェクトで返ってきます
    dataset = loader.load_raw_dataset()
    dataset.print_information()
