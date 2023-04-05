from autoencoders.data import MNISTDataModule


def test_simple():
  ds = MNISTDataModule(data_dir="dataset/MNIST/",
                       batch_size=32,
                       train_size=None)
  ds.prepare_data()
  ds.setup(stage="fit")
  ds.setup(stage="test")
  assert len(ds.mnist_train) == 55000
  assert len(ds.mnist_val) == 5000
  assert len(ds.mnist_test) == 10000
