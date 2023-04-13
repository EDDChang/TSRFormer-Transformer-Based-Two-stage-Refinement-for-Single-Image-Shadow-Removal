import os

from dataset import DataLoaderTrain, DataLoaderVal, DataLoaderTest, DataLoaderTestMask

def get_training_data(config):
    # assert os.path.exists(rgb_dir)
    return DataLoaderTrain(config)

def get_validation_data(config):
    # assert os.path.exists(rgb_dir)
    return DataLoaderVal(config)

def get_train_data_ISTD():
    return DataLoaderTest('/mnt/188/b/NTIRE23/data/shadow/ISTD/ISTD_Dataset/train/train_A', f'/mnt/188/b/NTIRE23/data/shadow/ISTD/ISTD_Dataset/train/pseudo_free')
def get_test_data_ISTD():
    return DataLoaderTest('/mnt/188/b/NTIRE23/data/shadow/ISTD/ISTD_Dataset/test/test_A', f'/mnt/188/b/NTIRE23/data/shadow/ISTD/ISTD_Dataset/test/pseudo_free')

def get_train_data(ver):
    return DataLoaderTest('/mnt/188/b/NTIRE23/data/shadow/train/shadow', f'/mnt/188/b/NTIRE23/data/shadow/train/pseudo_free_{ver}')
def get_val_data(ver):
    return DataLoaderTest('/mnt/188/b/NTIRE23/data/shadow/val/shadow', f'/mnt/188/b/NTIRE23/data/shadow/val/pseudo_free_{ver}')
def get_test_data(ver):
    return DataLoaderTest('/mnt/188/b/NTIRE23/data/shadow/test/shadow', f'/mnt/188/b/NTIRE23/data/shadow/test/pseudo_free_{ver}')