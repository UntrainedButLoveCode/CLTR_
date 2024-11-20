import os
import numpy as np
import argparse

if not os.path.exists('./npydata'):
    os.makedirs('./npydata')

'''please set your dataset path'''

parser = argparse.ArgumentParser(description='CLTR')
parser.add_argument('--high_path', type=str, default='D:\high_for_PET',
                    help='the data path of high')

args = parser.parse_args()
high_root = args.high_path

try:

    high_train_path = high_root + '/train_data/images_2048/'
    high_test_path = high_root + '/test_data/images_2048/'

    train_list = []
    for filename in os.listdir(high_train_path):
        if filename.split('.')[1] == 'jpg':
            train_list.append(high_train_path + filename)
    train_list.sort()
    np.save('./npydata/high_train.npy', train_list)

    test_list = []
    for filename in os.listdir(high_test_path):
        if filename.split('.')[1] == 'jpg':
            test_list.append(high_test_path + filename)
    test_list.sort()
    np.save('./npydata/high_test.npy', test_list)

    print("Generate HIGH image list successfully", len(train_list), len(test_list))# 350 150
except:
    print("The HIGH dataset path is wrong. Please check your path.")

