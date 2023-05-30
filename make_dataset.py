# -*- coding: utf-8 -*-
"""
Created on 2023-01
@author: ZQ
"""
import csv
import numpy as np
import pandas as pd

RATIO_TRAIN = 0.64
RATIO_VAL = 0.16
RATIO_TEST = 1 - RATIO_TRAIN - RATIO_VAL # 0.2
MAX = 1000000000

def read_csv(win_size, csv_path):
    with open(csv_path, "r") as csvfile:
        reader = csv.reader(csvfile)
        rows = [(np.array(row[0:2 * win_size + 2])) for row in reader]
    return rows

# when fold_num==-1, training set = all data
# when fold_num==-1, training set + val set = all data
def fold_n(fold_index, raw_data, fold_num):
    if fold_num == -1:
        train_set = []
        val_set = []
        test_set = [item[1:] for item in raw_data]
        print("train_set="+str(len(train_set)))
        print("val_set=" + str(len(val_set)))
        print("test_set=" + str(len(test_set)))
    elif fold_num == -2:
        patient_list = pd.unique([item[0] for item in raw_data])
        all_patient_num = len(patient_list)
        val_patient_num = int(all_patient_num*0.2)
        val_patient_list = patient_list[0:val_patient_num]
        train_patient_list = [item for item in patient_list if item not in val_patient_list]
        train_set = [item[1:] for item in raw_data if item[0] in train_patient_list]
        val_set = [item[1:] for item in raw_data if item[0] in val_patient_list]
        test_set = []
        print("train_set="+str(len(train_set)))
        print("val_set=" + str(len(val_set)))
        print("test_set=" + str(len(test_set)))
    else:
        patient_list = pd.unique([item[0] for item in raw_data])
        all_patient_num = len(patient_list)
        fold_size = int(all_patient_num/fold_num)
        val_patient_num = int((all_patient_num - fold_size)*0.2)
        test_patient_list = patient_list[fold_index*fold_size:fold_index*fold_size+fold_size]
        train_val_patient_list = [item for item in patient_list if item not in test_patient_list]
        val_patient_list = train_val_patient_list[0:val_patient_num]
        train_patient_list = [item for item in train_val_patient_list if item not in val_patient_list]
        train_set = [item[1:] for item in raw_data if item[0] in train_patient_list]
        val_set = [item[1:] for item in raw_data if item[0] in val_patient_list]
        test_set = [item[1:] for item in raw_data if item[0] in test_patient_list]
        print("train_set="+str(len(train_set)))
        print("val_set="+str(len(val_set)))
        print("test_set="+str(len(test_set)))

    return np.array(train_set).astype(np.float), np.array(val_set).astype(np.float), np.array(test_set).astype(np.float)

# when input_train_np==[], test set = all data
def make_dataset_from_fold_n(win_size, input_train_np, input_val_np, input_test_np):
    train = input_train_np
    val = input_val_np
    test = input_test_np
    if len(input_train_np) > 0:
        x1_train = train[:,:(win_size), None]
        x2_train = train[:, win_size:(win_size*2), None]
        y_train = train[:, win_size*2]
    else:
        x1_train = train[:, None, None]
        x2_train = train[:, None, None]
        y_train = train[:, None]
    if len(input_val_np) > 0:
        x1_val = val[:,:(win_size), None]
        x2_val = val[:, win_size:(win_size*2), None]
        y_val = val[:, win_size*2]
    else:
        x1_val = val[:, None, None]
        x2_val = val[:, None, None]
        y_val = val[:, None]
    if len(input_test_np) > 0:
        x1_test = test[:,:(win_size), None]
        x2_test = test[:, win_size:(win_size*2), None]
        y_test = test[:, win_size*2]
    else:
        x1_test = test[:, None, None]
        x2_test = test[:, None, None]
        y_test = test[:, None]
    return x1_train, x2_train, y_train, x1_val, x2_val, y_val, x1_test, x2_test, y_test

def down_sampling(x1_train, x2_train, y_train, x1_val, x2_val, y_val, x1_test, x2_test, y_test, down_sampling_grade=8):
    max_index = max(x1_train.shape[1], x1_val.shape[1], x1_test.shape[1])
    id = np.arange(0, max_index, down_sampling_grade)
    return x1_train.take(id,1), x2_train.take(id,1), y_train, \
           x1_val.take(id,1), x2_val.take(id,1), y_val, \
           x1_test.take(id,1), x2_test.take(id,1), y_test