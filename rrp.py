# -*- coding: utf-8 -*-
"""
Created on 2023-01
@author: ZQ
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import time
import random
from keras import optimizers
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from make_dataset import *
from make_model import *
from Utils import *
import tensorflow as tf

print("\033[1;31m ================================[  Respiratory Rate Prediction Starts  ]=================================\033[0m")
LOAD_FORM_SAVE = False
LOAD_MODEL = False
WIN_SIZE = 125*16
BATCH_SIZE = 64
EPOCHS = 500
DOWN_SAMPLING_GRADE = 8
LR = 0.001
CNN_FILTERS = 512
CNN_KERNEL = 20
LSTM_UNIT = 256
LSTM_DENSE = 1024
DENSE1_DIM = 512
DENSE2_DIM = 128
MAX = 1000000000
FOLD_NUM = 10
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=5, mode='auto', min_lr=0.0001)
early_stop = EarlyStopping(monitor="val_loss", patience=20, verbose=0, mode="min")

# dataset
csv_path = '/home/zz/respiratory_rate_prediction/data/bidmc_RR_16s_overlap87.5_vmd_zscore_RRscreen.csv'
# csv_path = '/home/zz/respiratory_rate_prediction/data/capnobase_RR_16s_overlap87.5_vmd_zscore_RRscreen_age5.csv'

raw_data = read_csv(WIN_SIZE, csv_path)

for fold_index in range(FOLD_NUM):
    input_train_np, input_val_np, input_test_np = fold_n(fold_index=fold_index, raw_data=raw_data, fold_num=FOLD_NUM)

    x1_train, x2_train, y_train, x1_val, x2_val, y_val, x1_test, x2_test, y_test = make_dataset_from_fold_n(
        win_size=WIN_SIZE, input_train_np=input_train_np, input_val_np=input_val_np, input_test_np=input_test_np)

    x1_train, x2_train, y_train, x1_val, x2_val, y_val, x1_test, x2_test, y_test = down_sampling(
        x1_train, x2_train, y_train, x1_val, x2_val, y_val, x1_test, x2_test, y_test, down_sampling_grade=DOWN_SAMPLING_GRADE)

    for repeat_index in range(1):
        TIME_STAMP = time.strftime('%Y-%m-%d-%H-%M', time.localtime(time.time()))

        seed_value = np.random.randint(MAX)
        random.seed(seed_value)
        np.random.seed(seed_value)
        tf.random.set_seed(seed_value)
        print("The seed value in {}th training is {}".format(repeat_index, seed_value))

        model = TransRR(250)
        model.compile(optimizer=optimizers.Adam(lr=LR), loss="mae")
        model.summary()

        history = model.fit(x=[x1_train, x2_train], y=[y_train],
                            validation_data=([x1_val, x2_val], [y_val]),
                            batch_size=64, shuffle=True, epochs=EPOCHS, verbose=1,
                            callbacks=[reduce_lr, early_stop])

        loss = history.history['loss']
        val_loss = history.history['val_loss']
        predicted_rr_train = model.predict(x=[x1_train, x2_train], verbose=1)
        predicted_rr_val = model.predict(x=[x1_val, x2_val], verbose=1)
        predicted_rr_test = model.predict(x=[x1_test, x2_test], verbose=1)
        real_rr_train = y_train
        real_rr_val = y_val
        real_rr_test = y_test

        predicted_rr_train = predicted_rr_train.squeeze()
        real_rr_train = real_rr_train.squeeze()
        predicted_rr_val = predicted_rr_val.squeeze()
        real_rr_val = real_rr_val.squeeze()
        predicted_rr_test = predicted_rr_test.squeeze()
        real_rr_test = real_rr_test.squeeze()

        rr_in_train = np.array(real_rr_train)
        rr_in_val = np.array(real_rr_val)
        rr_in_test = np.array(real_rr_test)
        print("[ fold_index-"+str(fold_index)+"(repeat" +str(repeat_index)+") Respiratory Rate Prediction Ends ]")
        print("val_loss:" + str(round(val_loss[-1],2)))
        print("test mae:", loss_mae(rr_in_test, predicted_rr_test))
        print("test e:", loss_e(rr_in_test, predicted_rr_test))
        print("test pcc:", loss_pcc(rr_in_test, predicted_rr_test))
        print("test loa:", loss_loa(rr_in_test, predicted_rr_test))

print("[ ALL Respiratory Rate Prediction Ends ]")
exit(0)