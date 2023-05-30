# -*- coding: utf-8 -*-
"""
Created on 2023-01
@author: ZQ
"""
import numpy as np

def loss_mae(rr_in_test, predicted_rr_test):
    diff = rr_in_test - predicted_rr_test
    diff_sq = abs(diff)
    diff_sq_mean = np.mean(diff_sq)
    return round(diff_sq_mean, 2)

def loss_e(rr_in_test, predicted_rr_test):
    diff = rr_in_test - predicted_rr_test
    diff_sq = abs(diff)
    diff_sq_mean = np.mean(diff_sq)
    return round(diff_sq_mean, 2)/np.mean(rr_in_test)

def loss_pcc(rr_in_test, predicted_rr_test):
    return np.corrcoef(rr_in_test, predicted_rr_test)[0][1]

def loss_loa(data1, data2):
    data1 = np.asarray(data1)
    data2 = np.asarray(data2)
    diff = data1 - data2
    md = np.mean(diff)
    sd = np.std(diff, axis=0)
    return (md - 1.96 * sd, md + 1.96 * sd)
