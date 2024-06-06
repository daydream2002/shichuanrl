#!sftp://tonnn!10.0.10.204:22/home/tonnn/xiu/venvs/scmj_ppo2/bin/python3.6
# -*- coding: utf-8 -*-
# @Time    : 2023/3/3
# @Author  : xiu
# @File    : read_npy.py
# @Description: 读数据-numpy格式
import numpy as np
import tensorflow as tf

#
# npy_file = np.load("./obs.npy")
# print(npy_file[0])

# logits = np.array([[0.6, 0.1, 0.05, 0.05, 0.03, 0.06, 0.01, 0.05, 0.05, 0.0],
#                   [0.09, 0.01, 0.05, 0.08, 0.025, 0.025, 0.02, 0.5, 0.1, 0.1],
#                   [0.9, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.02]
#                    ])
# logits = np.array([[0.6, 0.1, 0.05, 0.05, 0.03, 0.06, 0.01, 0.05, 0.05, 0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,],
#                   [0.09, 0.01, 0.05, 0.08, 0.025, 0.025, 0.02, 0.5, 0.1, 0.10,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,],
#                   [0.9, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.02,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,],
#                   [0.9, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.02,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,],
#                  [0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05]])

# logits = np.array([[0.9, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.02,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
#                   [0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05]])


# def calc_sbln_entropy(logits):
#     a_0 = logits - np.max(logits, axis=-1,keepdims=True)
#     exp_a_0 = np.exp(a_0)
#     z_0 = np.sum(exp_a_0,axis=-1,keepdims=True)
#     p_0 = exp_a_0 / z_0
#     return np.sum(p_0 * (np.log(z_0) - a_0), axis=-1)
# entropy = calc_sbln_entropy(logits)
# print(entropy)
# print(np.exp(0.6))
#
# if "":
#     print("ddd")

logitis = np.array([0] * 34)
logitis[1] = 0
logitis[2] = 0
logitis[5] = 100


# print(logitis)
def test_entory(x):
    a_0 = x - np.max(x, axis=-1, keepdims=True)
    exp_a_0 = np.exp(a_0)
    z_0 = np.sum(exp_a_0, axis=-1, keepdims=True)
    p_0 = exp_a_0 / z_0
    return np.sum(p_0 * (np.log(z_0) - a_0), axis=-1)


def test_prob(x):
    x_row_max = x.max(axis=-1)
    x_row_max = x_row_max.reshape(list(x.shape)[:-1] + [1])
    x = x - x_row_max
    x_exp = np.exp(x)
    x_exp_row_sum = x_exp.sum(axis=-1).reshape(list(x.shape)[:-1] + [1])
    softmax = x_exp / x_exp_row_sum
    return softmax


# def neglogp(logits, action):
#     # Note: we can't use sparse_softmax_cross_entropy_with_logits because
#     #       the implementation does not allow second-order derivatives...
#     one_hot_actions = np.eye(34)[action]
#     return tf.nn.softmax_cross_entropy_with_logits_v2(
#         logits=logits,
#         labels=one_hot_actions)
#     pass
l = [-3523.608, 962.382, -194.2013, -1094.9203, 697.1617,
     -454.791, -143.77692, -1236.8705, -307.96643, 902.8997,
     381.71927, -318.19107, -301.64737, -681.0222, 367.20377,
     -43.066666, -496.90472, 667.10785, -1263.6215, -953.20966,
     121.335335, -267.1639, -63.297802, -614.71313, 939.47284,
     1204.5238, -311.75195, 1042.7456, -1482.3445, 421.1765,
     2615.4922, -11.212275, 1334.0469, 606.4863, ]

if __name__ == '__main__':
    prob = test_prob(logitis)
    entory = test_entory(logitis)
    print(entory, prob)
    print(np.eye(34)[6])
    # print(np.log(0.1))
