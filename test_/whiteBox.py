#!sftp://tonnn!10.0.10.204:22/home/tonnn/xiu/venvs/scmj_ppo2/bin/python3.6
# -*- coding: utf-8 -*-
# @Time    : 2023/3/3
# @Author  : xiu
# @File    : whiteBox.py
# @Description:白盒测试

# import random
# import mah_tool.feature_extract_v10 as feature_extract
# import mah_tool.shangrao_hu as hu


# handcard = [1,1,1,1, 2,2, 3,3, 4,4, 7,7, 8, 9]  # 豪华七对宝吊 一个宝
#
# suit = []
# jing_card = 8
# catch_card = 7
# result = hu.haohua_qidui(handcard,suit, jing_card, catch_card)
# print("手牌：", handcard,"能胡吗：", result)



# handcard2 = [1,1,1, 2,2,2, 3,3,3, 4,4,4, 4,5]
# handcard2 = [1,1,2,2,3,3,3,4,5,5,6,6,7,7]
# handcard2 = [1,2,3, 4,5,6, 7,8,9, 11,12,13, 15,16]
# suit = []
# jing_card = 15
# catch_card = 16
# result = hu.is_hu(handcard2, suit, jing_card, catch_card)
# #
# # result = hu.normal_hu(handcard2, suit, jing_card, catch_card)
# print("手牌：", handcard2, "能胡吗：", result)
#
#
# # print(feature_extract.wait_types_comm([1,2,3, 4,5,6, 17,18,19, 22,22,22, 33,34], [], 0))
#
# print({1:3,2:4,3:2})
#
# import tensorflow as tf
#
# sess = tf.InteractiveSession()
#
# values = tf.constant([[0, 0, 0, 1],
#                       [0, 1, 0, 0],
#                       [0, 0, 1, 0]])
#
# T = tf.constant([[0, 1, 2 ,  3],
#                  [4, 5, 6 ,  7],
#                  [8, 9, 10, 11]])
#
# max_indices = tf.argmax(values, axis=1)
# print(sess.run(max_indices))
# print(sess.run(tf.stack([values,T],0)))
# print(sess.run(tf.concat([values,T],1)))
#
# str_split = "weight-ppo-40000".split("-")
# print(str_split[2])
#
# import numpy as np
# data = ["大", "小"]
# p = [0.99, 0.01]
#
# for i in range(10):
#     print(np.random.choice(data, p = p))
#
#
#
# max_indicies = tf.argmax(T, 1)
# import tensorflow as tf
#
# sess = tf.InteractiveSession()
#
# values = tf.constant([[0, 0, 0, 1],
#                       [0, 1, 0, 0],
#                       [0, 0, 1, 0]])
#
# T = tf.constant([[0, 1, 2 ,  3],
#                  [4, 5, 6 ,  7],
#                  [8, 9, 10, 11]])
#
# max_indices = tf.argmax(values, axis=1)
# print(max_indices.eval())
# # If T.get_shape()[0] is None, you can replace it with tf.shape(T)[0].
# result = tf.gather_nd(T, tf.stack((tf.range(T.get_shape()[0],
#                                             dtype=max_indices.dtype),
#                                    max_indices),
#                                   axis=1))
# print(result.eval())
#
# output = [0.,0.,0.,0.01788053,0.01788377,0.,
#  0.,        0.,       0.,        0.,        0.,        0.03023967,
#  0.01189825, 0.0127107,  0.03720286, 0.04555885 ,0.,         0.03005196,
#  0.,         0.04302652, 0.,         0.,         0.,         0.,
#  0.0408666,  0.0227937,  0.,         0.03081128, 0.02638525, 0.,
#  0.,         0. ,        0. ,        0.        ]
#
# # output= [2,3,4,0, 0,5,6, 0,0]
# output = [i*100 for i in output]
#
# def softmax(x):
#     x = np.array(x)
#     x_row_max = x.max(axis=-1)
#     x_row_max = x_row_max.reshape(list(x.shape)[:-1]+[1])
#     x = x - x_row_max
#     x_exp = np.exp(x)
#     x_exp_row_sum = x_exp.sum(axis=-1).reshape(list(x.shape)[:-1]+[1])
#     softmax = x_exp / x_exp_row_sum
#     return softmax
#
# print((softmax(output)))
#import tensorflow as tf
#import os
#from tensorflow.python.client import device_lib
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "99"
# print(device_lib.list_local_devices())

# 胡牌向听数测试
# import copy
# import mah_tool.feature_extract_v10 as feature
# import mah_tool.tool2 as tool2


# handcards = [1,1,1, 7, 11,12,13, 18,18,18, 19, 21, 22, 23]
# fulu = []
# jing_card = 1
# catch_card = 19
# result = hu.is_hu(handcards, fulu, jing_card, catch_card)
# print("is hu:", result)
#
#
# handcards_, fulu_, jing_card_, catch_card_ = hu.transform_params(handcards, fulu,jing_card,catch_card)
# xt_7, xt_91, xt_13, xt_h7, xt_normal = feature.wait_types_7(handcards_, fulu_, jing_card_), feature.wait_types_19(handcards_, fulu_),feature.wait_types_13(handcards_,fulu_), feature.wait_types_haohua7(handcards_,fulu_, jing_card_), feature.wait_types_comm(handcards_,fulu_,jing_card_)
#
# print("xt_7:{},xt_91:{},xt_13:{},xt_h7:{},xt_normal:{}".format(xt_7,xt_91,xt_13,xt_h7,xt_normal))
#
# import  numpy as np
# print(np.zeros([3,5,34,1]))
#
# class WW(object):
#     def __init__(self, name):
#         self.name = name
#         print("SB:", name)
#
# class Animal(WW):  # 定义一个父类
#     def __init__(self, name):  # 父类的初始化
#         self.name = name
#         self.role = 'parent'
#         print('I am father')
#
#
# class Dog(Animal):  # 定一个继承Animal的子类
#     def __init__(self,name,age):  # 子类的初始化函数，此时会覆盖父类Animal类的初始化函数
#         super(Animal, self).__init__(name)  # 在子类进行初始化时，也想继承父类的__init__()就通过super()实现,此时会对self.name= 'animal'
#         print('I am son')
#         self.age = age
#         self.name = 'dog'  # 定义子类的name属性,并且会把刚才的self.name= 'animal'更新为'dog'
#
#
# # animal = Animal()#I am father
# xbai = Dog('dog',15)  # I am father,I am son
# print(xbai.name)  # 'dog'
# # print(xbai.role)  # 'parent'
# print(xbai.age)
#
# def longestValidParentheses(s: str) -> int:
#     str_len = len(s)
#     dp = []
#     mark = [0] * str_len
#     for i in range(str_len):
#         if s[i] == "(":
#             dp.append(i)
#         else:
#             if len(dp) == 0:
#                 mark[i] = 1  # 多余的右括号是不需要的，标记
#             else:
#                 dp.pop()  # 右括号匹配成功，无需标记
#
#     while len(dp) > 0:
#         mark[dp.pop()] = 1
#     print(mark)
#     ans = 0
#     len_ = 0
#     for i in mark:
#         if i == 1:
#             len_ = 0
#             continue
#         len_ += 1
#         ans = max(len_, ans)
#     return ans
#
# ans = longestValidParentheses(")(()())")
# print(ans)
import numpy as np
#

import tensorflow as tf
# with tf.Session() as sess:
#     print(sess.run(tf.exp(2.0)))

import pandas as pd
# reward_7453 = pd.read_table("indicator/win_rate_0_7453reward.txt")
# print(reward_7453.columns.values)0
# # columns = {'当前EPOCh ':"EPOCH", ' 当前胜率  ':"WIN_RATE", ' 当前败率 ':"FAIL_RATE", ' 流局率':"ABORT_RATE"}
# columns = {"EPOCH", "WIN_RATE", "FAIL_RATE", "ABORT_RATE"}
# reward_7453.rename(columns=columns)
# print(reward_7453.columns.values)

# L = [1,2,3,4]
# L.insert(0, 3)
# print(L)
# rev = False if 1%2 == 1 else True
# print(rev)

# import numpy as np
#
# obs = np.load("../sbln_learning/obs.npy")
# test_obs = obs[0]
# callback = lambda x, y: (x, y)
# print(callback(1,2))
#
# xt_91 = feature_extract.wait_types_19(tool2.list10_to_16([1,2,3,9,9,11,13,14]),[ [1,2,3],[17,17,17]])
# print(xt_91)

test_op = {1:[], 2:[], 3:[]}
print(sorted(test_op.keys(),reverse=True))