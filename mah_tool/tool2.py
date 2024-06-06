# -*- coding: utf-8 -*-
# !/usr/bin/python
import random
import copy
import logging
# import mahjongEnv.mah_tool.feature_extract_v8 as feature_extract
# import interface.interface_v1.feature_extract_v8 as feature_extract
from mah_tool import feature_extract_v10 as feature_extract


# from mah_tool.suphx_extract_features import feature_extract_v10 as feature_extract_suphx

# 十进制转十六进制
def f10_to_16(num):  # 用16进制表示
    a = int(num / 10)
    b = num - a * 10
    return a * 16 + b


# 十六进制转十进制
def f16_to_10(num):  # 用十进制表示
    a = int(num / 16)
    b = num - 16 * a
    return a * 10 + b


def list10_to_16(cardlist):
    cardlist2 = []
    for i in cardlist:
        cardlist2.append(int(f10_to_16(i)))
    cardlist2.sort()
    return cardlist2


def list10_to_16_2(cardlist):
    cardlist2 = []
    for i in cardlist:
        cardlist2.append(int(f10_to_16(i)))
    # cardlist2.sort()
    return cardlist2


# 16转10进制的批量转换代码
def list16_to_10(cardlist):
    cardlist2 = []
    for card in cardlist:
        cardlist2.append(f16_to_10(card))
    return cardlist2


def fulu_translate(fulu):
    actions2 = []
    for i in fulu:
        actions2.append(list10_to_16(i))
    return actions2


def translate3(op_card):  # 16进制op_card转换到 0-33 34转换    #######################################
    if 1 <= op_card <= 9:
        op_card = op_card - 1
    elif 17 <= op_card <= 25:
        op_card = op_card - 8
    elif 33 <= op_card <= 41:
        op_card = op_card - 15
    elif 49 <= op_card <= 55:
        op_card = op_card - 22
    elif op_card == 255:
        op_card = 34
    return op_card


def discard_translate(cardlist):
    cardlist2 = [0] * 34
    for i in cardlist:
        temp = f10_to_16(i)
        temp2 = translate3(temp)
        cardlist2[temp2] += 1
    return cardlist2


# 检查生成的手牌是否合理
def check(cardlist):
    for i in cardlist:
        if cardlist.count(i) > 4:
            return False
    return True


# 随机发放副露（未检查）
def random_deal_fulu(random_deal_handcards):
    shunzi = [  # 所有顺子
        # 万
        [1, 2, 3],
        [2, 3, 4],
        [3, 4, 5],
        [4, 5, 6],
        [5, 6, 7],
        [6, 7, 8],
        [7, 8, 9],
        # 条
        [11, 12, 13],
        [12, 13, 14],
        [13, 14, 15],
        [14, 15, 16],
        [15, 16, 17],
        [16, 17, 18],
        [17, 18, 19],
        # 筒
        [21, 22, 23],
        [22, 23, 24],
        [23, 24, 25],
        [24, 25, 26],
        [25, 26, 27],
        [26, 27, 28],
        [27, 28, 29], ]

    kezi = [  # 所有刻子
        [1, 1, 1],
        [2, 2, 2],
        [3, 3, 3],
        [4, 4, 4],
        [5, 5, 5],
        [6, 6, 6],
        [7, 7, 7],
        [8, 8, 8],
        [9, 9, 9],

        [11, 11, 11],
        [12, 12, 12],
        [13, 13, 13],
        [14, 14, 14],
        [15, 15, 15],
        [16, 16, 16],
        [17, 17, 17],
        [18, 18, 18],
        [19, 19, 19],

        [21, 21, 21],
        [22, 22, 22],
        [23, 23, 23],
        [24, 24, 24],
        [25, 25, 25],
        [26, 26, 26],
        [27, 27, 27],
        [28, 28, 28],
        [29, 29, 29],

        [31, 31, 31],
        [32, 32, 32],
        [33, 33, 33],
        [34, 34, 34],
        [35, 35, 35],
        [36, 36, 36],
        [37, 37, 37],
    ]
    list1 = shunzi + kezi
    length = len(random_deal_handcards)  # 手牌长度
    s = 4 - (length - 2)  # 应该构建的幅露长度
    fulu = random.sample(list1, s)
    return fulu


# 列表的减法
def list_sub(list1, list2):
    temp = deepcopy(list1)
    for i in list2:
        temp.remove(i)  # 从list1中去除list2
    return temp


# 给某为玩家发指定数量的牌
def distribution_cards(card_library, num):
    cards_list = []
    for i in range(num):
        cards_list.append(card_library.pop())  # 发牌
    cards_list.sort()
    return cards_list


def translate34_to_136(cardlist):
    output = [0] * 136
    for i in range(len(cardlist)):
        if cardlist[i] != 0:
            output[i * 4 + cardlist[i] - 1] = 1
    return output


import numpy as np


def nonzeros_len(list1):
    count = 0
    for i in list1:
        if i != 0:
            count = count + 1
    return count


# 对某一数据进行onehot编码,max为数据中的最大值
def one_hot(max, x):
    newlist = [0] * max
    newlist[x - 1] = 1
    return newlist


# 为某一连续特征进行onehot编码
def batch_one_hot(max, list1):
    feature = []
    for i in list1:
        fea = one_hot(max, i)
        feature.extend(fea)
    return feature


# 调用监督模型，为强化学习缩减搜索范围
def model_based_rl_recommend(ret1, num):
    ret1 = ret1.tolist()[0]
    newlist = deepcopy(ret1)
    newlist.sort()
    newlist.reverse()
    print("new_list_len", len(newlist))
    # print nonzeros_len(newlist)
    if nonzeros_len(newlist) >= num:
        search = newlist[0:num]

    else:
        search = newlist[0:nonzeros_len(newlist)]
    search_id = []

    for i in search:
        id = ret1.index(i)
        search_id.append(id)
    search_id.sort()

    result = []
    for ii in search_id:
        ii = ii + 1
        ii = translate(ii)
        result.append(ii)
    result.sort()

    # print search
    # print newlist
    return result


def translate(i):  # 1-34转换成对应的牌
    if 1 <= i <= 9:
        return i
    elif 10 <= i <= 18:
        return i + 1
    elif 19 <= i <= 27:
        return i + 2
    elif 28 <= i <= 34:
        return i + 3
    else:
        logging.info('tool2/translate:Error !')


def card_preprocess_dqn(state):  # 特征工程，预处理入入据据
    handCards2 = list10_to_16(state.handcards)
    actions2 = []
    for i in state.fulu:
        actions2.append(list10_to_16(i))

    feature_noking = feature_extract.calculate_noking_sys_2_dqn(handCards2, actions2)  # 察察值  action是副露
    feature_noking.append(35)
    testData = np.array(feature_noking)
    # data input
    x_data = np.array(testData[0:-1])
    sign = 1
    test_data = x_data
    return test_data


# 特征编码
def card_preprocess(state):  # 特征工程，预处理入入据据

    if isinstance(state, np.ndarray):
        print("hello")
        print("state.handcards", state.handcards, state)
    handCards2 = list10_to_16(state.handcards)
    actions2 = []
    for i in state.fulu:
        actions2.append(list10_to_16(i))

    feature_noking = feature_extract.calculate_noking_sys_2(handCards2, actions2)  # 察察值
    feature_noking.append(35)
    testData = np.array(feature_noking)
    # data input
    x_data = np.array(testData[0:-1])
    sign = 1
    test_data = x_data
    return test_data


def index_to_card(index):
    if 0 <= index <= 8:
        return index + 1
    elif 9 <= index <= 17:
        return index + 2
    elif 18 <= index <= 26:
        return index + 3
    elif 27 <= index <= 33:
        return index + 4
    else:
        print("下标:", index, "转换成十进制card错误！")


def card_to_index(card):
    '''
    十进制card转换成下标
    :param card:
    :return:
    '''
    if 1 <= card <= 9:
        card -= 1
    elif 11 <= card <= 19:
        card -= 2
    elif 21 <= card <= 29:
        card -= 3
    elif 31 <= card <= 37:
        card -= 4
    else:
        print("十进制card", card, "转换错误")
    return card


def sub_is_list(l):
    for i in l:
        if isinstance(i, list):
            return False
    return True


def deepcopy(src):
    dst = []
    for sub_s in src:
        if isinstance(sub_s, list) and not sub_is_list(sub_s):
            print("sub", sub_s)
            sub_s = deepcopy(sub_s)
        dst.append(copy.copy(sub_s))
    return dst


# zengw 20.0703
def card_preprocess_sr(state):
    '''
    上饶麻将特征提取,模仿suphx
    :param state:
    :return:
    '''
    # handcard 4 * 34 表示
    if isinstance(state, np.ndarray):
        print("hello")
        print("state.handcards", state.handcards, state)

    handCards2 = list10_to_16(state.handcards)
    features = feature_extract.suphx_cards_feature_code(handCards2, 4)  # 4 * 34

    discards = list10_to_16_2(state.game.discards)
    feature_discards = feature_extract.suphx_cards_feature_code(discards, 4)  # 弃牌表 4 * 34
    features.extend(feature_discards)

    fulu = []
    for i in state.fulu:
        fulu.append(list10_to_16(i))

    return features


# zengw 20.0703

def get_card_preprocess_sr_king_param(state, seat_id=0):
    # 获取特征里的参数，模块重用
    if isinstance(state, np.ndarray):
        print("hello")
        print("state.handcards", state.players[seat_id].handcards, state)

    handCards2 = list10_to_16(state.players[seat_id].handcards)
    fulu_ = []
    for i in state.player_fulu[seat_id]:
        fulu_.append(list10_to_16(i))

    king_card = f10_to_16(state.jing_card)
    discards = list10_to_16(state.player_discards[seat_id])

    fei_king = discards.count(king_card)
    king_nums = handCards2.count(king_card)
    fei = 1 if (len(discards) and discards[-1] == king_card) else 0

    all_discards = list10_to_16(state.discards)  # 场面弃牌表信息

    round_ = state.round

    return handCards2, all_discards, fulu_, king_card, fei_king, king_nums, fei, round_


# zengw 20.1105
def card_preprocess_sr_suphx(state, search=False, global_state=False, dropout_prob=0):
    '''
    上饶麻将特征提取,模仿suphx
    :param state:
    :return:
    '''

    features = feature_extract.calculate_king_sys_suphx(state, state.seat_id, search, global_state, dropout_prob)

    features = np.array(features)
    features = features.T
    features = np.expand_dims(features, 0)
    features = features.transpose([2, 1, 0])  # 更换位置  转换成c × 34 × 1的格式

    return features


# zengw 21.97
def card_preprocess_suphx_sc(state, search=False, global_state=False, dropout_prob=0):
    '''
    四川麻将特征提取,模仿suphx
    :param state:
    :return:
    '''

    features = feature_extract.calculate_sys_suphx_sc(state, state.seat_id, search, global_state, dropout_prob)

    features = np.array(features)
    features = features.T
    features = np.expand_dims(features, 0)
    features = features.transpose([2, 1, 0])  # 更换位置  转换成c × 34 × 1的格式

    return features


# 获取花色对应标志
def splitColorFlags(cards, fulu):
    # 输入十进制，转成十六进制
    handcarAndFulu = []
    colorFlag = [0, 0, 0]

    handcarAndFulu.extend(cards)
    for fulu_ in fulu:
        handcarAndFulu.extend(fulu_)
    handcarAndFulu = list10_to_16(handcarAndFulu)
    for card in handcarAndFulu:
        if colorFlag[0] == 0 and card & 0xf0 == 0:
            colorFlag[0] = 1
        elif colorFlag[1] == 0 and card & 0xf0 == 0x10:
            colorFlag[1] = 1
        elif colorFlag[2] == 0 and card & 0xf0 == 0x20:
            colorFlag[2] = 1
    return colorFlag


# 获取花色分离后的牌
def splitColor(cards):
    color = [[], [], []]
    for card in cards:
        if card & 0xf0 == 0:
            color[0].append(card)
        elif card & 0xf0 == 0x10:
            color[1].append(card)
        elif card & 0xf0 == 0x20:
            color[2].append(card)
    return color


# 获取手中的定缺牌
def get_dingQue_cards(handcards, choose_color, Dec=True):
    # 获取定缺牌列表  十进制牌
    DingQue_cards = []
    if Dec:  # 默认十进制
        for card in handcards:
            if card // 10 == choose_color:
                DingQue_cards.append(card)
    else:  # 十六进制
        for card in handcards:
            if card // 16 == choose_color:
                DingQue_cards.append(card)
    return DingQue_cards
# 获取花色
#
# if __name__ == '__main__':
#     # 测试
#     print(splitColorFlags([1,2,3,12,13,14,15,15],[[5,5,5],[9,9,9,9]]))
#     pass

# index_list = [index for index in range(170 + 1709)]
# del_indexs = random.sample(index_list, int(1879 * 1))
# print(del_indexs, len(del_indexs), len(set(del_indexs)))
# from mahEnv import  MahjongEnv
#
# env = MahjongEnv()
# for i_episode in range(10):
#
#     # 获取回合 i_episode 第一个 observation
#     observation = env.reset()
#     print("i_episode:", i_episode, "   observation:(手牌：", observation.handcards, "  副露：", observation.fulu,
#           "精牌：", observation.jing_card, "   抓牌：", observation.catch_card)
#
#     while True:
#         action = observation.handcards[0]  # 选行为
#         print("action:", action)
#         observation, reward, done, info = env.step(action)  # 获取下一个 state
#         print("reward:", reward, "   observation:(手牌：", observation.handcards, "  副露：", observation.fulu,
#               "精牌：", observation.jing_card, "   抓牌：", observation.catch_card)
#         # obs_ = card_preprocess(observation)
#         obs = card_preprocess_sr_king(observation)
#         obs_ = get_reshape_obs_(obs, 34, 34)
#         if done:
#             print(env.game.win_result)
#             break

# result = [[1,2,3], [4,5,6], [7,8,9]]
# result1 = [[3,4,5],[6,7,8],[9,10,11]]
# result.extend(result1)
# print(result)
# L = set([1,2,3,4,4,3,2,1])
# for i in L:
#     print(i)
