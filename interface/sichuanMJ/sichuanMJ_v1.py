#!/usr/bin/python2.7
# -*- coding: utf-8 -*-
# @Time    : 2020/12/24 17:02
# @Author  : Ren
# @Email   : 1543042088@qq.com
# @File    : shangraoMJ_v5_3.py
# @Software: PyCharm

# -*- coding:utf-8 -*-
# cython: language_level=2
# python 2.0　两整数相处会自动取整，需要人为给被除数添加float型

import copy
import time
from interface.sichuanMJ import lib_MJ_v1 as MJ
import logging
# import opp_srmj as DFM  # 对手建模
import itertools
import datetime

# 需要调参的位置： line 40  418  443  1336     # 碰牌和自摸牌的比重，番型倍数，有效牌路径计算，向听数的扩展度（*如果变成 +2 +更多，就可以达到看得更远

# 日志输出
logger = logging.getLogger("sichuanMJ_log_v1")
logger.setLevel(level=logging.DEBUG)
time_now = datetime.datetime.now()
handler = logging.FileHandler("log/log_v1_%i%i%i.txt" % (time_now.year, time_now.month, time_now.day))
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.info("sichuanMJ_v1 compile finished...")

# global variable
TIME_START = time.time()
w_type = 0  # lib_MJ的权重选择
ROUND = 0  # 轮数
t3Set = MJ.get_t3info()
t2Set, t2Efc, efc_t2index = MJ.get_t2info()
REMAIN_NUM = 108  # 剩余牌数    136-》108
KING = None  # 宝牌
fei_king = 0  # 飞宝数

mul = 2  # 番型放大倍数
# xtspp=2     #平胡向听数的扩展数;;这里也可以根据剩余牌来调整，前期=2（牌墙40-60  中期=1（20-40 后期=0（0-20
w_bb = 1  # 组成对子的权重，只有自摸所以=1

T_SELFMO = [0] * 34  # 自摸概率表，牌存在于牌墙中的概率表
LEFT_NUM = [0] * 34  # 未出现的牌的数量表
RT1 = [[0] * 34, [0] * 34]  # 其他玩家的状态表[table1,table2] ，不需要的牌table1与需要的牌table2，
RT2 = [[0] * 34, [0] * 34]  # table1:计算吃碰的概率，table2：计算危险度
RT3 = [[0] * 34, [0] * 34]

# 生成t1,t2转化为t3的状态集合，便于搜索直接使用
# t1tot2_dict = MJ.t1tot2_info()
t1tot3_dict = MJ.t1tot3_info()  # t1转化为t3
t2tot3_dict = MJ.t2tot3_info()  # t2转化为t3


class SwitchTiles:
    def __init__(self, hand, n=3):
        """

        :param hand: 手牌
        :param n: 换牌张数，默认为换3张
        """
        self.hand = hand
        self.type = n
        self.color = MJ.splitColor(hand)
        # print(self.color)

    def judge_cs_value(self, cs):

        """
        输入一个cs(包含杠牌)，返回其最大组合评估值,不包含手牌基础分
        """
        value = 0
        for gang in cs[0]:  # 杠牌
            value += 50
        for aaa in cs[1]:  # kezi
            if MJ.c_num(aaa[0]) in [1, 2, 8, 9]:
                value += 25
            else:
                value += 18
        for abc in cs[2]:
            value += 10
        for aa in cs[3]:
            if MJ.c_num(aa[0]) in [1, 2, 8, 9]:
                value += 6
            else:
                value += 5
        for ab in cs[4]:  # 搭子这里需要分4次判断
            if ab[0] + 1 == ab[1]:
                if MJ.c_num(ab[0]) in [1, 8]:  # 12,89
                    value -= 1
                else:
                    value += 4
            else:
                if MJ.c_num(ab[0]) in [3, 4, 5]:  # 35、46、57
                    value += 2
                else:
                    value += 1
        for a in cs[-1]:  # 孤张
            if MJ.c_num(a) in [1, 9]:
                value -= 2
                print(11111)
            if MJ.c_num(a) in [2, 8]:
                value -= 1
        return value

    # def choose_color_index(self):
    #     len_color = [len(self.color[0]), len(self.color[1]), len(self.color[2])]
    #     # 闲家手牌为13张
    #     # [[6, 3], [7, 1], [7, 2], [7, 3], [8, 0], [8, 1], [8, 2], [9, 0], [9, 1], [9, 2], [10, 0], [10, 1], [11, 0],
    #     # [11, 1], [12, 0], [13, 0]]
    #     max_len = max(len_color)
    #     min_len = min(len_color)
    #
    #     if len(self.hand) == 13:
    #         index = self.t13.index([max_len, min_len])
    #
    #
    #         color_n = self.s13[index]
    #         color_index = MJ.get_index(len_color, color_n)
    #         return color_index
    #
    #     elif len(self.hand) == 14:
    #         index = self.t14.index([max_len, min_len])
    #         color_n = self.s14[index]
    #         color_index = MJ.get_index(len_color, color_n)
    #         return color_index
    #
    #     else:
    #         print('SwitchTiles, choose_color ERROR! len(self.hand)=13 or 14, but =' + len(self.hand))
    #         return -1

    def choose_color_xiao(self):
        """
        换三张开始前选择大于3张的最拉胯花色
        """
        len_color = [len(self.color[0]), len(self.color[1]), len(self.color[2])]
        color_3 = []
        for i in range(3):  # 获取大于三的花色
            if len_color[i] >= 3:
                color_3.append(i)
        min_value = 1000
        min_value_index = -1
        for i in color_3:
            color_cards = self.color[i]
            all_cs = MJ.tree_expand_gang(color_cards)
            one_max = 0  # 记载每种花色的最大值
            for cs in all_cs:
                value_cs = MJ.judge_cs_value(cs) + len_color[i] * 10
                # print(i,cs,value_cs)
                one_max = max(one_max, value_cs)
            # print(one_max)
            if one_max < min_value:
                min_value = one_max
                min_value_index = i
        return min_value_index

    def choose_color_final(self):
        """
        换三张开始前选择大于3张的最拉胯花色
        """
        len_color = [len(self.color[0]), len(self.color[1]), len(self.color[2])]
        color_3 = []
        for i in range(3):  # 加入所有花色
            color_3.append(i)
        min_value = 1000
        min_value_index = -1
        for i in color_3:
            color_cards = self.color[i]
            all_cs = MJ.tree_expand_gang(color_cards)
            one_max = 0  # 记载每种花色的最大值
            for cs in all_cs:
                value_cs = MJ.judge_cs_value(cs) + len_color[i] * 10
                # print(i,cs,value_cs)
                one_max = max(one_max, value_cs)
            # print(one_max)
            if one_max < min_value:
                min_value = one_max
                min_value_index = i
        return min_value_index

    def choose_3card(self):
        """
        从选择好的最拉胯花色选择3张最拉胯的牌
        """
        choose_color = self.choose_color_xiao()
        choose_c_cards = self.color[choose_color]
        gap = -1000
        for cards_3 in itertools.combinations(choose_c_cards, 3):  # 从所选择的花色中取出3个
            cards_3 = list(cards_3)
            cards_other = list(choose_c_cards)
            for card in cards_3:
                cards_other.remove(card)
            cards_3_cs = MJ.tree_expand_gang(cards_3)
            one_max3 = 0  # 记载组合的最大值
            for cs in cards_3_cs:
                value_cs = MJ.judge_cs_value(cs) + 30
                if value_cs >= one_max3:
                    one_max3 = value_cs
                    max_cs_3 = cs  # 最大权值对应组合
            # print(one_max3, max_cs_3)
            one_max_o = 0
            cards_other_cs = MJ.tree_expand_gang(cards_other)
            for cs in cards_other_cs:
                value_cs = MJ.judge_cs_value(cs) + 10 * len(cards_other)
                if value_cs >= one_max_o:
                    one_max_o = value_cs
                    max_cs_o = cs
            # print(one_max_o, max_cs_o)

            gap_tmp = one_max_o - one_max3
            # print('gap',gap_tmp)
            if gap_tmp > gap:
                gap = gap_tmp
                choose_3cards = cards_3
        # print('-----------------------------------f_gap',gap)
        return choose_3cards

    def choose_n_cards(self, cards=[]):
        cc = []
        cv = [0] * len(cards)
        for i in range(len(cards)):
            card = cards[i]
            cv[i] = MJ.assess_card(card=card)

        for j in range(self.type):
            index = cv.index(min(cv))
            cc.append(cards[index])
            cv[index] = 0xff
        return cc

    def switch_cards(self):
        color_index = self.choose_color_index()
        if len(color_index) == 1:
            cmb = MJ.tree_expand(cards=self.color[color_index[0]])
            while (len(cmb[0][-1]) < self.type):
                # todo 搭子需排序
                if cmb[0][3] != []:
                    cmb[0][-1].extend(cmb[0][3][-1])
                    cmb[0][3].pop()
                elif cmb[0][2] != []:
                    cmb[0][-1].extend(cmb[0][2][-1])
                    cmb[0][2].pop()
                elif cmb[0][1] != []:
                    cmb[0][-1].extend(cmb[0][1][-1])
                    cmb[0][1].pop()
                else:
                    cmb[0][-1].extend(cmb[0][0][-1])
                    cmb[0][0].pop()

            switch_cards = self.choose_n_cards(cmb[0][-1])
            return switch_cards
        # 当有多种花色需要选择时，对此进行评估
        value = [-1, -1, -1]  # 花色的评估值
        cc = [[], [], []]  # 花色的待选牌
        for i in color_index:
            cmb = MJ.tree_expand(cards=self.color[i])
            cmb0 = cmb[0]
            v = 100 * len(cmb0[0] + cmb0[1]) + 30 * len(cmb0[2]) + 10 * len(cmb0[3])
            for card in cmb0[-1]:
                v += MJ.assess_card(card)
            value[i] = v
            cc[i] = self.choose_n_cards(cards=self.color[i])
        # 选择最小的花色
        for j in range(3):
            if value[j] == -1:
                value[j] = 0xff
        min_color_index = value.index(min(value))
        return cc[min_color_index]

    def choose_color(self):
        # 选花色中牌数量最少的，如果最少的和第二少的相差为1，则进行比较
        len_color = [len(self.color[0]), len(self.color[1]), len(self.color[2])]
        min_color = min(len_color)
        index_dq = []
        for i in range(3):
            if len_color[i] == min_color or len_color[i] - 1 == min_color:
                index_dq.append(i)

        value = [0xff, 0xff, 0xff]
        # cc = [[], [], []]
        for i in index_dq:
            cmb = MJ.tree_expand(cards=self.color[i])
            cmb0 = cmb[0]
            v = 100 * len(cmb0[0] + cmb0[1]) + 30 * len(cmb0[2]) + 10 * len(cmb0[3])
            for card in cmb0[-1]:
                v += MJ.assess_card(card)
            value[i] = v
            # cc[i] = self.choose_n_cards(n=3, cards=self.color[i])
        # 选择最小的花色
        min_color_index = value.index(min(value))
        return min_color_index


class Node_PH:
    def __init__(self, take=None, AAA=[], ABC=[], jiang=[], T2=[], T1=[], raw=[], taking_set=[], taking_set_w=[],
                 king_num=0,
                 fei_king=0, baohuanyuan=False):
        self.take = take  # 缺的牌？？
        self.AAA = AAA  # 刻子
        self.ABC = ABC  # 顺子
        self.jiang = jiang  # 将牌
        self.T2 = T2  # T2组合
        self.T1 = T1  # T1组合
        self.raw = raw  # 待扩展集合
        self.king_num = king_num  # 宝牌数量
        self.fei_king = fei_king  # 飞宝数量
        self.children = []  # 孩子节点
        self.taking_set = taking_set  # 缺失牌
        self.baohuanyuan = baohuanyuan  # 宝还原
        self.taking_set_w = taking_set_w  # 数量，补齐类型？？权重

    def add_child(self, child):
        self.children.append(child)

    def node_info(self):
        print(self.AAA, self.ABC, self.jiang, "T1=", self.T1, "T2=", self.T2, self.raw, self.taking_set, self.king_num,
              self.fei_king, self.baohuanyuan)


class Node_SSL:
    def __init__(self, take=None, taking_set=[], wan=[], tiao=[], tong=[], zi=[], T1=[], raw=[]):
        self.wan = wan
        self.tiao = tiao
        self.tong = tong
        self.zi = zi
        self.T1 = T1
        self.take = take
        self.raw = raw
        self.taking_set = taking_set
        self.children = []

    def add_child(self, node):
        self.children.append(node)

    def node_info(self):
        print(self.wan, self.tiao, self.tong, self.zi, "take ", self.take, "taking_set ", self.taking_set)


class SearchTree_PH():
    """
    平胡搜索模块
    """

    def __init__(self, hand, suits, combination_sets, king_card=None, fei_king=0):
        """
        类变量初始化
        :param hand: 手牌
        :param suits: 副露
        :param combination_sets: 拆分组合集合
        :param king_card: 宝牌
        :param fei_king: 飞宝数
        """
        self.hand = hand
        self.suits = suits
        self.combination_sets = combination_sets
        self.xts = combination_sets[0][-2]
        self.tree_dict = []
        self.king_card = king_card
        self.fei_king = fei_king

        if king_card != None:
            self.king_num = hand.count(king_card)
        else:
            self.king_num = 0
        self.discard_score = {}  # 出牌集合的评估值集合
        self.discard_state = {}  # 出牌集合的状态集合
        self.node_num = 0  # 统计节点数目（观测值）
        self.chang_num = 0  # 统计状态不同但分数相同的节点 （观测值）

    def expand_node(self, node):
        """
        节点扩展.首先扩展将牌，再扩展t3:先扩展t2->t3,再t1->t3,使用itertools.combinations生成待扩展集合可以有效减少重复计算量
        :param node:
        :return: None
        """
        # node.node_info()
        # 先定将
        if node.jiang == []:  # 没有将牌
            has_jiang = False
            if node.king_num >= 2:  # 宝还原   2  3  4
                has_jiang = True  # 有宝还原时不再搜索无将情况
                child = Node_PH(take=-1, AAA=node.AAA, ABC=node.ABC, jiang=[self.king_card, self.king_card],
                                T2=node.T2,
                                T1=node.T1,
                                taking_set=node.taking_set, taking_set_w=node.taking_set_w,
                                king_num=node.king_num - 2,  # 消耗2张宝牌
                                fei_king=node.fei_king, baohuanyuan=node.baohuanyuan)
                node.add_child(child=child)
                self.expand_node(child)

            if node.king_num > 0:  # 宝吊，其他12张牌组4个T3，剩下一张宝打任意胡  1  2  3  4
                has_jiang = True  # 宝吊不再搜索无将
                taking_set = copy.copy(node.taking_set)
                taking_set.append(0)  # 填充0---------已选定牌的ID
                taking_set_w = copy.copy(node.taking_set_w)
                taking_set_w.append(w_bb)  # -----------已选定牌的权重，为一，表示只能自摸获得
                child = Node_PH(take=0, AAA=node.AAA, ABC=node.ABC, jiang=[0, 0], T2=node.T2,
                                T1=node.T1,
                                taking_set=taking_set, taking_set_w=taking_set_w, king_num=node.king_num - 1,  # 消耗1张宝牌
                                fei_king=node.fei_king, baohuanyuan=False)
                node.add_child(child=child)
                self.expand_node(child)

            if node.king_num <= 1:  # 0  1
                for t2 in node.T2:  # T2组合有将牌
                    T2 = MJ.deepcopy(node.T2)
                    # 从t2中找到对子作为将牌
                    if t2[0] == t2[1]:
                        has_jiang = True  # 有将不再搜索无将
                        T2.remove(t2)
                        child = Node_PH(take=-1, AAA=node.AAA, ABC=node.ABC, jiang=t2, T2=T2,
                                        T1=node.T1,
                                        taking_set=node.taking_set, taking_set_w=node.taking_set_w,
                                        king_num=node.king_num,
                                        fei_king=node.fei_king, baohuanyuan=False)  # 非宝吊宝还原
                        node.add_child(child=child)
                        self.expand_node(node=child)

            # 无将的情况
            if not has_jiang:  # 经历了上面3个if还没有将   todo 这里可以考虑有将时也扩展
                jiangs = copy.copy(node.T1)  # 复制T1，尝试用T1扩展成将牌
                # todo 可以在有T1时也扩展该部分
                if jiangs == []:  # 没有T1,则选择一个t2来扩展
                    for t2 in node.T2:  # t2: 2张牌的搭子
                        jiangs = t2
                        T2 = MJ.deepcopy(node.T2)
                        T2.remove(t2)
                        for t1 in jiangs:  # jiangs总共2张，依次挑一张
                            taking_set = copy.copy(node.taking_set)
                            taking_set.append(t1)
                            taking_set_w = copy.copy(node.taking_set_w)
                            taking_set_w.append(w_bb)
                            T1 = copy.copy(jiangs)
                            T1.remove(t1)  # 搭子剩余的那张牌，退回成T1
                            child = Node_PH(take=t1, AAA=node.AAA, ABC=node.ABC, jiang=[t1, t1], T2=T2,
                                            T1=T1,
                                            taking_set=taking_set, taking_set_w=taking_set_w, king_num=node.king_num,
                                            fei_king=node.fei_king, baohuanyuan=False)
                            node.add_child(child=child)
                            self.expand_node(node=child)
                # 从T1中选择一张作为将
                else:  # 存在孤张，使用孤张扩展成对搭
                    for t1 in jiangs:
                        if t1 == -1:  # op填充的-1不作扩展
                            continue
                        taking_set = copy.copy(node.taking_set)
                        taking_set.append(t1)
                        taking_set_w = copy.copy(node.taking_set_w)
                        taking_set_w.append(w_bb)  # 要更改。。
                        T1 = copy.copy(jiangs)
                        T1.remove(t1)
                        child = Node_PH(take=t1, AAA=node.AAA, ABC=node.ABC, jiang=[t1, t1], T2=node.T2,
                                        T1=T1,
                                        taking_set=taking_set, taking_set_w=taking_set_w, king_num=node.king_num,
                                        fei_king=node.fei_king, baohuanyuan=False)
                        node.add_child(child=child)
                        self.expand_node(node=child)

        # 胡牌判断，此时有将牌
        elif len(node.AAA) + len(node.ABC) == 4:
            if node.king_num > 0:  # 扩展结束后，还有多的宝牌
                node.fei_king += node.king_num  # 多余的宝牌没使用，作为弃牌飞掉
                node.king_num = 0
                if node.baohuanyuan and node.fei_king == self.king_num + self.fei_king:  # 宝牌全部飞完了，所以就不是宝还原了
                    node.baohuanyuan = False  # 原来是宝还原，把宝飞掉，胡另一边 eg:123->23(4)
            return

        # T3扩展，此时有将牌
        else:
            # 这里分为是否有3组t3
            # 若有，进行全扩展（可剪枝
            # 若无，使用原来逻辑，（但是要在组合的时候-1

            # 当待扩展集合不为空时，使用该集合进行扩展
            if node.raw != []:
                tn = node.raw[-1]  # 取最后一个待扩展
                raw = copy.copy(node.raw)  # 深度搜索后面的节点会改变raw，回退可能导致前面的节点raw不正确，这里需要copy
                raw.pop()
                if type(tn) == list:  # 使用t2扩展t3
                    t2 = tn  # 某个T2
                    for item in t2tot3_dict[str(t2)]:  # item:即info: [[待补t2],[补齐后的t3],[原t2剩余牌],补齐牌,补齐类型]
                        if item[1][0] == item[1][1]:  # 刻子
                            AAA = MJ.deepcopy(node.AAA)  # 复制刻子集合
                            AAA.append(item[1])  # 刻子集合加1
                            ABC = node.ABC  # 复制顺子集合
                        else:  # 顺子
                            AAA = node.AAA
                            ABC = MJ.deepcopy(node.ABC)
                            ABC.append(item[1])
                        if node.king_num > 0 and item[-2] == self.king_card:  # 宝还原，用宝的原ID补齐
                            child = Node_PH(take=-1, AAA=AAA, ABC=ABC, jiang=node.jiang, T2=node.T2,
                                            T1=node.T1, raw=raw, taking_set=node.taking_set,
                                            taking_set_w=node.taking_set_w,
                                            king_num=node.king_num - 1,
                                            fei_king=node.fei_king, baohuanyuan=node.baohuanyuan)
                            node.add_child(child=child)
                            self.expand_node(node=child)

                        elif node.king_num > 0 and (0 in node.jiang):  # 宝牌补一张,jiang=[0,0]，宝吊
                            child = Node_PH(take=0, AAA=AAA, ABC=ABC, jiang=node.jiang, T2=node.T2,
                                            T1=node.T1, raw=raw, taking_set=node.taking_set,
                                            taking_set_w=node.taking_set_w,
                                            king_num=node.king_num - 1,
                                            fei_king=node.fei_king, baohuanyuan=False)
                            node.add_child(child=child)
                            self.expand_node(node=child)
                        else:  # 正常打法
                            taking_set = copy.copy(node.taking_set)
                            taking_set_w = copy.copy(node.taking_set_w)
                            taking_set.append(item[-2])  # 补齐牌
                            taking_set_w.append(item[-1])  # 补齐类型 aa2 ab6
                            child = Node_PH(take=item[-2], AAA=AAA, ABC=ABC, jiang=node.jiang, T2=node.T2,
                                            T1=node.T1, raw=raw, taking_set=taking_set, taking_set_w=taking_set_w,
                                            king_num=node.king_num,
                                            fei_king=node.fei_king, baohuanyuan=node.baohuanyuan)
                            node.add_child(child=child)
                            self.expand_node(node=child)
                # t1扩展为t3
                elif type(tn) == int:  # 单张牌，使用t1 扩展 t3
                    t1 = tn
                    for item in t1tot3_dict[str(t1)]:  # 同理 item:info:[[扩展成的t3],[缺失的2张牌],[1，补齐类型]]
                        flag2 = False
                        if node.king_num > 0:  # 用于处理宝还原
                            for card in item[1]:  # item[1]:缺失的2张牌
                                if card == self.king_card:
                                    flag2 = True  # 宝还原标识
                                    raw_copy = copy.copy(raw)
                                    raw_copy.append(sorted([card, t1]))  # 此时待扩展的变成t2
                                    child = Node_PH(take=-1, AAA=node.AAA, ABC=node.ABC, jiang=node.jiang, T2=node.T2,
                                                    T1=node.T1, raw=raw_copy,
                                                    taking_set=node.taking_set, taking_set_w=node.taking_set_w,
                                                    king_num=node.king_num - 1, fei_king=node.fei_king,
                                                    baohuanyuan=node.baohuanyuan)
                                    node.add_child(child=child)
                                    self.expand_node(node=child)
                        if flag2:  # 上述宝还原后不再继续扩展
                            continue

                        if item[0][0] == item[0][1]:  # 扩展成刻子
                            AAA = MJ.deepcopy(node.AAA)
                            AAA.append(item[0])
                            ABC = node.ABC
                        else:  # 扩展成顺子
                            AAA = node.AAA
                            ABC = MJ.deepcopy(node.ABC)
                            ABC.append(item[0])

                        if node.king_num >= 2:  # 宝牌有2张以上，直接补2张，即使其中有一张被作为宝还原也不影响
                            child = Node_PH(take=[0, 0], AAA=AAA, ABC=ABC, jiang=node.jiang, T2=node.T2, T1=node.T1,
                                            raw=raw,
                                            taking_set=node.taking_set, taking_set_w=node.taking_set_w,
                                            king_num=node.king_num - 2, fei_king=node.fei_king,
                                            baohuanyuan=False)
                            node.add_child(child=child)
                            self.expand_node(node=child)

                        elif node.king_num == 0:  # 宝数量为0 的处理
                            take = item[1]  # take = 缺失的牌
                            take_w = item[-1]  # 补齐类型

                            taking_set = copy.copy(node.taking_set)
                            taking_set.extend(take)
                            taking_set_w = copy.copy(node.taking_set_w)
                            taking_set_w.extend(take_w)
                            child = Node_PH(take=take, AAA=AAA, ABC=ABC, jiang=node.jiang, T2=node.T2, T1=node.T1,
                                            raw=raw,
                                            taking_set=taking_set, taking_set_w=taking_set_w,
                                            king_num=node.king_num, fei_king=node.fei_king,
                                            baohuanyuan=node.baohuanyuan)
                            node.add_child(child=child)
                            self.expand_node(node=child)

                        elif node.king_num == 1:  # king_num=1 ,补一张牌
                            # 用1张宝牌
                            for i in range(len(item[1])):
                                card = item[1][i]  # item：[[扩展成的t3],[缺失的2张牌],[1，补齐类型]]
                                take = [0, card]  # 0是宝牌补缺，card是正常缺的

                                taking_set = copy.copy(node.taking_set)
                                taking_set.append(card)
                                taking_set_w = copy.copy(node.taking_set_w)
                                taking_set_w.append(w_bb)  # 要更改

                                child = Node_PH(take=take, AAA=AAA, ABC=ABC, jiang=node.jiang, T2=node.T2, T1=node.T1,
                                                raw=raw,
                                                taking_set=taking_set, taking_set_w=taking_set_w,
                                                king_num=node.king_num - 1, fei_king=node.fei_king,
                                                baohuanyuan=False)
                                node.add_child(child=child)
                                self.expand_node(node=child)
                        else:  # 发生错误！！宝牌0 1 2+ 情况都if了
                            logger.error("node.king_num==%s", (node.king_num))
                else:  # 错误！！待扩展集合为list int 都if了
                    logger.error("tn Error")
            # 当待扩展集合为空时
            else:
                if node.T2 != []:  # 1、先扩展T2为T3
                    t2_sets = node.T2
                    T2 = copy.copy(node.T2)
                    # 生成待扩展集合                       从t2_sets中选min(差几个T3，当前T2数量)个出来 的全组合
                    for t2_set in itertools.combinations(t2_sets, min(4 - len(node.AAA) - len(node.ABC), len(t2_sets))):
                        node.T2 = copy.copy(T2)
                        node.raw = list(t2_set)  # 待扩展加入t2_set
                        for t2 in node.raw:
                            node.T2.remove(t2)  # T2移除t2
                        self.expand_node(node=node)
                #  生成T1扩展T3集合
                elif node.T1 != []:
                    t1_sets = copy.copy(node.T1)
                    # 这里移除了填充的-1，不作扩展
                    if -1 in t1_sets:
                        t1_sets.remove(-1)
                    T1 = copy.copy(node.T1)
                    for t1_set in itertools.combinations(t1_sets, min(4 - len(node.AAA) - len(node.ABC), len(t1_sets))):
                        node.T1 = copy.copy(T1)
                        node.raw = list(t1_set)
                        for t1 in node.raw:
                            node.T1.remove(t1)
                        self.expand_node(node=node)

    def generate_tree(self):
        """
        生成树
        :return: None
        """
        kz = []
        sz = []
        # 将副露加入到节点的AAA和ABC状态中
        for t3 in self.suits:
            if t3[0] == t3[1]:
                kz.append(t3)
            else:
                sz.append(t3)
        # 使用拆分组合生成树
        for cs in self.combination_sets:
            root = Node_PH(take=None, AAA=cs[0] + kz, ABC=cs[1] + sz, jiang=[], T2=cs[2] + cs[3], T1=cs[-1],
                           taking_set=[], taking_set_w=[], king_num=self.king_num,
                           fei_king=self.fei_king, baohuanyuan=self.king_num > 0)  # 简单的传参，初始化类
            # 每一棵树都存储到树集合中
            self.tree_dict.append(root)
            self.expand_node(node=root)  # 扩展树

    def cal_chance(self, node, index_hu):
        """
        节点评估值计算模块-概率子模块
        :param node:
        :param index_hu:有效牌被选作胡牌的id,相同的有效牌只会选中一次
        :return: float 路径概率
        """
        value = 1
        if node.taking_set_w != []:
            # 上饶麻将中胡牌需要自摸，获取权重为1。这里将具有最小获取权重的牌的权重置为1.是一种权重最大化的处理.todo 可以尝试其他的处理
            node.taking_set_w[node.taking_set_w.index(min(node.taking_set_w))] = 1

            bal = 0
            for i in range(len(node.taking_set)):  # 遍历缺失牌
                if i == index_hu:  # 跳过某一张，作为胡牌
                    continue
                card = node.taking_set[i]  # 取一张缺失牌
                bal += 1
                if card == 0:  # 宝吊的任意牌,获取概率为1
                    taking_rate = 1.0  # 这张缺失牌是宝吊任意胡，获取概率为1
                    print("bug，出现宝牌，---------------------！！！")
                else:  # 其他牌的获取概率计算
                    # taking  = 自摸 * bal ； 因为缺2张牌的时候，这2张牌来的顺序可以有先后*2  eg:3张牌 *6  其他的就按照真实倍率写就行
                    taking_rate = T_SELFMO[MJ.convert_hex2index(card)] * bal  # 自摸概率表，开始的时候，多走一步
                value *= taking_rate * node.taking_set_w[i]  # todo 需要结合其他玩家打出这张牌的概率来计算，将获取权重具体化

        # 摸牌概率修正，当一张牌被重复获取时，T_selfmo修改为当前数量占未出现牌数量的比例
        taking_set = list(set(node.taking_set))  # 缺失牌集合
        taking_set_num = [node.taking_set.count(i) for i in taking_set]  # 缺失牌数量
        for i in range(len(taking_set_num)):
            n = taking_set_num[i]
            j = 0
            while n > 1:
                j += 1
                index = MJ.convert_hex2index(taking_set[i])  # 返回 0-33 序数
                if LEFT_NUM[
                    index] >= n:  # value * 3/4 * 2/4  eg:需要3张 value=4/136 * 4/136 * 4/136 * 3/4 * 2/4 修正概率为 4/136 3/136 2/136
                    value *= float(LEFT_NUM[index] - j) / LEFT_NUM[index]
                else:  # 摸牌数超过了剩余数，直接舍弃；；需要缺失牌的数量<可能获取的数量
                    value = 0
                    return value, 0
                n -= 1

        # 修正数用量误差，如果被选择的胡牌在路径中大于1 要修正。如果等于1 不修改
        # 增加返回，可胡牌的剩余数量

        # print("n=",index_hu)
        # print("len=",len(node.taking_set))
        hu_need = node.taking_set.count(node.taking_set[index_hu])  # 胡牌被路径需要几张
        hu_left = LEFT_NUM[MJ.convert_hex2index(node.taking_set[index_hu])]  # 胡牌剩余几张
        num = hu_left - hu_need + 1  # 可胡张数

        # 如果路径中需求数量 大于1，即一些是路径，一些是胡牌。这修正路径概率
        if hu_need > 1:
            value *= float(hu_left) / num

        return value, num

    def cal_score(self, node):
        """
        节点评估值计算模块-分数子模块
        :param node:
        :return: float 分数
        """

        # fan计算
        fan = Fan_PH(kz=node.AAA, sz=node.ABC, jiang=node.jiang, fei_king=node.fei_king,
                     using_king=self.king_num + self.fei_king - node.fei_king,
                     baohuanyuan=node.baohuanyuan).fanDetect()  # -------？？这样算的吗？

        # 单吊翻倍（在四川麻将，叫金钓钓 4
        if len(self.suits) == 4:
            fan *= 4

        return fan

    def calculate_path_expectation(self, node):
        """
        计算整条路径的上的评估值，并将其赋予为所有出牌的评估值
        :param node:
        :return:
        """
        # 深度搜索。搜索胡牌的叶子节点 todo 这里不变，依然搜索到胡牌节点，然后进行反选 n-1 + 1
        if len(node.AAA) + len(node.ABC) == 4 and node.jiang != []:
            self.node_num += 1  # 可胡牌节点  +1
            discard_set = []  # 出牌集合

            # 将没有使用的T2加入到出牌中
            for t2 in node.T2:
                discard_set.extend(t2)
            discard_set.extend(node.T1)  # 多余的牌
            taking_set_sorted = sorted(node.taking_set)  # 缺失的牌
            if discard_set != []:
                # 这里计算的是 一条完整的路径的分数，没有进行-1
                # 大家的分是一样的，但是路径会不一样，
                # 将cal_score的功能分离，分别算 路径概率(cal_chance)和分数(cal_score)
                score = self.cal_score(node=node)  # 放到外面统一计算，减少耗时
            else:
                return

            # todo 这里拆解，for
            taking_set_ = list(set(node.taking_set))
            for i in range(len(taking_set_)):
                # 这个是路径概率+可胡张数，少一长度的
                chance, hu_num = self.cal_chance(node=node, index_hu=node.taking_set.index(taking_set_[i]))
                taking_set_sorted2 = copy.deepcopy(taking_set_sorted)
                taking_set_sorted2.remove(taking_set_[i])
                taking_set_lable = str(taking_set_sorted2)

                for card in list(set(discard_set)):
                    if card not in self.discard_state.keys():  # 弃牌不在 状态里
                        self.discard_state[card] = {}
                    if taking_set_lable not in self.discard_state[card].keys():  # 有效牌集不在 状态[弃牌]里
                        self.discard_state[card][taking_set_lable] = [[], []]
                    if taking_set_[i] not in self.discard_state[card][taking_set_lable][0]:
                        self.discard_state[card][taking_set_lable][0].append(taking_set_[i])  # 加入
                        self.discard_state[card][taking_set_lable][1].append(chance * score * hu_num)
                    else:
                        index = self.discard_state[card][taking_set_lable][0].index(taking_set_[i])
                        if chance * score * hu_num > self.discard_state[card][taking_set_lable][1][index]:
                            self.discard_state[card][taking_set_lable][1][index] = chance * score * hu_num

            '''
            taking_set_lable = str(taking_set_sorted)  # 转化为str可以加快查找

            # todo 这种按摸牌的评估方式是否唯一准确
            # todo 在上面进行 n-1 + 1的拆分，这里discard_state[card] = [[n-1], [1], [score]]
            # discard_state[弃牌] = {有效牌集-1}
            # discard_state[弃牌][有效牌集-1]=[[可胡牌],[对应分数]]
            for card in list(set(discard_set)):  # 遍历多余的牌
                if card not in self.discard_state.keys():  # 如果牌不在弃牌状态集dict
                    self.discard_state[card] = [[], []]  # 加入当前弃牌 dict[弃牌]=[[缺失牌集1，缺失牌集2],[缺失牌1评分，缺失牌2评分]]
                if taking_set_lable not in self.discard_state[card][0]:  # 缺失牌不在状态集
                    self.discard_state[card][0].append(taking_set_lable)  # 加入牌
                    self.discard_state[card][-1].append(score)  # 加入分数
                else:  # 缺失牌在状态集
                    index = self.discard_state[card][0].index(taking_set_lable)  # 返回当前缺失牌的序号
                    if score > self.discard_state[card][-1][index]:  # 如果这个分数更高（分数：这一串有效牌的评分）
                        self.chang_num += 1
                        self.discard_state[card][-1][index] = score  # 替换分数？？这里这里用加法，是不是刚好可以满足路径发散度的计算
                        # print("分数替换")
            '''
        # 当前节点不能胡牌，继续搜索其 子节点
        elif node.children != []:
            for child in node.children:
                self.calculate_path_expectation(node=child)

    def get_discard_score(self):
        """
        总接口。获取出牌的评估值
        :return: dict. 出牌的评估值集合
        """
        # t1 = time.time()
        self.generate_tree()  # 生成树，将组合好的手牌，用生成树的方式计算所有扩展组合
        # t2 = time.time()
        for root in self.tree_dict:  # 扩展树的过程集合
            self.calculate_path_expectation(root)  # 会更新 状态集： state[弃牌]=[[缺失牌集],[缺失牌分数集]]
        # t3 = time.time()
        # print ("tree time:",t2 - t1, "value time:",t3 - t2)
        state_num = 0

        # todo 这里的
        # discard_state={}
        # discard_state[弃牌]={}
        # discard_state[弃牌][有效牌集]=[[可胡牌],[可胡牌全胡的分数]]

        for discard in self.discard_state.keys():
            if discard not in self.discard_score.keys():
                self.discard_score[discard] = 0
            all_score = 0
            for take in self.discard_state[discard]:
                all_score += sum(self.discard_state[discard][take][-1])
            self.discard_score[discard] = all_score

        # recommend_card = max(self.discard_score, key=lambda x: self.discard_score[x])
        # print("当前弃牌：",recommend_card)
        # for take in self.discard_state[recommend_card]:
        #     print("\n对应的有效牌：", take)
        #     print("对应的可胡牌(*)：", self.discard_state[recommend_card][take][0])
        #     print("对应的可胡张数：",end="")
        #     for card in self.discard_state[recommend_card][take][0]:
        #         print(LEFT_NUM[MJ.convert_hex2index(card)]," ",end="")
        #     print("\n对应的可能得分：", self.discard_state[recommend_card][take][-1])
        #
        # for discard in self.discard_state.keys():
        #     print("\n弃牌：",discard)
        #     for take in self.discard_state[discard]:
        #         print("对应的有效牌：",take)
        #         print("对应的可胡牌(*)：",self.discard_state[discard][take][0])

        '''
        for discard in self.discard_state.keys():
            if discard not in self.discard_score.keys():  # 如果不在分数集里，先添加一个key
                self.discard_score[discard] = 0
            self.discard_score[discard] = sum(self.discard_state[discard][-1])  # 分数累加
            # print("分支数量：",len(self.discard_state[discard][-1]))
            state_num += len(self.discard_state[discard][-1])
            '''

        # print ("discard_state", self.discard_state)
        # print ("discard_score", self.discard_score)
        # print ("leaf node ", self.node_num)
        # print ("state_num", state_num)
        # print ("chang_num", self.chang_num)

        return self.discard_score


class ShiSanLan:
    """
    十三烂类
    """

    def __init__(self, cards, suits, king_card, fei_king, padding=[]):
        """
        类变量初始化
        :param cards:  手牌
        :param suits: 副露
        :param king_card:  宝牌
        :param fei_king: 飞宝数
        :param padding: 填充牌，op操作前，填充-1，使手牌达到14张
        """
        self.cards = cards
        self.suits = suits
        self.king_card = king_card
        self.discard_state = {}
        self.discard_score = {}
        self.tree_list = []
        self.fei_king = fei_king
        self.padding = padding
        # 具有单花色的3张的状态集合
        self.ssl_three_table = [[1, 6, 9],  # 6
                                [1, 4, 9],  # 3
                                [1, 4, 7],  # 1
                                [1, 4, 8],  # 2
                                [1, 5, 9],  # 5
                                [1, 5, 8],  # 4
                                [2, 5, 9],  # 8
                                [2, 5, 8],  # 7
                                [3, 6, 9],  # 10
                                [2, 6, 9]]  # 9

        # 2张，经过了筛选，不含【1，2】这种
        self.ssl_two_table = [[1, 4], [1, 5], [1, 6], [1, 7], [1, 8], [1, 9],
                              [2, 5], [2, 6], [2, 7], [2, 8], [2, 9],
                              [3, 6], [3, 7], [3, 8], [3, 9],
                              [4, 7], [4, 8], [4, 9],
                              [5, 8], [5, 9],
                              [6, 9]]
        # 2张的有效牌集合  #与上面对应
        self.ssl_two_efc = [[[7], [8], [9]], [[8], [9]], [[9]], [[4]], [[4], [5]], [[4], [5], [6]],
                            [[8], [9]], [[9]], [[]], [[5]], [[5], [6]],
                            [[9]], [[]], [[]], [[6]],
                            [[1]], [[1]], [[1]],
                            [[1]], [[1]],
                            [[1], [2], [3]]
                            ]

        # 一张    #隐含了重要性排名
        self.ssl_one_table = [[1], [9], [2], [8], [3], [7], [4], [6], [5]]
        # 一张的有效牌集合
        self.ssl_one_efc = [[[4, 7], [4, 8], [4, 9], [5, 8], [5, 9], [6, 9]],  # 1
                            [[1, 4], [1, 5], [1, 6], [2, 5], [2, 6], [3, 6]],  # 9
                            [[5, 8], [5, 9], [6, 9], [7], [8]],  # 2
                            [[1, 4], [1, 5], [2, 5], [3]],  # 8
                            [[6, 9], [7], [8]],  # 3
                            [[1, 4], [2], [3]],  # 7
                            [[1, 7], [8], [9]],  # 4
                            [[1], [2], [3, 9]],  # 6
                            [[1, 8], [1, 9], [2, 8], [2, 9]]  # 5
                            ]
        # 0张的有效牌集合
        self.ssl_zero_efc = [[1, 6, 9],
                             [1, 4, 9],
                             [1, 4, 7],
                             [1, 4, 8],
                             [1, 5, 9],
                             [1, 5, 8],
                             [2, 5, 9],
                             [2, 5, 8],
                             [3, 6, 9],
                             [2, 6, 9],
                             [2, 7], [3, 7], [3, 8]]

    def add_color(self, list, color):
        """
        给移除花色的烂牌添加花色
        :param list: 烂牌
        :param color: 花色:0x00,0x10,0x20
        :return: list. 添加花色后的牌
        """
        return [i + color for i in list]

    def color_info(self, cards, color):
        """
        计算单花色的相关信息, 包含烂牌和移除烂牌后的无用牌
        :param cards: 单花色的手牌（需要已经移除花色）  万条筒的序号(1234
        :param color: 花色                        万0x00  条0x10  筒ox20
        :return: list[[]].[ssl_cards, T1],ssl_cards:烂牌,T1:抽取烂牌后的牌
        """
        CSs = []
        tiles = list(set(cards))
        # 计算单花色的有用牌的最大成组数量
        # Max1可以=0 1 2 3     等于2时可以通过补齐变成3
        waitnumMax1 = max((tiles.count(1) + tiles.count(4) + tiles.count(7)),
                          (tiles.count(1) + tiles.count(4) + tiles.count(8)),
                          (tiles.count(1) + tiles.count(4) + tiles.count(9)),
                          (tiles.count(1) + tiles.count(5) + tiles.count(8)),
                          (tiles.count(1) + tiles.count(5) + tiles.count(9)),
                          (tiles.count(1) + tiles.count(6) + tiles.count(9)),
                          (tiles.count(2) + tiles.count(5) + tiles.count(8)),
                          (tiles.count(2) + tiles.count(5) + tiles.count(9)),
                          (tiles.count(2) + tiles.count(6) + tiles.count(9)),
                          (tiles.count(3) + tiles.count(6) + tiles.count(9)))
        # Max2=0 1 2     此时不能通过补齐变成3
        waitnumMax2 = max((tiles.count(2) + tiles.count(7)),
                          (tiles.count(3) + tiles.count(7)),
                          (tiles.count(3) + tiles.count(8)), )

        if max(waitnumMax1, waitnumMax2) == 3:  # 当有用牌数量为3 的时候 直接返回3，无有效牌(缺失牌)
            for tb in self.ssl_three_table:  # 遍历 3表，对于每一种可以存在的组合，都加入CSs
                if tb[0] in cards and tb[1] in cards and tb[2] in cards:  # 三个数同时存在
                    tmp = copy.copy(cards)  # 复制手牌
                    tmp.remove(tb[0])  # 移除这三张牌
                    tmp.remove(tb[1])
                    tmp.remove(tb[2])
                    CSs.append([self.add_color(tb, color), self.add_color(tmp, color)])  # [确定留下的牌，剩下的牌]
        elif max(waitnumMax1, waitnumMax2) == 2:  # 当有用牌为2 的时候 返回向听数；同上
            for i in range(len(self.ssl_two_table)):
                tb = self.ssl_two_table[i]
                if tb[0] in cards and tb[1] in cards:
                    tmp = copy.copy(cards)
                    tmp.remove(tb[0])
                    tmp.remove(tb[1])
                    CSs.append([self.add_color(tb, color), self.add_color(tmp, color)])
        elif max(waitnumMax1, waitnumMax2) == 1:  # 当有用牌只有1 的时候
            for card in range(1, 10):
                if card in tiles:
                    tmp = copy.copy(cards)
                    tmp.remove(card)
                    CSs.append([[card + color], self.add_color(tmp, color)])

        else:  # 此时没有这种花色的牌
            CSs.append([[], []])
        return CSs

    def ssl_CS(self):
        """
        计算十三烂的拆分组合
        :return: []，返回拆分结果 [[wan],[tiao],[tong],[zi],[left],xts]
        """
        CSs = []
        if self.suits != []:  # 进行过 吃碰  则不可能组成十三烂
            return [[[], [], 14]]
        # 花色分离
        wan, tiao, tong, zi = MJ.split_type_s(self.cards)
        wan_CS = self.color_info(wan, 0)  # wan_CS[0]:    [[十三烂牌],[剩余牌]]
        tiao_CS = self.color_info([i & 0x0f for i in tiao], 0x10)
        tong_CS = self.color_info([i & 0x0f for i in tong], 0x20)
        # zi
        zi_ssl = list(set(zi))  # 字牌的set集，就是十三烂集
        zi_T1 = copy.copy(zi)
        for card in zi_ssl:
            zi_T1.remove(card)
            # zi_efc.remove(card)
        zi_CS = [[zi_ssl, zi_T1]]  # zi_CS只有 [0]，此时的[[[zi_ssl],[zi_T1]]]是为了统一格式
        for cs_wan in wan_CS:  # cs_wan:[[十三烂牌],[剩余牌]]
            for cs_tiao in tiao_CS:
                for cs_tong in tong_CS:
                    for cs_zi in zi_CS:
                        xts = 14 - len(cs_wan[0]) - len(cs_tiao[0]) - len(cs_tong[0]) - len(cs_zi[0])
                        CSs.append([cs_wan[0], cs_tiao[0], cs_tong[0], cs_zi[0],
                                    cs_wan[-1] + cs_tiao[-1] + cs_tong[-1] + cs_zi[-1], xts])
        CSs.sort(key=lambda k: k[-1], reverse=False)  # 这里用true??  应该小->大 ，其实不差
        return CSs

    def efc_ssl(self, cards, type):
        """
        计算ssl的有效牌
        :param cards: 单花色的有效牌
        :param type: 花色
        :return: [],合理的最大化有效牌组，例如[1,9] 的 有效牌返回[[4],[5],[6]]
        """
        # print("cards:",cards)  #经过了筛选 的。。只会含table里有的
        cards_cp = [i & 0x0f for i in cards]  # 序号化
        # print("cards_cp",cards_cp)
        if type <= 0x20:  # 万条筒
            if len(cards_cp) == 0:
                efc = self.ssl_zero_efc  # 直接返回3位、2位的组合，因为当前花色为0张
            elif len(cards_cp) == 1:
                efc = self.ssl_one_efc[self.ssl_one_table.index(cards_cp)]  # 返回one_table[5]的序号，再用序号去定位efc
            elif len(cards_cp) == 2:
                efc = self.ssl_two_efc[self.ssl_two_table.index(cards_cp)]
            elif len(cards_cp) == 3:
                efc = [[]]
            efc_c = []
            for s in efc:  # 2020.12.28 bug解决，这里没有花色还原
                efc_c.append([i + type for i in s])
            efc = efc_c
        else:
            efc = [[]]
            for card in range(0x31, 0x38):
                if card not in cards:
                    efc[0].append(card)
        return efc

    def expand_node(self, node):
        """
        ssl搜索节点扩展，首先会生成所有可能的摸牌组合，对摸牌组合进行节点的扩展
        :param node: 待扩展的节点
        :return: None
        """
        # 胡牌判断
        if len(node.wan) + len(node.tiao) + len(node.tong) + len(node.zi) == 14:
            # node.node_info()
            return
        else:
            # 与平胡一样。待扩展集合是否为空，不为空直接进行扩展，否则生成该组合
            if node.raw != []:  # 因为是深度优先，每个node的raw都是当前组合的缺失牌
                raw = copy.copy(node.raw)  # raw=[1，4，7]  每个raw给与手牌都可以胡十三烂
                card = raw[-1]  # 取最后一个元素
                raw.pop()  # 方便这里pop
                type = card & 0xf0
                taking_set = copy.copy(node.taking_set)
                taking_set.append(card)  # 这里的take不是缺失牌吗？--是缺失牌，因为这里的raw提前转置成了缺失牌集合
                if type == 0x00:
                    wan = copy.copy(node.wan)
                    wan.append(card)  # 更新 万牌
                    child = Node_SSL(take=card, taking_set=taking_set, wan=wan, tiao=node.tiao, tong=node.tong,
                                     zi=node.zi, T1=node.T1, raw=raw)
                elif type == 0x10:
                    tiao = copy.copy(node.tiao)
                    tiao.append(card)
                    child = Node_SSL(take=card, taking_set=taking_set, wan=node.wan, tiao=tiao, tong=node.tong,
                                     zi=node.zi, T1=node.T1, raw=raw)
                elif type == 0x20:
                    tong = copy.copy(node.tong)
                    tong.append(card)
                    child = Node_SSL(take=card, taking_set=taking_set, wan=node.wan, tiao=node.tiao, tong=tong,
                                     zi=node.zi, T1=node.T1, raw=raw)
                elif type == 0x30:
                    zi = copy.copy(node.zi)
                    zi.append(card)
                    child = Node_SSL(take=card, taking_set=taking_set, wan=node.wan, tiao=node.tiao, tong=node.tong,
                                     zi=zi, T1=node.T1, raw=raw)
                node.add_child(child)
                self.expand_node(node=child)
            else:  # 生成raw
                # 对每种花色进行有效牌组合的计算，然后生成待扩展的集合
                for wan_efc in self.efc_ssl(node.wan, 0):  # 因为这里的node.wan是筛选过的，必定会在table里
                    for tiao_efc in self.efc_ssl(node.tiao, 0x10):
                        for tong_efc in self.efc_ssl(node.tong, 0x20):
                            for zi_efc in self.efc_ssl(node.zi, 0x30):
                                efcs = wan_efc + tiao_efc + tong_efc + zi_efc
                                xts = 14 - len(node.wan) - len(node.tiao) - len(node.tong) - len(node.zi)
                                for efc in itertools.combinations(efcs, xts):  # 从efcs取xts个元素出来
                                    node.raw = list(efc)
                                    self.expand_node(node)

    def generate_tree(self):
        """
        生成树
        :return:  None，结果保留在类变量中tree_list
        """
        CSs = self.ssl_CS()  # [万，条，筒，字，剩余牌，向听数]  这里的万是筛选过的。。
        CSs.sort(key=lambda k: k[-1], reverse=False)  # 又倒排！！
        # print("CSs:",CSs)
        # 取xts最小的一组
        min_xts = CSs[0][-1]
        CSs_min_xts = []
        for cs in CSs:
            if cs[-1] == min_xts:
                CSs_min_xts.append(cs)

        for cs in CSs_min_xts:
            node = Node_SSL(take=None, taking_set=[], wan=cs[0], tiao=cs[1], tong=cs[2], zi=cs[3], T1=cs[4], raw=[])
            self.tree_list.append(node)
            self.expand_node(node=node)

    def cal_score(self, node):
        """
            计算节点的评估值
        :param node: 节点
        :return: 评估值
        """

        value = 1
        # print node.taking_set
        for card in node.taking_set:  # 遍历缺失集
            value *= T_SELFMO[MJ.convert_hex2index(card)]  # 乘以自摸概率表
        # fan检测
        fan = 8  # 十三烂不存在番型
        # 飞宝
        fei_king = self.fei_king + node.T1.count(self.king_card)
        fan *= 2 ** fei_king
        # 七星
        if len(node.zi) == 7:
            fan *= 2
        score = value * fan
        return score

    def evaluate(self, node):
        """
        胡牌后的节点的评估值计算
        :param node:
        :return:
        """
        if node.children == []:  # 叶子节点，可以进行计算的节点
            if len(node.wan) + len(node.tiao) + len(node.tong) + len(node.zi) == 14:  # 有必要  2 2 2 7=13不够14
                score = self.cal_score(node)  # 计算这个节点的分数---now
                taking_set_sorted = sorted(node.taking_set)  # 需要的牌
                discards = node.T1 + self.padding  # 弃牌
                for discard in discards:
                    if discard not in self.discard_state.keys():  # 当前弃牌不在状态集里
                        self.discard_state[discard] = [[], []]  # 状态集添加此key
                        self.discard_state[discard][0].append(taking_set_sorted)  # [0] ：缺失牌
                        self.discard_state[discard][-1].append(score)  # [-1]：评估值
                    elif taking_set_sorted not in self.discard_state[discard][0]:  # 当前弃牌在状态集里，但这个弃牌组不在
                        self.discard_state[discard][0].append(taking_set_sorted)  # 添加行新的弃牌组，不用加key这里
                        self.discard_state[discard][-1].append(score)  # 评估值
        else:
            for child in node.children:
                self.evaluate(node=child)

    def get_discard_score(self):
        """
        对外总接口，生成所有合理出牌的评估值
        :return:
        """
        self.generate_tree()  # 生成树，每个节点都有当前牌self，和缺失牌raw，和假设已有牌take
        # 已经生成完所有的树
        for tree in self.tree_list:
            self.evaluate(node=tree)  # 计算路径评估值==以下同理
        for discard in self.discard_state.keys():
            if discard not in self.discard_score.keys():
                self.discard_score[discard] = sum(self.discard_state[discard][-1])
        return self.discard_score


class JiuYao:
    def __init__(self, cards, suits, king_card, fei_king, padding=[]):
        """
        九幺类变量初始化
        :param cards: 手牌
        :param suits: 副露
        :param king_card: 宝牌
        :param fei_king: 飞宝数
        :param padding: 填充，op操作时填充为-1
        """
        self.cards = cards
        self.suits = suits
        self.king_card = king_card
        self.fei_king = fei_king
        self.padding = padding
        self.discard_score = {}
        self.discard_state = {}
        self.yaojiu = [1, 9, 0x11, 0x19, 0x21, 0x29, 0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37]

    def yaojiu_CS(self):
        """
        生成九幺的拆分组合
        :return:
        """
        # 判断是否是91
        CS = [[], [], 14]
        flag = True
        for suit in self.suits:  # 如果副露就已经不满足九幺，立刻返回不可能
            for card in suit:
                if card not in self.yaojiu:
                    flag = False
        if not flag:
            return CS
        CS[-1] -= len(self.suits) * 3
        for card in self.cards:
            if card in self.yaojiu:
                CS[0].append(card)
            else:
                CS[1].append(card)
        CS[-1] -= len(CS[0])
        return CS

    def get_discard_score(self):
        """
        计算九幺的所有出牌的评估值
        :return: {card:score}，
        """
        CS = self.yaojiu_CS()
        if CS[-1] != 14:  # ！=14代表有可能胡
            value = 1
            yaojiu_take = 0
            n = 0
            for card in self.yaojiu:
                if CS[0].count(card) > 2 and CS[-1] != 1:  # 如果已有3 4 张，向听数又>1 todo 待完善
                    w = 6  # 应该是考虑3+1杠牌？？
                else:
                    w = 1
                n += LEFT_NUM[MJ.convert_hex2index(card)]  # n+= 这张牌可能的剩余数
                yaojiu_take += T_SELFMO[MJ.convert_hex2index(card)] * w  # todo 重复摸牌的处理，不能同理平胡处理

            value *= yaojiu_take ** CS[-1]  # 如果向听数=3 ，==yaojiu_take^3

            xt = CS[-1]
            j = 0
            while xt > 1:
                j += 1
                value *= float(n - j) / n  # 重复摸牌的处理
                xt -= 1
                n -= 1
            # fan计算
            fan = 4
            fei_king = self.fei_king + CS[1].count(self.king_card)
            fan *= 2 ** fei_king
            if len(self.suits) == 4:  # 单吊(金勾勾
                fan *= 2
            # 有七星91吗？有？TODO
            score = value * fan
            discards = CS[1] + self.padding
            for discard in discards:
                if discard not in self.discard_score:
                    self.discard_score[discard] = score
        return self.discard_score


class Node_Qidui:
    def __init__(self, take=None, AA=[], T1=[], raw=[], taking_set=[], king_num=0):
        """
        七对节点变量初始化
        :param take: 摸牌
        :param AA: 对子集合
        :param T1: 单张牌集合
        :param raw: 待扩展集合
        :param taking_set: 已摸牌集合
        :param king_num: 未使用的宝数量
        """
        self.take = take
        self.AA = AA
        self.T1 = T1
        self.raw = raw
        self.taking_set = taking_set
        self.king_num = king_num
        self.children = []

    def add_child(self, child):
        self.children.append(child)

    def node_info(self):
        print(self.AA, self.T1, self.raw, self.taking_set, self.king_num)


class Qidui:
    def __init__(self, cards, suits, king_card, fei_king, padding=[]):
        """
        七对类变量初始化
        :param cards: 手牌
        :param suits: 副露
        :param king_card: 宝牌
        :param fei_king: 飞宝数量
        :param padding: 填充牌，op操作时填充-1 ，一般来说，七对不会有这种操作
        """
        self.cards = cards
        self.suits = suits
        self.king_card = king_card
        self.fei_king = fei_king
        self.discard_score = {}
        self.king_num = cards.count(king_card)
        self.padding = padding
        self.tree_list = []
        self.discard_state = {}

    def qidui_CS(self):
        """
        计算七对组合的生成
        :return:
        """
        CS = [[], [], 14]
        if self.suits != []:  # 进行过碰吃，直接不可能七对
            return CS
        cards_rm_king = copy.copy(self.cards)
        # for i in range(self.king_num):
        #     cards_rm_king.remove(self.king_card)    #移除宝牌
        for card in list(set(cards_rm_king)):
            n = cards_rm_king.count(card)  # n=牌数
            if n == 1:
                CS[1].append(card)  # 1孤张
            elif n == 2:
                CS[0].append([card, card])  # 一对
            elif n == 3:
                CS[0].append([card, card])  # 一对+1张
                CS[1].append(card)
            elif n == 4:
                CS[0].append([card, card])  # 2对
                CS[0].append([card, card])
        # king_num = self.king_num
        # 这里把宝用掉
        # while king_num > 0:
        #     if len(CS[0]) + king_num > 7:   #对子数量+宝牌数量>7    6+2  or  5+3  or  4+ 4  反正宝牌>=2
        #         CS[0].append([self.king_card, self.king_card])
        #         king_num -= 2       #对子、宝牌数量够，做宝还原
        #     else:       #对子数量不够，做宝牌任意对
        #         CS[0].append([0, 0])
        #         king_num -= 1
        CS[-1] -= len(CS[0]) * 2 + (7 - len(CS[0]))  # 向听数= 7-已有对子数量
        # CS[-1]+=2  # todo 这里给七对的xt+2，减少后面选择打七对的概率
        if CS[-1] >= 4:  # todo  如果对子的数量过少，不建议打七对
            CS[-1] += 3
        if CS[-1] < 0:
            CS[-1] = 0
        return CS

    def expand_node(self, node):
        """
        节点扩展
        :param node:
        :return:
        """
        # 与平胡类似，先生成待扩展集合，再进行节点扩展
        if len(node.AA) == 7:
            return
        else:
            if node.raw != []:  # 这里的raw是待扩展，即没有被选上作为 对子or半对子的  T1牌
                # for card in node.raw:
                card = node.raw[-1]
                node.raw.pop()
                AA = copy.copy(node.AA)
                AA.append([card, card])
                taking_set = copy.copy(node.taking_set)
                taking_set.append(card)
                child = Node_Qidui(take=card, AA=AA, T1=node.T1, raw=node.raw, taking_set=taking_set,
                                   king_num=node.king_num)
                node.add_child(child=child)
                self.expand_node(node=child)
            else:
                if node.T1 != []:
                    t1_sets = copy.copy(node.T1)  # T1复制
                    # if -1 in t1_sets:
                    #     t1_sets.remove(-1)
                    T1 = copy.copy(node.T1)
                    # 从孤张选 xts张牌出来作为备选对子
                    for t1_set in itertools.combinations(t1_sets, min(7 - len(node.AA), len(t1_sets))):
                        node.T1 = copy.copy(T1)
                        node.raw = list(t1_set)
                        for t1 in node.raw:
                            node.T1.remove(t1)
                        self.expand_node(node=node)

    def generate_tree(self):
        """
        生成树
        :return:
        """
        CS = self.qidui_CS()
        # print "qidui CS",CS
        node = Node_Qidui(take=None, AA=CS[0], T1=CS[1], taking_set=[], king_num=self.king_num)
        self.tree_list.append(node)
        self.expand_node(node=node)

    def fan(self, node):
        """
        七对番型
        :param node:
        """
        # fei_king = self.fei_king + node.T1.count(self.king_card)
        # if self.king_num == 0 or fei_king == self.fei_king + self.king_num: #无宝或宝还原  (总之就是没使用癞子权限
        #     fan = 16        #这么特殊。？
        # else:
        #     fan = 12        #刚好可以契合 番数？
        # fan *= 2 ** fei_king
        # # 91
        # jiuyao = [1, 9, 0x11, 0x19, 0x21, 0x29, 0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37]   #九幺七对
        # ziyise = [0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37]                                 #字一色七对
        # flag_jiuyao = True
        # for t2 in node.AA:
        #     if t2[0] not in jiuyao:
        #         flag_jiuyao = False
        #         break
        # flag_ziyise = True
        # for t2 in node.AA:
        #     if t2[0] not in ziyise:
        #         flag_ziyise = False
        #         break
        # if flag_jiuyao:
        #     fan *= 2
        # if flag_ziyise:
        #     fan *= 2
        # return fan

        fan = 4  # 七对的基础倍率是 4 倍 （平胡为1

        # 附加倍率有  龙七对2 断一九2 一九牌4 清一色4
        yijiu = [1, 9, 0x11, 0x19, 0x21, 0x29]  # 一九牌 4   断一九 2

        flag_yijiu = True
        for t2 in node.AA:
            if t2[0] not in yijiu:
                flag_yijiu = False
                break

        flag_duanyijiu = True
        for t2 in node.AA:
            if t2[0] in yijiu:
                flag_duanyijiu = False

        flag_qinyise = True
        yise = node.AA[0][0] & 0xf0
        for t2 in node.AA:
            if t2[0] & 0xf0 != yise:
                flag_qinyise = False

        take = []
        longqidui = 0
        for t2 in node.AA:
            take.append(t2[0])
        for t in take:
            if take.count(t) == 2:
                longqidui += 1

        if flag_yijiu:
            fan *= 4
        if flag_duanyijiu:
            fan *= 2
        if flag_qinyise:
            fan *= 4
        fan *= 2 ** longqidui

        return fan

    def evaluate(self, node):
        """
        节点评估值计算
        :param node:
        :return:
        """
        if node.children == []:
            if len(node.AA) == 7:  # 这里一般是会通过的
                # node.node_info()
                taking_set_sorted = sorted(node.taking_set)  # 缺失牌
                value = 1
                for card in taking_set_sorted:
                    # print "card",card
                    if card == -1:  # -1代表 填充牌
                        print("bug,出现填充牌-------------！")
                        value = 1.0 / 34
                    else:
                        value *= T_SELFMO[MJ.convert_hex2index(card)]
                fan = self.fan(node=node)  # --=

                score = value * fan  # 概率*番  （这里的概率分母都是一样的，分子就是当前剩余牌数，所以是等价的。。可以
                discards = node.T1 + self.padding
                for discard in discards:
                    if discard not in self.discard_state.keys():
                        self.discard_state[discard] = [[], []]
                        self.discard_state[discard][0].append(taking_set_sorted)
                        self.discard_state[discard][-1].append(score)
                    elif taking_set_sorted not in self.discard_state[discard][0]:
                        self.discard_state[discard][0].append(taking_set_sorted)
                        self.discard_state[discard][-1].append(score)  # 这里精髓，可以让当前弃牌产生的价值，全部汇聚于这里
        else:
            for child in node.children:
                self.evaluate(child)

    def get_discard_score(self):
        """
        生成所有合理出牌的评估值
        :return: {card:score}
        """
        # t1 = time.time()
        self.generate_tree()  # 从T1选不同的牌 尝试组成对子
        # t2 = time.time()
        for tree in self.tree_list:
            self.evaluate(tree)  # --
        # t3=time.time()
        # print ("qidui time",t2-t1,t3-t2)
        for discard in self.discard_state.keys():
            if discard not in self.discard_score:
                self.discard_score[discard] = 0
            self.discard_score[discard] = sum(self.discard_state[discard][-1])
        return self.discard_score


'''
番数计算类
'''


class Fan_PH():
    def __init__(self, kz, sz, jiang, fei_king=0, using_king=0, baohuanyuan=False):
        """
        初始化类变量
        :param kz: 刻子
        :param sz: 顺子
        :param jiang: 将
        :param node: 待检测的结点
        :param fei_king: 飞宝数
        """
        self.kz = kz
        self.sz = sz
        self.jiang = jiang
        self.fei_king = fei_king
        self.using_king = using_king
        self.baohuanyuan = baohuanyuan
        # self.mul = 2

    # 碰碰胡
    def pengPengHu(self):
        """
        碰碰胡检测
        是否刻子树数达到４个
        :return: bool
        """
        if len(self.kz) == 4:
            # if self.usingKing==0:
            return True
        else:
            return False

    # 宝还原 x2
    # def baoHuanYuan(self):
    #
    #     if self.baohuanyuan:
    #         return True
    #     else:
    #         return False

    # 清一色 x2
    def qingYiSe(self):
        """
        清一色检测
        手牌为同一花色
        :return: bool
        """
        # todo 宝吊无法检测清一色，因为将牌无法确定
        w = 0
        ti = 0
        to = 0
        z = 0
        # print self.kz + self.sz+ self.jiang
        for t in self.kz + self.sz + [self.jiang]:
            card = t[0]
            if card != 0:
                if card & 0xf0 == 0x00:
                    w = 1
                elif card & 0xf0 == 0x10:
                    ti = 1
                elif card & 0xf0 == 0x20:
                    to = 1
                else:
                    return False

        if w + ti + to <= 1:
            return True
        else:
            return False

    # 一九牌 4
    def yiJiuPai(self):
        yijiu = [1, 9, 0x11, 0x19, 0x21, 0x29]  # 一九牌 4   断一九 2

        flag_yijiu = True
        for T in self.kz + self.sz + [self.jiang]:
            for t in T:
                flag_yijiu = False
                if t in yijiu:
                    flag_yijiu = True
                if (flag_yijiu == False):
                    return False
        return True

    # 断一九 2
    def duanYiJiu(self):
        yijiu = [1, 9, 0x11, 0x19, 0x21, 0x29]  # 一九牌 4   断一九 2

        flag_duanyijiu = True
        for T in self.kz + self.sz + [self.jiang]:
            for t in T:
                if t in yijiu:
                    flag_duanyijiu = False

        return flag_duanyijiu

    def fanDetect(self):
        """
        番数计算
        基础分４分，通过调用上述的番种检测来增加基础分
        :return: int 番数
        """
        # # 基础分判定
        # score = 4
        # if self.pengPengHu():
        #     # print "0"
        #     score *= self.mul       # mul=2
        #     if self.using_king == 0 or self.baohuanyuan:    #不使用宝，或者宝还原
        #         score *= self.mul   # mul=2
        #     score *= 2  # 碰碰胡再给2倍分
        #
        # # 翻倍机制
        # # 飞宝 当可以宝吊时，将飞宝倍数得到提高
        # # if 0 in self.jiang:
        # #     for i in range(self.fei_king):
        # #         score *= 2.5
        # # else:
        # for i in range(self.fei_king):          #飞宝2
        #     # print "1"
        #     score *= self.mul
        #
        # # # 宝还原　x2
        # if self.baohuanyuan:                    #宝还原2
        #     # print score, self.baohuanyuan,self.jiang,
        #     # print "2"
        #     score *= self.mul
        #
        # # 清一色
        # if self.qingYiSe():                     #清一色2
        #     score *= self.mul
        #     # print "3"
        # # 单吊　x2
        # # if len
        # # 这里无法处理，宝吊需要吃碰杠吃碰杠处理
        # # if score>16: #得分大于16时，分数评估提高
        # #     score*=1.5
        # # print
        # return score

        fan = 1  # 平胡基础分 1
        # 碰碰胡，断一九，一九牌， 金钓钓（需要副露信息，在外面判定）， 清一色  5种
        # mul=1

        # 碰碰胡 2
        if self.pengPengHu():
            fan *= 2

        # 清一色 4
        if self.qingYiSe():
            fan *= 4

        # 一九牌 4
        if self.yiJiuPai():
            fan *= 4

        # 断一九 2
        if self.duanYiJiu():
            fan *= 2

        # 金勾勾，杠牌 -未写

        return fan


'''
平胡类，相关处理方法
分为手牌拆分模块sys_info，评估cost,出牌决策，吃碰杠决策等部分
'''


class PingHu:
    '''
    平胡类模块
    '''

    def __init__(self, cards, suits, kingCard=None, fei_king=0, padding=[], round=0):
        """
        类变量初始化
        :param cards: 手牌　
        :param suits:副露
        :param leftNum:剩余牌数量列表
        :param discards:弃牌
        :param discards_real:实际弃牌
        :param discardsOp:场面副露
        :param round:轮数
        :param remainNum:牌墙剩余牌数量
        :param seat_id:座位号
        :param kingCard:宝牌
        :param fei_king:飞宝数
        :param padding: 填充牌。用于计算：op缺一张牌时，填充-1。
        :param op_card:动作操作牌
        """
        cards.sort()
        self.cards = cards
        self.suits = suits
        self.kingCard = kingCard
        self.fei_king = fei_king
        self.padding = padding
        self.kingNum = cards.count(kingCard)
        self.round = round

    @staticmethod
    def split_type_s(cards=[]):
        """
        功能：手牌花色分离，将手牌分离成 万 条 筒  字各色后输出
        :param cards: 手牌　[]
        :return: 万,条,筒,字　[],[],[],[]
        """
        cards_wan = []
        cards_tiao = []
        cards_tong = []
        cards_zi = []
        for card in cards:
            if card & 0xF0 == 0x00:  # 如果16进制只是为了在这里好算，那10进制也可以
                cards_wan.append(card)
            elif card & 0xF0 == 0x10:
                cards_tiao.append(card)
            elif card & 0xF0 == 0x20:
                cards_tong.append(card)
            elif card & 0xF0 == 0x30:
                cards_zi.append(card)
        return cards_wan, cards_tiao, cards_tong, cards_zi

    @staticmethod
    def get_effective_cards(dz_set=[]):
        """
        获取有效牌
        :param dz_set: 搭子集合 list [[]]   只有连续搭和中间搭
        :return: 有效牌 list []
        """
        effective_cards = []
        for dz in dz_set:
            if len(dz) == 1:  # 孤张组将牌
                effective_cards.append(dz[0])
            elif dz[1] == dz[0]:  # 对搭
                effective_cards.append(dz[0])
            elif dz[1] == dz[0] + 1:  # 连续搭
                if int(dz[0]) & 0x0F == 1:  # 12类型
                    effective_cards.append(dz[0] + 2)
                elif int(dz[0]) & 0x0F == 8:  # 89类型
                    effective_cards.append((dz[0] - 1))
                else:
                    effective_cards.append(dz[0] - 1)
                    effective_cards.append(dz[0] + 2)
            elif dz[1] == dz[0] + 2:  # 中间搭
                effective_cards.append(dz[0] + 1)
        effective_cards = set(effective_cards)  # set 和list的区别？ set 不能重复，天然的去重
        return list(effective_cards)

    # 判断３２Ｎ是否存在于ｃａｒｄｓ中
    @staticmethod
    def in_cards(t32=[], cards=[]):
        """
        判断３２Ｎ是否存在于ｃａｒｄｓ中
        :param t32: 某一种 3N或2N组合牌
        :param cards: 本次判断的手牌
        :return: bool
        """
        for card in t32:
            if card not in cards:
                return False
        return True

    @staticmethod
    def get_32N(cards=[]):
        """
        功能：计算所有存在的手牌的３Ｎ与２Ｎ的集合，例如[3,4,5]　，将得到[[3,4],[3,5],[4,5],[3,4,5]]
        思路：为减少计算量，对长度在12张以上的单花色的手牌，当存在顺子时，不再计算搭子;但存在刻子时，不再计算对搭
        #等我手写的时候，考虑是否可以 T3 no T2T1    T2 no T1
        :param cards: 手牌　[]
        :return: 3N与2N的集合　[[]]
        """
        cards.sort()
        kz = []
        sz = []
        aa = []
        ab = []
        ac = []
        lastCard = 0
        # 对长度在12张以上的单花色的手牌，当存在顺子时，不再计算搭子（连续搭和中间搭）
        # if True:
        if len(cards) >= 12:
            for card in cards:
                if card == lastCard:  # 如果现在判定的牌和上一张一样，就跳过
                    continue
                else:
                    lastCard = card  # 记录刚刚判定过的牌

                if cards.count(card) >= 3:  # 刻子和对搭  二者取一 （刻子优先,,因为将牌的原因，不选择这样
                    kz.append([card, card, card])
                if cards.count(card) >= 2:  # 对搭
                    aa.append([card, card])

                if card + 1 in cards and card + 2 in cards:  # 顺子和 连续搭中间搭  二者取一 （顺子优先
                    sz.append([card, card + 1, card + 2])
                else:
                    if card + 1 in cards:
                        ab.append([card, card + 1])
                    if card + 2 in cards:
                        ac.append([card, card + 2])
        else:
            for card in cards:
                if card == lastCard:
                    continue
                else:
                    lastCard = card

                # 同一花色少于12张，载入所有可能的 小组合; 防止另一种花色只有一组刻子，还拆不开
                if cards.count(card) >= 3:
                    kz.append([card, card, card])
                if cards.count(card) >= 2:
                    aa.append([card, card])
                if card + 1 in cards and card + 2 in cards:
                    sz.append([card, card + 1, card + 2])  # 因为做了4种牌的分离，所以可以这样写
                if card + 1 in cards:
                    ab.append([card, card + 1])
                if card + 2 in cards:
                    ac.append([card, card + 2])
        return kz + sz + aa + ab + ac  # 这里没有做移除，而是返回所有可能的 T3 T2

    def extract_32N(self, cards=[], t32_branch=[], t32_set=[]):
        """
        功能：递归计算手牌的所有组合信息，并存储在t32_set，
        思路: 每次递归前检测是否仍然存在３２N的集合,如果没有则返回出本此计算的结果，否则在手牌中抽取该３２N，再次进行递归
        :param cards: 手牌
        :param t32_branch: 本次递归的暂存结果
        :param t32_set: 所有组合信息
        :return: 结果存在t32_set中
        """
        t32N = self.get_32N(cards=cards)  # 返回所有可能的T3T2  t32N:List[list] :kz + sz + aa + ab + ac

        if len(t32N) == 0:  # 当不存在t32N时，处理递归 的终结
            t32_set.extend(t32_branch)  # extend :追加一个数组[]  追加一个同级别的
            # t32_set.extend([cards])
            t32_set.append(0)  # append :追加一个元素             追加一个次级别的
            t32_set.extend([cards])  # entend :追加保存剩下的孤张[]

        else:  # 一般流程
            for t32 in t32N:  # 从t32N中 取出一个 t3t2 组合
                if self.in_cards(t32=t32, cards=cards):  # 这一步？？  应该是多余的，因为下面会用复制品来删除牌。。
                    cards_r = copy.copy(cards)
                    for card in t32:  # 对组合的单个牌 进行移除
                        cards_r.remove(card)  # 从复制手牌中 移除该T3/T2 组合，原手牌不影响，方便下一次移除

                    t32_branch.append(t32)  # 将这种组合暂存在branch
                    self.extract_32N(cards=cards_r, t32_branch=t32_branch, t32_set=t32_set)  # 递归，cards里的牌会逐步转移到branch中
                    if len(t32_branch) >= 1:
                        t32_branch.pop(-1)  # 移除刚刚载入的t32，因为下一次循环时，会把branch传参

    def tree_expand(self, cards):
        """
        功能：对extract_32N计算的结果进行处理同一格式，计算万条筒花色的组合信息
        思路：对t32_set的组合信息进行格式统一，分为[kz,sz,aa,ab,xts,leftCards]保存，并对划分不合理的地方进行过滤，例如将３４５划分为35,4为废牌的情况
        当牌可以组 T3 就不尝试T2T1   T2 no T1  ??不知道这样行不行
        :param cards: cards [] 万条筒其中一种花色手牌
        :return: allDeWeight　[kz,sz,aa,ab,xts,leftCards] 去除不合理划分情况的组合后的组合信息
        """
        all = []
        t32_set = []
        # t32_set的值是  [[1,2,3],[5,5,5],0,[1,6]  ,  [3,4,5],0,[]] 这样子的
        self.extract_32N(cards=cards, t32_branch=[], t32_set=t32_set)  # 将该花色的 t32集合存在t32_set里
        # print("t32_set:T3T2+孤张：", t32_set)
        # print(type(t32_set))
        # logger.info("t32_set::%s", t32_set)
        kz = []
        sz = []
        t2N = []
        aa = []
        length_t32_set = len(t32_set)
        i = 0
        # for i in range(len(t32_set)):
        while i < length_t32_set:
            t = t32_set[i]  # 取第一个，按这个顺序排的 kz + sz + aa + ab + ac + 0 + a
            flag = True  # 本次划分是否合理
            if t != 0:
                if len(t) == 3:  # 刻子 顺子
                    if t[0] == t[1]:
                        kz.append(t)  # 长度为3 且牌相同  ->刻子
                    else:
                        sz.append(t)  # print (sub)     -> 顺子

                elif len(t) == 2:  # 搭子
                    if t[1] == t[0]:
                        aa.append(t)  # 长度为2 且牌相同 -> 对搭
                    else:
                        t2N.append(t)  # ->连续搭 和 中间搭

            else:  # t==0  t3、t2和t1 的分割线  #[[1,2,3] , 0 , [5,9]]
                '修改，使计算时间缩短'
                leftCards = t32_set[i + 1]  # 剩下的孤张牌
                efc_cards = self.get_effective_cards(dz_set=t2N)  # t2N中不包含aa，只有中间搭和连续搭
                # 去除划分不合理的情况，例如345　划分为34　或35等，对于333 划分为33　和3的情况，考虑有将牌的情况暂时不做处理
                # for card in leftCards:
                #     if card in efc_cards:
                #         flag = False
                #         break
                #   我认为不需要，在前面就应 避免将这种组合加入进来

                if flag:
                    all.append([kz, sz, aa, t2N, 0, leftCards])  # 碰到0的时候，就表示只剩下孤张了，把0和孤张也组合起来，就完整了
                kz = []  # 将4种集合 重置
                sz = []
                aa = []
                t2N = []
                i += 1
            i += 1

        # print("all:",all)
        allSort = []  # 给每一个元素排序
        allDeWeight = []  # 排序去重 后

        # all：List[List[list or int]] 由[kz + sz + aa + ab + ac + 0 + a]为元素，组成的list
        for e in all:  # e:[kz + sz + aa + ab + ac + 0 + a]
            for f in e:  # f:kz + sz + aa + ab + ac + 0 + a  中的一种
                if f == 0:  # 0是xts位，int不能排序
                    continue
                else:
                    f.sort()  # 主要是对孤张排序，，搭子其实不用怎么排
            allSort.append(e)

        for a in allSort:
            if a not in allDeWeight:  # a 是手牌的一种组合方式，不加入重复的组合方式
                allDeWeight.append(a)

        allDeWeight = sorted(allDeWeight, key=lambda k: (len(k[0]), len(k[1]), len(k[2])), reverse=True)  # 居然可以这样排序！！
        # 排序方式，按照T3的数量，若T3相同 再按T2  reverse:从大到小，（默认小->大）
        return allDeWeight

    @staticmethod
    def zi_expand(cards=[]):
        """
        功能：计算字牌组合信息
        思路：字牌组合信息需要单独计算，因为没有字顺子，迭代计算出各张字牌的２Ｎ和３Ｎ的情况，由于某些情况下，可能只会需要ａａ作为将牌的情况，同时需要刻子和ａａ的划分结果
        :param cards: 字牌手牌
        :return: ziBranch　字牌的划分情况　[kz,sz,aa,ab,xts,leftCards]
        """
        cardList = []
        for i in range(7):
            cardList.append([])
        ziCards = [0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37]
        for card in ziCards:
            index = (card & 0x0f) - 1  # 转成  0-6，然后一张一张判断
            # print(index)

            if cards.count(card) == 4:
                # 此结构为[3N,2N,leftCards]
                cardList[index].append([[[card, card, card]], [], [], [], 0, [card]])

            elif cards.count(card) == 3:
                cardList[index].append([[[card, card, card]], [], [], [], 0, []])
                cardList[index].append([[], [], [[card, card]], [], 0, [card]])

            elif cards.count(card) == 2:
                cardList[index].append([[], [], [[card, card]], [], 0, []])

            elif cards.count(card) == 1:
                cardList[index].append([[], [], [], [], 0, [card]])

            else:
                cardList[index].append([[], [], [], [], 0, []])

        ziBranch = []
        # for i in cardList:
        #     print("组合数，预估大部分1：",len(i))
        # c1-c7的值在  1-2中波动，所以要循环
        for c1 in cardList[0]:
            for c2 in cardList[1]:
                for c3 in cardList[2]:
                    for c4 in cardList[3]:
                        for c5 in cardList[4]:
                            for c6 in cardList[5]:
                                for c7 in cardList[6]:
                                    branch = []
                                    for n in range(6):
                                        branch.append(c1[n] + c2[n] + c3[n] + c4[n] + c5[n] + c6[n] + c7[n])
                                    ziBranch.append(branch)
        return ziBranch

    def pinghu_CS(self, cards=[], suits=[], t1=[]):
        if cards == []:
            cards = self.cards
            suits = self.suits
        cards.sort()
        # 加入一些特殊的处理 例如碰碰胡的CS生成
        CS = self.pinghu_CS2(cards, suits, t1)
        kingNum = 0
        RM_King = copy.copy(cards)
        if self.kingCard != None:
            kingNum = cards.count(self.kingCard)
            for i in range(kingNum):
                RM_King.remove(self.kingCard)
        pph = MJ.pengpengHu(outKingCards=RM_King, suits=suits, kingNum=kingNum)
        if pph[0] not in CS and pph[0][-2] <= CS[0][-2] + 1:
            CS += pph
        return CS

    def pinghu_CS2(self, cards=[], suits=[], t1=[]):
        """
        功能：综合计算手牌的组合信息
        思路：对手牌进行花色分离后，单独计算出每种花色的组合信息　，再将其综合起来，计算每个组合向听数，最后输出最小向听数及其加一的组合
        :param cards: 手牌
        :param suits: 副露
        :param left_num: 剩余牌
        :param kingCard: 宝牌
        :return: 组合信息
        """
        # 去除宝牌计算信息，后面出牌和动作决策再单独考虑宝牌信息
        if cards == []:
            cards = self.cards
            suits = self.suits
        RM_King = copy.copy(cards)
        kingNum = 0
        # if self.kingCard != None:
        #     kingNum = cards.count(self.kingCard)
        #     for i in range(kingNum):  # 有几张宝牌，就移除几次
        #         RM_King.remove(self.kingCard)  # 从手牌中移除宝牌

        # 花色分离
        wan, tiao, tong, zi = self.split_type_s(RM_King)
        wan_expd = self.tree_expand(cards=wan)  # 返回的是 List[List[list or int]]：[kz,sz,aa,ab,xts,leftCards]: [[1,2,3],0]
        tiao_expd = self.tree_expand(cards=tiao)  # [  [[kz],[sz],[aa],[ab],xts,[leftCards]] , [同左边]  ]
        tong_expd = self.tree_expand(cards=tong)  # 这三个一样，都是平胡分支的 小分支
        zi_expd = self.zi_expand(cards=zi)  # 关于字牌，只需要搜索 刻子，对搭，孤张

        all = []
        for i in wan_expd:  # 进行组合搭配
            for j in tiao_expd:
                for k in tong_expd:
                    for m in zi_expd:
                        branch = []
                        # 将每种花色的4个字段合并成一个字段
                        for n in range(6):
                            branch.append(i[n] + j[n] + k[n] + m[n])

                        branch[-1] += self.padding + t1  # 2个都为空
                        # print("len(最终组合)",len(branch)) =6，因为上面这句是向孤张组合加入
                        all.append(branch)

        # 将获取概率为０的组合直接丢弃到废牌中 todo 由于有宝，这里也可能会被宝代替
        # 移到了出牌决策部分处理
        if self.kingNum <= 1:  # 这里只考虑出牌、宝做宝吊的情况
            for a in all:  # 所有牌组合的一种 a: [kz + sz + aa + t2N + 0 + a]
                for i in range(len(a[3]) - 1, -1, -1):  # for(int i=len-1;i>=0;i--)
                    ab = a[3][i]  # 找出 连续搭和中间搭
                    efc = self.get_effective_cards([ab])  # 得到有效牌,有效牌是16进制，要转成0-33的连续数
                    if sum([LEFT_NUM[MJ.convert_hex2index(e)] for e in efc]) == 0:  # 如果剩余牌为0
                        a[3].remove(ab)  # 将t2N移出 T2
                        a[-1].extend(ab)  # 放入 T1  --ps.可写，前面也有一个可写
                        # logger.info("remove rate 0 ab,%s,%s,%s,a=%s",self.cards,self.suits,self.kingCard,a)

        # 计算向听数
        # 计算拆分组合的向听数
        all = MJ.cal_xts(all, suits, kingNum)  # --ok

        # 获取向听数最小的all分支
        min_index = 0
        # xtspp=max(0,2-self.round/5)
        for i in range(len(all)):  # all 中向听数已经按照 小->大 排序
            if all[i][4] > all[0][4]:  # 找到向听数变大的 index，舍弃后面的组合;; 这里给与一位的扩展数
                min_index = i
                break

        if min_index == 0:  # 如果全部都匹配，则min_index没有被赋值，将min_index赋予ａｌｌ长度
            min_index = len(all)

        all = all[:min_index]  # 只保留 0-min_index 的组合

        # 处理向听数为0时的情况，需要从中依次选择一张牌作为t1.
        # ###上饶麻将中，由碰牌形成的可胡牌组合，此时不能胡牌，只能舍弃一张牌。（此时，不论舍弃哪张，向听数都=1
        if all[0][-2] == 0 and all[0][-1] == []:  # 向听数==0  and  孤张没有
            all = []  # 重新算过all组合
            for card in list(set(cards)):
                cards_ = copy.copy(cards)
                cards_.remove(card)
                all += self.pinghu_CS2(cards=cards_, suits=suits, t1=[card])  # 去掉一张牌的all
        return all

    def left_card_weight(self, card, left_num, need_jiang=False):
        """
        功能：对废牌组合中的每张废牌进行评估，计算其成为３Ｎ的概率
        思路：每张牌能成为３N的情况可以分为先成为搭子，在成为３Ｎ２步，成为搭子的牌必须自己摸到，而成为kz,sz可以通过吃碰。刻子为获取２张相同的牌，顺子为其邻近的２张牌
        :param card: 孤张牌
        :param left_num: 剩余牌
        :return: 评估值
        """

        # if self.remainNum==0:
        #     remainNum=1
        # else:
        #     remainNum = self.remainNum
        # remainNum = 1
        i = convert_hex2index(card)

        if need_jiang:
            return left_num[i]
        # d_w = 0

        # if left_num[i] == self.remainNum:
        #     sf = float(self.leftNum[i])
        # else:
        #     sf = float(left_num[i]) / remainNum * float((left_num[i] - 1)) / remainNum * 6

        if left_num[i] > 1:
            aa = left_num[i] * (left_num[i] - 1) * 4
        else:
            aa = left_num[i]
        if card >= 0x31:  # kz概率
            # todo if card == fengwei:
            # if card >= 0x35 and left_num[i] >= 2:
            #     d_w = left_num[i] * left_num[i] * 2  # bug 7.22 修正dw-d_w
            # else:
            d_w = aa  # 7.22 １６:３５ 去除字牌
        elif card % 16 == 1:  # 11＋２3
            d_w = aa + left_num[i + 1] * left_num[i + 2] * 2
        elif card % 16 == 2:  # 22+13+3(14)+43   222 123 234
            d_w = aa + left_num[i - 1] * left_num[i + 1] * 2 + left_num[i + 1] * left_num[i + 2] * 2
        elif card % 16 == 8:  # 888 678 789
            d_w = aa + left_num[i - 2] * left_num[i - 1] * 2 + left_num[i - 1] * left_num[i + 1] * 2
        elif card % 16 == 9:  # 999 789
            d_w = aa + left_num[i - 2] * left_num[i - 1] * 2
        # 删除多添加的×２
        else:  # 555 345 456 567
            # print (left_num)
            d_w = aa + left_num[i - 2] * left_num[i - 1] * 2 + left_num[i - 1] * left_num[i + 1] * 2 + left_num[i + 1] * \
                  left_num[
                      i + 2] * 2
        # if card<=0x31:
        #     if (card%0x0f==3 or card %0x0f==7): #给金3银7倍数
        #         d_w*=1.5
        #     elif card%0x0f==5:
        #         d_w*=1.2
        print("i=", i, d_w)
        return d_w


def translate16_33(i):
    """
    将牌值１６进制转化为０－３３的下标索引
    :param i: 牌值
    :return: 数组下标
    """
    i = int(i)
    if i >= 0x01 and i <= 0x09:
        i = i - 1
    elif i >= 0x11 and i <= 0x19:
        i = i - 8
    elif i >= 0x21 and i <= 0x29:
        i = i - 15
    elif i >= 0x31 and i <= 0x37:
        i = i - 22
    else:
        # i=1/0
        print("translate16_33 is error,i=%d" % i)
        i = -1
    return i


def convert_hex2index(a):
    """
    将牌值１６进制转化为０－３３的下标索引
    :param a: 牌
    :return: 数组下标
    """
    if a > 0 and a < 0x10:
        return a - 1
    if a > 0x10 and a < 0x20:
        return a - 8
    if a > 0x20 and a < 0x30:
        return a - 15
    if a > 0x30 and a < 0x40:
        return a - 22


def trandfer_discards(discards, discards_op, handcards):
    """
    获取场面剩余牌数量
    计算手牌和场面牌的数量，再计算未知牌的数量
    :param discards: 弃牌
    :param discards_op: 场面副露
    :param handcards: 手牌
    :return: left_num, discards_list　剩余牌列表，已出现的牌数量列表
    """
    discards_map = {0x01: 0, 0x02: 1, 0x03: 2, 0x04: 3, 0x05: 4, 0x06: 5, 0x07: 6, 0x08: 7, 0x09: 8, 0x11: 9, 0x12: 10,
                    0x13: 11, 0x14: 12, 0x15: 13, 0x16: 14, 0x17: 15, 0x18: 16, 0x19: 17, 0x21: 18, 0x22: 19, 0x23: 20,
                    0x24: 21, 0x25: 22, 0x26: 23, 0x27: 24, 0x28: 25, 0x29: 26, 0x31: 27, 0x32: 28, 0x33: 29, 0x34: 30,
                    0x35: 31, 0x36: 32, 0x37: 33, }
    # print ("discards=",discards)
    # print ("discards_op=",discards_op)
    left_num = [4] * 34
    discards_list = [0] * 34
    for per in discards:  # 减去弃牌集
        for item in per:
            discards_list[discards_map[item]] += 1
            left_num[discards_map[item]] -= 1
    for seat_op in discards_op:  # 减去副露集
        for op in seat_op:
            for item in op:
                discards_list[discards_map[item]] += 1
                left_num[discards_map[item]] -= 1
    for item in handcards:  # 减去自己的手牌
        left_num[discards_map[item]] -= 1
    # print("discards_lsit:", discards_list)
    # print("letf_num:     ", left_num)
    return left_num, discards_list


# 获取ｌｉｓｔ中的最小值和下标        ##只考虑最小的那个吗？如果有相同向听数的组合？
def get_min(list=[]):
    """
    获取最小ｘｔｓ的下标
    :param list: 向听数列表
    :return: 返回最小向听数及其下标
    """
    min = 14
    index = 0
    for i in range(len(list)):
        if list[i] < min:
            min = list[i]
            index = i
    return min, index


def pre_king(king_card=None):
    """
    计算宝牌的前一张
    :param king_card: 宝牌
    :return:宝牌的前一张牌
    """
    if king_card == None:
        return None
    if king_card == 0x01:
        return 0x09
    elif king_card == 0x11:
        return 0x19
    elif king_card == 0x21:
        return 0x29
    elif king_card == 0x31:
        return 0x37
    else:
        return king_card - 1


def value_t1(card):
    """
    计算出牌的危险度评估值，由该牌转化为t3的概率组成
    :param card:
    :return:
    """
    value = 0
    if card != -1:
        for e in t1tot3_dict[str(card)]:
            v = 1
            for i in range(len(e[1])):
                v *= T_SELFMO[MJ.convert_hex2index(e[1][i])] * e[-1][i]
            value += v
    return value


def get_score_dict(cards, suits, king_card, fei_king, padding=[], max_xts=14, round=0):
    """
    计算各牌型的评估值
    :param cards: 手牌
    :param suits: 副露
    :param king_card:  宝牌
    :param fei_king: 飞宝
    :param padding: 填充牌。用于计算：op缺一张牌时，填充-1。# 13张手牌到14张手牌 缺的那张牌
    :param max_xts: 允许的最大向听数，否则停止计算，用于处理：op中非平胡牌型的吃碰杠处理，例如十三烂牌型吃碰导致需要计算平胡牌型的出牌评估值，从而导致超时
    :return: score_dict,min_xts ，index 各出牌的评估值与本轮计算的最小向听数及最小向听数对应的牌型下标 0-平胡 1-七对（用于op中对比操作前后时）
    """
    # 寻找向听数在阈值内的牌型
    PH = PingHu(cards=cards, suits=suits, kingCard=king_card, fei_king=fei_king, padding=padding,
                round=round)  # 传参数，初始化类
    # SSL = ShiSanLan(cards=cards, suits=suits, king_card=king_card, fei_king=fei_king, padding=padding)
    # JY = JiuYao(cards=cards, suits=suits, king_card=king_card, fei_king=fei_king, padding=padding)
    QD = Qidui(cards=cards, suits=suits, king_card=king_card, fei_king=fei_king, padding=padding)

    # 组合信息
    CS_PH = PH.pinghu_CS2()  # 平胡的组合信息，最后只保留了向听数 最少的，[刻子，顺子，搭子，向听数，剩余牌]
    # CS_SSL = SSL.ssl_CS()       #十三烂的组合信息，[万，条，筒，字，剩余牌，向听数]
    # CS_JY = JY.yaojiu_CS()      #九幺的组合信息，[九幺牌，剩余牌，向听数]
    CS_QD = QD.qidui_CS()  # 七对的组合信息，[对子，剩余牌，向听数]       对子=【0，0】表示用宝牌的任意对
    # 向听数

    # xts_list = [CS_PH[0][-2], CS_SSL[0][-1], CS_JY[-1], CS_QD[-1]]  # 记录每一种胡牌的最少向听数
    xts_list = [CS_PH[0][-2], CS_QD[-1]]
    # print("xts_list PH,SSL,JY,QD", xts_list)
    # logger.info("xts PH,SSL,JY,QD:%s", xts_list)
    min_xts, index = get_min(xts_list)
    # op中吃碰后向听数增加的情况，特别是打非平胡的牌型
    if min_xts > max_xts + 1:
        logger.info("min_xts > max_xts + 1 in get_score_dict function")
        return {cards[-1]: 0}, min_xts, index
    type_list = []  # 需搜索的牌型
    for i in range(2):
        if xts_list[i] <= min_xts + 1:  # 比最少向听数大1 的组合也考虑一下，因为按概率给路径，xts大的一般就只有影响作用，不能决定
            type_list.append(i)  # 比如平胡向听数=1，九幺向听数=2 .考虑到九幺容易胡，所以九幺也去计算
    # type_list.append(3)

    score_list = []
    time_start = time.time()
    time_list = []
    for i in type_list:  # 0123分别代表 4中胡牌类型。
        if i == 0:
            search_PH = SearchTree_PH(hand=cards, suits=suits, combination_sets=CS_PH, king_card=king_card,
                                      fei_king=fei_king)  # 同上，初始化类
            score_list.append(search_PH.get_discard_score())  # 返回dict: 包含每一张弃牌的评估值
        # elif i == 1:
        #     score_list.append(SSL.get_discard_score())      #返回dict: 包含每一张弃牌的评估值
        # elif i == 2:
        #     score_list.append(JY.get_discard_score())       #返回dict: 包含每一张弃牌的评估值
        if i == 1:
            score_list.append(QD.get_discard_score())  # 返回dict: 包含每一张弃牌的评估值==now
        time_list.append(time.time() - time_start - sum(time_list))

    # print time_list
    # logger.info("time use%s", time_list)
    # 计算总的评估值，有所有选中的牌型的评估值之和
    score_dict = {}
    # print("score_list", score_list)     #score_list:dict: 包含每一张弃牌的评估值
    # print("type(score_list):",type(score_list))        #type=list
    # print("type(score_list[0]):",type(score_list[0]))  #type=dict
    for score in score_list:  # 取第一个分数集，所有牌都会有一个分数 （在上文，如果多种牌型向听数接近，就会叠加考虑）
        for key in score.keys():  # key ==牌ID  如果key未在dict中，存入一个震荡值；如果已经在，追加权值（叠加考虑？？）
            if key not in score_dict.keys():
                score_dict[key] = score[key] - float(value_t1(key)) / (10 ** (min_xts + 1) / 2)  # 用来区分相同权重的出牌，同价值浮动
            else:
                score_dict[key] += score[key]  # XXX这里的相加是人为的xxx
    # # 飞宝的权重增加
    # if king_card in score_dict.keys():
    #     score_dict[king_card] *= 1.2  # XXX这里的乘1.2是人为的xxx

    # print(score_dict)
    # print(type(score_dict))

    # print("score_dict:", score_dict)

    return score_dict, min_xts, index


def recommend_switch_cards(hand_cards=[], switch_n_cards=3):
    switch_cards = SwitchTiles(hand=hand_cards, n=switch_n_cards).choose_3card()
    return switch_cards


def recommend_choose_color(hand_cards=[], switch_n_cards=3):
    choose_color = SwitchTiles(hand=hand_cards, n=switch_n_cards).choose_color_final()
    return choose_color


def recommend_card(cards=[], suits=[], king_card=None, discards=[], discards_op=[], fei_king=0, remain_num=108,
                   round=0, seat_id=0, self_lack=0, is_RL=False):
    """
    功能：推荐出牌接口
    思路：使用向听数作为牌型选择依据，对最小ｘｔｓ的牌型，再调用相应的牌型类出牌决策
    :param cards:
    :param suits: 自己的副露手牌
    :param king_card: 宝牌
    :param discards: 弃牌
    :param discards_op: 场面副露
    :param fei_king: 飞宝数
    :param remain_num: 剩余牌
    :param is_RL: 是否强化学习推荐出牌
    :return: outCard 推荐出牌
    """

    # logger.info("cards = %s", cards)
    # print("cards = ", cards)

    list_lack = []
    for i in range(len(cards)):
        if int(cards[i] // 16) == self_lack:
            list_lack.append(cards[i])
    if len(list_lack) > 0:
        list_lack.sort(key=lambda x: abs(x % 16 - 5), reverse=True)  # 优先返回边张1 9，最后 5
        # print("out_card=", list_lack[0])
        if is_RL:
            return 0, list_lack[0]
        return list_lack[0]

    # print("card:", cards)

    # logger.info("recommond card start...")
    # 更新全局变量
    ##T_SELFMO:自摸表  LEFT_NUM:剩余表    RT1：其他玩家的状态表[不要,需要]     RT2：[吃碰概率，危险度]
    global T_SELFMO, LEFT_NUM, TIME_START, RT1, RT2, RT3, ROUND  # , t2tot3_dict, t1tot3_dict
    ROUND = round  # 轮数
    MJ.KING = king_card  # 宝牌ID
    TIME_START = time.time()
    # 计算获取概率
    LEFT_NUM, _ = trandfer_discards(discards=discards, discards_op=discards_op, handcards=cards)  # 利用场面所有已知信息，求出剩余牌
    # LEFT_NUM[translate16_33(pre_king(king_card))] -= 1  # 计算宝牌时，它的前一张牌也是弃牌
    REMAIN_NUM = max(1, min(sum(LEFT_NUM), remain_num))  # 约等于  REMAIN_NUM=LEFT_NUM.size();约定了上下限的

    if True:
        # if round <8:
        T_SELFMO = [float(i) / REMAIN_NUM for i in LEFT_NUM]  # 求出每一张牌的概率  分子：这张牌的剩余数量 分母：剩余牌总数
        # print T_SELFMO
        RT1 = []
        RT2 = []
        RT3 = []

    # 计算所有可能出牌的评估值
    score_dict, _, paixing_idx = get_score_dict(cards, suits, king_card, fei_king,
                                                round=round)  # score_dict: 每种牌的评估值-----------now
    if score_dict != {}:
        recommend_card = max(score_dict, key=lambda x: score_dict[x])  # 输出score_dict里最大值的keys（用values比较）
    else:  # 手牌可能已经胡了，这里出一张牌，一般不可能发生
        recommend_card = cards[-1]
        # logger.error("no card be recommonded,cards=%s,suits=%s,king_card=%s", cards, suits, king_card)
    end = time.time()
    if end - TIME_START > 3:  # 超时输出
        logger.error("overtime %s,%s,%s,%s", end - TIME_START, cards, suits, king_card)
    # logger.info("recommend_card %s\n", recommend_card)
    if is_RL:
        return paixing_idx, recommend_card
    return recommend_card


def recommend_op(op_card, cards=[], suits=[], king_card=None, discards=[], discards_op=[], canchi=False,
                 self_turn=False, fei_king=0, isHu=False, round=0):
    """
    功能：动作决策接口
    思路：使用向听数作为牌型选择依据，对最小ｘｔｓ的牌型，再调用相应的牌型类动作决策
    :param op_card: 操作牌，别人回合的弃牌
    :param cards: 手牌
    :param suits: 副露
    :param king_card: 宝牌
    :param discards: 弃牌
    :param discards_op: 场面副露
    :param canchi: 吃牌权限
    :param self_turn: 是否是自己回合
    :param fei_king: 飞宝数
    :param isHu: 是否胡牌
    :return: [],isHu 动作组合牌，是否胡牌
    """
    if isHu:
        return [], True

    # 更新全局变量
    global T_SELFMO, LEFT_NUM, t2tot3_dict, t1tot3_dict, TIME_START
    MJ.KING = king_card
    TIME_START = time.time()
    LEFT_NUM, discards_list = trandfer_discards(discards=discards, discards_op=discards_op, handcards=cards)
    # LEFT_NUM[translate16_33(pre_king(king_card))] -= 1
    # if remain_num == 0:
    #     remain_num = 1
    REMAIN_NUM = sum(LEFT_NUM)
    if round > 100:
        T_SELFMO = []
        RT1 = []
        RT2 = []
        RT3 = []
    else:
        T_SELFMO = [float(i) / REMAIN_NUM for i in LEFT_NUM]
        RT1 = []
        RT2 = []
        RT3 = []

    # t1tot3_dict = MJ.t1tot3_info(T_selfmo=T_SELFMO, RT1=[], RT2=[], RT3=[])
    # t2tot3_dict = MJ.t2tot3_info(T_selfmo=T_SELFMO, RT1=[], RT2=[], RT3=[])

    # 计算操作前评估值
    cards_pre = copy.copy(cards)
    # cards_pre.append(-1) #加入一张0作为下次摸到的牌，并提升一定的概率a
    score_dict_pre, min_xts_pre, _ = get_score_dict(cards_pre, suits, king_card, fei_king, padding=[-1])
    # xts_pre = min
    if score_dict_pre != {}:

        score_pre = max(score_dict_pre.values())
    else:
        score_pre = 0

    # 计算操作后的评估值
    # 确定可选动作
    set_cards = list(set(cards))
    if self_turn:  # 自己回合，暗杠或补杠
        for card in set_cards:
            if cards.count(card) == 4:
                return [card, card, card, card], False  # 暗杠必杠
        for suit in suits:
            if suit.count(suit[0]) == 3 and suit[0] in cards:  # 碰牌副露 + 自摸一张 = 补杠
                return suit + [suit[0]], False

    else:  # 其他玩家回合 #明杠，吃碰
        if cards.count(op_card) == 3:
            return [op_card, op_card, op_card, op_card], False

        op_sets = []
        if canchi:
            # 可操作的集合
            # 计算可吃组合
            if op_card < 0x30:  # 字牌不能吃
                rm_sets = [[op_card - 2, op_card - 1], [op_card - 1, op_card + 1], [op_card + 1, op_card + 2]]
            else:
                rm_sets = []
            for op_set in rm_sets:
                if op_set[0] in cards and op_set[1] in cards:  # 我们有可以吃的牌型
                    op_sets.append(op_set)
            # 碰
            if cards.count(op_card) >= 2:
                op_sets.append([op_card, op_card])  # 要存在2次，因为 if else只走一个
        else:
            if cards.count(op_card) >= 2:
                op_sets.append([op_card, op_card])

        score_set = []
        for op_set in op_sets:
            cards_ = copy.copy(cards)
            cards_.remove(op_set[0])  # 进行了吃碰，从手牌移除
            cards_.remove(op_set[1])

            suits_ = MJ.deepcopy(suits)  # 加入副露中
            suits_.append(sorted(op_set + [op_card]))
            score_dict, _, _ = get_score_dict(cards=cards_, suits=suits_, king_card=king_card, fei_king=fei_king,
                                              max_xts=min_xts_pre)
            # max_discard = max(score_dict, key=lambda x: score_dict[x])
            # print "score_dict",score_dict
            if score_dict != {}:
                score = max(score_dict.values())
                score_set.append(score)
        if time.time() - TIME_START > 3:
            logger.warning("op time out %s", time.time() - TIME_START)
        if score_set == []:
            return [], False
        else:
            max_score = max(score_set)
            # print max_score, score_pre
            if max_score > score_pre * 1.05:
                return sorted(op_sets[score_set.index(max_score)] + [op_card]), False

    return [], False


def recommend_card_rf(cards=[], suits=[], round=0, remain_num=(9 + 0) * 4 * 3, discards=[], discards_real=[],
                      discards_op=[], seat_id=0, choose_color=[], hu_cards=[[], [], [], []], hu_fan=[[], [], [], []]):
    # 兼容以前的接口，参数改成一致
    # return paixing_idx, recommend_card
    return recommend_card(cards, suits, None, discards, discards_op, 0, remain_num,
                          round, seat_id, choose_color[seat_id], is_RL=True)


def recommend_op_rf(op_card, cards=[], suits=[], round=0, remain_num=136, discards=[], discards_real=[], discards_op=[],
                    self_turn=False, seat_id=0, choose_color=[], hu_cards=[[], [], [], []], hu_fan=[[], [], [], []],
                    isHu=False):
    return recommend_op(op_card, cards, suits, king_card=None, discards=discards, discards_op=discards_op, canchi=False,
                        self_turn=self_turn, fei_king=0, isHu=isHu, round=round)


def findmin(cards):
    cards_wan = 0
    cards_tiao = 0
    cards_tong = 0
    for card in cards:
        if card & 0xF0 == 0x00:  # 如果16进制只是为了在这里好算，那10进制也可以
            cards_wan += 1
        elif card & 0xF0 == 0x10:
            cards_tiao += 1
        elif card & 0xF0 == 0x20:
            cards_tong += 1

    if (cards_wan < 3):
        cards_wan += 10
    if (cards_tiao < 3):
        cards_tiao += 10
    if (cards_tong < 3):
        cards_tong += 10

    min_card = min(cards_wan, min(cards_tiao, cards_tong))
    if (cards_wan == min_card):
        # print("定缺万字牌")
        return 0
    if (cards_tiao == min_card):
        # print("定缺条字牌")
        return 1
    if (cards_tong == min_card):
        # print("定缺筒字牌")
        return 2


def change_three(cards):
    cards_wan = []
    cards_tiao = []
    cards_tong = []
    for card in cards:
        if card & 0xF0 == 0x00:  # 如果16进制只是为了在这里好算，那10进制也可以
            cards_wan.append(card)
        elif card & 0xF0 == 0x10:
            cards_tiao.append(card)
        elif card & 0xF0 == 0x20:
            cards_tong.append(card)
    min_card = findmin(cards)

    if min_card == 0:
        return cards_wan[:3]

    if min_card == 1:
        return cards_tiao[:3]

    if min_card == 2:
        return cards_tong[:3]


def def_lack(cards):
    min_card = findmin(cards)
    if (0 == min_card):
        print("定缺万字牌")
        return 0
    if (1 == min_card):
        print("定缺条字牌")
        return 1
    if (2 == min_card):
        print("定缺筒字牌")
        return 2

# if __name__ == "__main__":
#     import random
#
#     walls = [1, 2, 3, 4, 5, 6, 7, 8, 9, 17, 18, 19, 20, 21, 22, 23, 24, 25, 33, 34, 35, 36, 37, 38, 39, 40, 41] * 4
#     for i in range(1000):
#         cards = random.sample(walls, 13)
#         cards.sort()
#         suits = []
#         t32_set = []
#
#         choose_3cards = recommend_switch_cards(cards)
#         final_c = recommend_choose_color(cards)
#
#         print(cards)
#         print(choose_3cards)
#         print(final_c)
