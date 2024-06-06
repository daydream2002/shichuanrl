# -*- coding:utf-8 -*-
# cython: language_level=2
# ｂｕｇ：递归迭代中有些情况没有返回值而无法返回
# input 一种花色手牌，９^5种
# python 2.0　两整数相处会自动取整，需要人为给被除数添加float型
'''
改进版本
2020.0805
金币场版本
todo 评估模块耗时问题

拆搭子的情况在搭子的有效牌数量为0
搭子的数量多于待需数量
'''

import copy
import numpy as np
import time

# import numpy as np
import math
import interface.sichuanMJ.lib_MJ as MJ
import logging
# import opp_srmj as DFM
import datetime
import random
# import thread
import os

# logger = logging.getLogger("sichuanMJ_log_v2")
# logger.setLevel(level=logging.DEBUG)
# log_path = "/home/tonnn/recommondsrv_qipai/app/recommond/shangraoMJ/"
# log_file = "shangraoMJlog.txt"
# if not os.path.isfile(log_path):
#     os.mknod(log_path) #windows不存在node
#     os.mkdir(log_path)
#     with open(os.path.join(log_path,log_file),'a+') as fp:
#
#         fp.close()

# handler = logging.FileHandler("/home/tonnn/recommondsrv_qipai/app/recommond/shangraoMJ/shangraoMJlog.txt")
# time_now = datetime.datetime.now()
# # log_path = "./%i_log.txt"%time_now.day
# # print log_path
# handler = logging.FileHandler("./%i%i%i_log.txt" % (time_now.year, time_now.month, time_now.day))
#
# handler.setLevel(logging.INFO)
#
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# handler.setFormatter(formatter)
#
# logger.addHandler(handler)
# logger.info("compile finished...")

# global variable
TIME_START = time.time()

w_ways = 1
w_aa = (1 + 3 * w_ways)
w_ab = (1 + 0 * w_ways)
w_type = 0
ROUND = 0
t3Set = MJ.get_t3info()
t2Set, t2Efc, efc_t2index = MJ.get_t2info()

T_SELFMO = [0] * 34  # 自摸概率表，牌存在于牌墙中的概率表
LEFT_NUM = [0] * 34  # 未出现的牌的数量表
RT1 = [[0] * 34, [0] * 34]  # 危险度表
RT2 = [[0] * 34, [0] * 34]
RT3 = [[0] * 34, [0] * 34]

t1tot2_dict = {}
t1tot3_dict = {}
t2tot3_dict = {}


class SwitchTiles:
    def __init__(self, hand, n=3):
        """

        :param hand: 手牌
        :param n: 换牌张数，默认为换3张
        """
        self.hand = hand
        self.type = n
        self.color = MJ.splitColor(hand)
        if self.type == 3:
            self.t13 = [[5, 3], [5, 4], [6, 1], [6, 2], [6, 3], [7, 0], [7, 1], [7, 2], [7, 3], [8, 0], [8, 1], [8, 2],
                        [9, 0], [9, 1], [9, 2], [10, 0], [10, 1], [11, 0], [11, 1], [12, 0], [13, 0]]
            self.s13 = [[3], [5, 4], [6], [6, 5], [4, 3], [7, 6], [5], [4], [3], [5], [4], [3], [4], [3], [9], [3],
                        [10], [11], [11], [12], [13]]
            self.t14 = [[5, 4], [6, 2], [6, 3], [6, 4], [7, 0], [7, 1], [7, 2], [7, 3], [8, 0], [8, 1], [8, 2], [8, 3],
                        [9, 0], [9, 1], [9, 2], [10, 0], [10, 1], [10, 2], [11, 0], [11, 1], [12, 0], [12, 1], [13, 0],
                        [14, 0]]
            self.s14 = [[5, 4], [6], [3], [4], [7], [7, 6], [5], [4, 3], [6], [5], [4], [3], [5], [4], [3], [4], [3],
                        [10], [3], [11], [12], [12], [13], [14]]
        elif self.type == 4:
            self.t13 = [[5, 3], [5, 4], [6, 1], [6, 2], [6, 3], [7, 0], [7, 1], [7, 2], [7, 3], [8, 0], [8, 1], [8, 2],
                        [9, 0], [9, 1], [9, 2], [10, 0], [10, 1], [11, 0], [11, 1], [12, 0], [13, 0]]
            self.s13 = [[5, 5], [5, 4, 4], [6, 6], [6, 5], [4], [7, 6], [5], [4], [7], [5], [4], [8], [4], [9], [9],
                        [10], [10], [11], [11], [12], [13]]
            self.t14 = [[5, 4], [6, 2], [6, 3], [6, 4], [7, 0], [7, 1], [7, 2], [7, 3], [8, 0], [8, 1], [8, 2], [8, 3],
                        [9, 0], [9, 1], [9, 2], [10, 0], [10, 1], [10, 2], [11, 0], [11, 1], [12, 0], [12, 1], [13, 0],
                        [14, 0]]
            self.s14 = [[5, 5, 4], [6, 6], [6, 5], [4, 4], [7, 7], [7, 6], [5], [4], [6], [5], [4], [8], [5], [4], [9],
                        [4], [10], [10], [11], [11], [12], [12], [13], [14]]
        elif self.type == 0:
            pass
        else:
            print("SwitchTiles ERROR! input type error,n=0, 3 or 4.")
        pass

    def choose_color_index(self):
        len_color = [len(self.color[0]), len(self.color[1]), len(self.color[2])]
        # 闲家手牌为13张
        # [[6, 3], [7, 1], [7, 2], [7, 3], [8, 0], [8, 1], [8, 2], [9, 0], [9, 1], [9, 2], [10, 0], [10, 1], [11, 0],
        # [11, 1], [12, 0], [13, 0]]
        max_len = max(len_color)
        min_len = min(len_color)
        if len(self.hand) == 13:
            index = self.t13.index([max_len, min_len])

            color_n = self.s13[index]
            color_index = MJ.get_index(len_color, color_n)
            return color_index

        elif len(self.hand) == 14:
            index = self.t14.index([max_len, min_len])
            color_n = self.s14[index]
            color_index = MJ.get_index(len_color, color_n)
            return color_index

        else:
            print('SwitchTiles, choose_color ERROR! len(self.hand)=13 or 14, but =' + len(self.hand))
            return -1

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


'''
抓牌结点类
功能：保存抓牌结点的相关信息，包括抓牌，获取概率，本路径的所有抓过的牌，弃牌等,以及本路径现有的sz,kz,jiang 
'''


class CatchNode:
    def __init__(self, cards=[], catchCard=None, leftNum=[], remainNum=136, t2=[], level=0, kingCard=None, t2N=[],
                 ocards=[], baoHuanYuan=0):
        """
        功能：类变量初始化
        :param cards: 手牌
        :param catchCard:抓牌
        :param leftNum: 剩余牌数量ｌｉｓｔ
        :param remainNum: 剩余牌总数
        :param t2: 抓牌搭子
        :param level: 所处搜索树层数
        :param ocards: 出牌结点策略集合
        :param t2N: 抓牌结点扩炸集合
        :param kingCard: 宝牌
        """
        self.type = 2
        self.cards = cards
        self.leftNum = leftNum
        self.catchCard = catchCard
        self.rate = 1
        if catchCard != None:  # 获取概率
            if len(t2) == 1:  # 单张牌，凑将
                if t2[0] == kingCard:  # 宝吊处理
                    self.rate = 1
                else:  # 无宝摸将
                    self.rate = float(
                        leftNum[MJ.convert_hex2index(catchCard)]) / remainNum * 1
            elif len(t2) == 2:
                if t2[0] == t2[1]:
                    self.rate = float(leftNum[MJ.convert_hex2index(catchCard)]) / remainNum * 8
                else:
                    self.rate = float(leftNum[MJ.convert_hex2index(catchCard)]) / remainNum * 2
            else:
                print('CatchNode Error 2!', catchCard, t2)
        # print catchCard ,t2,  self.rate
        # if self.rate==0:
        # print ('rate=0,catchCard',catchCard)
        self.t2 = t2
        self.level = level  # 在树中的层数

        self.kz = []
        self.sz = []
        self.jiang = 0x00
        self.parent = None  # todo 可以使用ｈａｓｈ表来存，可能会快一点
        self.children = []
        self.formerCatchCards = []
        self.formerOutCards = []
        # 增加宝牌的处理
        self.kingCard = kingCard
        self.feiKingNum = 0  # 飞宝数
        # self.noUseKingNum=0#待用宝牌数
        # self.usingKing=0 #
        self.baoHuanYuan = baoHuanYuan
        self.addKing = False
        self.t2N = t2N
        self.ocards = ocards
        self.firstOutCard = 0x00

    # def ac(self,t2):
    #     if t2[0]+2==t2[1]:
    #         return True
    #     elif

    def setParent(self, parent):
        """
        设置父结点
        :param parent:父结点
        """
        self.parent = parent

    def addChild(self, child):
        """
        增加子结点
        :param child:子结点
        """
        self.children.append(child)

    def equal(self, newNode):
        """
        判断结点与本结点是否是同一结点
        :param newNode: 待比较的结点
        :return: bool 是否相同
        """
        if newNode.catchCard == self.catchCard and newNode.kz == self.kz and newNode.sz == self.sz and newNode.jiang == self.jiang:
            return True

        return False

    def __repr__(self):
        # return "{%d,%s,%s}".format(self.type,self.cards,self.catchCard)
        return self.type, self.cards, self.catchCard, self.level

    # def nodeInfo(self):
    #     print ['type', self.type, 'cards', self.cards, 'catchCard', self.catchCard, 'rate', self.rate, 't2', self.t2,
    #            'level', self.level, 'ocards', self.ocards, 't2N', self.t2N, 'kz', self.kz, 'sz', self.sz, 'jiang',
    #            self.jiang, 'formerCatchCards', self.formerCatchCards, 'formerOutCards', self.formerOutCards, 'kingCard',
    #            self.kingCard, 'baoHuanYuan', self.baoHuanYuan]


'''
出牌结点类
功能：保存出牌结点相关信息，包括出牌，出牌危险度，本路径所有出的牌，抓的牌，以及本路径现有的sz,kz,jiang 
'''


class OutNode:
    def __init__(self, cards=[], outCard=[], level=0, dgRate=[], kingCard=None, t2N=[], ocards=[], baoHuanYuan=0):
        """
        初始化出牌结点类变量
        :param cards: 手牌
        :param outCard: 出牌
        :param level: 所处的搜索树层数
        :param ocards: 本路径的出牌策略结合
        :param t2N: 本路径的抓牌策略集合
        :param dgRate: 危险概率表
        :param kingCard: 宝牌
        """
        self.type = 1
        self.cards = cards
        self.outCard = outCard
        self.level = level  # 在树中的层数
        self.parent = None
        self.children = []
        self.kz = []
        self.sz = []
        self.jiang = 0x00

        self.formerCatchCards = []
        self.formerOutCards = []
        self.rate = dgRate[MJ.convert_hex2index(outCard)]  # 危险概率
        # 增加宝牌的处理信息
        self.kingCard = kingCard
        self.feiKingNum = 0
        self.addKing = False
        self.t2N = t2N
        self.ocards = ocards
        self.baoHuanYuan = baoHuanYuan
        self.firstOutCard = 0x00

    def setParent(self, parent):
        """
        设置父结点
        :param parent:父结点
        """
        self.parent = parent

    def addChild(self, child):
        """
        设置子结点
        :param child:子结点
        """
        self.children.append(child)

    def equal(self, newNode):
        """
        判断结点是否相同
        :param newNode:待比较的结点
        :return: bool 是否相同
        """
        if newNode.outCard == self.outCard and newNode.kz == self.kz and newNode.sz == self.sz and newNode.jiang == self.jiang:
            return True

        return False

    # def nodeInfo(self):
    #     """
    #     打印结点信息
    #     """
    #     print('type', self.type, 'cards', self.cards, 'outCard', self.outCard, 'rate', self.rate, 'level', self.level,
    #           'ocards', self.ocards, 't2N', self.t2N, 'kz', self.kz, 'sz', self.sz, 'jiang', self.jiang,
    #           'formerCatchCards', self.formerCatchCards, 'formerOutCards', self.formerOutCards, 'kingCard',
    #           self.kingCard, 'baoHuanYuan', self.baoHuanYuan)


'''
搜索树类，用于搜索最佳出牌
'''


class SearchTree:
    def __init__(self, cards, suits, leftNum, all, remainNum, dgtable, kingCard, feiKingNum=0):
        """
        初始化类变量，以及搜索树的根结点
        :param cards: 手牌
        :param suits: 副露
        :param leftNum: 剩余牌
        :param all: 组合信息
        :param remainNum: 剩余牌
        :param dgtable: 危险度
        :param kingCard: 宝牌
        :param feiKingNum: feiKingNum飞宝数
        """
        # print('leftNum', leftNum)
        # print  ('xts', all[0][4])
        # print('search tree : all',all)
        self.root = CatchNode(cards=cards, catchCard=None, leftNum=leftNum, remainNum=remainNum, t2=[], level=0,
                              kingCard=kingCard)
        self.kingNum = cards.count(kingCard)
        self.root.feiKingNum = feiKingNum
        self.kingCard = kingCard
        self.cards = cards
        self.suits = suits
        self.leftNum = leftNum
        self.all = all
        self.xts = all[0][4]
        self.xts_min = all[0][4]
        self.remainNum = remainNum
        self.dgtable = dgtable
        self.stateSet = {}
        self.fei_king = feiKingNum
        # self.op_card=op_card
        # self.type=type
        self.scoreDict = {}
        self.t2Nw_Set = {}
        for suit in suits:
            if suit[0] != suit[1]:
                self.root.sz.append(suit[0])
            else:
                self.root.kz.append(suit[0])
        self.maxScore = [0, 0]
        # CI修正，当ｔ２Ｎ溢出时，将概率最低的２Ｎ加入废牌区
        # CI = copy.deepcopy(all)
        # bl = 4 - len(suits)
        # for a in all:
        #     ab = copy.deepcopy(a[3])
        #     if a[2]!=[] and self.kingNum==0:
        #         lenofT2Set=len(a[2])+len(a[3])-1
        #     else:
        #         lenofT2Set=len(a[2])+len(a[3])
        #     if lenofT2Set>bl-len(a[0])-len(a[1]):
        #         CI.remove(a)
        #         ab_efc,w=self.get_effective_cards_w(a[3])
        #
        #         for i in range(len(w)):
        #             ab[i].append(w[i])
        #
        #         ab.sort(key=lambda k: k[2], reverse=True)
        #         min_ab=[]
        #         for ab_ in ab:
        #             if ab_[2]==ab[0][2]:
        #                 min_ab.append([ab_[0],ab_[1]])
        #         for m_ab in min_ab:
        #             C = copy.deepcopy(a)
        #             # print (T2Set[-1])
        #             # if ab[-1][0]==ab[-1][1]:
        #             #     C[2].remove([ab[-1][0],ab[-1][1]])
        #             # else:
        #             C[3].remove([m_ab[0], m_ab[1]])
        #             C[-1].append(m_ab[0])
        #             C[-1].append(m_ab[1])
        #             CI.append(C)
        #
        # self.all=CI
        # print ('CI',CI)
        self.minList = self.minOut()
        # print (self.all)

    def minOut(self):
        minList = [0] * 34
        for i in range(34):
            if i in [0, 9, 18]:
                minList[i] = self.leftNum[i] * 2 + self.leftNum[i + 1] + self.leftNum[i + 2]
            elif i in [8, 17, 26]:
                minList[i] = self.leftNum[i] * 2 + self.leftNum[i - 1] + self.leftNum[i - 2]
            elif i in [1, 10, 19]:
                minList[i] = self.leftNum[i - 1] + self.leftNum[i] * 2 + self.leftNum[i + 1] + self.leftNum[i + 2]
            elif i in [7, 16, 25]:
                minList[i] = self.leftNum[i - 2] + self.leftNum[i - 1] + self.leftNum[i] * 2 + self.leftNum[i + 1]
            elif i >= 27:
                minList[i] = self.leftNum[i]
            else:
                minList[i] = self.leftNum[i - 2] + self.leftNum[i - 1] + self.leftNum[i] * 2 + self.leftNum[i + 1] + \
                             self.leftNum[i + 2]
        return minList

    def inChild(self, node, newNode):
        """
        判断搜索树结点是否已经创建，用于重复结点的判断
        :param node: 父结点
        :param newNode: 新创建的结点
        :return: 是否已经创建
        """
        # flag=False
        # node的类型是出牌结点，子结点为抓牌结点，抓牌为t2
        if node.type == 1:
            for c in node.children:
                if c.equal(newNode):
                    return c
        # node的类型时抓牌结点,子结点为出牌结点，即出的牌在子节点中
        if node.type == 2:
            for c in node.children:
                if c.equal(newNode):
                    return c
        return None

    def get_effective_cards_w(self, dz_set=[]):
        """
        有效牌及其概率获取
        :param dz_set: 搭子集合 list[[]],剩余牌　[]
        :param left_num: 有效牌集合[], 有效牌概率　[]
        :return:
        """
        left_num = self.leftNum
        cards_num = self.remainNum
        effective_cards = []
        w = []
        for dz in dz_set:
            if len(dz) == 1:
                effective_cards.append(dz[0])
                w.append(float(left_num[MJ.convert_hex2index(dz[0])]) / cards_num)
            elif dz[1] == dz[0]:
                effective_cards.append(dz[0])
                w.append(float(
                    left_num[MJ.convert_hex2index(dz[0])]) / cards_num * 8.1)  # 修改缩进,发现致命错误panic 忘了写float,这里写６是因为评估函数计算的缺陷

            elif dz[1] == dz[0] + 1:
                if int(dz[0]) & 0x0F == 1:
                    effective_cards.append(dz[0] + 2)
                    w.append(float(left_num[MJ.convert_hex2index(dz[0] + 2)]) / cards_num * 2)
                elif int(dz[0]) & 0x0F == 8:
                    effective_cards.append((dz[0] - 1))
                    w.append(float(left_num[MJ.convert_hex2index(dz[0] - 1)]) / cards_num * 2)
                else:
                    effective_cards.append(dz[0] - 1)
                    effective_cards.append(dz[0] + 2)
                    w.append(float(left_num[MJ.convert_hex2index(int(dz[0]) - 1)] + left_num[
                        MJ.convert_hex2index(int(dz[0]) + 2)]) / cards_num * 2)
            elif dz[1] == dz[0] + 2:
                effective_cards.append(dz[0] + 1)
                w.append(float(left_num[MJ.convert_hex2index(int(dz[0]) + 1)]) / cards_num * 2)
        return effective_cards, w

    def getEffectiveCards(self, dz):
        """
        功能：获取搭子的有效牌，用于抓牌结点的扩展
        思路：特定情景下，计算搭子的有效牌
        :param dz: 搭子
        :return: 有效牌集合
        """
        # 获取有效牌,输入为搭子集合,
        combineCards = []

        # 单张牌的扩展,todo 只扩展将牌
        if len(dz) == 1:
            fCard = dz[0]
            # combineCards.append([fCard,[fCard, fCard]])
            # if fCard == self.kingCard:
            #     combineCards.append([fCard, [fCard, fCard]])
            if fCard > 0x30 or fCard == self.kingCard:
                combineCards.append([fCard, [fCard, fCard]])
            elif fCard & 0x0f == 1:
                combineCards.append([fCard + 1, [fCard, fCard + 1]])
                combineCards.append([fCard + 2, [fCard, fCard + 2]])
                combineCards.append([fCard, [fCard, fCard]])
            elif fCard & 0x0f == 2:
                combineCards.append([fCard - 1, [fCard - 1, fCard]])
                combineCards.append([fCard, [fCard, fCard]])
                combineCards.append([fCard + 1, [fCard, fCard + 1]])
                combineCards.append([fCard + 2, [fCard, fCard + 2]])
            elif fCard & 0x0f == 9:
                combineCards.append([fCard - 2, [fCard - 2, fCard]])
                combineCards.append([fCard - 1, [fCard - 1, fCard]])
                combineCards.append([fCard, [fCard, fCard]])
            elif fCard & 0x0f == 8:
                combineCards.append([fCard - 2, [fCard - 2, fCard]])
                combineCards.append([fCard - 1, [fCard - 1, fCard]])
                combineCards.append([fCard, [fCard, fCard]])
                combineCards.append([fCard + 1, [fCard, fCard + 1]])

            else:
                combineCards.append([fCard - 2, [fCard - 2, fCard]])
                combineCards.append([fCard - 1, [fCard - 1, fCard]])
                combineCards.append([fCard, [fCard, fCard]])
                combineCards.append([fCard + 1, [fCard, fCard + 1]])
                combineCards.append([fCard + 2, [fCard, fCard + 2]])
        elif dz[1] == dz[0]:
            combineCards.append([dz[0], [dz[0], dz[0], dz[0]]])

        elif dz[1] == dz[0] + 1:
            if int(dz[0]) & 0x0F == 1:
                combineCards.append([dz[0] + 2, [dz[0], dz[0] + 1, dz[0] + 2]])
            elif int(dz[0]) & 0x0F == 8:
                combineCards.append([dz[0] - 1, [dz[0] - 1, dz[0], dz[0] + 1]])
            else:
                combineCards.append([dz[0] - 1, [dz[0] - 1, dz[0], dz[0] + 1]])
                combineCards.append([dz[0] + 2, [dz[0], dz[0] + 1, dz[0] + 2]])
        elif dz[1] == dz[0] + 2:
            combineCards.append([dz[0] + 1, [dz[0], dz[0] + 1, dz[0] + 2]])

        return combineCards

    def expandNode_2(self, node, ocards, t2N, kingNum=0, kz=[], sz=[]):
        """
        功能：结点扩展方法
        思路：递归结点扩展，先判断是否已经胡牌，若已经胡牌则停止扩展，再判断是否超过搜索深度,若是则停止扩展
            对出牌结点进行出牌扩展，直接将出牌集合加入到扩展策略，若本路径的出牌集合已空，则分别将２Ｎ或宝牌加入到出牌集合，再次递归。出牌结点创建后需更新所有的出牌结点信息，再次递归
            对抓牌结点进行抓牌扩展，直接将２Ｎ的有效牌加入到抓牌结点，若２Ｎ已空，则遍历出牌结点，获取该张牌的邻近牌，加入到２Ｎ中，再次递归。抓牌结点创建后，需更新抓牌结点信息，再次递归
        :param node: 本次需扩展的结点
        :param ocards: 出牌集合
        :param t2N: ２Ｎ集合
        :param kingNum: 未使用的宝数量
        :param baoHuanYuan: 是否作为宝还原进行扩展
        :param kz: 顺子
        :param sz: 刻子
        :return: 搜索树
        """
        # node.nodeInfo()

        if len(node.sz) + len(node.kz) == 4 and node.jiang != 0x00:
            # #少搜索一层的奖励，×２概率
            # if self.kingNum!=0:
            #     if node.level==self.xts*2:
            #         node.rate*=2
            #         return
            return

        # 宝吊多一层　
        if self.kingNum > 0 and node.feiKingNum + node.baoHuanYuan < self.kingNum + self.fei_king:
            if node.level >= (self.xts + 1) * 2:
                node.rate = 0
                return
        else:
            if node.level >= (self.xts) * 2:
                node.rate = 0
                return

        # 出牌结点
        if node.type == 2:
            # 当ocards为空时，分支为其中一个２Ｎ或者kingCard
            if ocards == []:
                # 分支１：t2N添加到ocards中
                if t2N != []:

                    # _,t2Nw = self.get_effective_cards_w(t2N)
                    # min_w_set=[]
                    # min_w=min(t2Nw)
                    # for i in range(len(t2N)):
                    #     if t2Nw[i]==min_w:
                    #         min_w_set.append(t2N[i])
                    # for t2 in min_w_set:
                    #     ocardsCP = copy.copy(t2)
                    #     t2NCP = copy.deepcopy(t2N)
                    #     t2NCP.remove(t2)
                    #     # 更新了ocards,t2N
                    # 全遍历，将所有２Ｎ轮流加入到ocards中
                    for t2 in t2N:
                        t2NCP = MJ.deepcopy(t2N)
                        t2NCP.remove(t2)
                        ocardsCP = copy.copy(ocards)
                        ocardsCP.extend(t2)
                        self.expandNode(node, ocardsCP, t2NCP, kingNum=kingNum, kz=kz, sz=sz)
                # 分支２　：kingCard加入到ocards中
                if kingNum != 0:
                    # 当有ab/ac时，不出宝牌
                    # for t2 in t2N:
                    # if t2[0]+2==t2[1]:
                    #     return

                    ocardsCPaddKing = [self.kingCard]
                    # print (ocardsCPaddKing)
                    self.expandNode(node, ocardsCPaddKing, t2N, kingNum=kingNum - 1, kz=kz, sz=sz)
                # 结束分支
                return
            # 胡牌多宝时，将宝放入ocards中，看是否宝吊
            # elif kingNum >= 2:
            #     ocards_KingMore2 = copy.copy(ocards)
            #     ocards_KingMore2.append(self.kingCard)
            #     self.expandNode(node, ocards_KingMore2, t2N, kingNum=kingNum - 1, baoHuanYuan=baoHuanYuan, kz=kz, sz=sz)
            #     return
            else:
                ocardsTMP = copy.copy(ocards)
                # t2NCP = t2N

            # 极小值出牌 merit 加快搜索树效率，可能会导致遗漏部分情况
            # min_ocards_w=[]
            # for tile in ocardsTMP:
            #     min_ocards_w.append(self.minList[convert_hex2index(tile)])
            # min_ocards=[]
            # min_w=min(min_ocards_w)
            #
            # for i in range(len(min_ocards_w)):
            #     if min_ocards_w[i]==min_w:
            #         min_ocards.append(ocardsTMP[i])

            # ocardsTMP=min_ocards

            for out in ocardsTMP:
                # 已经摸过的牌，不需要再出
                if out in node.formerCatchCards:
                    continue
                # if out==self.op_card:
                #     continue
                ocardsCP = copy.copy(ocardsTMP)
                ocardsCP.remove(out)

                cardsCP = copy.copy(node.cards)
                cardsCP.remove(out)

                oNode = OutNode(cards=cardsCP, outCard=out, level=node.level + 1,
                                dgRate=self.dgtable, kingCard=self.kingCard, t2N=t2N, ocards=ocardsCP,
                                baoHuanYuan=node.baoHuanYuan)

                oNode.feiKingNum = node.feiKingNum
                if out == self.kingCard:
                    oNode.feiKingNum += 1
                oNode.kz = copy.copy(node.kz)
                oNode.kz.extend(kz)
                oNode.sz = copy.copy(node.sz)
                oNode.sz.extend(sz)
                oNode.jiang = node.jiang

                # 重复结点检测，如果子结点与现在要扩充的结点一致，则用子结点代替现有结点进行扩充
                # child = self.inChild(node, oNode)
                # if child != None:
                # print('hello', out)
                # continue
                # print ('inChild', child.type)
                # self.expandNode(child, ocardsCP, t2NCP)
                # continue
                # 更新出抓牌状态
                oNode.formerCatchCards = copy.copy(node.formerCatchCards)
                oNode.formerOutCards = copy.copy(node.formerOutCards)
                oNode.formerOutCards.append(out)
                oNode.formerOutCards.sort()

                oNode.setParent(node)
                node.addChild(oNode)

                oNode.kz.sort()
                oNode.sz.sort()
                # if oNode.jiang!=0:
                self.expandNode(oNode, ocardsCP, t2N, kingNum=kingNum)
                # elif kingNum!=0:
                #     有宝，分为将为宝与宝吊打法

        # 抓牌结点
        if node.type == 1:

            # 当t2N为空时，分支为将ocards中的一张牌加入到t2N中，或将kingCard加入到t2N中
            if t2N == []:
                # 分支１：将ocards中的一张牌加入到t2N中
                if ocards != []:
                    for card in ocards:
                        # print ('ocardsCP',ocardsCP)
                        t2NCP = [[card]]
                        ocardsCP = copy.copy(ocards)
                        ocardsCP.remove(card)
                        self.expandNode(node, ocardsCP, t2NCP, kingNum=kingNum, kz=kz, sz=sz)  # continue
                # 分支２　当ocards也为空，但是kingNum不为空时，将kingCard加入到t2N中，这时已经宝吊胡牌了
                if ocards == [] and kingNum != 0:
                    t2NCP = [[self.kingCard]]
                    self.expandNode(node, ocards, t2NCP, kingNum=kingNum - 1, kz=kz, sz=sz)
                return

            # 正式处理抓牌结点
            else:
                # ocardsCP = ocards
                t2NCPTMP = t2N

            # 极大值抓牌
            # t2Nw=[]
            # for t2 in t2NCPTMP:
            #
            #     if str(t2) in self.t2Nw_Set.keys():
            #         t2Nw.append(self.t2Nw_Set[str(t2)])
            #     else:
            #         _,w=self.get_effective_cards_w([t2])
            #         t2Nw.append(w[0])
            # maxw=max(t2Nw)
            # maxw_t2N=[]
            # for i in range(len(t2NCPTMP)):
            #     if t2Nw[i]==maxw:
            #         maxw_t2N.append(t2NCPTMP[i])

            for t2 in t2NCPTMP:
                # print ('t2NCPTMP',t2NCPTMP)
                t2NCP = MJ.deepcopy(t2NCPTMP)
                t2NCP.remove(t2)

                combineCards = self.getEffectiveCards(t2)
                # print ('combineCards',combineCards)
                if combineCards == []:
                    print('Error combineCards is []')
                else:
                    for e in combineCards:  # e[0] catchcard e[1] t2N
                        # 已经出过的牌，不需要再摸到。这样路径会变长没有意义
                        if e[0] in node.formerOutCards:
                            continue

                        # #宝还原，让node的父结点生成一个复制结点
                        if kingNum != 0 and e[0] == self.kingCard:
                            # nodeCopy = copy.deepcopy(node)
                            t2N_BHY = MJ.deepcopy(t2N)
                            t2N_BHY.remove(t2)
                            oNode = OutNode(cards=node.cards, outCard=node.outCard, level=node.level,
                                            dgRate=self.dgtable,
                                            kingCard=self.kingCard, t2N=t2N_BHY, ocards=ocards,
                                            baoHuanYuan=node.baoHuanYuan + 1)
                            # oNode.rate=1
                            oNode.feiKingNum = node.feiKingNum
                            oNode.kz = copy.copy(node.kz)
                            # oNode.kz.extend(kz)
                            oNode.sz = copy.copy(node.sz)
                            # oNode.sz.extend(sz)
                            oNode.jiang = node.jiang

                            oNode.formerCatchCards = copy.copy(node.formerCatchCards)
                            oNode.formerOutCards = copy.copy(node.formerOutCards)

                            oNode.setParent(node.parent)
                            node.parent.addChild(oNode)

                            # 更新结点信息　

                            if len(e[1]) == 2:

                                t2N_BHY.append(e[1])

                            elif len(e[1]) == 3:

                                if e[1][0] == e[1][1]:
                                    oNode.kz.append(e[1][0])
                                    oNode.kz.sort()
                                else:
                                    oNode.sz.append(e[1][0])
                                    oNode.sz.sort()
                            self.expandNode(oNode, ocards, t2N_BHY, kingNum=kingNum - 1, kz=[], sz=[])
                            continue

                        cardsCP = copy.copy(node.cards)
                        cardsCP.append(e[0])
                        cardsCP.sort()

                        cNode = CatchNode(cards=cardsCP, catchCard=e[0], leftNum=self.leftNum, remainNum=self.remainNum,
                                          t2=t2, level=node.level + 1,
                                          kingCard=self.kingCard, t2N=t2NCP, ocards=ocards,
                                          baoHuanYuan=node.baoHuanYuan)
                        # todo 可能存在ｂｕｇ
                        # if self.xts == 0 and t2NCP == [] and ocardsCP.count(self.kingCard) + kingNum >= 2:
                        #     cNode.catchCard = self.kingCard
                        #     cNode.rate = 1

                        cNode.feiKingNum = node.feiKingNum
                        cNode.kz = copy.copy(node.kz)
                        cNode.kz.extend(kz)
                        cNode.sz = copy.copy(node.sz)
                        cNode.sz.extend(sz)
                        cNode.jiang = node.jiang

                        t2NCP2 = MJ.deepcopy(t2NCP)
                        if len(e[1]) == 3:
                            if e[1][0] == e[1][1]:
                                cNode.kz.append(e[1][0])
                            else:
                                cNode.sz.append(e[1][0])
                        elif len(e[1]) == 2:
                            # if e[1][0] == e[1][1]:

                            t2NCP2.append(e[1])

                        # 胡牌判断
                        # 已胡牌，补充信息
                        # kingNumall = ocardsCP.count(self.kingCard) + kingNum
                        if len(cNode.kz) + len(cNode.sz) == 4:
                            if (len(t2NCP2) == 1 and t2NCP2[0][0] == t2NCP2[0][1]):  # 普通无宝胡牌，包括了宝吊（搜索时另一张牌也赋予了宝牌值）的情况
                                # if baoHuanYuan and self.kingCard in cNode.cards:
                                #     cNode.baoHuanYuan = True
                                cNode.jiang = t2NCP2[0][0]

                            elif kingNum == 2:  # 宝还原　宝做将　胡牌
                                #     cNode.baoHuanYuan = True
                                cNode.jiang = self.kingCard
                                # elif self.xts == 0 and kingNumall == 1:  # 飞宝后这里会使搜索多一层，todo 这里应该搜索不到吧
                                #     cNode.jiang = self.kingCard

                        # 多宝胡牌判断
                        kingNum_remain = kingNum
                        trans_t2N = []
                        if kingNum >= 2:
                            # 一张宝做宝吊，其他宝牌做任意牌

                            useking = kingNum - 1  # 宝吊牌

                            t3NKz = []
                            t3NSz = []
                            for i in range(len(t2NCP2)):
                                # eFCards = self.getEffectiveCards(t2NCP[i])
                                if t2NCP2[i][0] == t2NCP2[i][1]:
                                    t3NKz.append(t2NCP2[i][0])

                                else:
                                    if t2NCP2[i][0] & 0x0f == 8:
                                        t3NSz.append(t2NCP2[i][0] - 1)

                                    else:
                                        t3NSz.append(t2NCP2[i][0])
                                trans_t2N.append(t2NCP2[i])
                                useking -= 1

                            if useking >= 0:
                                # 上述处理，已经在２Ｎ中使用了宝牌变成了３N,所以这里必须有２个以上的宝牌才能凑成３Ｎ　
                                # 由于４宝会直接杠掉，这里不处理
                                if useking >= 2:
                                    # noKingCard = 0
                                    for card in ocards:
                                        if card != self.kingCard:
                                            # noKingCard = card
                                            if useking - 2 >= 0:
                                                t3NKz.append(card)
                                                useking -= 2
                                            else:
                                                break
                                                # if noKingCard != 0:
                                                #     t3NKz.append(noKingCard)
                                                #     useking-=2

                                if len(cNode.kz) + len(cNode.sz) + len(t3NSz) + len(t3NKz) == 4:
                                    cNode.kz.extend(t3NKz)
                                    cNode.sz.extend(t3NSz)
                                    # if baoHuanYuan and self.kingCard in cNode.cards:
                                    #     cNode.baoHuanYuan = True
                                    # 所有的２Ｎ都已用宝牌配完，这里直接置[]
                                    for t2tmp in trans_t2N:
                                        t2NCP2.remove(t2tmp)
                                    kingNum_remain = useking  # 填胡了，才将宝牌更新

                        # child = self.inChild(node, cNode)
                        # 重复结点检测，如果子结点与现在要扩充的结点一致，则用子结点代替现有结点进行扩充
                        # if child != None:
                        # self.expandNode(child, ocardsCP, t2NCP)
                        # continue
                        # 更新出抓牌状态
                        cNode.formerCatchCards = copy.copy(node.formerCatchCards)
                        cNode.formerCatchCards.append(cNode.catchCard)
                        cNode.formerCatchCards.sort()
                        cNode.formerOutCards = copy.copy(node.formerOutCards)
                        cNode.setParent(node)
                        node.addChild(cNode)

                        # 排序
                        cNode.kz.sort()
                        cNode.sz.sort()
                        self.expandNode(cNode, ocards, t2NCP2, kingNum=kingNum_remain)

    def expandNode_(self, node, ocards, t2N, kingNum=0, baoHuanYuan=False, kz=[], sz=[]):
        # print('expandNode','node.kz,sz,jiang',node.kz,node.sz,node.jiang,'kz,sz',kz,sz,'ocards,t2N',ocards,t2N,'node.cards',node.cards,'node.type,level,rate',node.level,node.type,node.rate)

        if node.level >= self.xts * 2:  # todo 此处修改为深度为ｘｔｓ　，不再为ｘｔｓ＋１
            # if ocards==[] and len(t2N)==1 and t2N[0][0]==t2N[0][1]:
            # 胡牌
            if len(node.sz) + len(node.kz) == 4 and node.jiang != 0x00:
                return
            else:
                node.rate = 0
                return

        # 出牌结点
        if node.type == 2:
            if ocards == [] and t2N != []:
                ocardsTMP = t2N[-1]
                t2NCP = MJ.deepcopy(t2N)
                t2NCP.remove(t2N[-1])
            else:
                ocardsTMP = ocards
                t2NCP = t2N
            for out in ocardsTMP:
                # if out==self.op_card:
                #     continue

                ocardsCP = copy.copy(ocardsTMP)
                ocardsCP.remove(out)

                cardsCP = copy.copy(node.cards)
                cardsCP.remove(out)
                oNode = OutNode(cards=cardsCP, outCard=out, level=node.level + 1, dgRate=self.dgtable,
                                kingCard=self.kingCard)
                oNode.kz = copy.copy(node.kz)
                oNode.kz.extend(kz)
                oNode.sz = copy.copy(node.sz)
                oNode.sz.extend(sz)

                # 重复结点检测，如果子结点与现在要扩充的结点一致，则用子结点代替现有结点进行扩充
                # child = self.inChild(node, oNode)
                # if child != None:
                # print('hello', out)
                # continue
                # print ('inChild', child.type)
                # self.expandNode(child, ocardsCP, t2NCP)
                # continue
                # 更新出抓牌状态
                oNode.formerCatchCards = copy.copy(node.formerCatchCards)
                oNode.formerOutCards = copy.copy(node.formerOutCards)
                oNode.formerOutCards.append(out)
                oNode.formerOutCards.sort()

                oNode.setParent(node)
                node.addChild(oNode)

                oNode.kz.sort()
                oNode.sz.sort()
                self.expandNode(oNode, ocardsCP, t2NCP)

        # 抓牌结点
        if node.type == 1:
            # 近胡牌状态，只有２张废牌，另一张做将
            if t2N == [] and len(ocards) == 1:
                t2NCPTMP = [copy.copy(ocards)]
                ocardsCP = []
            elif t2N != []:
                ocardsCP = ocards
                t2NCPTMP = t2N
            else:  # todo 无成型的２N抓，现在省略掉了
                # ocardsCP = ocards
                # t2NCPTMP = t2N
                print('Error expandNode', self.cards, node.cards, ocards, t2N, node.level)
                node.rate = 0
                return
            for t2 in t2NCPTMP:
                t2NCP = MJ.deepcopy(t2NCPTMP)
                t2NCP.remove(t2)

                effectiveCards = self.getEffectiveCards(t2)
                if effectiveCards == []:
                    print('Error effectiveCards is []')
                else:
                    for e in effectiveCards:
                        cardsCP = copy.copy(node.cards)
                        cardsCP.append(e[0])
                        cardsCP.sort()
                        cNode = CatchNode(cards=cardsCP, catchCard=e[0], leftNum=self.leftNum,
                                          remainNum=self.remainNum,
                                          t2=t2, level=node.level + 1, kingCard=self.kingCard)
                        cNode.kz = copy.copy(node.kz)
                        cNode.kz.extend(kz)
                        cNode.sz = copy.copy(node.sz)
                        cNode.sz.extend(sz)

                        t2tmp = copy.copy(t2)
                        t2tmp.append(e)
                        t2tmp.sort()

                        # 已胡牌,这里是补将牌
                        if len(t2tmp) == 2:
                            cNode.jiang = t2tmp[0]
                        elif t2tmp[0] == t2tmp[1]:
                            cNode.kz.append(t2tmp[0])
                        else:
                            cNode.sz.append(t2tmp[0])
                        # 已胡牌，这里不是补将牌，补的其他２Ｎ
                        if len(cNode.kz) + len(cNode.sz) == 4 and ocardsCP == [] and len(t2NCP) == 1 and t2NCP[0][0] == \
                                t2NCP[0][1]:
                            # if len(cNode.sz)+len(cNode.kz)!=5:
                            #     print ('No hu Error',cNode.kz,cNode.sz,ocardsCP,t2NCP,self.cards,self.suits,cNode.level,node.level)
                            # if node.kz==[24] and node.sz==[]
                            cNode.jiang = t2NCP[0][0]
                            # t2NCP=[]
                            # child = self.inChild(node, cNode)
                            # 重复结点检测，如果子结点与现在要扩充的结点一致，则用子结点代替现有结点进行扩充
                            # if child != None:
                            # self.expandNode(child, ocardsCP, t2NCP)
                            # continue
                        # 更新出抓牌状态
                        cNode.formerCatchCards = copy.copy(node.formerCatchCards)
                        cNode.formerCatchCards.append(e)
                        cNode.formerCatchCards.sort()
                        cNode.formerOutCards = copy.copy(node.formerOutCards)
                        cNode.setParent(node)
                        node.addChild(cNode)
                        # 排序
                        cNode.kz.sort()
                        cNode.sz.sort()
                        self.expandNode(cNode, ocardsCP, t2NCP)

    def expandNode(self, node, ocards, t2N, kingNum=0, kz=[], sz=[], xts=14):
        """
        功能：结点扩展方法
        思路：递归结点扩展，先判断是否已经胡牌，若已经胡牌则停止扩展，再判断是否超过搜索深度,若是则停止扩展
            对出牌结点进行出牌扩展，直接将出牌集合加入到扩展策略，若本路径的出牌集合已空，则分别将２Ｎ或宝牌加入到出牌集合，再次递归。出牌结点创建后需更新所有的出牌结点信息，再次递归
            对抓牌结点进行抓牌扩展，直接将２Ｎ的有效牌加入到抓牌结点，若２Ｎ已空，则遍历出牌结点，获取该张牌的邻近牌，加入到２Ｎ中，再次递归。抓牌结点创建后，需更新抓牌结点信息，再次递归
        :param node: 本次需扩展的结点
        :param ocards: 出牌集合
        :param t2N: ２Ｎ集合
        :param kingNum: 未使用的宝数量
        :param baoHuanYuan: 是否作为宝还原进行扩展
        :param kz: 顺子
        :param sz: 刻子
        :return: 搜索树
        """
        # node.nodeInfo()

        if len(node.sz) + len(node.kz) == 4 and node.jiang != 0x00 and node.type == 2:
            # #少搜索一层的奖励，×２概率
            # if self.kingNum!=0:
            #     if node.level==self.xts*2:
            #         node.rate*=2
            #         return
            # if node.jiang!=self.kingCard:
            #     return
            # else:
            #     if ocards==[] and len(t2N)==1 and t2N[0][1]==self.kingCard:
            #         return
            return

        # 宝吊多一层　
        if self.kingNum > 0 and node.feiKingNum + node.baoHuanYuan < self.kingNum + self.fei_king:

            # if node.jiang==self.kingCard:
            if node.level >= (xts + 1) * 2:
                node.rate = 0
                return
        else:
            if node.level >= (xts) * 2:
                node.rate = 0
                return

        # 出牌结点
        if node.type == 2:
            # 当ocards为空时，分支为其中一个２Ｎ或者kingCard

            if ocards == []:
                # 分支１：t2N添加到ocards中
                if t2N != []:

                    # _,t2Nw = self.get_effective_cards_w(t2N)
                    # min_w_set=[]
                    # min_w=min(t2Nw)
                    # for i in range(len(t2N)):
                    #     if t2Nw[i]==min_w:
                    #         min_w_set.append(t2N[i])
                    # for t2 in min_w_set:
                    #     ocardsCP = copy.copy(t2)
                    #     t2NCP = copy.deepcopy(t2N)
                    #     t2NCP.remove(t2)
                    #     self.expandNode(node, ocardsCP, t2NCP, kingNum=kingNum, kz=kz, sz=sz,xts=xts)

                    #     # 更新了ocards,t2N
                    # 全遍历，将所有２Ｎ轮流加入到ocards中
                    for t2 in t2N:
                        t2NCP = MJ.deepcopy(t2N)
                        t2NCP.remove(t2)
                        ocardsCP = copy.copy(ocards)
                        ocardsCP.extend(t2)
                        self.expandNode(node, ocardsCP, t2NCP, kingNum=kingNum, kz=kz, sz=sz, xts=xts)
                # 分支２　：kingCard加入到ocards中
                if kingNum != 0:
                    # 当有ab/ac时，不出宝牌
                    # for t2 in t2N:
                    # if t2[0]+2==t2[1]:
                    #     return

                    ocardsCPaddKing = [self.kingCard]
                    # print (ocardsCPaddKing)
                    self.expandNode(node, ocardsCPaddKing, t2N, kingNum=kingNum - 1, kz=kz, sz=sz, xts=xts)
                # 结束分支
                return
            # 胡牌多宝时，将宝放入ocards中，看是否宝吊
            # elif kingNum >= 2:
            #     ocards_KingMore2 = copy.copy(ocards)
            #     ocards_KingMore2.append(self.kingCard)
            #     self.expandNode(node, ocards_KingMore2, t2N, kingNum=kingNum - 1, baoHuanYuan=baoHuanYuan, kz=kz, sz=sz)
            #     return
            else:
                ocardsTMP = copy.copy(ocards)
                # t2NCP = t2N

            # 极小值出牌 merit 加快搜索树效率，可能会导致遗漏部分情况
            # min_ocards_w=[]
            # for tile in ocardsTMP:
            #     min_ocards_w.append(self.minList[convert_hex2index(tile)])
            # min_ocards=[]
            # min_w=min(min_ocards_w)
            #
            # for i in range(len(min_ocards_w)):
            #     if min_ocards_w[i]==min_w:
            #         min_ocards.append(ocardsTMP[i])

            # ocardsTMP=min_ocards

            for out in ocardsTMP:
                # 已经摸过的牌，不需要再出
                if out in node.formerCatchCards:
                    continue
                # if out==self.op_card:
                #     continue
                ocardsCP = copy.copy(ocardsTMP)
                ocardsCP.remove(out)

                cardsCP = copy.copy(node.cards)
                cardsCP.remove(out)

                oNode = OutNode(cards=cardsCP, outCard=out, level=node.level + 1,
                                dgRate=self.dgtable, kingCard=self.kingCard, t2N=t2N, ocards=ocardsCP,
                                baoHuanYuan=node.baoHuanYuan)
                # if oNode.level==1:
                #     oNode.firstOutCard=out
                # else:
                #     oNode.firstOutCard=node.firstOutCard

                oNode.feiKingNum = node.feiKingNum
                if out == self.kingCard:
                    oNode.feiKingNum += 1
                    if self.kingNum > 1 and kingNum > 1:
                        xts += 1
                oNode.kz = copy.copy(node.kz)
                oNode.kz.extend(kz)
                oNode.sz = copy.copy(node.sz)
                oNode.sz.extend(sz)
                oNode.jiang = node.jiang

                # 重复结点检测，如果子结点与现在要扩充的结点一致，则用子结点代替现有结点进行扩充
                # child = self.inChild(node, oNode)
                # if child != None:
                # print('hello', out)
                # continue
                # print ('inChild', child.type)
                # self.expandNode(child, ocardsCP, t2NCP)
                # continue
                # 更新出抓牌状态
                oNode.formerCatchCards = copy.copy(node.formerCatchCards)
                oNode.formerOutCards = copy.copy(node.formerOutCards)
                oNode.formerOutCards.append(out)
                oNode.formerOutCards.sort()

                oNode.setParent(node)
                node.addChild(oNode)

                oNode.kz.sort()
                oNode.sz.sort()
                # if oNode.jiang!=0:

                self.expandNode(oNode, ocardsCP, t2N, kingNum=kingNum, xts=xts)
                # elif kingNum!=0:
                #     有宝，分为将为宝与宝吊打法

        # 抓牌结点
        if node.type == 1:

            # 当t2N为空时，分支为将ocards中的一张牌加入到t2N中，或将kingCard加入到t2N中
            # if t2N==[]:
            #     print (t2N)
            if t2N == []:
                # 分支１：将ocards中的一张牌加入到t2N中
                if ocards != []:
                    for card in ocards:
                        # print ('ocardsCP',ocardsCP)
                        t2NCP = [[card]]
                        ocardsCP = copy.copy(ocards)
                        ocardsCP.remove(card)
                        self.expandNode(node, ocardsCP, t2NCP, kingNum=kingNum, xts=xts)  # continue
                # 分支２　当ocards也为空，但是kingNum不为空时，将kingCard加入到t2N中，这时已经宝吊胡牌了

                # print ('test',ocards,kingNum,node.jiang)
                if ocards == []:
                    if kingNum != 0:
                        t2NCP = [[self.kingCard]]
                        self.expandNode(node, ocards, t2NCP, kingNum=kingNum - 1, xts=xts)
                    elif node.jiang == self.kingCard:
                        t2NCP = [[self.kingCard]]
                        self.expandNode(node, ocards, t2NCP, kingNum=kingNum, xts=xts)
                return

            # 正式处理抓牌结点
            else:
                # ocardsCP = ocards
                t2NCPTMP = t2N

            # 极大值抓牌
            # t2Nw=[]
            # for t2 in t2NCPTMP:
            #
            #     if str(t2) in self.t2Nw_Set.keys():
            #         t2Nw.append(self.t2Nw_Set[str(t2)])
            #     else:
            #         _,w=self.get_effective_cards_w([t2])
            #         t2Nw.append(w[0])
            # maxw=max(t2Nw)
            # maxw_t2N=[]
            # for i in range(len(t2NCPTMP)):
            #     if t2Nw[i]==maxw:
            #         maxw_t2N.append(t2NCPTMP[i])

            for t2 in t2NCPTMP:
                # print ('t2NCPTMP',t2NCPTMP)
                t2NCP = MJ.deepcopy(t2NCPTMP)
                t2NCP.remove(t2)

                combineCards = self.getEffectiveCards(t2)
                # print ('combineCards',combineCards)
                if combineCards == []:
                    print('Error combineCards is []')
                else:
                    for e in combineCards:  # e[0] catchcard e[1] t2N
                        # 已经出过的牌，不需要再摸到。这样路径会变长没有意义
                        if e[0] in node.formerOutCards:
                            continue

                        # #宝还原，让node的父结点生成一个复制结点
                        if len(t2) == 2 and e[0] == self.kingCard:
                            BHY_ocards = copy.copy(ocards)
                            BHY_kingNum = kingNum
                            if kingNum != 0:
                                BHY_kingNum = kingNum - 1
                            elif self.kingCard in ocards:
                                BHY_ocards.remove(self.kingCard)
                            else:
                                continue
                            # node.nodeInfo()

                            # nodeCopy = copy.deepcopy(node)
                            t2N_BHY = MJ.deepcopy(t2N)
                            t2N_BHY.remove(t2)
                            oNode = OutNode(cards=node.cards, outCard=node.outCard, level=node.level,
                                            dgRate=self.dgtable,
                                            kingCard=self.kingCard, t2N=t2N_BHY, ocards=BHY_ocards,
                                            baoHuanYuan=node.baoHuanYuan + 1)

                            # if oNode.level == 1:
                            #     oNode.firstOutCard = out
                            # else:
                            oNode.firstOutCard = node.firstOutCard

                            # oNode.rate=1
                            oNode.feiKingNum = node.feiKingNum
                            oNode.kz = copy.copy(node.kz)
                            # oNode.kz.extend(kz)
                            oNode.sz = copy.copy(node.sz)
                            # oNode.sz.extend(sz)
                            oNode.jiang = node.jiang

                            oNode.formerCatchCards = copy.copy(node.formerCatchCards)
                            oNode.formerOutCards = copy.copy(node.formerOutCards)

                            oNode.setParent(node.parent)
                            node.parent.addChild(oNode)

                            # 更新结点信息　

                            if len(e[1]) == 2:

                                t2N_BHY.append(e[1])

                            elif len(e[1]) == 3:

                                if e[1][0] == e[1][1]:
                                    oNode.kz.append(e[1][0])
                                    oNode.kz.sort()
                                else:
                                    oNode.sz.append(e[1][0])
                                    oNode.sz.sort()
                            # oNode.nodeInfo()
                            self.expandNode(oNode, BHY_ocards, t2N_BHY, kingNum=BHY_kingNum, kz=[], sz=[], xts=xts)
                            continue

                        cardsCP = copy.copy(node.cards)
                        cardsCP.append(e[0])
                        cardsCP.sort()

                        cNode = CatchNode(cards=cardsCP, catchCard=e[0], leftNum=self.leftNum, remainNum=self.remainNum,
                                          t2=t2, level=node.level + 1,
                                          kingCard=self.kingCard, t2N=t2NCP, ocards=ocards,
                                          baoHuanYuan=node.baoHuanYuan)
                        # todo 可能存在ｂｕｇ
                        # if self.xts == 0 and t2NCP == [] and ocardsCP.count(self.kingCard) + kingNum >= 2:
                        #     cNode.catchCard = self.kingCard
                        #     cNode.rate = 1
                        # if cNode.level == 1:
                        #     cNode.firstOutCard = out
                        # else:
                        #     cNode.firstOutCard = node.firstOutCard
                        cNode.feiKingNum = node.feiKingNum
                        cNode.kz = copy.copy(node.kz)
                        # cNode.kz.extend(kz)
                        cNode.sz = copy.copy(node.sz)
                        # cNode.sz.extend(sz)
                        cNode.jiang = node.jiang

                        t2NCP2 = MJ.deepcopy(t2NCP)
                        if len(e[1]) == 3:
                            if e[1][0] == e[1][1]:
                                cNode.kz.append(e[1][0])
                            else:
                                cNode.sz.append(e[1][0])
                        elif len(e[1]) == 2:
                            # if e[1][0] == e[1][1]:

                            t2NCP2.append(e[1])

                        # 胡牌判断
                        # 已胡牌，补充信息
                        # kingNumall = ocardsCP.count(self.kingCard) + kingNum
                        if len(cNode.kz) + len(cNode.sz) == 4:
                            if (len(t2NCP2) == 1 and t2NCP2[0][0] == t2NCP2[0][1]):  # 普通无宝胡牌，包括了宝吊（搜索时另一张牌也赋予了宝牌值）的情况
                                # if baoHuanYuan and self.kingCard in cNode.cards:
                                #     cNode.baoHuanYuan = True
                                cNode.jiang = t2NCP2[0][0]

                            elif kingNum == 2:  # 宝还原　宝做将　胡牌
                                #     cNode.baoHuanYuan = True
                                cNode.jiang = self.kingCard
                                # elif self.xts == 0 and kingNumall == 1:  # 飞宝后这里会使搜索多一层，todo 这里应该搜索不到吧
                                #     cNode.jiang = self.kingCard

                        # 多宝胡牌判断
                        kingNum_remain = kingNum
                        trans_t2N = []
                        if kingNum >= 2:
                            # 一张宝做宝吊，其他宝牌做任意牌

                            useking = kingNum - 1  # 宝吊牌

                            t3NKz = []
                            t3NSz = []
                            for i in range(len(t2NCP2)):
                                # eFCards = self.getEffectiveCards(t2NCP[i])
                                if t2NCP2[i][0] == t2NCP2[i][1]:
                                    t3NKz.append(t2NCP2[i][0])

                                else:
                                    if t2NCP2[i][0] & 0x0f == 8:
                                        t3NSz.append(t2NCP2[i][0] - 1)

                                    else:
                                        t3NSz.append(t2NCP2[i][0])
                                trans_t2N.append(t2NCP2[i])
                                useking -= 1

                            if useking >= 0:
                                # 上述处理，已经在２Ｎ中使用了宝牌变成了３N,所以这里必须有２个以上的宝牌才能凑成３Ｎ　
                                # 由于４宝会直接杠掉，这里不处理
                                if useking >= 2:
                                    # noKingCard = 0
                                    for card in ocards:
                                        if card != self.kingCard:
                                            # noKingCard = card
                                            if useking - 2 >= 0:
                                                t3NKz.append(card)
                                                useking -= 2
                                            else:
                                                break
                                                # if noKingCard != 0:
                                                #     t3NKz.append(noKingCard)
                                                #     useking-=2

                                if len(cNode.kz) + len(cNode.sz) + len(t3NSz) + len(t3NKz) == 4:
                                    cNode.kz.extend(t3NKz)
                                    cNode.sz.extend(t3NSz)
                                    # if baoHuanYuan and self.kingCard in cNode.cards:
                                    #     cNode.baoHuanYuan = True
                                    # 所有的２Ｎ都已用宝牌配完，这里直接置[]
                                    for t2tmp in trans_t2N:
                                        t2NCP2.remove(t2tmp)
                                    kingNum_remain = useking + 1  # 填胡了，才将宝牌更新 todo 忘了加１

                        # child = self.inChild(node, cNode)
                        # 重复结点检测，如果子结点与现在要扩充的结点一致，则用子结点代替现有结点进行扩充
                        # if child != None:
                        # self.expandNode(child, ocardsCP, t2NCP)
                        # continue
                        # 更新出抓牌状态
                        cNode.formerCatchCards = copy.copy(node.formerCatchCards)
                        cNode.formerCatchCards.append(cNode.catchCard)
                        cNode.formerCatchCards.sort()
                        cNode.formerOutCards = copy.copy(node.formerOutCards)
                        cNode.setParent(node)
                        node.addChild(cNode)

                        # 排序
                        cNode.kz.sort()
                        cNode.sz.sort()
                        self.expandNode(cNode, ocards, t2NCP2, kingNum=kingNum_remain, xts=xts)

    def generateTree(self):
        """
        功能：搜索树创建，用于初始化路径的相关变量，包括出牌集合　抓牌集合２Ｎ　顺子　刻子等
        思路：使用了组合信息进行创建和扩展树，将３Ｎ直接加入到结点信息中，不再处理，将２Ｎ加入到抓牌结点扩展策略集合中，将孤张leftCards加入到出牌结点扩展策略中
        并增加了宝还原处理
        """
        # if self.type==2:
        #     node = self.root
        # else:#扩展了ｏｐ中的结点
        #     node=self.root.children[0]

        node = self.root
        for a in self.all:
            # t2N = copy.deepcopy(a[2] + a[3])
            # efc_cards, t2_w = self.get_effective_cards_w(t2N)
            #
            # for i in range(len(t2N)):
            #     if str(t2N[i]) not in self.t2Nw_Set.keys():
            #         self.t2Nw_Set[str(t2N[i])]=t2_w[i]
            #     t2N[i].append(t2_w[i])
            #
            # t2N[:len(a[2])] = sorted(t2N[:len(a[2])], key=lambda k: k[2], reverse=True)
            # t2N[len(a[2]):] = sorted(t2N[len(a[2]):], key=lambda k: k[2], reverse=True)
            # # t2N[len():] = sorted(t2N[len(a[2]):], key=lambda k: k[2], reverse=True) #修改为１＋
            # # 扩展出牌结点
            ocards = copy.copy(a[-1])
            t2NCP = []
            t2NCP.extend(a[2] + a[3])

            # for t2 in t2N:
            #     t2NCP.append([t2[0], t2[1]])
            kz = []
            sz = []
            for k in a[0]:
                kz.append(k[0])
            for s in a[1]:
                sz.append(s[0])

            # print ('ocards', ocards, 't2NCP', t2NCP)

            # if self.kingNum>0:
            #     #溢出
            #     if len(t2NCP)>4-len(self.suits):
            #
            #         for ab in a[3]:
            #             if
            # t1=time.time()
            self.expandNode(node=node, ocards=ocards, t2N=t2NCP, kingNum=self.kingNum, kz=kz, sz=sz, xts=a[4])
            # t2=time.time()
            # print ('t21',t2-t1)
            # if self.kingNum!=0:
            #     # if self.kingNum<=2:
            #     #宝吊打法,最快胡牌
            #     node.jiang=self.kingCard
            #     self.expandNode(node=node, ocards=ocards, t2N=t2NCP, kingNum=self.kingNum-1, kz=kz, sz=sz,xts=self.xts)
            #     #飞宝打法，
            #     ocardsKing=copy.copy(ocards)

            # ocardsKing.append(self.kingCard)
            # self.expandNode(node=node, ocards=ocardsKing, t2N=t2NCP, kingNum=self.kingNum-1, kz=kz, sz=sz,xts=self.xts)

            #     if self.kingNum > 1:
            #         KN=self.kingNum - 1
            #     else:
            #         KN=self.kingNum
            #
            #
            #     for i in range(KN):
            #         ocardsKing.append(self.kingCard)
            #     #只有一张宝做宝吊，其他全部打掉
            #     if self.kingNum==1:
            #         for aa in a[2]:
            #             node.jiang = aa[0]
            #             t2NCP_rmJ = copy.deepcopy(t2NCP)
            #             t2NCP_rmJ.remove(aa)
            #             self.expandNode(node=node, ocards=ocardsKing, t2N=t2NCP_rmJ, kingNum=self.kingNum-KN, kz=kz, sz=sz,xts=self.xts+KN)
            #             return
            #     # #宝牌全部打掉
            #     else:
            #         # 留一宝
            #         node.jiang = self.kingCard
            #         self.expandNode(node=node, ocards=ocards, t2N=t2NCP, kingNum=self.kingNum-KN, kz=kz, sz=sz,xts=self.xts+KN)
            #         #
            #         #全打
            #         ocards_allKing=copy.copy(ocards)
            #         for i in range(self.kingNum):
            #             ocards_allKing.append(self.kingCard)
            #         for aa in a[2]:
            #             node.jiang = aa[0]
            #             t2NCP_rmJ = copy.deepcopy(t2NCP)
            #             t2NCP_rmJ.remove(aa)
            #             self.expandNode(node=node, ocards=ocards_allKing, t2N=t2NCP_rmJ, kingNum=0,kz=kz, sz=sz,xts=self.xts+self.kingNum)
            # #
            # elif len(a[2])!=0:
            #     #aa做将打法
            #     for aa in a[2]:
            #         node.jiang=aa[0]
            #         t2NCP_rmJ=copy.deepcopy(t2NCP)
            #         t2NCP_rmJ.remove(aa)
            #         # print ocards,t2NCP_rmJ
            #         self.expandNode(node=node, ocards=ocards, t2N=t2NCP_rmJ, kingNum=self.kingNum, kz=kz,sz=sz,xts=self.xts)
            #
            #     #aa不做将打法
            #     node.jiang=0
            #     self.expandNode(node=node, ocards=ocards, t2N=t2NCP, kingNum=self.kingNum, kz=kz,sz=sz,xts=self.xts)
            # else:
            #     node.jiang=0
            #     self.expandNode(node=node, ocards=ocards, t2N=t2NCP, kingNum=self.kingNum, kz=kz, sz=sz,xts=self.xts)

            # # 宝还原处理
            # if self.kingNum != 0:
            #     allBaoHuanYuan = pinghu(self.cards, self.suits, self.leftNum).sys_info_V3(self.cards, self.suits,
            #                                                                               self.leftNum)
            #     for a in allBaoHuanYuan:
            #         t2N = copy.deepcopy(a[2] + a[3])
            #         efc_cards, t2_w = pinghu(cards=self.cards, suits=self.suits,
            #                                  leftNum=self.leftNum).get_effective_cards_w(dz_set=t2N, left_num=self.leftNum)
            #         for i in range(len(t2N)):
            #             t2N[i].append(t2_w[i])
            #         t2N[:len(a[2])] = sorted(t2N[:len(a[2])], key=lambda k: k[2], reverse=True)
            #         t2N[len(a[2]):] = sorted(t2N[len(a[2]):], key=lambda k: k[2], reverse=True)
            #         # t2N[len():] = sorted(t2N[len(a[2]):], key=lambda k: k[2], reverse=True) #修改为１＋
            #         # 扩展出牌结点
            #         ocards = a[-1]
            #         t2NCP = []
            #         for t2 in t2N:
            #             t2NCP.append([t2[0], t2[1]])
            #         kz = []
            #         sz = []
            #         for k in a[0]:
            #             kz.append(k[0])
            #         for s in a[1]:
            #             sz.append(s[0])
            #         # 这里宝还原,将kingNum置０
            #         self.kingNum = 0
            #         kingNum = self.kingNum
            #         # for i in range(self.kingNum):
            #         #     ocards.append(self.kingCard)
            #         node.noUseKingNum = kingNum
            #         self.expandNode(node=node, ocards=ocards, t2N=t2NCP, kingNum=0, baoHuanYuan=True, kz=kz, sz=sz)

    def getRate(self, node, rate):
        """
        功能：对叶结点进行评估
        思路：递归计算评估值，当叶结点已经胡牌时，计算评估值＝胡牌概率×危险度×分数，并进行了去重处理，对具有相同的抓牌路径视为同一路径，对视为相同的路径只取最大的评估值的路径作为最后的路径
        :param node: 本次需要计算的结点
        :param rate: 本路径现有的评估值
        :return: 更新了类变量中各路径的评估值与路径的胡牌信息，包括stateSet　胡牌状态集合　scoreDict　分数集合
        """
        # print ('nodeInfo',node.kz,node.sz,node.jiang,node.rate,node.children == [] )
        # if node.level!=0:
        #     print ('getRate',node.level)
        children = node.children
        if children == []:

            # 胡牌结点
            # 有宝的树
            # print (node.feiKingNum,self.kingNum+self.fei_king,node.level)
            # if self.kingNum>0 and node.feiKingNum==self.kingNum+self.fei_king and node.level>self.xts*2:
            #     print (node.feiKingNum,self.kingNum+self.fei_king)
            #     return
            if len(node.sz) + len(node.kz) == 4 and node.jiang != 0 and node.rate != 0 and node.type == 2:
                # if
                # node.nodeInfo()
                # if node.rate != 0 and node.type == 2 and len(node.t2) == 2:
                #     # print('here')
                #     if node.t2[0] != node.t2[1]:
                #         # node.rate = float(self.leftNum[convert_hex2index(node.catchCard)]) / self.remainNum * 1
                #         node.rate *= 0.5
                #     else:
                #         node.rate *= 0.25  # print (node.rate)  # else:  #     node.rate*2.0/3

                if node.type == 2 and len(node.t2) == 2:
                    # if node.t2==[3,3]:
                    #     print (self.leftNum[convert_hex2index(node.catchCard)])
                    node.rate = float(self.leftNum[MJ.convert_hex2index(node.catchCard)]) / self.remainNum
                    if node.t2[0] == node.t2[1]:
                        node.rate *= 1.5

                        # print ('rate',node.rate,node.catchCard,self.leftNum[convert_hex2index(node.catchCard)])
                # else:
                #     print ('ERROR rate rate')
                #     return

                rate *= node.rate
                if rate != 0 and node.jiang == 0:
                    print ('getRate Error', node.cards, node.kz, node.sz, node.level)
                # todo 可以优化时间
                if rate != 0:
                    # if node.level==4:
                    #     print (node.nodeInfo(),node.parent.parent.nodeInfo())
                    catchCards = node.formerCatchCards
                    outCards = node.formerOutCards
                    state = []
                    state.append(node.kz)
                    state.append(node.sz)
                    state.append(node.jiang)
                    # if node.feiKingNum==0:
                    # print(node.feiKingNum)
                    fan = Fan2(node.kz, node.sz, node.jiang, node, node.feiKingNum,
                               self.kingNum + self.fei_king).fanDetect()
                    if fan > self.maxScore[0]:
                        # self.maxScore[2]=self.maxScore[1]
                        self.maxScore[1] = self.maxScore[0]
                        self.maxScore[0] = fan

                    # print ('fan',fan)
                    # print (rate)
                    score = rate * fan
                    # if catchCards==[8,23]:
                    #     print ("tree1",rate,fan,score)
                    state.append(fan)

                    # score = rate * (2 + sum(fanList))
                    # if [catchCards,outCards]==[[3, 8, 19], [4, 22, 23]]:
                    #     print node.t2,node.parent.parent.t2,node.parent.parent.parent.parent.t2
                    #     print node.rate,node.parent.parent.rate,node.parent.parent.parent.parent.rate

                    # if node.firstOutCard==0:
                    #     print ('firstOutCard Error')
                    for card in outCards:
                        # if card ==22:
                        #     # print ('state',state,score)
                        #     if state== [[24], [1, 7, 17, 39], 6, 0]:
                        #         print (score,node.rate,node.t2,node.parent.parent.rate,node.parent.parent.t2)
                        if card not in self.stateSet.keys():
                            self.stateSet[card] = [[], [], []]
                            self.stateSet[card][0].append(catchCards)
                            self.stateSet[card][1].append(outCards)
                            self.stateSet[card][2].append(state)

                            self.scoreDict[card] = []
                            self.scoreDict[card].append(score)
                        else:
                            if catchCards not in self.stateSet[card][0]:
                                self.stateSet[card][0].append(catchCards)
                                self.stateSet[card][1].append(outCards)
                                self.stateSet[card][2].append(state)
                                self.scoreDict[card].append(score)
                            else:

                                index = self.stateSet[card][0].index(catchCards)
                                if score > self.scoreDict[card][index]:
                                    self.scoreDict[card][index] = score
                                    # self.stateSet[card][0][index] = catchCards
                                    self.stateSet[card][1][index] = outCards
                                    self.stateSet[card][2][index] = state
                return
        else:
            rate *= node.rate
            for child in children:
                self.getRate(child, rate)

    def getCardScore(self):
        """
        功能：计算每张出牌的评估值，并输出评估值最高的牌作为最佳出牌
        思路：对类变量中的scoreDict　累加计算出牌的评估值
        :return: outCard　最佳出牌
        """
        # 建树
        t1 = time.time()
        self.generateTree()
        t2 = time.time()
        outCardsNodes = self.root.children
        for i in range(len(outCardsNodes)):
            rate = 1
            node = outCardsNodes[i]
            self.getRate(node=node, rate=rate)
        nodeNum = 0
        t3 = time.time()
        print ('scoreDict', self.scoreDict)
        for k in self.scoreDict.keys():

            nodeNum += len(self.scoreDict[k])
            k_score = 0
            for i in range(len(self.scoreDict[k])):
                # print (k)
                # n=self.stateSet[k][0][i][1].count(k)
                # print (k,n)
                if self.stateSet[k][2][i][3] >= 16:
                    k_score += self.scoreDict[k][i] * 2
                else:
                    k_score += self.scoreDict[k][i]
            # self.scoreDict[k] = sum(self.scoreDict[k])
            self.scoreDict[k] = k_score

        # print ('score', self.scoreDict)
        # print ('stateSet', self.stateSet)
        # print('nodeNum', nodeNum)
        # print ('usetime', t2 - t1, t3 - t2)
        return self.scoreDict


class Discard_Node:
    def __init__(self, discard=None, AAA=[], ABC=[], jiang=[], T2=[], T1=[], taking_set=[], king_num=0, fei_king=0,
                 baohuanyuan=True):
        self.discard = discard
        self.AAA = AAA
        self.ABC = ABC
        self.jiang = jiang
        self.T2 = T2
        self.T1 = T1
        self.king_num = king_num
        self.fei_king = fei_king
        # self.T_selfmo = copy.copy(T_selfmo)
        self.children = []
        self.taking_set = taking_set
        self.baohuanyuan = baohuanyuan
        # self.value = 1
        self.taking_set_w = []

    def add_child(self, child):
        self.children.append(child)

    def is_exist(self, nodes):
        for node in nodes:
            if node.discard == self.discard and node.AAA == self.AAA and node.ABC == self.ABC and node.T2 == self.T2 and node.T1 == self.T1 and node.king_num == self.king_num and node.fei_king == self.fei_king:
                return True
                # else:
                #     return False
        return False

    def node_info(self):
        print(self.AAA, self.ABC, self.jiang, "T1=", self.T1, "T2=", self.T2, self.taking_set, self.king_num, self.fei_king, self.baohuanyuan)


class Take_Node:
    def __init__(self, take=None, AAA=[], ABC=[], jiang=[], T2=[], T1=[], taking_set=[], taking_set_w=[]):
        self.take = take
        self.AAA = AAA
        self.ABC = ABC
        self.jiang = jiang
        self.T2 = T2
        self.T1 = T1
        self.children = []
        self.taking_set = taking_set
        # self.value = value
        self.taking_set_w = taking_set_w

    def add_child(self, child):
        self.children.append(child)

    def node_info(self):
        print (self.AAA, self.ABC, self.jiang, "T1=", self.T1, "T2=", self.T2, self.taking_set)

    # def is_exist(self, nodes):
    #     for node in nodes:
    #         if node.take == self.take and node.AAA == self.AAA and node.ABC == self.ABC and node.T2 == self.T2 and node.T1 == self.T1 and node.king_num == self.king_num and node.fei_king == self.fei_king:
    #             return True
    #     return False


class SearchTree_take:
    def __init__(self, hand, suits, combination_sets):
        self.hand = hand
        self.suits = suits
        self.combination_sets = combination_sets
        self.xts = combination_sets[0][-2]
        self.tree_dict = []
        self.discard_score = {}
        self.discard_state = {}
        self.node_num = 0
        self.chang_num = 0

    def expand_node(self, node):
        # 胡牌判断
        # print "a"
        if len(node.AAA) + len(node.ABC) == 4 and node.jiang != []:
            return

        # 超时终止
        if time.time() - TIME_START > 2.5:
            # logger.warning("time out!,%s,%s,%s", self.hand, self.suits)
            return
        # 节点扩展，只考虑摸牌
        # 判断需要扩展哪类
        # 当T3的数量不够时
        # if node.king_num == 0:
        if len(node.AAA) + len(node.ABC) < 4:
            if node.T2 != []:  # 1、先扩展T2为T3
                for t2 in node.T2:
                    for item in t2tot3_dict[str(t2)]:
                        if item[1][0] == item[1][1]:
                            AAA = MJ.deepcopy(node.AAA)
                            AAA.append(item[1])
                            ABC = node.ABC
                        else:
                            AAA = node.AAA
                            ABC = MJ.deepcopy(node.ABC)
                            ABC.append(item[1])
                        T2 = MJ.deepcopy(node.T2)
                        T2.remove(t2)
                        T1 = node.T1

                        taking_set = copy.copy(node.taking_set)
                        taking_set_w = copy.copy(node.taking_set_w)
                        taking_set.append(item[-2])
                        taking_set_w.append(item[-1])
                        child = Take_Node(take=item[-2], AAA=AAA, ABC=ABC, jiang=node.jiang, T2=T2,
                                          T1=T1, taking_set=taking_set, taking_set_w=taking_set_w)
                        node.add_child(child=child)
                        self.expand_node(node=child)

            if node.T2 == []:  # or (node.king_num == 0 and len(node.T2) == 1 and node.T2[0][0] == node.T2[0][1]):  # 2、扩展T1为T3？ "t1":[[t3,t2,p]] 这里无宝要留将的打法
                for t1 in node.T1:
                    for item in t1tot3_dict[str(t1)]:
                        T1 = copy.copy(node.T1)
                        T1.remove(t1)
                        # 用于处理废牌存在于T1中的特殊情况
                        # flag1 = False
                        # for card in item[1]:
                        #     if card in T1:
                        #         T1.remove(card)
                        #         T2 = MJ.deepcopy(node.T2)
                        #         T2.append(sorted([card, t1]))
                        #         # logger.info("merge T1 to T2,%s,%s", t1, T2)
                        #         child = Take_Node(take=-1, AAA=node.AAA, ABC=node.ABC, jiang=node.jiang, T2=T2, T1=T1,
                        #                           taking_set=node.taking_set, taking_set_w=node.taking_set_w,
                        #                           king_num=node.king_num, fei_king=node.fei_king,
                        #                           baohuanyuan=node.baohuanyuan)
                        #         node.add_child(child=child)
                        #         self.expand_node(node=child)
                        #         flag1 = True
                        #         break
                        # if flag1:
                        #     continue

                        if item[0][0] == item[0][1]:
                            AAA = MJ.deepcopy(node.AAA)
                            AAA.append(item[0])
                            ABC = node.ABC
                        else:
                            AAA = node.AAA
                            ABC = MJ.deepcopy(node.ABC)
                            ABC.append(item[0])

                        take = item[1]
                        take_w = item[-1]

                        taking_set = copy.copy(node.taking_set)
                        taking_set.extend(take)
                        taking_set_w = copy.copy(node.taking_set_w)
                        taking_set_w.extend(take_w)
                        # taking_set_w.append(take_w[0]) #区别顺序
                        # taking_set_w.append(take_w[1]+1)
                        child = Take_Node(take=take, AAA=AAA, ABC=ABC, jiang=node.jiang, T2=node.T2, T1=T1,
                                          taking_set=taking_set, taking_set_w=taking_set_w)
                        node.add_child(child=child)
                        self.expand_node(node=child)

        else:  # 添加将牌

            # 判断是否已经达到胡牌状态
            # 非双宝做将加宝还原的不算宝还原
            if len(node.AAA) + len(node.ABC) == 4:
                has_jiang = False
                for t2 in node.T2:  # 有将
                    T2 = MJ.deepcopy(node.T2)
                    # 从t2中找到对子作为将牌
                    if t2[0] == t2[1]:
                        has_jiang = True
                        T2.remove(t2)
                        child = Take_Node(take=-1, AAA=node.AAA, ABC=node.ABC, jiang=t2, T2=T2,
                                          T1=node.T1,
                                          taking_set=node.taking_set, taking_set_w=node.taking_set_w)  # 非宝吊宝还原
                        node.add_child(child=child)
                        self.expand_node(node=child)
                # break #移除，好像也不影响，后面评估去重是按摸牌来确定的，这里也不会摸牌了

            if not has_jiang:
                jiangs = copy.copy(node.T1)
                for t2 in node.T2:  # 将T2中的牌也加入到将牌中
                    jiangs.extend(t2)
                for t1 in jiangs:
                    taking_set = copy.copy(node.taking_set)
                    taking_set.append(t1)
                    taking_set_w = copy.copy(node.taking_set_w)
                    taking_set_w.append(1)
                    T1 = copy.copy(jiangs)
                    T1.remove(t1)
                    child = Take_Node(take=t1, AAA=node.AAA, ABC=node.ABC, jiang=[t1, t1], T2=[],
                                      T1=T1,
                                      taking_set=taking_set, taking_set_w=taking_set_w)
                    node.add_child(child=child)
                    self.expand_node(node=child)


    def generate_tree(self):
        kz = []
        sz = []
        for t3 in self.suits:
            if t3[0] == t3[1]:
                kz.append(t3)
            else:
                sz.append(t3)
        for cs in self.combination_sets:
        # 超时处理，直接返回
        # time.sleep(2)
        # if time.time()-TIME_START>2:
        #     logger.warning("time out!%s,%s,%s",self.hand,self.suits,self.king_card)
        #     return
        # t1=time.time()
        # 这里只搜素非胡牌的出牌
            root = Take_Node(take=None, AAA=cs[0] + kz, ABC=cs[1] + sz, jiang=[], T2=cs[2] + cs[3], T1=cs[-1],
                         taking_set=[], taking_set_w=[])

            self.tree_dict.append(root)
            self.expand_node(node=root)


    def cal_score(self, node):
        value = 1

        # 矫正1->3的权重
        # t13 = []
        # w_set = []
        # taking_set_w = copy.copy(node.taking_set_w)
        # for k in range(len(node.taking_set_w)):
        #     if node.taking_set_w[k] == MJ.w_aa + 1 or node.taking_set_w[k] == MJ.w_ab + 1:
        #         taking_set_w[k] -= 1
        #         t13.append(k - 1)  # t13的前一张牌不能被作为最后一张摸到的牌，这里有顺序

        w = 0

        for i in range(len(node.taking_set)):
            card = node.taking_set[i]
            # value *= T_SELFMO[MJ.convert_hex2index(card)] #todo bug 连续乘了2个
            if i != len(node.taking_set) - 1:
                w_ = node.taking_set_w[i]
            # elif node.taking_set_w[i]==MJ.w_aa:
            #     w_=1.5
            else:
                w_ = 4

            value *= T_SELFMO[MJ.convert_hex2index(card)] * w_


        # 摸牌概率修正，当一张牌被重复获取时，T_selfmo修改为当前数量占未出现牌数量的比例 0.4s
        taking_set = list(set(node.taking_set))
        taking_set_num = [node.taking_set.count(i) for i in taking_set]
        for i in range(len(taking_set_num)):
            if taking_set[i] == 0:
                continue
            n = taking_set_num[i]
            index = MJ.convert_hex2index(taking_set[i])
            if LEFT_NUM[index] < n:
                return 0
            for j in range(n - 1):
                value *= float((LEFT_NUM[index] - j - 1)) / (LEFT_NUM[index])
            # len_taking=len(node.taking_set)
            # xts=self.combination_sets[0][-2]
            # 摸牌次数越多，危险度越大
            # if len_taking==xts:
            #     value = 1
            # else:
            #     value=1
            #     for i in range(len_taking-xts):
            #         value *= 1 - (0.02*(i+1))
        fan = Fan(kz=node.AAA, sz=node.ABC, jiang=node.jiang).fanDetect()

        score = fan * value
        return score

    def calculate_path_expectation(self, node):
        # 深度搜索
        # node.node_info()
        # print value
        # value_ = value
        # print node.AAA,node.ABC,node.jiang

        if len(node.AAA) + len(node.ABC) == 4 and node.jiang != []:
            self.node_num += 1
        # 测试：最快胡牌 #可能搜索到了一些不应该出现的局面，这些概率影响了
        # xts = self.combination_sets[0][-2]
        # layer = len(node.taking_set)
        # if node.take!=0:
        #     if layer>xts:
        #         return
        # elif layer-1>xts:
        #     return
        # 弃牌不应该出现在摸牌中 done  先去掉已出牌不再摸的情况
            discard_set = []
            for t2 in node.T2:
                discard_set.extend(t2)
            discard_set.extend(node.T1)
            if self.combination_sets[0][-2] != 0:

                for i in range(len(discard_set) - 1, -1, -1):  #
                    card = discard_set[i]

                # 出了对牌，但是最后没有将牌的情况应该舍去，
                    if discard_set.count(card) >= 2 and node.take not in [0, -1]:
                        return
                    # 出牌存在于摸牌中
                    if card in node.taking_set:
                    # logger.warning("remove disicard card in takingset,%s,%s,%s,%s,%s",discard_set,node.taking_set,self.hand,self.suits,self.king_card)
                        return

            # node.AAA.sort()
            # node.ABC.sort()
            taking_set_sorted = sorted(node.taking_set)
            # taking_set_sorted = node.taking_set
            if discard_set != []:
                score = self.cal_score(node=node)  # 放到外面减耗时
                if score == 0:  # 胡牌概率为0
                    return
            else:
                return
            # todo 这种按摸牌的评估方式是否唯一准确
            for card in list(set(discard_set)):

                # for card in [discard]:
                if card not in self.discard_state.keys():
                    self.discard_state[card] = [[], []]
                if taking_set_sorted not in self.discard_state[card][0]:
                    self.discard_state[card][0].append(taking_set_sorted)

                    # self.discard_state[discard][1].append([node.AAA,node.ABC,node.jiang])
                    self.discard_state[card][-1].append(score)
                # elif time.time() - TIME_START < 2.3:  # 时间处理3
                else:
                    index = self.discard_state[card][0].index(taking_set_sorted)
                    if score > self.discard_state[card][-1][index]:
                        self.chang_num += 1
                        # self.discard_state[discard][1][index]=([node.AAA, node.ABC, node.jiang])
                        self.discard_state[card][-1][index] = score

        elif node.children != []:
            for child in node.children:
                self.calculate_path_expectation(node=child)

    def get_discard_score(self):

        t1 = time.time()
        self.generate_tree()
        t2 = time.time()
        for root in self.tree_dict:
            self.calculate_path_expectation(root)
        t3 = time.time()
        # print ("tree time:", t2 - t1, "value time:", t3 - t2)
        state_num = 0
        for discard in self.discard_state.keys():
            if discard not in self.discard_score.keys():
                self.discard_score[discard] = 0
            self.discard_score[discard] = sum(self.discard_state[discard][-1])
            state_num += len(self.discard_state[discard][-1])

        # print ("discard_state", self.discard_state)
        # print ("discard_score", self.discard_score)
        # print ("leaf node ", self.node_num)
        # print ("state_num", state_num)
        # print ("chang_num", self.chang_num)
        return self.discard_score


'''
番数计算类
'''


class Fan():
    def __init__(self, kz, sz, jiang):
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
        self.mul = 2

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

    def fanDetect(self):
        """
        番数计算
        基础分４分，通过调用上述的番种检测来增加基础分
        :return: int 番数
        """
        # 基础分判定
        score = 1
        if self.pengPengHu():
            # print "0"
            score *= self.mul
            # score *= 2  # 碰碰胡再给2倍分

        # 翻倍机制


        # 清一色
        if self.qingYiSe():
            score *= self.mul
            # print "3"
        # 单吊　x2
        # 这里无法处理，宝吊需要吃碰杠吃碰杠处理
        # if score>16: #得分大于16时，分数评估提高
        #     score*=1.5
        # print
        return score


class Fan2():
    def __init__(self, kz, sz, jiang, node=None, fei_king=0, kingNum=0):
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
        self.baoHuanYuan = False
        self.noKing = True

        # self.usingKing=node.usingKing
        if node != None:
            self.feiKingNum = node.feiKingNum
            if node.baoHuanYuan + node.feiKingNum == kingNum:
                # self.NoKing=True
                if kingNum != 0 and node.baoHuanYuan != 0:
                    self.baoHuanYuan = True
            else:
                self.noKing = False


        else:
            self.feiKingNum = fei_king

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
    #     if self.baoHuanYuan:
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
        cards = copy.copy(self.kz + self.sz)
        cards.append(self.jiang)
        w = 0
        ti = 0
        to = 0
        z = 0
        for card in cards:
            if card & 0xf0 == 0x00:
                w = 1
            elif card & 0xf0 == 0x10:
                ti = 1
            elif card & 0xf0 == 0x20:
                to = 2
            else:
                return False

        if w + ti + to + z <= 1:
            return True
        else:
            return False

    def fanDetect(self):
        """
        番数计算
        基础分４分，通过调用上述的番种检测来增加基础分
        :return: int 番数
        """
        # 基础分判定
        score = 4
        if self.pengPengHu():

            score = 8
            if self.noKing:
                score = 16

        # 翻倍机制
        # 飞宝
        for i in range(self.feiKingNum):
            score *= 2
        # # 宝还原　x2
        if self.baoHuanYuan:
            score *= 2
        # 单吊　x2
        # 这里无法处理，宝吊需要吃碰杠吃碰杠处理

        return score


'''
平胡类，相关处理方法
分为手牌拆分模块sys_info，评估cost,出牌决策，吃碰杠决策等部分
'''


class pinghu:
    '''
    '''

    def __init__(self, cards, suits, leftNum=LEFT_NUM, discards=[], discards_real=[], discardsOp=[], round=0,
                 remainNum=134,
                 fengWei=0, seat_id=0, choose_color=[], hu_cards=[],
                 hu_fan=[]):
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
        :param op_card:动作操作牌
        """
        cards.sort()
        self.cards = cards
        self.suits = suits
        self.discards = discards
        self.discards_real = discards_real
        self.discardsOp = discardsOp
        self.remainNum = max(1, remainNum)
        self.leftNum = leftNum
        self.round = round
        self.seat_id = seat_id

        self.choose_color = choose_color
        self.dingque = choose_color[self.seat_id]
        self.hu_cards = hu_cards
        self.hu_fan = hu_fan

        # 定缺的牌
        self.split = self.split_type_s(cards)
        if self.dingque == 0:
            self.dingque_cards = copy.copy(self.split[0])
        elif self.dingque == 1:
            self.dingque_cards = copy.copy(self.split[1])
        elif self.dingque == 2:
            self.dingque_cards = copy.copy(self.split[2])

        self.fei_king = 0
        # print ('self.leftNum',self.leftNum)
        # if self.leftNum == []:
        #     leftNum, discardsList = trandfer_discards(discards, discardsOp, cards)
        #     self.leftNum = leftNum

        # self.fengWei = fengWei
        self.kingCard = 0
        self.kingNum = 0
        self.preKingCard = 0
        # self.preKingCard = pre_king(kingCard)
        # self.op_card = op_card
        # if kingCard != None:
        #     self.kingNum = cards.count(kingCard)
        # else:
        #     self.kingNum = 0  # print('kingNum111',leftNum[convert_hex2index(self.kingCard)],self.cards)

    @staticmethod
    def get_effective_cards(dz_set=[]):
        """
        获取有效牌
        :param dz_set: 搭子集合 list [[]]
        :return: 有效牌 list []
        """
        effective_cards = []
        for dz in dz_set:
            if len(dz) == 1:
                effective_cards.append(dz[0])
            elif dz[1] == dz[0]:
                effective_cards.append(dz[0])
            elif dz[1] == dz[0] + 1:
                if int(dz[0]) & 0x0F == 1:
                    effective_cards.append(dz[0] + 2)
                elif int(dz[0]) & 0x0F == 8:
                    effective_cards.append((dz[0] - 1))
                else:
                    effective_cards.append(dz[0] - 1)
                    effective_cards.append(dz[0] + 2)
            elif dz[1] == dz[0] + 2:
                effective_cards.append(dz[0] + 1)
        effective_cards = set(effective_cards)  # set 和list的区别？
        return list(effective_cards)

    def get_effective_cards_w(self, dz_set=[], left_num=[]):
        """
        有效牌及其概率获取
        :param dz_set: 搭子集合 list[[]],剩余牌　[]
        :param left_num: 有效牌集合[], 有效牌概率　[]
        :return:
        """
        cards_num = max(1, self.remainNum)
        effective_cards = []
        w = []
        for dz in dz_set:
            if dz[1] == dz[0]:
                effective_cards.append(dz[0])
                # if dz[0]>=0x31 and dz[0]<=0x37 and left_num[translate16_33(dz[0])]>0:#添加字牌权重
                #     w.append(float((left_num[translate16_33(dz[0])]+0.5) * w_aa) / cards_num)
                # else:
                w.append(float(
                    left_num[MJ.convert_hex2index(dz[0])]) / (max(1, cards_num)) * w_aa)  # 修改缩进,发现致命错误panic 忘了写float,这里写６是因为评估函数计算的缺陷

            elif dz[1] == dz[0] + 1:
                if int(dz[0]) & 0x0F == 1:
                    effective_cards.append(dz[0] + 2)
                    w.append(float(left_num[MJ.convert_hex2index(dz[0] + 2)]) / cards_num * w_ab)
                elif int(dz[0]) & 0x0F == 8:
                    effective_cards.append((dz[0] - 1))
                    w.append(float(left_num[MJ.convert_hex2index(dz[0] - 1)]) / cards_num * w_ab)
                else:
                    effective_cards.append(dz[0] - 1)
                    effective_cards.append(dz[0] + 2)
                    w.append(float(left_num[MJ.convert_hex2index(dz[0] - 1)] + left_num[
                        MJ.convert_hex2index(dz[0] + 2)]) / cards_num * w_ab)
            elif dz[1] == dz[0] + 2:
                effective_cards.append(dz[0] + 1)
                w.append(float(left_num[MJ.convert_hex2index(int(dz[0]) + 1)]) / cards_num * w_ab)
        return effective_cards, w

    @staticmethod
    def split_type_s(cards=[]):
        """
        功能：手牌花色分离，将手牌分离成万条筒字各色后输出
        :param cards: 手牌　[]
        :return: 万,条,筒,字　[],[],[],[]
        """
        cards_wan = []
        cards_tiao = []
        cards_tong = []
        cards_zi = []
        for card in cards:
            if card & 0xF0 == 0x00:
                cards_wan.append(card)
            elif card & 0xF0 == 0x10:
                cards_tiao.append(card)
            elif card & 0xF0 == 0x20:
                cards_tong.append(card)
            elif card & 0xF0 == 0x30:
                cards_zi.append(card)
        return cards_wan, cards_tiao, cards_tong, cards_zi

    @staticmethod
    def get_32N(cards=[]):
        """
        功能：计算所有存在的手牌的３Ｎ与２Ｎ的集合，例如[3,4,5]　，将得到[[3,4],[3,5],[4,5],[3,4,5]]
        思路：为减少计算量，对长度在12张以上的单花色的手牌，当存在顺子时，不再计算搭子
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
        # 对长度在12张以上的单花色的手牌，当存在顺子时，不再计算搭子
        if len(cards) >= 12:
            for card in cards:
                if card == lastCard:
                    continue
                else:
                    lastCard = card
                if cards.count(card) >= 3:
                    kz.append([card, card, card])
                elif cards.count(card) >= 2:
                    aa.append([card, card])
                if card + 1 in cards and card + 2 in cards:
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
                if cards.count(card) >= 3:
                    kz.append([card, card, card])
                if cards.count(card) >= 2:
                    aa.append([card, card])
                if card + 1 in cards and card + 2 in cards:
                    sz.append([card, card + 1, card + 2])
                if card + 1 in cards:
                    ab.append([card, card + 1])
                if card + 2 in cards:
                    ac.append([card, card + 2])
        return kz + sz + aa + ab + ac

    # 判断３２Ｎ是否存在于ｃａｒｄｓ中
    @staticmethod
    def in_cards(t32=[], cards=[]):
        """
        判断３２Ｎ是否存在于ｃａｒｄｓ中
        :param t32: ３Ｎ或2N组合牌
        :param cards: 本次判断的手牌
        :return: bool
        """
        for card in t32:
            if card not in cards:
                return False
        return True

    def extract_32N(self, cards=[], t32_branch=[], t32_set=[]):
        """
        功能：递归计算手牌的所有组合信息，并存储在t32_set，
        思路: 每次递归前检测是否仍然存在３２N的集合,如果没有则返回出本此计算的结果，否则在手牌中抽取该３２N，再次进行递归
        :param cards: 手牌
        :param t32_branch: 本次递归的暂存结果
        :param t32_set: 所有组合信息
        :return: 结果存在t32_set中
        """
        t32N = self.get_32N(cards=cards)

        if len(t32N) == 0:
            t32_set.extend(t32_branch)
            # t32_set.extend([cards])
            t32_set.append(0)
            t32_set.extend([cards])
        else:
            for t32 in t32N:
                if self.in_cards(t32=t32, cards=cards):
                    cards_r = copy.copy(cards)
                    for card in t32:
                        cards_r.remove(card)
                    t32_branch.append(t32)
                    self.extract_32N(cards=cards_r, t32_branch=t32_branch, t32_set=t32_set)
                    if len(t32_branch) >= 1:
                        t32_branch.pop(-1)

    def tree_expand(self, cards):
        """
        功能：对extract_32N计算的结果进行处理同一格式，计算万条筒花色的组合信息
        思路：对t32_set的组合信息进行格式统一，分为[kz,sz,aa,ab,xts,leftCards]保存，并对划分不合理的地方进行过滤，例如将３４５划分为35,4为废牌的情况
        :param cards: cards [] 万条筒其中一种花色手牌
        :return: allDeWeight　[kz,sz,aa,ab,xts,leftCards] 去除不合理划分情况的组合后的组合信息
        """
        all = []
        t32_set = []
        self.extract_32N(cards=cards, t32_branch=[], t32_set=t32_set)
        kz = []
        sz = []
        t2N = []
        aa = []
        length_t32_set = len(t32_set)
        i = 0
        # for i in range(len(t32_set)):
        while i < length_t32_set:
            t = t32_set[i]
            flag = True  # 本次划分是否合理
            if t != 0:
                if len(t) == 3:

                    if t[0] == t[1]:
                        kz.append(t)
                    else:
                        sz.append(t)  # print (sub)
                elif len(t) == 2:
                    if t[1] == t[0]:
                        aa.append(t)
                    else:
                        t2N.append(t)

            else:
                '修改，使计算时间缩短'
                leftCards = t32_set[i + 1]
                efc_cards = self.get_effective_cards(dz_set=t2N)  # t2N中不包含ａａ
                # 去除划分不合理的情况，例如345　划分为34　或35等，对于333 划分为33　和3的情况，考虑有将牌的情况暂时不做处理
                for card in leftCards:
                    if card in efc_cards:
                        flag = False
                        break

                if flag:
                    all.append([kz, sz, aa, t2N, 0, leftCards])
                kz = []
                sz = []
                aa = []
                t2N = []
                i += 1
            i += 1

        allSort = []  # 给每一个元素排序
        allDeWeight = []  # 排序去重后

        for e in all:
            for f in e:
                if f == 0:  # 0是xts位，int不能排序
                    continue
                else:
                    f.sort()
            allSort.append(e)

        for a in allSort:
            if a not in allDeWeight:
                allDeWeight.append(a)

        allDeWeight = sorted(allDeWeight, key=lambda k: (len(k[0]), len(k[1]), len(k[2])), reverse=True)  # 居然可以这样排序！！
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
            index = (card & 0x0f) - 1
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

    def pengpengHu(self, outKingCards, suits, kingNum):
        """
        功能：碰碰胡检测
        思路：计算碰碰胡的组合情况，只考虑kz和aa，当副露中存在sz时，返回[[],[],[],[],14,[]]，其中xts为１４表示不可能胡碰碰胡
        :param outKingCards: 去除宝牌后的手牌
        :param suits: 副露
        :param kingNum: 宝数量
        :return: all_PengPengHu　碰碰胡的组合情况
        """
        all_PengPengHu = [[], [], [], [], 14, []]

        for suit in suits:
            if suit[0] != suit[1]:
                return []

        for card in set(outKingCards):

            if outKingCards.count(card) == 1:
                all_PengPengHu[-1].append(card)
            elif outKingCards.count(card) == 2:
                all_PengPengHu[2].append([card, card])
            elif outKingCards.count(card) == 3:
                all_PengPengHu[0].append([card, card, card])
            elif outKingCards.count(card) == 4:
                all_PengPengHu[0].append([card, card, card])
                all_PengPengHu[-1].append(card)
        all_PengPengHu = self.xts([all_PengPengHu], suits, kingNum)
        return all_PengPengHu

    @staticmethod
    def xts(all=[], suits=[], kingNum=0):
        """
         功能：计算组合的向听数
        思路：初始向听数为１４，减去相应已成型的组合（kz,sz为３，aa/ab为２，宝直接当１减去），当２Ｎ过剩时，只减去还需要的２Ｎ，对２Ｎ不足时，对还缺少的３Ｎ减去１，表示从孤张牌中选择一张作为３Ｎ的待选
        :param all: [[]]组合信息
        :param suits: 副露
        :param kingNum: 宝牌数量
        :return: all　计算向听数后的组合信息
        """
        for i in range(len(all)):
            t3N = all[i][0] + all[i][1]
            all[i][4] = 14 - (len(t3N) + len(suits)) * 3
            # 有将牌
            has_aa = False
            if len(all[i][2]) > 0:
                has_aa = True

            if has_aa and kingNum == 0:  # has do 当２Ｎ与３Ｎ数量小于4时，存在没有减去相应待填数，即废牌也会有１张作为２Ｎ或３Ｎ的待选位,
                # print()all_src
                if len(suits) + len(t3N) + len(all[i][2]) + len(all[i][3]) - 1 >= 4:

                    all[i][4] -= (4 - (len(suits) + len(t3N))) * 2 + 2
                else:
                    all[i][4] -= (len(all[i][2]) + len(all[i][3]) - 1) * 2 + 2 + 4 - (
                            len(suits) + len(t3N) + len(all[i][2]) + len(all[i][3]) - 1)  # 0717 17:24
            # 无将牌
            else:
                if len(suits) + len(t3N) + len(all[i][2]) + len(all[i][3]) >= 4:

                    all[i][4] -= (4 - (len(suits) + len(t3N))) * 2 + 1

                else:
                    all[i][4] -= (len(all[i][2]) + len(all[i][3])) * 2 + 1 + 4 - (
                            len(suits) + len(t3N) + len(all[i][2]) + len(all[i][3]))
            all[i][4] -= kingNum
            if all[i][4] < 0:
                all[i][4] = 0
        all.sort(key=lambda k: (k[4], len(k[-1])))
        return all

    @staticmethod
    def is_related(card=[], ndCards=[]):
        """
        功能：判断孤张牌是否与次级废牌能成为搭子２Ｎ关系
        思路：先计算该张废牌的相关牌为临近２张牌，判断其是否在次级废牌中ndCards
        :param card: 废牌
        :param ndCards: 次级废牌组合
        :return: bool
        """
        if card > 0x30:
            return False
        relatedSet = [card - 2, card - 1, card, card + 1, card + 2]
        for card in relatedSet:
            if card in ndCards:
                return True
        return False

    def sys_info_V3(self, cards, suits):
        """
        功能：综合计算手牌的组合信息
        思路：对手牌进行花色分离后，单独计算出每种花色的组合信息　，再将其综合起来，计算每个组合向听数，最后输出最小向听数及其加一的组合
        :param cards: 手牌
        :param suits: 副露
        :param left_num: 剩余牌
        :param kingCard: 宝牌
        :return: 组合信息
        """
        # 花色分离
        wan, tiao, tong, zi = self.split_type_s(cards)

        # dingque_cards = []
        if self.dingque == 0:
            # dingque_cards = copy.copy(wan)
            wan = []

        elif self.dingque == 1:
            # dingque_cards = copy.copy(tiao)
            tiao = []

        elif self.dingque == 2:
            # dingque_cards = copy.copy(tong)
            tong = []
        wan_expd = self.tree_expand(cards=wan)
        tiao_expd = self.tree_expand(cards=tiao)
        tong_expd = self.tree_expand(cards=tong)
        zi_expd = self.zi_expand(cards=zi)

        all = []
        for i in wan_expd:
            for j in tiao_expd:
                for k in tong_expd:
                    for m in zi_expd:
                        branch = []
                        # 将每种花色的4个字段合并成一个字段
                        for n in range(6):
                            branch.append(i[n] + j[n] + k[n] + m[n])
                        # branch[-1].extend(dingque_cards)
                        all.append(branch)

        # 将获取概率为０的组合直接丢弃到废牌中 todo 由于有宝，这里也可能会被宝代替
        # 移到了出牌决策部分处理
        # if len(cards) % 3 == 1 and self.kingNum <= 1:#这里只考虑出牌、宝做宝吊的情况
        #     for a in all:
        #         for i in range(len(a[3]) - 1, -1, -1):
        #             ab = a[3][i]
        #             efc = self.get_effective_cards([ab])
        #             if sum([LEFT_NUM[MJ.convert_hex2index(e)] for e in efc]) == 0:
        #                 a[3].remove(ab)
        #                 a[-1].extend(ab)
        #                 logger.info("remove rate 0 ab,%s,%s,%s,a=%s",self.cards,self.suits,self.kingCard,a)

        # 对废牌区的牌都是与ａａ/ab区相联系时，将价值最低的ab丢弃到废牌区
        # for a in all:
        #     ndCards = []
        #     for aa_ab in a[2] + a[3]:
        #         ndCards.extend(aa_ab)
        #     Flag = True
        #     for card in a[-1]:
        #         if not self.is_related(card, ndCards):
        #             Flag = False
        #             break
        #     if Flag:
        #         # print ('all are related',a[3])
        #
        #         for i in range(len(a[3]) - 1, -1, -1):
        #             ab = a[3][i]
        #             efc = self.get_effective_cards([ab])
        #             if sum([left_num[convert_hex2index(e)] for e in efc]) <= 2:
        #                 a[3].remove(ab)
        #                 a[-1].extend(ab)

        # 计算向听数
        # 计算拆分组合的向听数
        all = self.xts(all, suits)

        # 获取向听数最小的all分支
        min_index = 0
        for i in range(len(all)):
            if all[i][4] > all[0][4] + 1:  # xts+１以下的组合
                min_index = i
                break

        if min_index == 0:  # 如果全部都匹配，则min_index没有被赋值，将min_index赋予ａｌｌ长度
            min_index = len(all)

        all = all[:min_index]
        # print("all_terminal", all)
        return all

    def left_card_weight_bak(self, card, left_num):
        """
        功能：对废牌组合中的每张废牌进行评估，计算其成为３Ｎ的概率
        思路：每张牌能成为３N的情况可以分为先成为搭子，在成为３Ｎ２步，成为搭子的牌必须自己摸到，而成为kz,sz可以通过吃碰。刻子为获取２张相同的牌，顺子为其邻近的２张牌
        :param card: 孤张牌
        :param left_num: 剩余牌
        :return: 评估值
        """
        if self.remainNum == 0:
            remainNum = 1
        else:
            remainNum = self.remainNum
        # remainNum = 1
        i = MJ.convert_hex2index(card)
        # d_w = 0

        if left_num[i] == self.remainNum:
            sf = float(self.leftNum[i]) / remainNum * 6
        else:
            sf = float(left_num[i]) / remainNum * float((left_num[i] - 1)) / remainNum * 6
        if card >= 0x31:  # kz概率
            # todo if card == fengwei:
            # if card >= 0x35 and left_num[i] >= 2:
            #     d_w = left_num[i] * left_num[i] * 2  # bug 7.22 修正dw-d_w
            # else:
            d_w = sf  # 7.22 １６:３５ 去除字牌
        elif card % 16 == 1:  # 11＋２3
            d_w = sf + float(left_num[i + 1]) / remainNum * float(left_num[i + 2]) / remainNum * 2
        elif card % 16 == 2:  # 22+13+3(14)+43   222 123 234
            d_w = sf + float(left_num[i - 1]) / remainNum * float(left_num[i + 1]) / remainNum * 2 + float(
                left_num[i + 1]) / remainNum * float(left_num[
                                                         i + 2]) / remainNum * 2  # d_w = left_num[i - 1] + left_num[i] * 3 + left_num[i + 1] * 2 + left_num[i + 2]
        elif card % 16 == 8:  # 888 678 789
            d_w = sf + float(left_num[i - 2]) / remainNum * float(left_num[i - 1]) / remainNum * 2 + float(
                left_num[i - 1]) / remainNum * float(left_num[
                                                         i + 1]) / remainNum * 2  # d_w = left_num[i - 2] + left_num[i - 1] * 2 + left_num[i] * 3 + left_num[i + 1]
        elif card % 16 == 9:  # 999 789
            d_w = sf + float(left_num[i - 2]) / remainNum * float(left_num[
                                                                      i - 1]) / remainNum * 2  # d_w = left_num[i - 2] + left_num[i - 1] + left_num[i] * 3  # 删除多添加的×２
        else:  # 555 345 456 567
            # print (left_num)
            d_w = sf + float(left_num[i - 2]) / remainNum * float(left_num[i - 1]) / remainNum * 2 + float(
                left_num[i - 1]) / remainNum * float(left_num[i + 1]) / remainNum * 2 + float(
                left_num[i + 1]) / remainNum * float(left_num[
                                                         i + 2]) / remainNum * 2
        # print("i=", i, d_w)
        return d_w

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
        i = MJ.convert_hex2index(card)

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
        # print("i=", i, d_w)
        return d_w

    # t2N列表最后的ａａ
    @staticmethod
    def is_last_aa(t2N=[]):
        """
        在计算评估值时，用于判断是否是最后一个ａａ
        判断剩余搭子集合中是否还存在ａａ对子
        :param t2N:搭子集合
        :return: bool
        """
        for t in t2N:
            if t[0] == t[1]:
                return False
        return True

    def choose_n(self, t2N=[], n=0, rate=1, results=[], ab=False, abSet=[]):
        """
        采用递归的方式，计算所有可能的胡牌的２Ｎ组合情况
        在t2N中选择ｎ个作为有效２Ｎ
        :param t2N: 待选搭子集合
        :param n: 待选数量
        :param rate: 本条路径的胡牌概率
        :param results: 计算结果列表形式　[]
        :param ab: 本条路径中是否有ａｂ的搭子
        :param abSet: 所有路径中是否存在ａｂ的集合
        :return:
        """
        if n == 0:
            results.append(rate)
            abSet.append(ab)
            return
        n_ = copy.copy(n)
        n_ -= 1
        for t2 in t2N:
            t2NCopy = MJ.deepcopy(t2N)
            t2NCopy.remove(t2)
            rate_ = copy.copy(rate)
            rate_ *= t2[2]
            if t2[0] != t2[1] or ab:
                ab_ = True
            else:
                ab_ = False

            self.choose_n(t2NCopy, n_, rate_, results, ab_, abSet)

    def calculate_path_w(self, a):
        """
        一条组合集的胡牌概率评估
        分为有宝和无宝，无宝中又分为有将和无将情况进行计算
        :param a: 组合集
        :param king_num: 宝牌数量
        :param feiKing: 飞宝数
        :return: 胡牌概率和废牌
        """
        path_w = [1, copy.copy(a[-1])]
        t2N = MJ.deepcopy(a[2] + a[3])
        efc_cards, t2_w = self.get_effective_cards_w(dz_set=t2N, left_num=self.leftNum)
        for i in range(len(t2N)):
            t2N[i].append(t2_w[i])
        bl = max(0, 4 - len(self.suits) - (len(a[0]) + len(a[1])))
        # print ("cost t2N", t2N)
        results = []
        abSet = []
        if True:  # 无宝
            # 对ａａ集合中选择一个作为将牌，在剩余的t2N中使用choose_n计算胡牌概率
            if a[2] != []:  # 定将
                t2N[:len(a[2])] = sorted(t2N[:len(a[2])], key=lambda k: k[2],
                                         reverse=True)  # 这里倒置会更好，如果aa的权重为０会导致整个评估为０
                t2N[len(a[2]):] = sorted(t2N[len(a[2]):], key=lambda k: k[2], reverse=True)
                # if len(a[2])-1 <= bl:
                #     self.jiang_rate(t2N[:len(a[2])])

                # has_ab = False
                # print ('bl', bl)
                if bl <= len(t2N) - 1:  # t2N溢出,需要出一张２N
                    # aa_rate=self.jiang_rate(aa)
                    t2NCP = MJ.deepcopy(t2N)
                    if a[-1] == [] and t2N != [] and a[4] != 0:  # 只添加最后的废牌: #只有当废牌区为空时，才将２Ｎ放入
                        path_w[1].append(t2N[-1][0])
                        path_w[1].append(t2N[-1][1])
                        t2NCP.remove(t2N[-1])
                    # yc=len(t2NCP)-1-bl

                    for aa in t2NCP[:len(a[2])]:
                        t2NCopy = MJ.deepcopy(t2NCP)
                        t2NCopy.remove(aa)
                        merge_rate = 0
                        t2NCopy.sort(key=lambda k: k[2], reverse=True)
                        i = 0
                        for t2 in t2NCopy[bl - 1:]:
                            i += 1
                            merge_rate += t2[2]
                        if i != 0:
                            for j in range(i - 1):
                                t2NCopy.pop(-1)
                            t2NCopy[-1][2] = merge_rate
                        # print t2NCP,i,'t2NCopy',t2NCopy

                        self.choose_n(t2N=t2NCopy, n=bl, rate=1, results=results, ab=False, abSet=abSet)
                    for i in range(len(abSet)):
                        if results[i] != 1:
                            if abSet[i]:
                                results[i] = float(results[i]) / w_ab
                            else:
                                results[i] = float(results[i]) / w_aa * 1.5
                    nums = math.factorial(bl)
                    path_w[0] *= float(sum(results)) / nums
                    # print ("results", results)


                else:
                    for i in range(bl - len(t2N) + 1):
                        path_w[0] *= (80.0) / (self.remainNum * self.remainNum)
                    # rateSet=[]
                    for aa in t2N[:len(a[2])]:
                        t2NCopy = MJ.deepcopy(t2N)
                        t2NCopy.remove(aa)
                        # todo 可以不用这种计算方法
                        self.choose_n(t2N=t2NCopy, n=len(t2NCopy), rate=1, results=results, ab=False,
                                      abSet=abSet)  # rate=1  # for t2 in t2NCopy:  #     rate*=t2[2]  # rateSet.append(rate)
                    # for i in range(len(abSet)):
                    #     if results[i] != 1:
                    #         if abSet[i]:
                    #             results[i] = float(results[i]) / (1+w_ways)
                    #         else:
                    #             results[i] = float(results[i]) / (1+3*w_ways)
                    nums = math.factorial(len(t2N) - 1)
                    path_w[0] *= float(sum(results)) / nums

            # 未定将牌
            # 同理，没有将牌的时候，直接在choose_n中计算胡牌概率
            else:
                if len(t2N) >= bl:
                    t2NCP = MJ.deepcopy(t2N)
                    if a[-1] == [] and t2N != []:  # 只添加最后的废牌: #只有当废牌区为空时，才将２Ｎ放入
                        path_w[1].append(t2N[-1][0])
                        path_w[1].append(t2N[-1][1])
                        t2NCP.remove(t2N[-1])
                    merge_rate = 0
                    t2NCP.sort(key=lambda k: k[2], reverse=True)
                    i = 0
                    for t2 in t2NCP[bl - 1:]:
                        i += 1
                        merge_rate += t2[2]
                    if i != 0:
                        for j in range(i - 1):
                            t2NCP.pop(-1)
                        t2NCP[-1][2] = merge_rate
                    self.choose_n(t2N=t2NCP, n=bl, rate=1, results=results)
                    nums = math.factorial(bl)
                    path_w[0] *= float(sum(results)) / nums
                else:
                    for i in range(bl - len(t2N)):
                        path_w[0] *= (80.0) / (self.remainNum * self.remainNum)
                    # todo
                    self.choose_n(t2N=t2N, n=len(t2N), rate=1, results=results)
                    nums = math.factorial(len(t2N))
                    path_w[0] *= float(sum(results)) / nums
                # 将概率获取
                left_cards = path_w[1]
                w_jiang = [0] * max(1, len(left_cards))
                for k in range(len(left_cards)):
                    if MJ.convert_hex2index(left_cards[k]) == -1:  # todo 添加的牌为0？
                        n = 3.0
                    else:
                        if MJ.convert_hex2index(left_cards[k])==None:
                            n = 0
                        else:
                            n = float(self.leftNum[MJ.convert_hex2index(left_cards[k])])
                    w_jiang[k] = float(n) / self.remainNum  # 可以摸到宝牌与其他废牌一起的概率+left_num[translate16_33(king_card)]
                path_w[0] *= max(w_jiang)  # 添加将牌概率
                if len(left_cards) > 1:  # 填胡状态下，差一个将牌胡牌,这里
                    path_w[1].remove(left_cards[w_jiang.index(max(w_jiang))])
                if a[-1] == [] and len(a[3]) == 1 and a[4] == 1:  # 添加没有将牌，但有刻子与２Ｎ的出牌情景
                    kz = []  # 存在刻子
                    for t in a[0]:
                        if t[0] == t[1]:
                            kz = t
                            break
                    if kz != []:
                        _, rate_out_3N = self.get_effective_cards_w(dz_set=a[3], left_num=self.leftNum)
                        if float(rate_out_3N[0]) / w_ab > path_w[0]:
                            path_w[0] = float(rate_out_3N[0]) / w_ab
                            path_w[1] = [kz[0]]


        return path_w

    def cost(self, all, suits):
        """
        功能：计算组合评估值－－胡牌概率，对组合中没有废牌的情况计算出废牌并输出
        思路：计算胡牌概率，摸到有效牌概率的累乘值，分为有将牌或宝牌和无将牌２种情况处理，无将牌中需要计算将牌概率，有宝牌情况将１张宝作为将牌，
            多余宝牌作为有效牌使用。对没有废牌的情况，将有效牌概率最低的搭子放入到废牌区
        :param all: 组合信息
        :param suits: 副露
        :param left_num: 剩余牌
        :param king_num: 宝数量
        :param king_card: 宝牌
        :return: path_w　[rate,leftCards]组合的评估值和废牌
        """
        # pengpenghu=True
        # for s in self.suits:
        #     if s[0]!=s[1]:
        #         pengpenghu=False
        #         break
        # path_w[0] 胡牌概率
        # path_w[1] 废牌表
        path_w = []  # 创建一个存储胡牌概率和废牌的list
        for i in range(len(all)):
            path_w.append([1.0, MJ.deepcopy(all[i][-1])])

        # 全部搜索会导致搜索空间极大
        for index_all in range(len(all)):  # 选出最大期望概率胡牌路径，选择该路径，从剩余牌中再选择最佳出牌顺序，局部最优

            path_w[index_all] = self.calculate_path_w(all[index_all])

        # print("path_w_end", path_w)
        return path_w

    def discards_w(self, discards=[], left_num=[], ndcards={}):
        """
         功能：计算废牌评估，并返回评估值最低的废牌作为最后的出牌
        思路：计算出每张废牌成为３Ｎ的概率，其中使用了搭子作为候选牌，例如废牌为５　，当有６６的情况时，将６６作为已获取牌，并在ｌｅｆｔＣａｒｄｓ中进行更新，将６的有效牌置为剩余牌总数
        :param discards: 废牌集合
        :param left_num: 剩余牌数量
        :param ndcards: 次级孤张牌
        :return: 最小评估值的废牌
        """
        discards_w = []
        if discards == []:
            return 0x00
        for card in discards:
            left_numCP = copy.copy(left_num)
            if ndcards != {}:
                if card in ndcards.keys():
                    for ndcard in ndcards[card]:
                        left_numCP[MJ.convert_hex2index(ndcard)] = self.remainNum
            discards_w.append(self.left_card_weight(card=card, left_num=left_numCP))  # 更新点：添加废牌权重
        return discards[discards_w.index(min(discards_w))]

    def get_efcCards(self, dz_set=[]):
        """
        获取所有搭子的有效牌，不去重
        :param dz_set: 搭子集合
        :return: effective_cards　[] 有效牌集合　不去重
        """
        effective_cards = []
        for dz in dz_set:
            if len(dz) == 1:
                effective_cards.append([dz[0]])
            elif dz[1] == dz[0]:
                effective_cards.append([dz[0]])
            elif dz[1] == dz[0] + 1:
                if int(dz[0]) & 0x0F == 1:
                    effective_cards.append([dz[0] + 2])
                elif int(dz[0]) & 0x0F == 8:
                    effective_cards.append([dz[0] - 1])
                else:
                    effective_cards.append([dz[0] - 1, dz[0] + 2])

            elif dz[1] == dz[0] + 2:
                effective_cards.append([dz[0] + 1])
        return effective_cards

    def contain(self, ab1=[], ab2=[]):
        """
        功能：计算组合是否存在包含关系，例如467中组合４６会包含于６７中，需要去除前者，避免重复计算。
        思路：分别判断２个组合中的搭子有效牌是否存在包含关系，先分别获取搭子的有效牌，如果某一组合的所有有效牌都包含于另一组合，则判定该组合包含于另一组合中。
            如果ab２有效牌全部包含于ab1 中，返回１，相反则返回２，没关系则返回０
        :param ab1: 组合１的搭子集合
        :param ab2: 组合２的搭子集合
        :return: int 如果ab２有效牌全部包含于ab1 中，返回１，相反则返回２，没关系则返回０
        """
        efc1 = self.get_effective_cards(ab1)
        efc2 = self.get_effective_cards(ab2)

        # 判断ab1 是否包含于ab2中
        contain1in2 = True
        for ab in ab1:
            if ab in ab2:
                continue
            else:
                efc = self.get_effective_cards([ab])
                if len(efc) == 2:
                    contain1in2 = False
                    break
                elif efc[0] in efc2:
                    continue
                else:
                    contain1in2 = False
                    break

        contain2in1 = True
        for ab in ab2:
            if ab in ab1:
                continue
            else:
                efc = self.get_effective_cards([ab])
                if len(efc) == 2:
                    contain2in1 = False
                    break
                elif efc[0] in efc1:
                    continue
                else:
                    contain2in1 = False
                    break
        if contain1in2:
            return 2
        elif contain2in1:
            return 1
        else:
            return 0

    def mergeSameall(self, all):
        """
        功能：对组合进行去重处理，去除有效牌全部包含于另一组合的情况，例如　３４５６会被拆分为　３４５　４５６　两种情况，５７８　会被拆分为５７和７８情况，避免了后面评估值计算时的重复计算
        思路：遍历组合，对本次组合后面的所有组合判断时候存在包含关系，当存在包含关系时，更新有效牌多的组合为本次的最终组合，并标记已被去除的组合，该组合不再被遍历
        :param all: 组合信息
        :return: 去重后的组合
        """
        used_index = []
        all3 = []
        # 合并去掉
        # todo 有效牌相同的组也可以合并
        for i in range(len(all)):  # 将２Ｎ相同的组合并
            a = MJ.deepcopy(all[i])
            if i in used_index:
                continue
            for j in range(i + 1, len(all)):
                if len(all[j][0]) + len(all[j][1]) == len(a[0]) + len(a[1]) and all[j][2] == a[2]:
                    if all[j][3] == a[3]:
                        used_index.append(j)
                        for card in all[j][-1]:
                            if card not in a[-1]:
                                a[-1].append(card)
                                # else:
                                #
                                #     relation = self.contain(a[3], all[j][3])
                                #     if relation == 1:
                                #         # a=copy.copy(all[j])
                                #         used_index.append(j)
                                #
                                #
                                #     elif relation == 2:  #todo 这样换可能会导致前面已经合并的被移除了．但是这种可能很少
                                #
                                #         a = copy.deepcopy(all[j])
                                #         used_index.append(j)
            all3.append(a)
        return all3

    def pre_qingYiSe(self):
        wan_len = len(self.split[0])
        tiao_len = len(self.split[1])
        tong_len = len(self.split[2])
        suit_type = [0, 0, 0]
        for suit in self.suits:
            if suit[0] & 0xf0 == 0:
                wan_len += 3
                suit_type[0] = 1
            elif suit[0] & 0xf0 == 1:
                tiao_len += 3
                suit_type[1] = 1
            else:
                tong_len += 3
                suit_type[2] = 1
        if sum(suit_type) > 1:
            return 0
        len_ = [wan_len, tiao_len, tong_len]
        max_len = max(len_)
        if max_len >= 10:
            max_index = len_.index(max_len)
        self.sys_info_V3(cards=self.split[max_index], suit=self.suits)
        return 1

    def pre_fan_detect(self, CS):
        pass
        # 清一色
        # for a in CS:
        #     if

    def defend_V2_2(self, all_combination):
        """
        功能：出牌策略
        思路：分为３阶段，第一阶段完全孤张牌出牌策略，计算出所有组合中都包含的孤张牌，出评估值最低的孤张牌，剩余牌与孤张牌的联系性最低
                    第二阶段：（当没有孤张）采用搜索树计算出最佳出牌
                    第三阶段：（当已经是胡牌的状态再进行出牌时），采用快速评估的方法计算出最佳出牌
        :param all_combination: 组合信息
        :return: 决策出牌
        """
        # 1、先出定缺牌 todo 等待完善
        if self.dingque_cards != []:
            return self.dingque_cards[0]

        # 2、观察是否有大番的牌型
        # 清一色
        '''
                           第一阶段：完全孤张牌出牌策略
                           原则：出相关性最低的孤张牌，剩余牌与孤张牌的联系性最低
                           现阶段只考虑ｘｔｓ最小的情况
        '''
        # # 有效牌数量为0的组合应该被视为废牌 todo 宝还原
        # if True:  # 这一段是必须的！
        #     if self.kingNum <= 1:  # 这里只考虑出牌、宝做宝吊的情况
        #         for a in all_combination:
        #             for i in range(len(a[3]) - 1, -1, -1):
        #                 ab = a[3][i]
        #                 efc = self.get_effective_cards([ab])
        #                 if sum([LEFT_NUM[MJ.convert_hex2index(e)] for e in efc]) <= 0:  # 先只算有效牌数量为0
        #                     a[3].remove(ab)
        #                     a[-1].extend(ab)
        #                     logger.info("remove ab with low getting rate, %s,%s,%s,a=%s", self.cards, self.suits,
        #                                 self.kingCard, a)
        #     all_combination = self.xts(all_combination, self.suits, self.kingNum)
        all_same_xts = []
        # all_same_xts_and_left = []

        min_xts = all_combination[0][-2]
        for a in all_combination:  # 获取ｘｔｓ相同的组合
            if a[-2] == min_xts:
                all_same_xts.append(a)
            # if a[-2] == min_xts and len(a[-1])==len(all_combination[0][-1]):
            #     all_same_xts_and_left.append(a)
        all_MG = copy.copy(all_same_xts)

        # 移除搭子有效牌被覆盖的划分 ，可能出现3 56的情况，3会获得更多的机会123，234,333，345
        # for a in all_same_xts:
        #     flag = False
        #     for t1 in a[-1]:
        #         if not flag:
        #             for t2 in a[2] + a[3]:
        #                 th = copy.copy(t2)
        #                 th.append(t1)
        #                 th.sort()
        #                 if th in MJ.T2_HALF:
        #                     if t2 not in MJ.T2_HALF_T2 or (
        #                             t2 in [[2, 4], [6, 8], [0x12, 0x14], [0x16, 0x18], [0x22, 0x24],
        #                                    [0x26, 0x28]] and t1 not in [1, 9, 0x11, 0x19, 0x21, 0x29]):
        #                         logger.info("remove duplication cs, %s,%s,%s", a, t2, t1)
        #                         all_MG.remove(a)
        #                         flag = True
        #                         break

        # if all_MG == []:
        #     all_MG = all_same_xts

        # 去重处理

        # for a in all_MG: #todo 20201013
        #     a_temp = MJ.deepcopy(a)
        #     for i in range(len(a_temp[3]) - 1, -1, -1):
        #         ab = a_temp[3][i]
        #         efc = self.get_effective_cards([ab])
        #         if sum([LEFT_NUM[MJ.convert_hex2index(e)] for e in efc]) <= 1:  #先只算有效牌数量为0
        #             a_temp[3].remove(ab)
        #             a_temp[-1].extend(ab)
        #             # logger.info("append ab with low getting rate, %s,%s,%s,a=%s", self.cards, self.suits, self.kingCard, a)
        #     if a_temp!=a:
        #         all_MG.append(a_temp)

        # print ('all_MG', all_MG)
        left_all_cards = []  # 全部组合的废牌集合

        for branch in all_MG:
            left_all_cards += branch[-1]
        unique_l = list(set(left_all_cards))
        left_cards = []  # 任何组合都包含的真正废牌
        left_cards_w = []
        need_jiang = False
        if all_MG[0][-2] == 1:
            if len(all_MG[0][0]) + len(all_MG[0][1]) + len(self.suits) == 4 and all_MG[0][-1] == 2:
                need_jiang = True

        for card in unique_l:
            if left_all_cards.count(card) == len(all_MG):
                left_cards.append(card)
                left_cards_w.append(
                    self.left_card_weight(card=card, left_num=LEFT_NUM, need_jiang=need_jiang))  # 更新点：添加废牌权重
        if left_cards != []:  # and all_MG[0][-2]>3:
            # if min(left_cards_w)<25: #当出37 5 的时候需要限制下
            # 这里也只能在搭子过多的情况下才会出，给的限制条件放宽点
            # if need_jiang or ((not need_jiang) and min(left_cards_w)<70):
            if True:
                # print ('state first')
                return left_cards[left_cards_w.index(min(left_cards_w))]

        '''
        第二阶段
        当unique_l不为空时，从所有废牌(unique_l)中出一张
        如果为空，从所有的t2Ｎ中出一张
        '''
        # 在ｘｔｓ<3的情况下，使用搜索树
        # if all_MG[0][4] <= 3:
        if False:
            Tree = SearchTree(cards=self.cards, suits=self.suits, leftNum=self.leftNum, all=all_same_xts,
                              remainNum=self.remainNum, dgtable=[1] * 34, kingCard=self.kingCard,
                              feiKingNum=self.fei_king)
            scoreDict = Tree.getCardScore()
            king_score = 0
            if self.kingCard in scoreDict.keys():
                king_score = scoreDict[self.kingCard]
            scoreDict = sorted(scoreDict.items(), key=lambda k: k[1], reverse=True)
            maxScoreCards = []
            # print ('scoreDict',scoreDict)
            if scoreDict != [] and king_score * 1.5 >= scoreDict[0][1]:
                return self.kingCard

            for i in range(len(scoreDict)):
                # print (scoreDict[i][1],scoreDict[0][1])
                if scoreDict[i][1] == scoreDict[0][1]:
                    maxScoreCards.append(scoreDict[i][0])
            # print ('maxScoreCards', maxScoreCards)
            # print(scoreDict)
            # if maxScoreCards != []:
            #     return self.discards_w(maxScoreCards, self.leftNum, ndcards={})

        # 加入处理概率过低的搭子的组合
        # todo 容易出现超时,增加向听数小于等于3的限制条件
        if False:
            # if all_MG[0][-2]<=3:
            supplement = []
            for a in all_MG:
                # print a
                a_copy = MJ.deepcopy(a)
                for ab in a[3]:
                    efc = self.get_effective_cards([ab])
                    # print ab,sum([LEFT_NUM[MJ.convert_hex2index(e)] for e in efc])
                    if sum([LEFT_NUM[MJ.convert_hex2index(e)] for e in efc]) <= 1:
                        a_copy[3].remove(ab)
                        a_copy[-1].extend(ab)
                        # logger.info("remove rate 0 ab,%s,%s,%s,a=%s", self.cards, self.suits, self.kingCard, a)
                        # break

                if len(a_copy[3]) != len(a[3]):
                    supplement.append(a_copy)
                    # logger.info("supplement a1=%s,a2=%s", a, a_copy)
            all_MG.extend(supplement)

        # 有效牌数量为0的组合应该被视为废牌 todo 宝还原
        if False:  # 这一段是必须的！
            if self.kingNum <= 1:  # 这里只考虑出牌、宝做宝吊的情况
                for a in all_MG:
                    for i in range(len(a[3]) - 1, -1, -1):
                        ab = a[3][i]
                        efc = self.get_effective_cards([ab])
                        if sum([LEFT_NUM[MJ.convert_hex2index(e)] for e in efc]) <= 0:  # 先只算有效牌数量为0
                            a[3].remove(ab)
                            a[-1].extend(ab)
                            # logger.info("remove ab with low getting rate, %s,%s,%s,a=%s", self.cards, self.suits,
                            #             self.kingCard, a)
                all_MG = self.xts(all_MG, self.suits, self.kingNum)

        if False:
            # 加入碰碰胡处理 加到后面并不影响孤张出牌，只在搜索中使用碰碰胡
            rm_king = copy.copy(self.cards)
            for i in range(self.kingNum):
                rm_king.remove(self.kingCard)
            a_pengpenghu = self.pengpengHu(outKingCards=rm_king, suits=self.suits, kingNum=self.kingNum)
            if a_pengpenghu != [] and a_pengpenghu[0][-2] - 1 <= all_MG[0][-2]:  # 现在用1
                if a_pengpenghu[0] not in all_MG:  # 有可能已经存在于all_MG
                    all_MG.append(a_pengpenghu[0])

        # print ("all_MG", all_MG)

        # 简化版搜索树
        if True:
            # if all_MG[0][-2]<=3:

            Tree = SearchTree_take(hand=self.cards, suits=self.suits, combination_sets=all_MG)
            t1 = time.time()
            scoreDict = Tree.get_discard_score()
            t2 = time.time()
            # if t2 - t1 > 2.9:  # 超时了
                # logger.error("time:%i,info:%s, %s, %s", t2 - t1, self.cards, self.suits, self.kingCard)
            king_score = 0  # 增加飞宝得分倍率1.5
            if self.kingCard in scoreDict.keys():
                king_score = scoreDict[self.kingCard]
            scoreDict = sorted(scoreDict.items(), key=lambda k: k[1], reverse=True)
            maxScoreCards = []
            # 希望给飞宝更多的分数,向听数越大飞宝概率越低,希望在接近胡牌时才会选择飞宝
            if scoreDict != [] and king_score != 0 and king_score * 1.5 >= scoreDict[0][1]:  # 9.23 增加2倍
                return self.kingCard
                # all_MG_cp = MJ.deepcopy(all_MG)
                # print self.xts(all_MG_cp,self.suits,self.kingNum-1)[0][-2],all_MG[0][-2]
                # if self.xts(all_MG_cp,self.suits,self.kingNum-1)[0][-2]==all_MG[0][-2]:
                #     w = 2
                # else:
                #     w = 1.5
                # w=random.uniform(1.0,2.0)
                # print w
                # if king_score * w >= scoreDict[0][1]:
                #     return self.kingCard
                #     if len(all_MG[0][2])==1 and len(all_MG[0][3])==1:
                #         # print "n",sum([LEFT_NUM[MJ.convert_hex2index(i)] for i in self.get_effective_cards(all_MG[0][3])])
                #         if sum([LEFT_NUM[MJ.convert_hex2index(i)] for i in self.get_effective_cards(all_MG[0][3])])>4:
                #             return self.kingCard
                #     elif len(all_MG[0][2])==2:
                #         if sum([LEFT_NUM[MJ.convert_hex2index(i)] for i in self.get_effective_cards(all_MG[0][2])])>2:
                #             return self.kingCard
                # else:
                #     if king_score * 1.5 >= scoreDict[0][1]
            for i in range(len(scoreDict)):
                if scoreDict[0][1] != 0 and scoreDict[i][1] == scoreDict[0][1]:
                    maxScoreCards.append(scoreDict[i][0])
            # print ('maxScoreCards2', maxScoreCards)
            if maxScoreCards != []:
                return self.discards_w(maxScoreCards, self.leftNum, ndcards={})
            # else:
                # logger.warning("recommond card is empty!%s,%s,%s,%s,%s", self.cards, self.suits, self.kingCard,
                #                self.discards, self.discardsOp)

        if True:
            path_w = self.cost(all=all_MG, suits=self.suits)  # , left_num=self.leftNum, king_num=self.kingNum,king_card=self.kingCard

            path_w.sort(key=lambda k: k[0], reverse=True)

            if path_w[0][-1] == []:  # 已经胡牌

                max_remove_3N = 0
                remove_card = 0
                # flag = False
                for a in all_MG:
                    if a[4] == 0:
                        # flag = True
                        if a[1] != []:
                            for t3 in a[0] + a[1]:
                                lc = self.get_effective_cards(dz_set=[[t3[1], t3[2]]])
                                ln = sum([self.leftNum[MJ.convert_hex2index(e)] for e in lc])

                                if ln >= max_remove_3N:
                                    max_remove_3N = ln
                                    remove_card = t3[0]
                                rc = self.get_effective_cards(dz_set=[[t3[0], t3[1]]])
                                rn = sum([self.leftNum[MJ.convert_hex2index(e)] for e in rc])
                                if rn > max_remove_3N:
                                    max_remove_3N = rn
                                    remove_card = t3[2]
                        elif len(a[2]) != []:  # 单吊
                            remove_card = a[2][0][0]
                # print("defend_V2_2,has Hu,and out a highest rate card", 1 / remove_card, remove_card)
                return remove_card
            out_card = self.discards_w(discards=path_w[0][-1], left_num=self.leftNum, ndcards={})
            return out_card
            # for i in range(len(all_MG)):
            #     for card in set(path_w[i][1]): #todo 修改点
            #         if card in discards_w.keys():
            #             # todo 需要加上场面剩余牌信息
            #             discards_w[card] += path_w[i][0]
            #         else:
            #             discards_w[card] = path_w[i][0]
            # discards_w = sorted(discards_w.items(), key=lambda k: k[1], reverse=True)
            # discards=[]
            # print ("discards_w", discards_w)
            # for tw in discards_w:
            #     if tw[1]==discards_w[0][1]:
            #         discards.append(tw[0])
            #
            # return int(self.discards_w(discards=discards, left_num=self.leftNum, ndcards=ndcards))

            # else:
            #   # 如果废牌区为空，使用搜索，出价值最低的２Ｎ
            #     path_w = self.cost(all=all_MG, suits=self.suits, left_num=self.leftNum, king_num=self.kingNum,
            #                        king_card=self.kingCard)
            #     path_w.sort(key=lambda k: k[0], reverse=True)
            #     if path_w[0][-1] == []:  # 已经胡牌
            #
            #         max_remove_3N = 0
            #         remove_card = 0
            #         # flag = False
            #         for a in all_MG:
            #             if a[4] == 0:
            #                 # flag = True
            #                 if a[0] + a[1] != []:
            #                     for t3 in a[0] + a[1]:
            #                         lc = self.get_effective_cards(dz_set=[[t3[1], t3[2]]])
            #                         ln = sum([self.leftNum[translate16_33(e)] for e in lc])
            #
            #                         if ln >= max_remove_3N:
            #                             max_remove_3N = ln
            #                             remove_card = t3[0]
            #                         rc = self.get_effective_cards(dz_set=[[t3[0], t3[1]]])
            #                         rn = sum([self.leftNum[translate16_33(e)] for e in rc])
            #                         if rn >= max_remove_3N:
            #                             max_remove_3N = rn
            #                             remove_card = t3[2]
            #                 elif len(a[2]) == 1:  # 单吊
            #                     remove_card = a[2][0][0]
            #         print("defend_V2_2,has Hu,and out a highest rate card", 1/remove_card,remove_card)
            #         return remove_card
            #
            #     out_card = self.discards_w(discards=path_w[0][-1], left_num=self.leftNum,ndcards=ndcards)
            #     print (path_w)
            #     print ("out_card", out_card)
            #     return out_card

    # 决策出牌
    def recommend_card(self):
        """
        推荐出牌接口
        :return: 返回最佳出牌
        """
        all = self.sys_info_V3(cards=self.cards, suits=self.suits)
        return self.defend_V2_2(all_combination=all)

    def hu_info(self, all, suits, kingNum):
        """
        功能：计算胡牌后的组合信息
        思路：当胡牌后，综合计算出组合信息和副露中的kz,sz,jiang
        :param all: 组合信息
        :param suits: 副露
        :param kingNum: kingNum宝牌数量
        :return: kz ,sz ,jiang
        """
        kz_suits = []
        sz_suits = []
        for suit in suits:
            if suit[0] == suit[1]:
                kz_suits.append(suit)
            else:
                sz_suits.append(suit)
        for a in all:
            kz = []
            kz.extend(kz_suits)
            sz = []
            sz.extend(sz_suits)

            jiang = 0x00

            if a[4] == 0:

                for kz_ in a[0] + a[2]:
                    # if
                    kz.append(kz_)
                for sz_ in a[1] + a[3]:
                    if sz_[0] != 8:
                        sz.append(sz_)
                    else:
                        sz.append(sz_ - 1)

                if kingNum != 0:
                    jiang = [0, 0]
                else:
                    jiang = a[2][0]
                return kz, sz, jiang
        return [], [], 0

    def recommend_op(self, op_card, canchi=False, self_turn=False, isHu=False):
        """
        功能：动作决策，包括吃碰杠胡的判断
        思路：胡牌判断：当有杠时，判断杠是否为暗杠，是则直接杠，
                                        否则判断杠后是否仍然胡牌，若是则杠，
                                                            否则接着判断，若本手胡牌基础分>8,则直接胡，否则杠，
                当有多宝时，如果飞宝能在３手内胡牌，则先飞宝，不胡，否则胡
            杠牌判断：有杠就杠
            吃碰：采用了反向胡牌概率比较策略，若吃碰后的概率大于不执行动作的概率，则执行吃碰，否则pass
        :param op_card: 操作牌
        :param canchi: 能否吃牌权限
        :param self_turn: 是否是自己回合
        :param isHu: 是否已经胡牌
        :return: [],isHu 前者为吃碰杠的组合　后者为是否胡牌
        """
        # 2项比较：前项计算胡牌ｒａｔｅ，吃碰杠后计算胡牌ｒａｔｅ比较,杠牌在不过多影响条件下都进行，其他需增加胡牌概率
        cards = self.cards
        suits = self.suits
        left_num = self.leftNum
        cards_former = copy.copy(cards)
        cards_former.append(0)
        all_former = self.sys_info_V3(cards=cards_former, suits=suits)
        # print ("recommend_op,all_former", all_former)
        # 计算前向胡牌概率 完全局部最优策略
        path_w_former = self.cost(all=all_former, suits=suits)  # , left_num=left_num, king_num=self.kingNum,
        # king_card=self.kingCard

        path_w_former.sort(key=lambda k: (k[0]), reverse=True)
        # print ("path_w_former", path_w_former)
        rate_former = path_w_former[0][0]  # 未执行动作的胡牌概率

        # 是否胡牌判断
        if isHu:
            # logger.info("deal with Hu...")
            # return [],True
            '''
                补杠如果能杠胡则杠，
                如果不能杠胡:本次手牌的分数较高则不杠直接胡，
                    如果本手牌分数为１２分(最低分)：如果杠了后胡牌几率陡降，不能胡了则不杠，
                                                如果杠了胡牌几率仍然较大，则先杠
            '''

            # 暗杠补杠判断
            for card in cards:
                # 暗杠２４　分必须要
                if cards.count(card) == 4:
                    # logger.info("choose AnGong,%s,%s,%s", self.cards, self.suits, self.kingCard)
                    return [card, card, card, card], False

            for card in cards:
                if [card, card, card] in suits:  # 处理补杠
                    cards_BuGang = copy.copy(cards)
                    cards_BuGang.remove(card)
                    all_BuGang = self.sys_info_V3(cards=cards_BuGang, suits=suits) # , left_num=left_num,kingCard=self.kingCard

                    asset = self.cost(all_BuGang, suits=suits)
                    #, left_num=left_num,
                    #                 king_num=cards_BuGang.count(self.kingCard), king_card=self.kingCard

                    asset.sort(key=lambda k: (k[0]), reverse=True)
                    buGangHuRate = asset[0][0]
                    # 如果补杠后也能胡，则直接杠，否则算期望
                    if buGangHuRate == 1:
                        # logger.info("choose buGang,%s,%s,%s", self.cards, self.suits, self.kingCard)
                        return [card, card, card, card], False
                    else:
                        return [], True
                        # kz, sz, jiang = self.hu_info(all_former, self.suits, kingNum=self.kingNum)
                        # if jiang == 0:
                        #     return [], True
                        # score = Fan(kz=kz, sz=sz, jiang=jiang, fei_king=self.fei_king, using_king=0, baohuanyuan=False)
                        # score = Fan(kz=kz, sz=sz, jiang=jiang, node=None, fei_king=self.fei_king)
                        # 胡牌分数高，则直接胡，否则，看几率
                        # if score >= 8:
                        #     return [], True
                        # else:
                        #     if buGangHuRate <= rate_former * 0.5:
                        #         return [], True
                        #     else:
                        #         return [card, card, card, card], False
            # return [],True
            # 手中有２张宝牌，先不胡，打掉一张宝牌后３手内的胡牌概率是否超过原有期望
            if self.kingNum >= 2:
                # 如果作为宝还原，宝吊则直接胡
                # if self.kingNum == 2:
                #     for a in all_former:
                #         if a[4] == 0 and len(a[0]) + len(a[1]) + len(suits) == 4:
                #             return [], True
                # return [], False

                cards_FeiBao = copy.copy(cards)
                cards_FeiBao.remove(self.kingCard)
                path_w_out1King = self.cost(all=all_former, suits=suits) #, left_num=left_num, king_num=self.kingNum - 1,
                                            # king_card=self.kingCard
                path_w_out1King.sort(key=lambda k: (k[0]), reverse=True)

                if path_w_out1King[0][0] * 2 < 1:

                    return [], True
                else:
                    # logger.info("abandon hu,%s,%s,%s", self.cards, self.suits, self.kingCard)
                    return [], False

            # 当手牌中只剩下一个面子，宝吊的概率
            # elif (self.kingNum == 1 and len(suits) == 3):
            #     rate = 0
            #     for a in all_former:
            #         if a[4] == 0:
            #             if len(a[0]) == 1:
            #                 # 碰３家没有自摸
            #                 rate += float(self.leftNum[convert_hex2index(a[0][0][0])] * 3) / self.remainNum
            #             elif len(a[1]) == 1:
            #                 cardSet = []
            #                 cardSet.extend(a[1][0])
            #                 if a[1][0][0] & 0x0f == 1:
            #                     cardSet.append(a[1][0][0] + 3)
            #                 elif a[1][0][0] & 0x0f == 9:
            #                     cardSet.append(a[1][0][0] - 1)
            #                 else:
            #                     cardSet.append(a[1][0][0] - 1)
            #                     cardSet.append(a[1][0][0] + 3)
            #                 for card in cardSet:
            #                     # 吃只能吃上家
            #                     rate += float(self.leftNum[convert_hex2index(card)]) / self.remainNum
            #     if rate * 2 * 2 <= 1or self.round>=10:
            #         return [], True
            #     else:
            #         return [], False

            else:
                return [], True

        # 杠牌限制，只杠已成型，且没有被用到的牌（在废牌区），杠牌没有分数奖励，只有多摸一张牌的机会
        # allSamexts = []
        # for a in all_former:
        #     if a[4] == all_former[0][4]:
        #         allSamexts.append(a)
        # 上饶麻将杠牌加分，这里直接能杠就杠
        if self_turn:  # 暗杠补杠
            # 是否存在暗杠,暗杠直接杠,补杠也杠
            for card in cards:
                if cards.count(card) == 4 or [card, card, card] in suits:
                    return [card, card, card,
                            card], False
        # 明杠
        if cards.count(op_card) == 3:
            return [op_card, op_card, op_card, op_card], False
        # prekingcard 得分点碰牌,这里算杠牌

        if op_card == self.preKingCard and cards.count(op_card) == 2:
            return [op_card, op_card, op_card], False

        cards_add_op = copy.copy(cards)
        cards_add_op.append(op_card)
        all_later = self.sys_info_V3(cards=cards_add_op, suits=suits) # , left_num=left_num, kingCard=self.kingCard
        val = []  # 记录满足条件的吃碰杠组合

        if canchi:  # 可以吃，碰
            for a in all_later:
                t3N = a[0] + a[1]
                # 针对上饶麻将单吊处理
                if op_card not in a[-1] and (
                        [op_card - 2, op_card - 1, op_card] in t3N or
                        [op_card - 1, op_card, op_card + 1] in t3N or
                        [op_card, op_card + 1, op_card + 2] in t3N or
                        [op_card, op_card, op_card] in t3N):
                    val.append(a)
        else:  # 只能碰
            for a in all_later:
                if (op_card not in a[-1]) and [op_card, op_card, op_card] in a[0]:
                    val.append(a)
        # print ("val", val)
        if val != []:
            path_w_later = self.cost(all=val, suits=suits) # , left_num=left_num, king_num=self.kingNum,king_card=self.kingCard

            # index记录有效的吃碰杠组合索引
            index = []
            for i_p in range(len(path_w_later)):
                if path_w_later[i_p][0] == 1 and self.kingNum == 0 and all_former[0][
                    4] == 1:  # 已胡牌,由于上饶麻将没有点炮胡，这里考虑下有效牌数量
                    efc_cards = []  # 未操作前的有效牌数量
                    max_remove_3N = 0  # 操作后，打掉一张３N的左或右边的一张牌，转变成２Ｎ后的有效牌数量
                    # aa+ab or aa+aa
                    for a in all_former:
                        if len(a[2]) == 1 and len(a[3]) == 1:
                            efc_cards.extend(self.get_effective_cards(dz_set=a[3]))
                            tianHu = True
                        elif len(a[2]) == 2 and len(a[3]) == 0:
                            efc_cards.extend(self.get_effective_cards(dz_set=a[2]))
                            tianHu = True
                        else:
                            tianHu = False
                        if tianHu:
                            if a[0] + a[1] != []:
                                for t3 in a[0] + a[1]:
                                    lc = self.get_effective_cards(dz_set=[[t3[1], t3[2]]])
                                    ln = sum([left_num[MJ.convert_hex2index(e)] for e in lc])
                                    # for card in lc:
                                    if ln > max_remove_3N:
                                        max_remove_3N = ln
                                    rc = self.get_effective_cards(dz_set=[[t3[0], t3[1]]])
                                    rn = sum([left_num[MJ.convert_hex2index(e)] for e in rc])
                                    if rn > max_remove_3N:
                                        max_remove_3N = rn
                            else:
                                # print a[2][0][0]
                                # 找到另一对被吃碰的牌，计算期望
                                t2Ns = a[2] + a[3]
                                for t2 in a[2] + a[3]:
                                    if op_card in self.get_effective_cards([t2]):
                                        t2Ns.remove(t2)
                                        break
                                # 单吊了
                                if self.leftNum[MJ.convert_hex2index(t2Ns[0][0])] * 2 > max_remove_3N:
                                    max_remove_3N = self.leftNum[MJ.convert_hex2index(t2Ns[0][0])]

                    efc_num = 0  # 胡牌的有效牌数量
                    efc_cards = set(efc_cards)
                    for card in efc_cards:
                        efc_num += left_num[MJ.convert_hex2index(card)]
                    # print ("efc_num,max_remove_3N", efc_num, max_remove_3N)
                    if max_remove_3N < efc_num * 1.2:  # or not (max_remove_3N==efc_num and len(cards)<=7):  # 如果有效牌数量增加，则执行此操作
                        return [], False  # continue

                # 有宝可以打宝吊，单吊
                # print (path_w_later[i_p][0], rate_former)
                if path_w_later[i_p][0] >= 1:
                    path_w_later[i_p][0] = 1
                if path_w_later[i_p][0] > rate_former:  # or (self.kingNum != 0 and len(cards) <= 4): #单吊
                    index.append([i_p, path_w_later[i_p][0]])
            index.sort(key=lambda k: k[1], reverse=True)
            if index != []:
                for t3 in val[index[0][0]][0] + val[index[0][0]][1]:  # 在最优吃碰杠组合中给出该３Ｎ,修正点，从all_later修正为ｖａｌ
                    # print ("op_ t3", t3)
                    if op_card in t3:
                        if canchi:
                            return t3, False
                        elif t3[0] == t3[1]:
                            return t3, False
        return [], False


class QiDui:
    def __init__(self, cards, suits=[], leftNum=[], discards=[], discards_real=[], discardsOp=[], round=0,
                 remainNum=136,
                 fengWei=0, seat_id=0, choose_color=[], hu_cards=[],
                 hu_fan=[]):
        cards.sort()
        self.cards = cards
        self.suits = suits
        self.discards = discards
        self.discards_real = discards_real
        self.discardsOp = discardsOp
        self.remainNum = remainNum
        self.remainNum = 50
        self.leftNum = leftNum

        self.round = min(13, round)

        self.seat_id = seat_id
        if leftNum == []:
            leftNum, discardsList = MJ.trandfer_discards(discards, discardsOp, cards, type=27)
            self.leftNum = leftNum
        T_selfmo = []  # 有效牌自摸概率表
        for i in self.leftNum:
            T_selfmo.append(float(i) / self.remainNum)

        self.fengWei = fengWei
        self.choose_color = choose_color
        self.hu_cards = hu_cards
        self.hu_fan = hu_fan

    def get_cards_num(self, cards=[]):
        """
        获取手牌中每张牌的数量
        :param cards: 手牌
        :return: cards_unique, cards_num　去重后的手牌及其数量
        """
        # if len(suits)!=0:
        #     return

        cards_unique = list(set(cards))
        cards_num = [0] * len(cards_unique)
        for i in range(len(cards_unique)):
            cards_num[i] = cards.count(cards_unique[i])

        return cards_unique, cards_num

    def qidui_info(self):
        """
        七对的相关信息｛｝
        包括cards_unique　去重后的手牌
        cards_num　每张牌的数量
        duipai　对牌
        left_cards　剩余牌
        xts　向听数
        :param cards:　手牌
        :param suits: 副露
        :param left_num:剩余牌
        :param king_num: 宝牌
        :return: ｛｝　qidui_info　字典格式存储的七对信息
        """
        cards = self.cards
        suits = self.suits
        qidui_info = {}
        if len(suits) != 0:
            qidui_info["xts"] = 14
            return qidui_info

        cards_unique, cards_num = self.get_cards_num(cards=cards)
        duipai = []
        left_cards = []
        for i in range(len(cards_unique)):
            if cards_num[i] == 4:
                duipai.append(cards_unique[i])
                duipai.append(cards_unique[i])
            elif cards_num[i] == 3:
                duipai.append(cards_unique[i])
                left_cards.append(cards_unique[i])
            elif cards_num[i] == 2:
                duipai.append(cards_unique[i])
            elif cards_num[i] == 1:
                left_cards.append(cards_unique[i])
        # for card in cards_unique:
        #     if
        qidui_info["cards_unique"] = cards_unique
        qidui_info["cards_num"] = cards_num
        qidui_info["duipai"] = duipai
        qidui_info["left_cards"] = left_cards
        if len(duipai) >= 3:
            qidui_info["xts"] = 14 - (len(duipai) * 2 + 7 - len(duipai))
        else:
            qidui_info["xts"] = 14

        return qidui_info

    def defend_V1(self, left_cards=[], left_num=[]):
        """
        七对出牌决策
        出剩余牌数量最低的牌
        :param left_cards:孤张
        :param left_num: 剩余牌数量
        :return: 最佳出牌
        """
        discards_order = [0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37, 0x01, 0x09, 0x11, 0x19, 0x21, 0x29, 0x02, 0x08,
                          0x12, 0x18, 0x22, 0x28, 0x03, 0x07, 0x13, 0x17, 0x23, 0x27, 0x04, 0x06, 0x14, 0x16, 0x24,
                          0x26, 0x05, 0x15, 0x25]
        effective_cards_num = [0] * len(left_cards)
        for i in range(len(left_cards)):
            # print("qidui,")
            effective_cards_num[i] = left_num[MJ.convert_hex2index(left_cards[i])]
        min_num = 4
        min_index = 0
        for i in range(len(effective_cards_num)):
            if min_num > effective_cards_num[i]:
                min_num = effective_cards_num[i]  # 忘了写了
                min_index = i
        # print ("ph.defend_V1,min_index",min_index)
        # print(left_cards)
        if left_cards == []:
            return 0x00
        else:
            return left_cards[min_index]
        return None

    def recommend_card(self):
        """
        七对出牌接口
        :param cards:手牌
        :param suits: 副露
        :param left_num: 剩余牌
        :param king_card: 宝牌
        :return: 最佳出牌
        """
        cards = self.cards
        suits = self.suits
        left_num = self.leftNum
        cards_copy = copy.deepcopy(cards)

        qidui_info = self.qidui_info()
        left_cards = qidui_info["left_cards"]
        return self.defend_V1(left_cards=left_cards, left_num=left_num)

    # 七对不考虑吃碰杠情况
    def recommend_op(self, op_card, cards=[], suits=[]):
        """
        七对不考虑动作决策，直接返回[]
        :param op_card: 操作牌
        :param cards: 手牌
        :param suits: 副露
        :return: []
        """
        return []


def paixing_choose(hand_cards=[], suits=[], discards=[], discards_op=[], op_card=None, choose_color=[-1, -1, -1, -1],
                   hu_cards=[[], [], [], []], hu_fan=[[], [], [], []]):
    """
    牌型选择
    通过计算向听数来判断
    :param cards: 手牌
    :param suits: 副露
    :param discards: 弃牌
    :param discards_op: 场面副露
    :param op_card: 操作牌
    :return: 牌型序号　０为平胡　１　为七对
    """
    left_num, discards_list = MJ.trandfer_discards(discards=discards, discards_op=discards_op, handcards=hand_cards)
    cards_op = copy.deepcopy(hand_cards)
    if op_card != None:
        cards_op.append(op_card)
    pinghu_info = pinghu(cards=hand_cards, suits=suits, discards=[], discards_real=[], discardsOp=discards_op, round=0,
                         remainNum=134,
                         fengWei=0, seat_id=0, choose_color=choose_color, hu_cards=hu_cards,
                         hu_fan=hu_fan).sys_info_V3(cards=cards_op, suits=suits)

    qidui_info = QiDui(cards=hand_cards, suits=suits, leftNum=left_num, discards=[], discards_real=[],
                       discardsOp=discards_op, round=0, remainNum=134,
                       fengWei=0, seat_id=0, choose_color=choose_color, hu_cards=hu_cards,
                       hu_fan=hu_fan).qidui_info()
    # print ("[pinghu_info[0], qidui_info[0]=",
    #        [pinghu_info[0][4], qidui_info["xts"]])

    min, index = MJ.get_min(
        list=[pinghu_info[0][4], qidui_info["xts"] + 1])
    return index


# 推荐出牌总接口
def recommend_card(cards=[], suits=[], round=0, remain_num=(9 + 0) * 4 * 3, discards=[], discards_real=[],
                   discards_op=[],
                   seat_id=0, choose_color=[], hu_cards=[[], [], [], []], hu_fan=[[], [], [], []]):
    # return cards[-1]
    # start = time.time()
    # print("start_recommend, time"+str(start))
    left_num, discards_list = MJ.trandfer_discards(discards=discards, discards_op=discards_op, handcards=cards, type=27)
    paixing = paixing_choose(hand_cards=cards, suits=suits, discards=discards, discards_op=discards_op, op_card=None,
                             choose_color=choose_color, hu_cards=hu_cards, hu_fan=hu_fan)
    if paixing == 0:
        # print 'choose_color',choose_color
        outCard = pinghu(cards, suits, leftNum=LEFT_NUM, discards=discards, discards_real=discards_real,
                         discardsOp=discards_op, remainNum=remain_num, round=round, fengWei=0, seat_id=seat_id,
                         choose_color=choose_color, hu_cards=hu_cards, hu_fan=hu_fan).recommend_card()
    elif paixing == 1:
        outCard = QiDui(cards, suits=suits, leftNum=left_num, discards=discards, discards_real=discards_real,
                        discardsOp=discards_op, round=round, remainNum=remain_num, fengWei=0, seat_id=seat_id,
                        choose_color=choose_color, hu_cards=hu_cards,
                        hu_fan=hu_fan).recommend_card()
    # end = time.time()
    # print('use time=', end - start)
    return outCard


# 推荐动作总接口
def recommend_op(op_card, cards=[], suits=[], round=0, remain_num=136, discards=[], discards_real=[], discards_op=[],
                 self_turn=False, seat_id=0, choose_color=[], hu_cards=[[], [], [], []], hu_fan=[[], [], [], []],
                 isHu=False):
    # return False,[]
    # start = time.time()
    # print("start_recommend_op, time" + str(start))
    if isHu == True:
        return [], True
    left_num, discards_list = MJ.trandfer_discards(discards=discards, discards_op=discards_op, handcards=cards, type=27)
    paixing = paixing_choose(hand_cards=cards, suits=suits, discards=discards, discards_op=discards_op, op_card=None,
                             choose_color=choose_color, hu_cards=hu_cards, hu_fan=hu_fan)
    if paixing == 0:
        return pinghu(cards, suits, leftNum=left_num, discards=discards, discards_real=discards_real,
                      discardsOp=discards_op, remainNum=remain_num, round=round, fengWei=0, seat_id=seat_id,
                      choose_color=choose_color, hu_cards=hu_cards, hu_fan=hu_fan).recommend_op(op_card=op_card,
                                                                                                self_turn=self_turn,
                                                                                                isHu=isHu)
        # end = time.time()
        # print('use time=', end - start)
    elif paixing == 1:
        # return QiDui().recommend_op()
        # end = time.time()
        # print('use time=', end - start)
        return [], False  # 七对不考虑
    # else:
    #     print ('recomend_op Error!')


def recommend_switch_cards(hand_cards=[], switch_n_cards=3):
    switch_cards = SwitchTiles(hand=hand_cards, n=switch_n_cards).switch_cards()
    return switch_cards


def recommend_choose_color(hand_cards=[], switch_n_cards=3):
    choose_color = SwitchTiles(hand=hand_cards, n=switch_n_cards).choose_color()
    return choose_color

# 推荐出牌总接口
def recommend_card_rf(cards=[], suits=[], round=0, remain_num=(9 + 0) * 4 * 3, discards=[], discards_real=[],
                   discards_op=[],
                   seat_id=0, choose_color=[], hu_cards=[[], [], [], []], hu_fan=[[], [], [], []]):
    # return cards[-1]
    # start = time.time()
    # print("start_recommend, time"+str(start))
    left_num, discards_list = MJ.trandfer_discards(discards=discards, discards_op=discards_op, handcards=cards, type=27)
    paixing = paixing_choose(hand_cards=cards, suits=suits, discards=discards, discards_op=discards_op, op_card=None,
                             choose_color=choose_color, hu_cards=hu_cards, hu_fan=hu_fan)
    if paixing == 0:
        # print 'choose_color',choose_color
        outCard = pinghu(cards, suits, leftNum=LEFT_NUM, discards=discards, discards_real=discards_real,
                         discardsOp=discards_op, remainNum=remain_num, round=round, fengWei=0, seat_id=seat_id,
                         choose_color=choose_color, hu_cards=hu_cards, hu_fan=hu_fan).recommend_card()
    elif paixing == 1:
        outCard = QiDui(cards, suits=suits, leftNum=left_num, discards=discards, discards_real=discards_real,
                        discardsOp=discards_op, round=round, remainNum=remain_num, fengWei=0, seat_id=seat_id,
                        choose_color=choose_color, hu_cards=hu_cards,
                        hu_fan=hu_fan).recommend_card()
    # end = time.time()
    # print('use time=', end - start)
    return paixing, outCard