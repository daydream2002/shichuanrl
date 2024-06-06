# -*- coding:utf-8 -*-
# ｂｕｇ：递归迭代中有些情况没有返回值而无法返回
# input 一种花色手牌，９^5种
# python 2.0　两整数相处会自动取整，需要人为给被除数添加float型
# 最快胡牌策略出牌
'''
v1的改进版
sys_info_v3 当２Ｎ与３Ｎ数量小于５时，存在没有减去相应待填数，即废牌也会有１张作为２Ｎ或３Ｎ的待选位,
２０１９／７／１９　改用搜索树处理ｘｔｓ在３以下的情况，搜索树使用了sys_info_v3的信息，算是结合了全局信息，并且在计算有效牌的概率时，综合了获取途径，
ａａ为４，ab/ac为２，在胡牌时，都为４
todo 给废牌添加一个权重哦，可以在对手模型中添加
todo 同一路径中的牌计算会有相同的权重，需要用对手模型加以区分
todo  优化结点的结构，可以用字典来存子结点 （暂时不处理，由于使用了sz,kz 来区分存储结点信息，ｋｅｙ也可以加上这些）

７.22
将权重进行了调整，ａａ为4.5,ab/ac为２，调整aa 为６
优化了tree_expand的结构，加快搜索
修改了不需要使用copy.deepcopy为copy.copy,加快计算速度
增加了对手模型，初始为0.9，需要进一步优化
todo 在只有孤张牌的时候，搜索树的抓牌结点无法扩展，所以该策略需要在第二阶段使用
todo 胡牌种类与数量的关系


7.23
更新ａａ，ａｂ／ａｃ的权重计算方式，改为自身摸牌与其他玩家出牌，ｐ＝num/remainNum+ways*class/34　todo 可以将３４进一步优化
将搜索深度改为４，并且不再扩展一层（由于策略集限制，扩展不平衡） 再次修正为３＋０　，４还是有少许不能在３ｓ搜索完，todo 测试3+0 3+1(half)  3+1(full)
verylongcat 在xts５和5以上的时候，使用估值函数快速胡牌，在４的时候使用４＋０　，在３及以下时，使用xts+1搜索
untest 给all_same_xts_and_left添加了去重处理



todo 番型检测中，可以设计结点的结构，fan: kz,sz,jiang,将副露也加入到该结构中，此结构非常方便于番型检测，

7.24
kz sz jiang都可以用第一张牌来代替，可以减少存储空间，并且在ｃｏｐｙ时可以不用ｄｅｅｐｃｏｐｙ
加入了番型检测
test1　将ａａ权重调到６，ｃｏｓｔ排序从ab,ac开始先出
    局部最优贯彻到底，３层出牌策略，先出孤张，当有２３３　类型牌时，用ｃｏｓｔ计算期望，当只有２Ｎ时，用ｃｏｓｔ计算最优的分组，再出没用的２Ｎ
    test2 表现好是由于第三层策略被搜索树替代了，所以不会输

７、２５
测试搜索树
todo　搜索树重复计算的结点要删除，ｖ１版可以考虑局部最优
移除了搜索树重复计算的结点
cost有重复计算的嫌疑
node 2 是加了番型检测的 (有ｂｕｇ呀)
tencent cloud  没有加番型检测


７、２６
优化ｏｐ ,在xts小于等于３的情况下使用搜索树的方法
调试搜索树ｂｕｇ（去重ｂｕｇ，创建结点的赋值kz,sz,bug）
　

7.27
清除重复结点的结算


7.29
更新了discards_w计算的ｌｅｆｔ_num
结点去重更新sz,kz排序,
更新 dict keys()

7.30

7.31
调试ｂｕｇ，将深度写反了e
精简代码
8.1
修改了昨天的ｂｕｇ（将t2NCP置空了）
测试opV1.2（使用v1.2的权重计算与杠判断）
测试opV2（在xts<3,使用搜索树）

test:1.测试纯净版（没有对手模型与搜索树，番型检测） test1
        改动点:废牌权重，移除手牌权重，ｃｏｓｔ中，２Ｎ排序改为１＋，不再添加ｌａｓｔaa概率，
    ２、加入搜索树，不加番型
        将ａａ权重　都调整为６，保留ａａ

    ３、加入搜索树，加入番型检测

    4、优化ｏｐ
'''

'''
此版本为四川麻将
框架分为换牌-定缺-出牌决策-动作决策-对手建模5部分
'''


"""
接口：
出牌
http://172.81.238.92:8081/sichuanMJ/v1/outcard


curl -XPOST -H 'Content-Type: application/json' http://172.81.238.92:8081/sichuanMJ/v1/outcard -d '{"catch_card":1,"discards":[],"discards_op":[[],[[23,24,25]],[[7,8,9]],[]],"discards_real":[],"remain_num":49,"round":6,"seat_id":0,"user_cards":{"hand_cards":[1,1,1,5,5,5,7,7,7,17,18,20,22,24],"operate_cards":[]},"choose_color":[2,0,0,0],"hu_cards":[[],[],[],[]],"hu_fan":[[],[],[],[]]}'

碰杠胡动作
http://172.81.238.92:8081/sichuanMJ/v1/operate

curl -XPOST -H 'Content-Type: application/json' http://172.81.238.92:8081/sichuanMJ/v1/operate -d '{"catch_card":1,"discards":[],"discards_op":[[],[[23,24,25]],[[7,8,9]],[]],"discards_real":[],"remain_num":49,"round":6,"seat_id":0,"user_cards":{"hand_cards":[1,1,1,5,5,5,7,7,7,17,18,20,22,24],"operate_cards":[]},"choose_color":[2,0,0,0],"hu_cards":[[],[],[],[]],"hu_fan":[[],[],[],[]]}'

换牌
http://172.81.238.92:8081/sichuanMJ/v1/switch_cards
curl -XPOST -H 'Content-Type: application/json' http://172.81.238.92:8081/sichuanMJ/v1/switch_cards -d '{"switch_n_cards":4,"user_cards":{"hand_cards":[2,4,4,5,6,17,18,19,24,33,34,38,40],"operate_cards":[]}}'

定缺
http://172.81.238.92:8081/sichuanMJ/v1/choose_color
curl -XPOST -H 'Content-Type: application/json' http://172.81.238.92:8081/sichuanMJ/v1/choose_color -d '{"catch_card":1,"discards":[],"discards_op":[[],[[23,24,25]],[[7,8,9]],[]],"discards_real":[],"remain_num":49,"round":6,"seat_id":0,"user_cards":{"hand_cards":[1,1,1,5,5,5,7,7,7,17,18,20,22,24],"operate_cards":[]},"choose_color":[2,0,0,0],"hu_cards":[[],[],[],[]],"hu_fan":[[],[],[],[]]}'

"""

import copy
# import numpy as np
import time
import random
#import matplotlib.pyplot as plt
import SCMJ.sichuanMJ.lib_MJ as MJ


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
            print "SwitchTiles ERROR! input type error,n=0, 3 or 4."
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
            print 'SwitchTiles, choose_color ERROR! len(self.hand)=13 or 14, but =' + len(self.hand)
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
        color_index=self.choose_color_index()
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
        #当有多种花色需要选择时，对此进行评估
        value = [-1, -1, -1]#花色的评估值
        cc = [[], [], []]#花色的待选牌
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


# 全局变量


# 向听数随轮数的分布表
Txts = [[4, 14, 81, 286, 680, 1294, 1999, 2906, 3709, 4525, 5314, 5964, 6518, 6990],
        [101, 361, 952, 1910, 2865, 3763, 4421, 4625, 4545, 4269, 3893, 3518, 3121, 2787],
        [994, 2125, 3268, 3961, 4109, 3601, 2835, 2100, 1536, 1094, 729, 480, 342, 209],
        [3284, 3951, 3776, 2948, 1951, 1145, 657, 319, 178, 93, 50, 27, 12, 9],
        [3708, 2752, 1607, 781, 345, 165, 61, 27, 13, 3, 3, 2, 0, 0],
        [1569, 670, 270, 87, 33, 13, 10, 7, 3, 4, 3, 2, 1, 2],
        [308, 114, 41, 23, 12, 13, 11, 12, 10, 8, 5, 2, 3, 0],
        [29, 12, 4, 3, 4, 5, 5, 2, 4, 2, 1, 3, 3, 2],
        [2, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0],
        ]
Txts_transpose = list(zip(*Txts))
[(4, 101, 994, 3284, 3708, 1569, 308, 29, 2), (14, 361, 2125, 3951, 2752, 670, 114, 12, 0),
 (81, 952, 3268, 3776, 1607, 270, 41, 4, 0), (286, 1910, 3961, 2948, 781, 87, 23, 3, 0),
 (680, 2865, 4109, 1951, 345, 33, 12, 4, 0), (1294, 3763, 3601, 1145, 165, 13, 13, 5, 0),
 (1999, 4421, 2835, 657, 61, 10, 11, 5, 0), (2906, 4625, 2100, 319, 27, 7, 12, 2, 1),
 (3709, 4545, 1536, 178, 13, 3, 10, 4, 1), (4525, 4269, 1094, 93, 3, 4, 8, 2, 1),
 (5314, 3893, 729, 50, 3, 3, 5, 1, 1), (5964, 3518, 480, 27, 2, 2, 2, 3, 1),
 (6518, 3121, 342, 12, 0, 1, 3, 3, 1), (6990, 2787, 209, 9, 0, 2, 0, 2, 0)]




for i in range(len(Txts_transpose)):
    Txts_transpose[i] = list(Txts_transpose[i])

    for j in range(len(Txts_transpose[i])):
        Txts_transpose[i][j] = float(Txts_transpose[i][j]) / 10000
print Txts_transpose
# 求每轮的平均向听数
avg_xts = [0] * 14
for round in range(len(Txts_transpose)):
    round_xts = Txts_transpose[round]
    for x in range(len(round_xts)):
        xts = round_xts[x]
        avg_xts[round] += (xts * (x + 1))
print avg_xts


def get_t2info():
    dzSet = [0] * (34 + 15 * 3)  # 34+15*3
    # 生成搭子有效牌表
    dzEfc = [0] * (34 + 15 * 3)
    for i in range(len(dzSet)):
        if i <= 33:  # aa
            card = int(i / 9) * 16 + i % 9 + 1
            dzSet[i] = [card, card]
            dzEfc[i] = [card]
        elif i <= 33 + 8 * 3:  # ab
            card = int((i - 34) / 8) * 16 + (i - 34) % 8 + 1
            dzSet[i] = [card, card + 1]
            if card & 0x0f == 1:
                dzEfc[i] = [card + 2]
            elif card & 0x0f == 8:
                dzEfc[i] = [card - 1]
            else:
                dzEfc[i] = [card - 1, card + 2]
        else:
            card = int((i - 34 - 8 * 3) / 7) * 16 + (i - 34 - 8 * 3) % 7 + 1

            dzSet[i] = [card, card + 2]
            dzEfc[i] = [card + 1]

    efc_dzindex = {}  # card->34+8+8+8+7+7+7

    cardSet = []
    for i in range(34):
        cardSet.append(i / 9 * 16 + i % 9 + 1)
    for card in cardSet:
        efc_dzindex[card] = []
        efc_dzindex[card].append(MJ.translate16_33(card))  # 加aa
        color = int(card / 16)
        if color != 3:
            if card & 0x0f == 1:
                efc_dzindex[card].append(33 + color * 8 + (card & 0x0f) + 1)

            elif card & 0x0f == 2:  # 13 34
                efc_dzindex[card].append(33 + 24 + color * 7 + (card & 0x0f) - 1)
                efc_dzindex[card].append(33 + color * 8 + (card & 0x0f) + 1)
            elif card & 0x0f == 8:
                efc_dzindex[card].append(33 + color * 8 + (card & 0x0f) - 2)
                efc_dzindex[card].append(33 + 24 + color * 7 + (card & 0x0f) - 1)
            elif card & 0x0f == 9:
                efc_dzindex[card].append(33 + color * 8 + (card & 0x0f) - 2)
            else:
                efc_dzindex[card].append(33 + color * 8 + (card & 0x0f) - 2)
                efc_dzindex[card].append(33 + 24 + color * 7 + (card & 0x0f) - 1)
                efc_dzindex[card].append(33 + color * 8 + (card & 0x0f) + 1)

    return dzSet, dzEfc, efc_dzindex


def get_t3info():
    t3Set = []
    for i in range(34):
        card = int(i / 9) * 16 + i % 9 + 1
        t3Set.append([card, card, card])
    for i in range(34, 34 + 7 * 3):
        card = int((i - 34) / 7) * 16 + (i - 34) % 7 + 1
        t3Set.append([card, card + 1, card + 2])
    return t3Set


dzSet, dzEfc, efc_dzindex = get_t2info()
t3Set = get_t3info()
[[1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7, 7], [8, 8], [9, 9], [17, 17], [18, 18], [19, 19], [20, 20],
 [21, 21], [22, 22], [23, 23], [24, 24], [25, 25], [33, 33], [34, 34], [35, 35], [36, 36], [37, 37], [38, 38], [39, 39],
 [40, 40], [41, 41], [49, 49], [50, 50], [51, 51], [52, 52], [53, 53], [54, 54], [55, 55], [1, 2], [2, 3], [3, 4],
 [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23], [23, 24], [24, 25],
 [33, 34], [34, 35], [35, 36], [36, 37], [37, 38], [38, 39], [39, 40], [40, 41], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7],
 [6, 8], [7, 9], [17, 19], [18, 20], [19, 21], [20, 22], [21, 23], [22, 24], [23, 25], [33, 35], [34, 36], [35, 37],
 [36, 38], [37, 39], [38, 40], [39, 41]]
[[1], [2], [3], [4], [5], [6], [7], [8], [9], [17], [18], [19], [20], [21], [22], [23], [24], [25], [33], [34], [35],
 [36], [37], [38], [39], [40], [41], [49], [50], [51], [52], [53], [54], [55], [3], [1, 4], [2, 5], [3, 6], [4, 7],
 [5, 8], [6, 9], [7], [19], [17, 20], [18, 21], [19, 22], [20, 23], [21, 24], [22, 25], [23], [35], [33, 36], [34, 37],
 [35, 38], [36, 39], [37, 40], [38, 41], [39], [2], [3], [4], [5], [6], [7], [8], [18], [19], [20], [21], [22], [23],
 [24], [34], [35], [36], [37], [38], [39], [40]]

{1: [0, 35], 2: [1, 58, 36], 3: [2, 34, 59, 37], 4: [3, 35, 60, 38], 5: [4, 36, 61, 39], 6: [5, 37, 62, 40],
 7: [6, 38, 63, 41], 8: [7, 39, 64], 9: [8, 40], 17: [9, 43], 18: [10, 65, 44], 19: [11, 42, 66, 45],
 20: [12, 43, 67, 46], 21: [13, 44, 68, 47], 22: [14, 45, 69, 48], 23: [15, 46, 70, 49], 24: [16, 47, 71], 25: [17, 48],
 33: [18, 51], 34: [19, 72, 52], 35: [20, 50, 73, 53], 36: [21, 51, 74, 54], 37: [22, 52, 75, 55], 38: [23, 53, 76, 56],
 39: [24, 54, 77, 57], 40: [25, 55, 78], 41: [26, 56], 49: [27], 50: [28], 51: [29], 52: [30], 53: [31], 54: [32],
 55: [33]}


# print (dzSet[65])
#
# print 'dzSet',dzSet
# print dzEfc

# print 'efc_dzindex',efc_dzindex


# import random
# 摸牌结点类
class CatchNode:
    def __init__(self, cards, catchCard, leftNum, remainNum, t2, level, ocards, t2N, p):
        self.type = 2
        self.cards = cards
        self.leftNum = leftNum
        self.catchCard = catchCard
        self.rate = 1
        # if catchCard != None:  # 获取概率
        #     if len(t2) == 1:  # 单张牌，凑将
        #         self.rate = float(leftNum[convert_hex2index(catchCard)]) / remainNum * 4
        #     elif len(t2) == 2:
        #         if t2[0] == t2[1]:
        #             self.rate = float(leftNum[convert_hex2index(catchCard)]) / remainNum * 6
        #         else:
        #             self.rate = float(leftNum[convert_hex2index(catchCard)]) / remainNum * 2
        #     else:
        #         print('CatchNode Error 2!', catchCard, t2)

        self.rate = p
        # if self.rate == 0:
        #     print ('rate=0,catchCard', catchCard)
        self.t2 = t2
        self.level = level  # 在树中的层数
        self.ocards = ocards
        self.t2N = t2N
        self.kz = []
        self.sz = []
        self.jiang = 0x00
        self.parent = None  # todo 可以使用ｈａｓｈ表来存，可能会快一点
        self.children = []
        self.formerCatchCards = []
        self.formerOutCards = []
        pass

    def setParent(self, parent):
        self.parent = parent

    def addChild(self, child):
        self.children.append(child)

    def addFormerState(self, formerState):
        if formerState not in self.formerState:
            self.formerState.append(formerState)
            return True
        else:
            return False

    def equal(self, newNode):
        if newNode.catchCard == self.catchCard and newNode.kz == self.kz and newNode.sz == self.sz and newNode.jiang == self.jiang:
            return True

        return False

    def __repr__(self):
        # return "{%d,%s,%s}".format(self.type,self.cards,self.catchCard)
        return self.type, self.cards, self.catchCard, self.level


# 出牌结点类
class OutNode:
    def __init__(self, cards, outCard, level, ocards, t2N, rt):
        self.type = 1
        self.cards = cards
        self.outCard = outCard
        self.level = level  # 在树中的层数
        self.parent = None
        self.children = []
        self.ocards = ocards
        self.t2N = t2N
        self.kz = []
        self.sz = []
        self.jiang = 0x00

        self.formerCatchCards = []
        self.formerOutCards = []
        self.rate = rt  # 危险概率
        pass

    def setParent(self, parent):
        self.parent = parent

    def addChild(self, child):
        self.children.append(child)

    # def addFormerState(self, formerState):
    #     if formerState not in self.formerState:
    #         # print(self.formerState)
    #         self.formerState.append(formerState)
    #         # print(self.formerState)
    #         return True
    #
    #     else:
    #         return False
    def equal(self, newNode):
        if newNode.outCard == self.outCard and newNode.kz == self.kz and newNode.sz == self.sz and newNode.jiang == self.jiang:
            return True

        return False

    def __repr__(self):
        # return "{%d,%s,%s}".format(self.type,self.cards,self.outCard)
        return self.type, self.cards, self.outCard, self.level


# 搜索树类
class SearchTree:
    def __init__(self, cards, suits, leftNum, all, all_fastWinning, all_highFan, remainNum, T_selfmo, RT1, RT2, RT3,
                 Txts_transpose, round):
        print('leftNum', leftNum)
        print('xts', all[0][4])
        self.root = CatchNode(cards, None, leftNum, remainNum, [], 0, [], [], 0)
        # if type==1:
        #     out = OutNode(cards=cards, outCard=cards[0], level=1, ocards=[], t2N=[],
        #                         dgRate=dgtable)
        #
        #     out.setParent(self.root)
        #     self.root.addChild(out)

        self.cards = cards
        self.suits = suits
        self.leftNum = leftNum
        self.all = all
        self.all_fastWinning = all_fastWinning
        self.all_highFan = all_highFan
        self.xts = all[0][4]
        self.remainNum = remainNum
        self.T_selfmo = T_selfmo
        self.RT1 = RT1
        self.RT2 = RT2
        self.RT3 = RT3
        self.Txts_transpose = Txts_transpose
        self.round = round
        self.stateSet = {}
        # self.op_card=op_card
        # self.type=type
        self.scoreDict = {}
        for suit in suits:
            if suit[0] != suit[1]:
                self.root.sz.append(suit[0])
            else:
                self.root.kz.append(suit[0])
                # if type == 1:
                #     out = OutNode(cards=cards, outCard=cards[0], level=1, ocards=[], t2N=[],
                #                   dgRate=dgtable)
                #     out.sz=copy.copy(self.root.sz)
                #     out.kz=copy.copy(self.root.kz)
                #     out.setParent(self.root)
                #     out.formerOutCards.append(0)
                #     self.root.addChild(out)

    # 子结点判断，用于避免重复结点的构建
    def inChild(self, node, newNode):
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

    # 有效牌获取
    def getEffectiveCards(self, dz):

        # 获取有效牌,输入为搭子集合,
        t2efc = []
        # leftCard=-1

        if dz == []:
            return []
        elif len(dz) == 1:
            t2efc.append([[dz[0]], dz[0], -1])
            # if dz[0] & 0xf0 == 0x30:
            #     t2efc.append([[dz[0]], dz[0], -1])
            # elif dz[0] & 0x0f == 0x01:
            #     t2efc.append([[dz[0]], dz[0], -1])
            #     t2efc.append([[dz[0]], dz[0] + 1, -1])
            #     t2efc.append([[dz[0]], dz[0] + 2, -1])
            # elif dz[0] & 0x0f == 0x02:
            #     t2efc.append([[dz[0]], dz[0] - 1, -1])
            #     t2efc.append([[dz[0]], dz[0], -1])
            #     t2efc.append([[dz[0]], dz[0] + 1, -1])
            #     t2efc.append([[dz[0]], dz[0] + 2, -1])
            # elif dz[0] & 0x0f == 0x08:
            #     t2efc.append([[dz[0]], dz[0] - 2, -1])
            #     t2efc.append([[dz[0]], dz[0] - 1, -1])
            #     t2efc.append([[dz[0]], dz[0], -1])
            #     t2efc.append([[dz[0]], dz[0] + 1, -1])
            # elif dz[0] & 0x0f == 0x09:
            #     t2efc.append([[dz[0]], dz[0] - 2, -1])
            #     t2efc.append([[dz[0]], dz[0] - 1, -1])
            #     t2efc.append([[dz[0]], dz[0], -1])
            # else:
            #     t2efc.append([[dz[0]], dz[0] - 2, -1])
            #     t2efc.append([[dz[0]], dz[0] - 1, -1])
            #     t2efc.append([[dz[0]], dz[0], -1])
            #     t2efc.append([[dz[0]], dz[0] + 1, -1])
            #     t2efc.append([[dz[0]], dz[0] + 2, -1])


        elif len(dz) == 3:
            if dz[0] + 4 == dz[2]:
                t2efc.append([[dz[0], dz[1]], dz[0] + 1, dz[2]])
                t2efc.append([[dz[1], dz[2]], dz[1] + 1, dz[0]])
            else:
                for t in dz:  # todo 去重
                    dzCP = copy.copy(dz)
                    dzCP.remove(t)
                    if abs(dzCP[0] - dzCP[1]) <= 2:
                        for a in self.getEffectiveCards(dzCP):
                            a[2] = t
                            t2efc.append(a)

        elif dz[1] == dz[0]:
            t2efc.append([dz, dz[0], -1])
        elif dz[1] == dz[0] + 1:
            if int(dz[0]) & 0x0F == 1:
                t2efc.append([dz, dz[0] + 2, -1])
            elif int(dz[0]) & 0x0F == 8:
                t2efc.append([dz, dz[0] - 1, -1])
            else:
                t2efc.append([dz, dz[0] - 1, -1])
                t2efc.append([dz, dz[0] + 2, -1])
        elif dz[1] == dz[0] + 2:
            t2efc.append([dz, dz[0] + 1, -1])

        t2efcRe = []
        for t in t2efc:
            if t not in t2efcRe:
                t2efcRe.append(t)
        return t2efcRe

    # node结点扩展
    def expandNode(self, node, ocards, t2N, kz=[], sz=[]):
        # print "t2N",t2N
        # print('expandNode','node.kz,sz,jiang',node.kz,node.sz,node.jiang,'kz,sz',kz,sz,'ocards,t2N',ocards,t2N,'node.cards',node.cards,'node.type,level,rate',node.level,node.type,node.rate)
        #
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
            # 只搜索具有最小权重的t2
            if ocards == [] and t2N != []:
                # 所有t2N都出一遍 对t2.5出其中一个，保留一个t2
                # t2_w=[]
                # for t2 in t2N:
                #     if t2[0]==t2[1]:
                #         w=sum([(self.T_selfmo[convert_hex2index(e)]*(4-self.RT1[2][convert_hex2index(e)]-self.RT2[2][convert_hex2index(e)]-self.RT3[2][convert_hex2index(e)])) for e in dzEfc[dzSet.index(t2)]])
                #     else:
                #
                #         w = sum([(self.T_selfmo[convert_hex2index(e)] * (
                #         2 - self.RT3[2][convert_hex2index(e)])) for e in dzEfc[dzSet.index(t2)]])
                #     t2_w.append(w)
                # min_w=min(t2_w)
                # min_index=[]
                # for i in range(len(t2_w)):
                #     if t2_w[i]==min_w:
                #         min_index.append(i)
                # for i in min_index:
                #     ocardsCP = t2N[i]
                #     t2NCP = copy.deepcopy(t2N)
                #     t2NCP.remove(t2N[i])
                #     self.expandNode(node, ocardsCP, t2NCP, kz, sz)
                for t2 in t2N:
                    if len(t2) == 2:
                        ocardsCP = t2
                        t2NCP = copy.deepcopy(t2N)
                        t2NCP.remove(t2)
                        self.expandNode(node, ocardsCP, t2NCP, kz, sz)
                # elif len(t2) == 3:  # aab ace abb aac acc
                #         if t2[2] - t2[0] == 4:
                #             ocardsCP = [t2[0]]
                #             t2NCP = copy.deepcopy(t2N)
                #             t2NCP.remove(t2)
                #             t2NCP.append([t2[1], t2[2]])
                #             self.expandNode(node, ocardsCP, t2NCP, kz, sz)
                #
                #             ocardsCP = [t2[2]]
                #             t2NCP = copy.deepcopy(t2N)
                #             t2NCP.remove(t2)
                #             t2NCP.append([t2[0], t2[1]])
                #             self.expandNode(node, ocardsCP, t2NCP, kz, sz)
                #
                #         else:
                #             for t in set(t2):
                #                 t2CP = copy.copy(t2)
                #                 t2CP.remove(t)
                #                 if abs(t2CP[0] - t2CP[1]) <= 2:
                #                     ocardsCP = [t]
                #                     t2NCP = copy.deepcopy(t2N)
                #                     t2NCP.remove(t2)
                #
                #                     t2NCP.append(t2CP)
                #                     self.expandNode(node, ocardsCP, t2NCP, kz, sz)
                return

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
                Pwn1 = self.Txts_transpose[self.round][0]
                index = MJ.translate16_33(out)
                rt1 = self.RT1[2][index]
                rt2 = self.RT2[2][index]
                rt3 = self.RT3[2][index]
                rt = 1 - (
                (1 - Pwn1) * (self.RT1[0][index] + self.RT2[0][index] + self.RT3[0][index]) + Pwn1 * (rt1 + rt2 + rt3))
                oNode = OutNode(cards=cardsCP, outCard=out, level=node.level + 1, ocards=ocardsCP, t2N=t2NCP,
                                rt=rt)
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
            # print t2N,ocards
            if t2N == [] and ocards != []:
                for t in ocards:
                    t2NCP = [[t]]
                    ocardsCP = copy.copy(ocards)
                    ocardsCP.remove(t)
                    self.expandNode(node, ocardsCP, t2NCP, kz, sz)
                return

            elif t2N != []:
                # ocardsCP = ocards
                t2NCPTMP = t2N
            # else:  # todo 无成型的２N抓，现在省略掉了
            #     # ocardsCP = ocards
            #     # t2NCPTMP = t2N
            #     print('Error expandNode', self.cards, node.cards, ocards, t2N, node.level)
            #     node.rate = 0
            #     return
            for t2 in t2NCPTMP:
                t2NCP = copy.deepcopy(t2NCPTMP)
                t2NCP.remove(t2)
                t2efc = self.getEffectiveCards(t2)  # [t3,efc,leftcard]
                # if len(t2)==3:
                #     print t2efc
                if t2efc == []:
                    print('Error effectiveCards is []', t2)
                else:
                    for e in t2efc:
                        # 更新t2.5的废牌
                        ocardsCP = copy.copy(ocards)
                        if e[2] != -1:
                            # print ocardsCP
                            ocardsCP.append(e[2])
                        cardsCP = copy.copy(node.cards)
                        cardsCP.append(e[1])
                        cardsCP.sort()
                        index = MJ.translate16_33(e[1])
                        rt1 = self.RT1[2][index]
                        rt2 = self.RT2[2][index]
                        rt3 = self.RT3[2][index]

                        if len(e[0]) == 2:
                            if e[0][0] == e[0][1]:  # aa*
                                p = self.T_selfmo[index] * (4 - rt1 - rt2 - rt3) * 1.2

                            else:
                                p = self.T_selfmo[index]
                        else:
                            p = self.T_selfmo[index]
                        # if e[0]==[0x18,0x19]:
                        #     print '17',p,self.T_selfmo[index],float(self.leftNum[index])/self.remainNum*2
                        # elif e[0]==[0x19,0x19]:
                        #     print '19',p,self.T_selfmo[index],float(self.leftNum[index])/self.remainNum*6
                        cNode = CatchNode(cards=cardsCP, catchCard=e[1], leftNum=self.leftNum,
                                          remainNum=self.remainNum,
                                          t2=e[0], level=node.level + 1, ocards=ocardsCP, t2N=t2NCP, p=p)
                        cNode.kz = copy.copy(node.kz)
                        cNode.kz.extend(kz)
                        cNode.sz = copy.copy(node.sz)
                        cNode.sz.extend(sz)

                        t2tmp = copy.copy(e[0])

                        t2tmp.append(e[1])
                        t2tmp.sort()
                        # 已胡牌,这里是补将牌
                        if len(t2tmp) == 2:
                            t2NCP.append(t2tmp)
                        elif t2tmp[0] == t2tmp[1]:
                            cNode.kz.append(t2tmp[0])
                        else:
                            cNode.sz.append(t2tmp[0])
                        # 已胡牌，这里不是补将牌，补的其他２Ｎ
                        # print len(cNode.kz) + len(cNode.sz) == 5
                        # print ocardsCP
                        # print len(t2NCP)
                        # print t2NCP

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
                        cNode.formerCatchCards.append(e[1])
                        cNode.formerCatchCards.sort()
                        cNode.formerOutCards = copy.copy(node.formerOutCards)
                        cNode.setParent(node)
                        node.addChild(cNode)
                        # 排序
                        cNode.kz.sort()
                        cNode.sz.sort()
                        self.expandNode(cNode, ocardsCP, t2NCP)

    # 构建树，使用了拆分信息，构建了扩展策略集合,快胡树
    def generateTree_fastWinning(self):
        # if self.type==2:
        #     node = self.root
        # else:#扩展了ｏｐ中的结点
        #     node=self.root.children[0]

        print 'self.all_fastWinning', self.all_fastWinning

        node = self.root
        for a in self.all_fastWinning:
            # t2N = copy.deepcopy(a[2] + a[3])
            # print 't2n',t2N
            # efc_cards, t2_w = pinghu(cards=self.cards, suits=self.suits, leftNum=self.leftNum).get_effective_cards_w(
            #     dz_set=t2N, left_num=self.leftNum)
            # for i in range(len(t2N)):
            #     t2N[i].append(t2_w[i])
            # t2N[:len(a[2])] = sorted(t2N[:len(a[2])], key=lambda k: k[2], reverse=True)
            # t2N[len(a[2]):] = sorted(t2N[len(a[2]):], key=lambda k: k[2], reverse=True)
            # # t2N[len():] = sorted(t2N[len(a[2]):], key=lambda k: k[2], reverse=True) #修改为１＋

            # 扩展出牌结点
            ocards = a[-1]
            t2NCP = []
            # for t2 in t2N:
            #     t2NCP.append([t2[0], t2[1]])
            t2NCP = a[2] + a[3] + a[4]
            kz = []
            sz = []
            for k in a[0]:
                kz.append(k[0])
            for s in a[1]:
                sz.append(s[0])
            self.expandNode(node=node, ocards=ocards, t2N=t2NCP, kz=kz, sz=sz)

    # 期望计算，评价函数
    def getRate(self, node, rate):
        # print ('nodeInfo',node.kz,node.sz,node.jiang,node.rate,node.children == [] )
        children = node.children
        if children == []:
            # 胡牌结点
            if node.level >= self.xts * 2 and len(node.sz) + len(node.kz) == 4 and node.jiang != 0:
                if node.rate != 0 and node.type == 2:
                    if (len(node.t2) == 2 and node.t2[0] != node.t2[1]) or len(node.t2) == 1:
                        index = MJ.translate16_33(node.catchCard)
                        rt1 = self.RT1[0][index] + self.RT1[1][index]
                        rt2 = self.RT2[0][index] + self.RT2[1][index]
                        rt3 = self.RT3[0][index] + self.RT3[1][index]

                        node.rate = self.T_selfmo[index] * (4 - rt1 - rt2 - rt3)
                        # node.rate = float(self.leftNum[convert_hex2index(node.catchCard)]) / self.remainNum * 4
                        # else:
                        #     node.rate*2.0/3
                rate *= node.rate
                if rate != 0 and node.jiang == 0:
                    print ('getRate Error', node.cards, node.kz, node.sz, node.level)
                # todo 可以优化时间
                if rate != 0:
                    catchCards = node.formerCatchCards
                    outCards = node.formerOutCards
                    state = []
                    state.append(node.kz)
                    state.append(node.sz)
                    state.append(node.jiang)

                    fanList = Fan(node.kz, node.sz, node.jiang, suits=self.suits).fanDetect()
                    state.append(sum(fanList))
                    score = rate * (2 + sum(fanList))

                    for card in node.formerOutCards:
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
                                    self.stateSet[card][1][index] = outCards
                                    self.stateSet[card][2][index] = state

                                    # # print ('self.stateSet[card][1]',index,self.stateSet[card][0],self.stateSet[card][1])
                                    # #原来是只要抓牌一样，就只算最高
                                    # #现在改为，抓牌与出牌不同才更新
                                    # if outCards == self.stateSet[card][1][index]:
                                    #     # print ('differ')
                                    #     # print (catchCards,self.stateSet[card][0][index])
                                    #     # print (outCards,self.stateSet[card][1][index])
                                    #     # print (state==self.stateSet[card][2][index])
                                    #     # if score != self.scoreDict[card][index]:
                                    #     #     #todo 抓牌与出牌都一样，但是胡牌概率不一样,这是因为吃在前面和吃在最后的区别
                                    #     #     print('Error',score,self.scoreDict[card][index])
                                    #     # # print ('here equal')
                                    #     # continue
                                    #     if score>self.scoreDict[card][index]:
                                    #         print('here')
                                    #         self.scoreDict[card][index]=score
                                    #         self.stateSet[card][1][index]=outCards
                                    # else:
                                    #     # print ('here')
                                    #     print('here', score, self.scoreDict[card][index])
                                    #     if score>self.scoreDict[card][index]:
                                    #         print('here')
                                    #         self.scoreDict[card][index]=score
                                    #         self.stateSet[card][1][index]=outCards
                return
        else:
            rate *= node.rate
            for child in children:
                self.getRate(child, rate)

    # 获取每张出牌的分数，整合信息并输出
    def getCardScore(self):
        # 建树
        self.generateTree_fastWinning()

        outCardsNodes = self.root.children
        for i in range(len(outCardsNodes)):
            rate = 1
            node = outCardsNodes[i]
            self.getRate(node=node, rate=rate)
        nodeNum = 0
        print ('scoreDict', self.scoreDict)
        for k in self.scoreDict.keys():
            nodeNum += len(self.scoreDict[k])
            self.scoreDict[k] = sum(self.scoreDict[k])

        print ('score', self.scoreDict)
        print ('stateSet', self.stateSet)
        print('nodeNum', nodeNum)
        return self.scoreDict


# 防守模型
class DefendModel:
    def __init__(self, cards, suits, discards, discardsOp, discardsReal, round, seat_id, Txts_transpose, M,
                 choose_color=[], hu_cards=[[], [], [], []], hu_fan=[[], [], [], []]):
        self.cards = cards
        self.suits = suits
        self.round = round
        self.seat_id = seat_id
        self.discards0 = discardsReal[seat_id]
        otherID = self.getOtherID()
        self.discards1 = discardsReal[otherID[0]]
        self.discards2 = discardsReal[otherID[1]]
        self.discards3 = discardsReal[otherID[2]]
        self.discardsOp0 = discardsOp[seat_id]
        self.discardsOp1 = discardsOp[otherID[0]]
        self.discardsOp2 = discardsOp[otherID[1]]
        self.discardsOp3 = discardsOp[otherID[2]]
        self.leftNum, _ = MJ.trandfer_discards(discards, discardsOp, cards, type=27)

        c3_d = {}
        for i in range(5):
            for j in range(5):
                for k in range(5):
                    key = i * 100 + j * 10 + k
                    # c3_d[key] = float(i * j * k) / 64
                    c3_d[key] = min([i, j, k]) * 0.5
                    # if c3_d[key]>1:
                    #     c3_d[key]=1
        self.c3_d = c3_d
        c5_d = {}
        for i in range(5):
            for j in range(5):
                for k in range(3, 5):
                    for x in range(5):
                        for y in range(5):
                            key = i * 10000 + j * 1000 + k * 100 + x * 10 + y
                            l = c3_d[i * 100 + j * 10 + k]
                            m = c3_d[j * 100 + k * 10 + x]
                            r = c3_d[k * 100 + x * 10 + y]
                            # c5_d[key]=1+0.25*(k-3)-max([l,m,r])

                            # c5_d[key] = k - max([l,m,r,0])
                            # c5_d[key] = max(0, k - 2 - (
                            # c3_d[i * 100 + j * 10 + k] + c3_d[j * 100 + k * 10 + x] + c3_d[k * 100 + x * 10 + y]))
        self.c5_d = c5_d
        self.Txts_transpose = Txts_transpose
        self.dzSet, self.dzEfc, self.efc_dzindex = self.get_t2info()
        self.M = M
        self.xts_zd = [[[5, 1]],
                       [[4, 2], [5, 0], [5, 1]],
                       [[4, 1], [3, 3], [3, 4], [4, 2]],
                       [[2, 4], [2, 5], [3, 2], [3, 3], [3, 4]],
                       [[3, 1], [2, 3], [2, 4], [2, 5]],
                       [[2, 5], [3, 0], [2, 2], [1, 4]],
                       [[2, 1], [1, 3]],
                       [[1, 2], [2, 0]],
                       [[0, 3]],
                       [[0, 2], [1, 0]],
                       [[0, 1]],
                       [0, 0]]
        # self.dgtable = [0.95] * 34
        self.choose_color = choose_color
        self.hu_cards = hu_cards
        self.hu_fan = hu_fan

    '''
    '''

    def getOtherID(self):
        if self.seat_id == 0:
            return [1, 2, 3]
        elif self.seat_id == 1:
            return [2, 3, 0]
        elif self.seat_id == 2:
            return [3, 0, 1]
        elif self.seat_id == 3:
            return [0, 1, 2]
        else:
            print('seat_id Error!', self.seat_id)
            return []

    # def get_distributionTable(self):
    #     t = [0] * (34 + 7 * 3)  # 34种刻子，21种顺子
    #     for i in range(27):
    #         if self.leftNum[i] <= 2:
    #             t[i] = 1
    #         else:
    #             c5 = [0, 0, 0, 0, 0]
    #             if i % 9 + 1 == 1:
    #                 c5 = [0, 0, self.leftNum[i], self.leftNum[i + 1], self.leftNum[i + 2]]
    #             elif i % 9 + 1 == 2:
    #                 c5 = [0, self.leftNum[i - 1], self.leftNum[i], self.leftNum[i + 1], self.leftNum[i + 2]]
    #             elif i % 9 + 1 == 8:
    #                 c5 = [self.leftNum[i - 2], self.leftNum[i - 1], self.leftNum[i], self.leftNum[i + 1], 0]
    #             elif i % 9 + 1 == 9:
    #                 c5 = [self.leftNum[i - 2], self.leftNum[i - 1], self.leftNum[i], 0, 0]
    #             else:
    #                 c5 = [self.leftNum[i - 2], self.leftNum[i - 1], self.leftNum[i], self.leftNum[i + 1],
    #                       self.leftNum[i + 2]]
    #             key = c5[0] * 10000 + c5[1] * 1000 + c5[2] * 100 + c5[3] * 10 + c5[4]
    #             t[i] = self.c5_d[key]
    #
    #     for i in range(28, 34):
    #         if self.leftNum[i] <= 2:
    #             t[i] == 0
    #         else:
    #             c5 = [0, 0, self.leftNum[i], 0, 0]
    #             key = c5[0] * 10000 + c5[1] * 1000 + c5[2] * 100 + c5[3] * 10 + c5[4]
    #             t[i] = self.c5_d[key]
    #     for i in range(34, 34 + 21):
    #         # if (i-34) % 9 + 1 >= 8:
    #         #     t[i]=0
    #         # else:
    #         index = int((i - 34) / 7) * 9 + (i - 34) % 7 + 1
    #         c3 = [self.leftNum[index], self.leftNum[index + 1], self.leftNum[index + 2]]
    #         key = c3[0] * 100 + c3[1] * 10 + c3[2]
    #         t[i] = self.c3_d[key]
    #
    #     return t

    def get_distributionTable(self, wall):
        t = [0] * (34 + 7 * 3)  # 34种刻子，21种顺子
        for i in range(27):
            if wall[i] > 3:
                t[i] = 2
                # t[i]=0
                # t[i]=float(wall[i])/2 #todo 有待商榷
                if i % 9 + 1 == 1:
                    t[i] += 2
                    if min(wall[i], wall[i + 1], wall[i + 2]) == 0:
                        t[i] += 4
                        # else:
                        # t[i]+=3.0/(1+min(wall[i],wall[i+1],wall[i+2]))
                elif i % 9 + 1 == 2:
                    t[i] += 1
                    if min([wall[i - 1], wall[i], wall[i + 1]]) + min([wall[i], wall[i + 1], wall[i + 2]]) == 0:
                        t[i] += 5
                        # t[i]+=3.0/(1+min([wall[i-1],wall[i],wall[i+1]])+min([wall[i],wall[i+1],wall[i+2]]))
                elif i % 9 + 1 == 8:
                    t[i] += 1
                    if min([wall[i - 2], wall[i - 1], wall[i]]) + min([wall[i - 1], wall[i], wall[i + 1]]) == 0:
                        # t[i]+=3.0/(1+min([wall[i-2],wall[i-1],wall[i]])+min([wall[i-1],wall[i],wall[i+1]]))
                        t[i] += 5
                elif i % 9 + 1 == 9:
                    t[i] += 2
                    if min([wall[i - 2], wall[i - 1], wall[i]]) == 0:
                        # t[i]+=3.0/(1+min([wall[i-2],wall[i-1],wall[i]]))
                        t[i] += 4
                else:
                    if min([wall[i - 2], wall[i - 1], wall[i]]) + min([wall[i - 1], wall[i], wall[i + 1]]) + min(
                            [wall[i], wall[i + 1], wall[i + 2]]) == 0:
                        # t[i]+=3.0/(1+(min([wall[i-2],wall[i-1],wall[i]])+min([wall[i-1],wall[i],wall[i+1]])+min([wall[i],wall[i+1],wall[i+2]])))
                        t[i] += 6
        for i in range(27, 34):
            if wall[i] > 3:
                t[i] = 8

        for i in range(34, 34 + 21):
            index = MJ.translate16_33(int((i - 34) / 7) * 16 + (i - 34) % 7 + 1)
            # if wall[index]==0 or wall[index+1]==0 or wall[index+2]==0:
            #     t[i]=0
            # c3 = [wall[index], wall[index + 1], wall[index + 2]]
            # key = c3[0] * 100 + c3[1] * 10 + c3[2]
            # t[i] = self.c3_d[key]
            t[i] = min([wall[index], wall[index + 1], wall[index + 2]])

        return t

    def erfen(self, list, key, left, right):
        if right > left:
            if key > list[(left + right) / 2]:
                left = (left + right) / 2
            else:
                right = (left + right) / 2
            self.erfen(list, key, left, right)
        else:
            return left

    def get_t2info(self):

        dzSet = [0] * (34 + 15 * 3)  # 34+15*3
        # 生成搭子有效牌表
        dzEfc = [0] * (34 + 15 * 3)
        for i in range(len(dzSet)):
            if i <= 33:  # aa
                card = int(i / 9) * 16 + i % 9 + 1
                dzSet[i] = [card, card]
                dzEfc[i] = [card]
            elif i <= 33 + 8 * 3:  # ab
                card = int((i - 34) / 8) * 16 + (i - 34) % 8 + 1
                dzSet[i] = [card, card + 1]
                if card & 0x0f == 1:
                    dzEfc[i] = [card + 2]
                elif card & 0x0f == 8:
                    dzEfc[i] = [card - 1]
                else:
                    dzEfc[i] = [card - 1, card + 2]
            else:
                card = int((i - 34 - 8 * 3) / 7) * 16 + (i - 34 - 8 * 3) % 7 + 1

                dzSet[i] = [card, card + 2]
                dzEfc[i] = [card + 1]

        efc_dzindex = {}  # card->34+8+8+8+7+7+7

        cardSet = []
        for i in range(34):
            cardSet.append(i / 9 * 16 + i % 9 + 1)
        for card in cardSet:
            efc_dzindex[card] = []
            efc_dzindex[card].append(MJ.translate16_33(card))
            color = int(card / 16)
            if color != 3:
                if card & 0x0f == 1:
                    efc_dzindex[card].append(33 + color * 15 + (card & 0x0f) + 1)

                elif card & 0x0f == 2:  # 13 34
                    efc_dzindex[card].append(33 + color * 15 + 8 + (card & 0x0f) - 1)
                    efc_dzindex[card].append(33 + color * 15 + (card & 0x0f) + 1)
                elif card & 0x0f == 8:
                    efc_dzindex[card].append(33 + color * 15 + (card & 0x0f) - 2)
                    efc_dzindex[card].append(33 + color * 15 + 8 + (card & 0x0f) - 1)
                elif card & 0x0f == 9:
                    efc_dzindex[card].append(33 + color * 15 + (card & 0x0f) - 2)
                else:
                    efc_dzindex[card].append(33 + color * 15 + (card & 0x0f) - 2)
                    efc_dzindex[card].append(33 + color * 15 + 8 + (card & 0x0f) - 1)
                    efc_dzindex[card].append(33 + color * 15 + (card & 0x0f) + 1)

        return dzSet, dzEfc, efc_dzindex

    def rand_N32(self, sum_xts, l_op):

        r = random.uniform(0, sum_xts[-1])
        xts = 0
        for i in range(0, len(sum_xts)):
            if r <= sum_xts[i]:
                xts = i
                break
        # 获得了xts
        # 再random一组t23
        xtSet = copy.deepcopy(self.xts_zd[xts])
        for x in self.xts_zd[xts]:
            if x[0] < l_op:
                xtSet.remove(x)
        if xtSet == []:
            return [l_op, 2]
        return random.choice(xtSet)

    def get_wall(self, P1_N32, P2_N32, P3_N32, wall_):
        wall = copy.copy(wall_)
        for i in range(P1_N32[0] + P2_N32[0] + P3_N32[0] - len(self.discardsOp1) - len(self.discardsOp2) - len(
                self.discardsOp3)):

            # 生成T3概率表
            t3 = self.get_distributionTable(wall)
            t3_sum = copy.copy(t3)

            for i in range(1, len(t3_sum)):
                t3_sum[i] = t3_sum[i - 1] + t3_sum[i]
            if t3_sum[-1] == 0:
                print ("failed to simulate T3")
                return wall_
            r = random.uniform(0, t3_sum[-1])
            j = 0
            flag = False
            while j < len(t3_sum) and not flag:
                if r <= t3_sum[j]:  # 下标
                    # index_set.append(i)
                    for card in t3Set[j]:
                        wall[MJ.convert_hex2index(card)] -= 1
                        flag = True
                j += 1
        return wall

    def simulate_t2(self, N2, discards, wall):
        P = [0] * (34 + 15 * 3)  # 不可分表
        for card in discards:
            for i in efc_dzindex[card]:
                P[i] += 1
        RT = [[0] * 34, [0] * 34]
        for y in range(N2):
            Pdz_index = []  # 可分配搭子
            t2 = [0] * (34 + (15 * 3))
            for i in range(len(wall)):
                if wall[i] >= 2:
                    t2[i] = 2
                # elif wall[i]==4:
                #     t2[i]= 4
                # t2[i]=int(wall[i]/2)
                # if wall[i] >= 2:
                #     t2[i] = float(wall[i])/2 #todo 有待商榷
                if i < 27:
                    color = int(i / 9)
                    if i % 9 + 1 == 8 or i % 9 + 1 == 1:  # 89
                        if sum([wall[MJ.convert_hex2index(e)] for e in
                                self.dzEfc[33 + color * 8 + (i % 9) + 1]]) != 0:
                            t2[33 + color * 8 + (i % 9) + 1] = min(wall[i], wall[i + 1]) * 0.7
                            # t2[33 + color * 15 + (i % 9) + 1] = 1
                    elif i % 9 + 1 == 9:
                        pass
                    else:
                        if sum([wall[MJ.convert_hex2index(e)] for e in
                                self.dzEfc[33 + color * 8 + (i % 9) + 1]]) != 0:
                            t2[33 + color * 8 + (i % 9) + 1] = min(wall[i], wall[i + 1]) * 0.8
                            # t2[33 + color * 15 + (i % 9) + 1] = 1

                        if sum([wall[MJ.convert_hex2index(e)] for e in
                                self.dzEfc[33 + 24 + color * 7 + (i % 9) + 1]]) != 0:
                            t2[33 + 24 + color * 7 + (i % 9) + 1] = min(wall[i], wall[i + 2]) * 0.7
                            # t2[33 + color * 15 + 8 + (i % 9) + 1] = 1
            # Pc=copy.copy(t2)
            Pc = [0] * (34 + (15 * 3))
            for x in range(len(t2)):
                if P[x] == 0:
                    Pc[x] = t2[x]
            for j in range(1, len(Pc)):
                Pc[j] = Pc[j] + Pc[j - 1]
                # if P[x] == 0 and t2[x] != 0:
                #     Pdz_index.append(x)
            # print 'dz_index',Pdz_index
            r = random.uniform(0, Pc[-1])
            for x in range(len(Pc)):
                if r <= Pc[x]:
                    # 更新wall
                    dz_index = x
                    for card in dzSet[dz_index]:
                        wall[MJ.convert_hex2index(card)] -= 1
                    if dz_index < 34:
                        for card in dzEfc[dz_index]:
                            RT[0][MJ.convert_hex2index(card)] += 1
                    else:
                        for card in dzEfc[dz_index]:
                            RT[1][MJ.convert_hex2index(card)] += 1
                    break
                    # dz_index = random.choice(Pdz_index)

        return RT, wall

    def getWTandRT(self):

        # # 生成T3概率表
        # t3 = self.get_distributionTable(wall)
        # print t3
        # t3_sum = copy.copy(t3)
        # for i in range(1, len(t3_sum)):
        #     t3_sum[i] = t3_sum[i - 1] + t3_sum[i]
        # xts轮数采样
        # xts = [4.6684, 4.1589, 3.6867, 3.2393, 2.858, 2.5237, 2.2456, 2.0018, 1.8313, 1.6837, 1.5572, 1.4617, 1.3894,
        #        1.3255]
        # zd = [[], [], [], [], [], [3, 1], [3, 2], [3, 2], [3, 3], [4, 2], [4, 2], [4, 2], [4, 2], [4, 2]]
        # for n in range(10):
        #     zd.append([4, 2])
        # 向听数与组和搭的关系表



        xts_round = self.Txts_transpose[self.round]
        sum_xts = copy.copy(xts_round)
        for i in range(1, len(xts_round)):
            sum_xts[i] = sum_xts[i] + sum_xts[i - 1]

        # 总表
        wall_sum = [0] * 34
        RT1_sum = [[0] * 34, [0] * 34]
        RT2_sum = [[0] * 34, [0] * 34]
        RT3_sum = [[0] * 34, [0] * 34]

        for dn in range(self.M):
            # rd = []  # 记录随机数的列表
            wall = copy.copy(self.leftNum)

            P1_N32 = self.rand_N32(sum_xts, len(self.discardsOp1))
            P2_N32 = self.rand_N32(sum_xts, len(self.discardsOp2))
            P3_N32 = self.rand_N32(sum_xts, len(self.discardsOp3))
            # 获取模拟T3后的wall
            wall = self.get_wall(P1_N32, P2_N32, P3_N32, wall)
            # print wall

            # 找到被分配的T3下标
            # index_set = []
            # for r in rd:
            #     for i in range(len(t3_sum)):
            #         if r <= t3_sum[i]:
            #             index_set.append(i)
            #             break
            # # 更新wall
            # for i in index_set:
            #     if i < 34:
            #         wall[i] = max(0, wall[i] - 3)
            #     else:
            #         index = int((i - 34) / 7 * 9) + (i - 34) % 7 + 1
            #         wall[index] = max(0, wall[index] - 1)
            #         wall[index + 1] = max(0, wall[index + 1] - 1)
            #         wall[index + 2] = max(0, wall[index + 2] - 1)
            # 二次采样 现在给T2采样
            # 分配剩下的t2
            # P1
            # RT1

            # 弃牌不再分配该组合的有效牌
            self.dzSet  # 搭子列表
            self.dzEfc  # 搭子有效牌列表
            self.efc_dzindex  # 有效牌的搭子列表

            RT1, wall = self.simulate_t2(P1_N32[1], self.discards1, wall)
            RT2, wall = self.simulate_t2(P2_N32[1], self.discards2, wall)
            RT3, wall = self.simulate_t2(P3_N32[1], self.discards3, wall)

            # P2 = [0] * (34 + 15 * 3)
            # for card in self.discards2:
            #     for i in self.efc_dzindex[card]:
            #         P2[i] += 1
            #
            # P3 = [0] * (34 + 15 * 3)
            # for card in self.discards2:
            #     for i in self.efc_dzindex[card]:
            #         P3[i] += 1
            #
            # # 进行 t2采样
            # # 生成t2可分配表
            #
            #
            # P2dz_index = []
            # P3dz_index = []
            #
            #
            #
            #
            #
            # for x in range(len(t2)):
            #     if P1[x] == 0 and t2[i]!=0:
            #         P1dz.append(x)
            # for x in range(len(t2)):
            #     if P1[x] == 0 and t2[i]!=0:
            #         P1dz.append(x)
            # sum1 = max(1,sum(P1dz))
            # RT1 = [[0] * 34, [0] * 34]
            #
            # # for x in range(len(P1dz)):
            # #     p = float(P1dz[x]) / sum1
            # #     for card in self.dzEfc[x]:
            # #         if x <= 33:
            # #             RT1[0][convert_hex2index(card)] += P1_N32[1] * p
            # #         else:
            # #             RT1[1][convert_hex2index(card)] += P1_N32[1] * p
            # #     for card in self.dzSet[x]:
            # #         wall[convert_hex2index(card)] -= P1_N32[1] * p
            #
            # sum2 = max(1,sum(P2dz))
            #
            # RT2 = [[0] * 34, [0] * 34]
            # for x in range(len(P2dz)):
            #     p = float(P2dz[x]) / sum2
            #     for card in self.dzEfc[x]:
            #         if x <= 33:
            #             RT2[0][convert_hex2index(card)] += P2_N32[1] * p
            #         else:
            #             RT2[1][convert_hex2index(card)] += P2_N32[1] * p
            #     for card in self.dzSet[x]:
            #         wall[convert_hex2index(card)] -= P2_N32[1] * p
            # sum3 = max(1,sum(P3dz))
            # RT3 = [[0] * 34, [0] * 34]
            # for x in range(len(P3dz)):
            #     p = float(P3dz[x]) / sum3
            #     for card in self.dzEfc[x]:
            #         if x <= 33:
            #             RT3[0][convert_hex2index(card)] += P3_N32[1] * p
            #         else:
            #             RT3[1][convert_hex2index(card)] += P3_N32[1] * p
            #     for card in self.dzSet[x]:
            #         wall[convert_hex2index(card)] -= P3_N32[1] * p

            # 给wallsum赋值
            # 给RT赋值
            for i in range(len(wall)):
                wall_sum[i] += wall[i]
                for j in range(2):
                    RT1_sum[j][i] += RT1[j][i]
                    RT2_sum[j][i] += RT2[j][i]
                    RT3_sum[j][i] += RT3[j][i]

        s_w = sum(wall_sum)
        T_selfmo = [0] * 34
        for i in range(len(wall_sum)):
            T_selfmo[i] = max(0, float(wall_sum[i]) / s_w)
            for j in range(2):
                RT1_sum[j][i] = float(RT1_sum[j][i]) / self.M
                RT2_sum[j][i] = float(RT2_sum[j][i]) / self.M
                RT3_sum[j][i] = float(RT3_sum[j][i]) / self.M

        return T_selfmo, RT1_sum, RT2_sum, RT3_sum


# 番型检测
class Fan:
    def __init__(self, kz, sz, jiang, suits):
        self.kz = kz
        self.sz = sz
        self.jiang = jiang
        self.suits=suits

    # 1 平胡
    def pingHu(self):
        if self.sz!=[]:
            return 1
        else:
            return 1
    # 2
    def pengPengHu(self):
        if len(self.kz)==4:
            return 2
        else:
            return 1

    # 3
    def qingYiSe(self):
        w = 0
        ti = 0
        to = 0
        cards = self.sz + self.kz
        cards.append(self.jiang)
        for card in cards:
            if card > 0x30:
                return 1
            elif card & 0xf0 == 0x00:
                w = 1
            elif card & 0xf0 == 0x10:
                ti = 1
            elif card & 0xf0 == 0x20:
                to = 1

        if w + ti + to > 1:
            return 1

        return 4


    def QiDui(self):
        pass

    # 4
    def jingGouGou(self):
        if self.pengPengHu()!=0 and len(self.suits)==4:
            return 4
        else:
            return 1

    # # 6
    # def qingPeng(self):
    #     if self.qingYiSe()!=0 and self.pengPengHu()!=0:
    #         return 8
    #     else:
    #         return 0
    #
    # # 7
    # def longQiDui(self):
    #
    #     pass
    #
    # # 8
    # def qingYiSeandQiDui(self):
    #     if self.qingYiSe()!=0 and self.qiDui()!=0:
    #         return 16
    #     else:
    #         return 0
    #
    #
    # # 9
    # def qingYiSeandJingGouGou(self):
    #     if self.qingYiSe()!=0 and self.jingGouGou()!=0:
    #         return 16
    #     else:
    #         return 0
    #
    #
    #
    # # 10
    # def tianHu(self):
    #     pass
    #
    # # 11
    # def diHu(self):
    #     pass
    #
    # # 12
    # def qingYiSeandLongQiDui(self):
    #     pass
    #
    # # 13
    # def shiBaLuoHan(self):
    #     if self.jingGouGou()!=0:
    #         num=0
    #         for suit in self.suits:
    #             if len(suit)==4:
    #                 num+=1
    #         if num==4:
    #             return 64
    #     return 0

    # 番互斥处理
    def mutexHandle(self, fanList):
        # 五暗刻２４不计４暗刻１８，３暗刻１１，碰碰胡１３
        if fanList[24] != 0:
            fanList[18] = 0
            fanList[11] = 0
            fanList[13] = 0

        if fanList[18] != 0:
            fanList[11] = 0

        # 4暗刻１８不计３暗刻１１
        if fanList[18] != 0:
            fanList[11] = 0
        # 纯带幺１５不计混带幺７
        if fanList[15] != 0:
            fanList[7] = 0
        # 清老头２５不计混老头１６
        if fanList[25] != 0:
            fanList[16] = 0
        # 清一色２２不计混一色１４
        if fanList[22] != 0:
            fanList[14] = 0
        # 大四喜２６不计小四喜２１
        if fanList[26] != 0:
            fanList[21] = 0
        # 字一色２３不计混一色１４
        if fanList[23] != 0:
            fanList[14] = 0
        # 大三元１９不计小三元１７
        if fanList[19] != 0:
            fanList[17] = 0
        # 大三元１９，小三元１７不计三元牌３
        if fanList[19] != 0 or fanList[17] != 0:
            fanList[3] = 0

        # 混老头１６不计混带幺７，不计碰碰胡１３
        if fanList[16] != 0:
            fanList[7] = 0
            fanList[13] = 0
        return fanList

    # combination [[kz],[sz],[aa],[ab/ac],xts,[left_cards]]
    def fanDetect(self):
        fanList = []
        # fanList.append(0)  # 添加一个空ｌｉｓｔ
        #0平胡
        fanList.append(self.pingHu())
        #1 碰碰胡
        fanList.append(self.pengPengHu())
        #2 清一色
        fanList.append(self.qingYiSe())
        #七对
        #3 金钩钩
        fanList.append(self.jingGouGou())
        if fanList[3]!=1:
            fanList[1]=1

        #计算底分
        score=1
        for fan in fanList:
            score*=fan

        #翻倍机制
        mul=0
        #根
        for suit in self.suits:
            if len(suit)==4:
                mul+=1
        #自摸
        #杠上开花
        #杠上炮
        #海底
        for i in range(mul):
            score*=2


        # fanList = self.mutexHandle(fanList)
        # print ('fanList', fanList)
        return score


'''
class HighFan:
    def __init__(self,a=[],suits=[]):
        self.a=a
        self.suits=suits
        pass

    #碰碰胡
    def pengpengHu(self):
        pengPengHu = copy.deepcopy(self.a)
        # 检测副露：
        for suit in self.suits:
            if suit[0] != suit[1]:
                return -1
        if len(pengPengHu[0] + len(pengPengHu[2])) >= 5:  # threshold 5
            for szab in pengPengHu[1] + pengPengHu[3]:
                pengPengHu[-1].extend(szab)
            # for ab in pengPengHu[3]:
            #     pengPengHu[-1].e
            pengPengHu[1] = []
            pengPengHu[3] = []
        pengPengHu = self.get_xts(a=pengPengHu)
        return pengPengHu

    #混一色
    def hunYiSe(self):
        hunYiSe = copy.deepcopy(self.a)
        wan = []
        wan_n = 0
        tiao = []
        tiao_n = 0
        tong = []
        tong_n = 0
        zi = []
        zi_n = 0
        #副露检测
        for suit in self.suits:
            if suit[0]&0xf0==0x00:
                wan_n+=3
            elif suit[0]&0xf0==0x10:
                tiao_n+=3
            elif suit[0]&0xf0==0x20:
                tong_n+=3
            else:
                zi_n+=3
        n=[wan_n,tiao_n,tong_n]
        if n.count(0)<2: #0 1 2 3
            return -1

        for t3t2_list in hunYiSe[0] + hunYiSe[1] + hunYiSe[2] + hunYiSe[3]:
            for t3t2 in t3t2_list:
                if t3t2[0] & 0xf0 == 0x00:
                    wan.append(t3t2)
                    wan_n+=len(t3t2)
                elif t3t2[0] & 0xf0 == 0x10:
                    tiao.append(t3t2)
                    tiao_n+=len(t3t2)
                elif t3t2[0] & 0xf0 == 0x20:
                    tong.append(t3t2)
                    tong_n+=len(t3t2)
                else:
                    zi.append(t3t2)
                    zi_n+=len(t3t2)
        for t1 in hunYiSe[-1]:
            if t1 & 0xf0 == 0x0:
                wan.append(t1)
                wan_n += len(t3t2)
            elif t1 & 0xf0 == 0x10:
                tiao.append(t1)
                tiao_n += len(t3t2)
            elif t1 & 0xf0 == 0x20:
                tong.append(t1)
                tong_n += len(t3t2)
            else:
                zi.append(t1)
                zi_n += len(t3t2)
        threshold=14
        choose=-1
        if wan_n+zi_n>=threshold:
            choose=0
        elif tiao_n+zi_n>=threshold:
            choose=1
        elif tong_n+zi_n>=threshold:
            choose=2
        if choose==-1:
            return -1

        hand = [wan, tiao, tong, zi]
        tmp_a=[]
        for color in range(len(hand)):
            if color==choose or color==3:
                for cmb in hand[color]:

                    if type(hand[color])==list:
                        pass



    def  highScoreFanDetection(self, a=[]):
        fan_combination=[]
        # fanList = []
        # fanList.append(0)  # 添加一个空ｌｉｓｔ
        # 断幺九1
        # duanYaoJiu = self.duanYaoJiu()
        # fanList.append(duanYaoJiu)
        # 门风刻2
        # menFengKe = self.menFengKe()
        # fanList.append(menFengKe)
        # 三元牌3
        # sanYuanPai = self.sanYuanPai()
        # fanList.append(sanYuanPai)
        # 双龙抱4
        # shuangLongBao = self.shuangLongBao()
        # fanList.append(shuangLongBao)
        # 全求人5
        # quanQiuRen=self.quanQiuRen()
        # fanList.append(0)
        # 2平胡6
        # pingHu=copy.deepcopy(a)
        # fan_combination.append(pingHu)
        # 2混带幺7
        # hunDaiYao = self.hunDaiYao()
        # fan_combination.append(hunDaiYao)
        # 2三色同顺8
        # sanSeTongShun = self.sanSeTongShun()
        # fan_combination.append(sanSeTongShun)
        # 2一条龙9
        # yiTiaoLong = self.yiTiaoLong()
        # fan_combination.append(yiTiaoLong)
        # 2双双龙抱10
        # self.shuangShuangLongBao()
        # fan_combination.append(0)
        # 三暗刻11
        # sanAnKe = self.sanAnKe()
        # fanList.append(sanAnKe)
        # 2 三色同刻12
        # sanSeTongKe = self.sanSeTongKe()
        # fan_combination.append(sanSeTongKe)
        # 4 碰碰胡13


        # 4 混一色14 阈值14



        # fan_combination.append(hunYiSe)

        # 4 纯带幺15
        chunDaiYao = self.chunDaiYao()
        fan_combination.append(chunDaiYao)
        # 4 混老头16
        hunLaoTou = self.hunLaoTou()
        fan_combination.append(hunLaoTou)
        # 4 小三元17
        xiaoSanYuan = self.xiaoSanYuan()
        fan_combination.append(xiaoSanYuan)
        # 6四暗刻18
        # siAnKe = self.siAnKe()
        # fan_combination.append(siAnKe)
        # 8 大三元19
        daSanYuan = self.daSanYuan()
        fan_combination.append(daSanYuan)

        # 四杠子20
        # siGangZi=self.siGangZi()
        # fanList.append(0)

        # 8小四喜21
        xiaoSiXi = self.xiaoSiXi()
        fan_combination.append(xiaoSiXi)
        # 8 清一色22
        qingYiSe = self.qingYiSe()
        fan_combination.append(qingYiSe)
        # 8 字一色23
        ziYiSe = self.ziYiSe()
        fan_combination.append(ziYiSe)
        # 五暗刻24
        # wuAnKe = self.wuAnKe()
        # fan_combination.append(wuAnKe)
        # 8 清老头25
        qingLaoTou = self.qingLaoTou()
        fan_combination.append(qingLaoTou)
        # 16 大四喜26
        daSiXi = self.daSiXi()
        fan_combination.append(daSiXi)
        # pass

        # fanList = self.mutexHandle(fanList)
        # print ('fanList', fanList)
        return fan_combination
'''


class pinghu:
    def __init__(self, cards, suits, leftNum=[], discards=[], discards_real=[], discardsOp=[], round=0, remainNum=134,
                 fengWei=0, seat_id=0, choose_color=[-1, -1, -1, -1], hu_cards=[[], [], [], []],
                 hu_fan=[[], [], [], []]):
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
        RT1 = [[0] * 34] * 3
        RT2 = [[0] * 34] * 3
        RT3 = [[0] * 34] * 3
        self.fengWei = fengWei
        self.choose_color = choose_color
        self.hu_cards = hu_cards
        self.hu_fan = hu_fan

    # 获取有效牌,输入为搭子集合,
    def get_effective_cards(self, dz_set=[]):
        """

        :param dz_set: 搭子组合 形如[[1,1],[2,3],[5,7]]
        :return:返回一个有效牌set 形如[1,4,6]
        """
        effective_cards = []
        for dz in dz_set:
            if len(dz) == 1:
                effective_cards.append(dz[0])
            elif dz[1] == dz[0]:
                effective_cards.append(dz[0])
            elif dz[1] == dz[0] + 1:
                if int(dz[0]) & 0x0F == 1:  # 靠边搭子的特殊处理
                    effective_cards.append(dz[0] + 2)
                elif int(dz[0]) & 0x0F == 8:
                    effective_cards.append((dz[0] - 1))
                else:
                    effective_cards.append(dz[0] - 1)
                    effective_cards.append(dz[0] + 2)
            elif dz[1] == dz[0] + 2:  # ac
                effective_cards.append(dz[0] + 1)
        effective_cards = set(effective_cards)  # set 和list的区别？
        return list(effective_cards)

    # 获取有效牌及概率
    def get_effective_cards_w(self, dz_set=[], left_num=[]):
        cards_num = self.remainNum
        effective_cards = []
        w = []
        for dz in dz_set:
            if dz[1] == dz[0]:
                effective_cards.append(dz[0])
                w.append(float(left_num[MJ.translate16_33(dz[0])]) / cards_num * 5)  # 修改缩进,发现致命错误panic 忘了写float

            elif dz[1] == dz[0] + 1:
                if int(dz[0]) & 0x0F == 1:
                    effective_cards.append(dz[0] + 2)
                    w.append(float(left_num[MJ.translate16_33(dz[0] + 2)]) / cards_num * 1)
                elif int(dz[0]) & 0x0F == 8:
                    effective_cards.append((dz[0] - 1))
                    w.append(float(left_num[MJ.translate16_33(dz[0] - 1)]) / cards_num * 1)
                else:
                    effective_cards.append(dz[0] - 1)
                    effective_cards.append(dz[0] + 2)
                    w.append(float(left_num[MJ.translate16_33(int(dz[0]) - 1)] +
                                   left_num[MJ.translate16_33(int(dz[0]) + 2)]) / cards_num * 1)
            elif dz[1] == dz[0] + 2:
                effective_cards.append(dz[0] + 1)
                w.append(float(left_num[MJ.translate16_33(int(dz[0]) + 1)]) / cards_num * 1)
        return effective_cards, w

    # 花色分离，输出为原来的牌
    def split_type_s(self, cards):
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

    # 获取面子及搭子
    def get_32N(self, cards):
        cards.sort()
        kz = []
        sz = []
        aa = []
        ab = []
        ac = []
        lastCard = 0
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
    def in_cards(self, t32=[], cards=[]):
        for card in t32:
            if card not in cards:
                return False
        return True

    # 生成所有的分类情况
    def extract_32N(self, cards=[], t32_branch=[], t32_set=[]):
        """

        :param cards: 手牌 list 形如[1,2,3,4,5,6]
        :param t32_branch:  有用牌组合 形如 [[1,1,1],[2,3]]
        :param t32_set: 有用组合和剩下的牌 中间用0隔开 形如[[[1,1,1],[2,3]，0,[6,9]], [...]]
        :return:t32_set 有用组合和剩下的牌 中间用0隔开 形如[[[1,1,1],[2,3]，0,[6,9]], [...]]
        """

        t32N = self.get_32N(cards=cards)  # 获取刻字面子搭子 如[[1,1,1], [5,5,5],[2,2],[3,4] ] kz sz aa ab ac

        if len(t32N) == 0:  # 都是孤张的情况
            t32_set.extend(t32_branch)
            # t32_set.extend([cards])
            t32_set.append(0)
            t32_set.extend([cards])
            return t32_set
        else:
            for t32 in t32N:  # 遍历
                if self.in_cards(t32=t32, cards=cards):  # 二次验证 防止bug？
                    cards_r = copy.copy(cards)
                    for card in t32:
                        cards_r.remove(card)  # 把有用的牌移除
                    t32_branch.append(t32)  # 把这个组合放到有用牌分支

                    # 递归 求出有用的组合牌 和剩下的牌 都放到t32_set中 例如[[1,1,1],[2,3,4],[5,9]]
                    self.extract_32N(cards=cards_r, t32_branch=t32_branch, t32_set=t32_set)
                    if len(t32_branch) >= 1:
                        t32_branch.pop(-1)  # 如果有用牌分支不为空，把最后一个分支移除 因为此时一次递归已经结束 回溯需要？

    # 万条筒花色的拆分信息
    '''
    sub=[[]]*4
    sub[0] kz
    sub[1] sz
    sub[2] aa
    sub[3] ２N
    sub[4] 得分
    sub[5] 废牌
    搜索理论一：如果同一分支的废牌（保留牌）一致，那么他们属于同一分支的不同分割类型，其有效牌、废牌共用
    '''

    def tree_expand(self, cards):
        """

        :param cards:
        :return:
        """
        all = []  # 全部的情况
        # all=[ [ [[1,1,1],[2,2,2]],  [[11,12,13],[14,15,16]],  [[21,21],[22,22]], [[26,27],[28,30]], 0, [] ], [...]  ]
        t32_set = []  # 有用组合和剩下的牌 中间用0隔开 形如[[1,1,1],[2,2,2],[3,3]，0,[6,9]]
        self.extract_32N(cards=cards, t32_branch=[], t32_set=t32_set)  # 运算出t32_branch t32_set
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
            if t != 0:  # t不等于0的时候说明是孤张前的顺子刻字搭子牌
                if len(t) == 3:  # 如果长度为3 可能为三张相同的刻字 也可能为顺子

                    if t[0] == t[1]:  # 前两张相同 是刻字
                        kz.append(t)
                    else:  # 否则是顺子
                        sz.append(t)
                        # print (sub)
                elif len(t) == 2:  # 两张牌 区分对子和ab ac
                    if t[1] == t[0]:
                        aa.append(t)
                    else:
                        t2N.append(t)

            else:  # 遍历到了0  即下一索引到了剩下的牌
                '修改，使计算时间缩短'
                leftCards = t32_set[i + 1]  # 废牌
                efc_cards = self.get_effective_cards(dz_set=t2N)  # t2N中不包含ａａ  有效牌集合[1,2,...]
                # 去除划分不合理的情况，例如345　划分为34　或35等，对于333 划分为33　和3的情况，考虑有将牌的情况暂时不做处理
                for card in leftCards:  # 不合理时这一搭子组合划分作废 看下一个搭子
                    if card in efc_cards:
                        flag = False
                        break

                if flag:
                    all.append([kz, sz, aa, t2N, 0, leftCards])
                    #  形如：all=[ [ [[1,1,1],[2,2,2]],  [[11,12,13],[14,15,16]],  [[21,21],[22,22]], [[26,27],[28,30]], 0, [] ], [...] ]

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

    # 修改了all结构，重新开始设计！
    # 字牌信息生成
    def zi_expand(self, cards):
        t3N = []
        t2N = []
        aa = []
        # ab = []
        # ac = []
        # t1289 = []
        efc = []
        left_cards = []
        unique = set(cards)
        for card in unique:
            if cards.count(card) == 4:
                t3N.append([card, card, card])
                left_cards.append(card)
            elif cards.count(card) == 3:
                t3N.append([card, card, card])
            elif cards.count(card) == 2:
                aa.append([card, card])
                efc.append(card)
            elif cards.count(card) == 1:
                left_cards.append(card)

        # return [[t3N,aa,ab,ac,t1289,efc,left_cards]]
        return [[t3N, [], aa, [], 0, left_cards]]

    # 获取向听数
    def get_xts(self, a):
        suits = self.suits
        t3N = a[0] + a[1]
        a[4] = 14 - (len(t3N) + len(suits)) * 3  # 原来是0的位置现在值改为向听数位
        # 有将牌
        has_aa = False
        if len(a[2]) > 0:
            has_aa = True
        # 计算向听数
        if has_aa:  # has do 当２Ｎ与３Ｎ数量小于５时，存在没有减去相应待填数，即废牌也会有１张作为２Ｎ或３Ｎ的待选位,
            # print()all_src
            if len(suits) + len(t3N) + len(a[2]) + len(a[3]) - 1 >= 4:

                a[4] -= (4 - (len(suits) + len(t3N))) * 2 + 2
            else:
                a[4] -= (len(a[2]) + len(a[3]) - 1) * 2 + 2 + 4 - (
                    len(suits) + len(t3N) + len(a[2]) + len(a[3]) - 1)  # 0717 17:24
        # 无将牌
        else:
            if len(suits) + len(t3N) + len(a[2]) + len(a[3]) >= 4:
                a[4] -= (4 - (len(suits) + len(t3N))) * 2 + 1
            else:
                a[4] -= (len(a[2]) + len(a[3])) * 2 + 1 + 4 - (
                    len(suits) + len(t3N) + len(a[2]) + len(a[3]))
        return a

    # 获取所有的拆分信息并输出
    def sys_info_V3(self, cards, suits, left_num=[4] * 34):

        # 去除宝牌计算信息，后面出牌和动作决策再单独考虑宝牌信息
        cards_copy = copy.copy(cards)
        # 花色分离
        cards_wan, cards_tiao, cards_tong, cards_zi = self.split_type_s(cards_copy)

        # all=[ [ [[1,1,1],[2,2,2]],  [[11,12,13],[14,15,16]],  [[21,21],[22,22]], [[26,27],[28,30]], 0, [] ], [...]  ]
        dingQue = self.choose_color[self.seat_id]
        dingQue_cards=[]
        if dingQue == 0:
            dingQue_cards = copy.copy(cards_wan)
            cards_wan=[]

        elif dingQue==1:
            dingQue_cards = copy.copy(cards_tiao)
            cards_tiao=[]

        elif dingQue==2:
            dingQue_cards = copy.copy(cards_tong)
            cards_tong=[]

        wan_expd = self.tree_expand(cards=cards_wan)  # 所有expand返回的结果都类似上行
        tiao_expd = self.tree_expand(cards=cards_tiao)
        tong_expd = self.tree_expand(cards=cards_tong)
        zi_expd = self.zi_expand(cards=cards_zi)

        all = []
        for i in wan_expd:
            for j in tiao_expd:
                for k in tong_expd:
                    for m in zi_expd:
                        branch = []
                        # 将每种花色的4个字段合并成一个字段
                        for n in range(6):
                            branch.append(i[n] + j[n] + k[n] + m[n])
                        # print ("dingQue_cards",dingQue_cards)
                        branch[-1].extend(dingQue_cards)
                        all.append(branch)
        # 按3N,2N数量排序，降序
        # all.sort(key=lambda k: (len(k[0]), len(k[1]), len(k[2])), reverse=True)
        # 向听数计算
        # xts=[14-3*len(suits)]*len(all)
        for i in range(len(all)):
            all[i] = self.get_xts(a=all[i])
        all.sort(key=lambda k: (k[4], len(k[-1])))
        # print ("all_src", all)

        xts_min = all[0][4]
        # 获取向听数最小的all分支  Q:这里是否是向听数最大？
        min_index = 0
        for i in range(len(all)):

            # min_index = i
            if all[i][4] > all[0][4] + 1:  # xts+１以下的组合  Q:这里是否应当是和all[min_index]比较？
                # A:不用 在前面已经根据向听数位来将不同组合排序好
                min_index = i  # 因为之前已经排好序，所以找是否有比最小向听数+1还大的组合 找到立即退出
                break

        if min_index == 0:  # 如果全部都匹配，则min_index没有被赋值，将min_index赋予ａｌｌ长度
            min_index = len(all)

        all = all[:min_index]  # 这里就是找xts+1及以下的所有组合 重新赋值
        # print("all_terminal", all)
        return all

    # 孤张权重表,给三元牌与风位添加bias　正向权重
    # todo 待更新
    # 策略:自己摸到该张牌，并且组合成３Ｎ的概率
    def left_card_weight(self, card, left_num):
        # print ('left_num',left_num)
        remainNum = self.remainNum  # 场面还剩下多少张牌
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
            d_w = sf + float(left_num[i + 1]) / remainNum * float(left_num[i + 2]) / remainNum * 1
        elif card % 16 == 2:  # 22+13+3(14)+43   222 123 234
            d_w = sf + float(left_num[i - 1]) / remainNum * float(left_num[i + 1]) / remainNum * 1 \
                  + float(left_num[i + 1]) / remainNum * float(left_num[i + 2]) / remainNum * 1
            # d_w = left_num[i - 1] + left_num[i] * 3 + left_num[i + 1] * 2 + left_num[i + 2]
        elif card % 16 == 8:  # 888 678 789
            d_w = sf + float(left_num[i - 2]) / remainNum * float(left_num[i - 1]) / remainNum * 1 \
                  + float(left_num[i - 1]) / remainNum * float(left_num[i + 1]) / remainNum * 1
            # d_w = left_num[i - 2] + left_num[i - 1] * 2 + left_num[i] * 3 + left_num[i + 1]
        elif card % 16 == 9:  # 999 789
            d_w = sf + float(left_num[i - 2]) / remainNum * float(left_num[i - 1]) / remainNum * 1
            # d_w = left_num[i - 2] + left_num[i - 1] + left_num[i] * 3  # 删除多添加的×２
        else:  # 555 345 456 567
            # print (left_num)
            d_w = sf + float(left_num[i - 2]) / remainNum * float(left_num[i - 1]) / remainNum * 1 \
                  + float(left_num[i - 1]) / remainNum * float(left_num[i + 1]) / remainNum * 1 \
                  + float(left_num[i + 1]) / remainNum * float(left_num[i + 2]) / remainNum * 1
            # d_w = left_num[i - 2] + left_num[i - 1] * 2 + left_num[i] * 3 + left_num[i + 1] * 2 + left_num[i + 2]
        print("i=", i, d_w)
        return d_w

    # t2N列表最后的ａａ
    def is_last_aa(self, t2N):
        for t in t2N:
            if t[0] == t[1]:
                return False
        return True

    '''
    胡牌概率计算
    br[0] kz
    br[1] sz
    br[2] aa
    br[3] t2N
    br[4] path_w
    br[5] 废牌
    '''

    def cost(self, all, suits, left_num=[]):
        # todo 没有区分听牌前与听牌时的２Ｎ获取权重
        print ('cost', all)
        # path_w[0] 胡牌概率
        # path_w[1] 废牌表
        path_w = []  # 创建一个存储胡牌概率和废牌的list
        for i in range(len(all)):
            path_w.append([1.0, copy.copy(all[i][-1])])
        # 全部搜索会导致搜索空间极大
        for index_all in range(len(all)):  # 选出最大期望概率胡牌路径，选择该路径，从剩余牌中再选择最佳出牌顺序，局部最优
            # TODO 之后考虑全局最优
            a = all[index_all]
            t2N = copy.deepcopy(a[2] + a[3])
            efc_cards, t2_w = self.get_effective_cards_w(dz_set=t2N, left_num=left_num)
            for i in range(len(t2N)):
                t2N[i].append(t2_w[i])
            bl = 4 - len(suits) - len(a[0]) - len(a[1])
            # print ('t2N', t2N)
            if a[2] != []:  # 定将
                t2N[:len(a[2])] = sorted(t2N[:len(a[2])], key=lambda k: k[2], reverse=True)
                t2N[len(a[2]):] = sorted(t2N[len(a[2]):], key=lambda k: k[2], reverse=True)
                # todo １＋　排序 ,测试但是效果不好就不用了
                # t2N[len(a[1]):].sort(key=lambda k: k[2], reverse=True)
                if bl <= len(t2N) - 1:  # t2N溢出,需要出一张２N
                    for i in range(1, bl + 1):
                        if t2N[i][0] == t2N[i][1] and ((i + 1 == len(t2N)) or self.is_last_aa(t2N[i + 1:])):
                            path_w[index_all][0] *= (t2N[i][2] + t2N[0][2])
                        else:
                            path_w[index_all][0] *= t2N[i][2]
                    # print bl + 1,len(t2N)

                    for j in range(bl + 1, len(t2N)):  # 废牌添加,
                        if a[-1] == [] and j == len(t2N) - 1:  # 只添加最后的废牌: #只有当废牌区为空时，才将２Ｎ放入
                            path_w[index_all][1].append(t2N[-1][0])
                            path_w[index_all][1].append(t2N[-1][1])

                else:
                    for i in range(1, len(t2N)):
                        if t2N[i][0] == t2N[i][1] and ((i + 1 == len(t2N)) or self.is_last_aa(t2N[i + 1:])):
                            path_w[index_all][0] *= (t2N[i][2] + t2N[0][2])
                        else:
                            path_w[index_all][0] *= t2N[i][2]
                    for j in range(bl - len(t2N) + 1):  # TODO 未填的３N ，这种处理方法有点粗糙
                        path_w[index_all][0] *= float(80.0) / (self.remainNum * self.remainNum)

            else:  # 未定将牌
                t2N = sorted(t2N, key=lambda k: k[2], reverse=True)
                if bl <= len(t2N):  # t2N溢出,需要出一张２N
                    for t in t2N[:bl]:  # 计算胡牌概率
                        path_w[index_all][0] *= t[2]
                    for j in range(bl, len(t2N)):  # 废牌添加,
                        if a[-1] == [] and j == len(t2N) - 1:  # 只添加最后的废牌: #只有当废牌区为空时，才将２Ｎ放入
                            path_w[index_all][1].append(t2N[-1][0])
                            path_w[index_all][1].append(t2N[-1][1])
                else:
                    for t in t2N:
                        path_w[index_all][0] *= t[2]
                    for j in range(bl - len(t2N)):  # 未填的３N
                        # TODO 3N补充，待改进
                        path_w[index_all][0] *= float(80.0) / (self.remainNum * self.remainNum)
                # 这里可能填入２N
                left_cards = path_w[index_all][1]
                w_jiang = [0] * len(left_cards)
                for k in range(len(left_cards)):
                    w_jiang[k] = float(left_num[MJ.translate16_33(left_cards[k])]) / self.remainNum
                path_w[index_all][0] *= max(w_jiang)  # 添加将牌概率
                if len(left_cards) > 1:  # 填胡状态下，差一个将牌胡牌,这里
                    path_w[index_all][1].remove(left_cards[w_jiang.index(max(w_jiang))])

                # 特殊情况的处理，添加没有将牌，但有刻子与２Ｎ的出牌情景
                if a[-1] == [] and len(a[3]) == 1:
                    kz = []  # 存在刻子
                    for t in a[0]:
                        if t[0] == t[1]:
                            kz = t
                            break
                    if kz != []:
                        _, rate_out_3N = self.get_effective_cards_w(dz_set=a[3], left_num=left_num)
                        if float(rate_out_3N[0]) / 2 > path_w[index_all][0]:
                            path_w[index_all][0] = float(rate_out_3N[0])
                            path_w[index_all][1] = [kz[0]]
        print("path_w", path_w)
        return path_w

    # 废牌权重
    def discards_w(self, discards=[], left_num=[], ndcards={}):
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
        # print ("discards_w",discards_w)
        return discards[discards_w.index(min(discards_w))]

    def opponent(self):
        T_selfmo = []  # 自摸概率表
        RT1 = [[0] * 34, [0] * 34]  # 玩家1有效牌概率表
        RT2 = [[0] * 34, [0] * 34]  # 玩家2有效牌概率表
        RT3 = [[0] * 34, [0] * 34]  # 玩家3有效牌概率表
        for i in self.leftNum:
            T_selfmo.append(float(i) / self.remainNum)
        return T_selfmo, RT1, RT2, RT3



        #
        # pass

    # [aaa,abc,aa,ab,aab,xts,t1]
    # 递归检测aab
    def get_a_fastWinning(self, flag=True, a_CB=[], all_fastWinning=[]):
        if flag == False:
            all_fastWinning.append(a_CB)
            return

        # a_CB
        flag = False
        for T1 in a_CB[-1]:
            for aa in a_CB[2]:
                if T1 + 1 in aa or T1 + 2 in aa or T1 - 1 in aa or T1 - 2 in aa:
                    a_CP = copy.deepcopy(a_CB)
                    a_CP[-1].remove(T1)
                    a_CP[2].remove(aa)
                    a_CP[4].append(sorted([aa[0], aa[1], T1]))
                    flag = True
                    self.get_a_fastWinning(flag, a_CP, all_fastWinning)

            for ab in a_CB[3]:
                if T1 + 1 in ab or T1 + 2 in ab or T1 - 1 in ab or T1 - 2 in ab:
                    a_CP = copy.deepcopy(a_CB)
                    a_CP[3].remove(ab)
                    a_CP[-1].remove(T1)
                    a_CP[4].append(sorted([ab[0], ab[1], T1]))
                    flag = True
                    self.get_a_fastWinning(flag, a_CP, all_fastWinning)
        if flag == False:
            self.get_a_fastWinning(flag, a_CB, all_fastWinning)

    # 出牌决策
    def defend_V2_2(self, all_combination):
        """
        :param all_combination: # 所有拆牌组合的可能
        # all=[ [ [[1,1,1],[2,2,2]],  [[11,12,13],[14,15,16]],  [[21,21],[22,22]], [[26,27],[28,30]], 0, [] ], [...]  ]
        :return: outcard
        """


        '''
                   第一阶段：完全孤张牌出牌次序
                   原则：出相关性最低的孤张牌，剩余牌与孤张牌的联系性最低
                   现阶段只考虑ｘｔｓ最小的情况
               '''
        # 只取xts最小的情况
        all = copy.deepcopy(all_combination)
        # all_same_xts_and_left = copy.deepcopy(all_combination)
        min_xts = all_combination[0][4]

        for a in all_combination:  # 获取ｘｔｓ相同的组合
            if a[4] != min_xts:
                all.remove(a)



        left_all_cards = []  # 全部组合的废牌集合
        left_cards = []  # 任何组合都包含的真正废牌
        left_cards_ndCards = {}  # 废牌与之联系的次级牌（包括２Ｎ，与废牌，用来计算手牌权重）

        jiangCards = []
        for branch in all_same_xts_and_left:
            # 在缺将牌胡牌的情况下
            # branch 形如[ [[1,1,1],[2,2,2]],  [[11,12,13],[14,15,16]],  [[21,21],[22,22]], [[26,27],[28,30]], 0, [废牌] ]
            if min_xts == 1 and branch[2] == [] and branch[3] == [] and len(branch[-1]) == 2:
                jiangCards.extend(branch[-1])  # 两张废牌可做将牌
            left_all_cards += branch[-1]  # 这个组合的废牌添加进全部废牌表中

            # 生成left_cards_ndCards
            for card in branch[-1]:  # 这是一个dict
                if card not in left_cards_ndCards.keys():
                    left_cards_ndCards[card] = []
                for t2 in branch[2] + branch[3]:
                    left_cards_ndCards[card].extend(t2)
                left_cards_ndCards[card].extend(branch[-1])
                left_cards_ndCards[card].remove(card)

                # 构造完成后的left_cards = {废牌1:[对应的所有aa和2n 1,1,4,6 ,除掉“废牌1”的废牌 ]，...}
        unique_l = list(set(left_all_cards))  # 无重复废牌
        left_cards_w = []  # 废牌权重
        for card in unique_l:
            if left_all_cards.count(card) == len(all_same_xts_and_left):  # 有多少个组合不要这张card 判断是否所有都不要
                left_numCP = copy.copy(self.leftNum)
                for c in set(left_cards_ndCards[card]):  # 如果所有组合都不要这张牌 那么遍历这张牌相关的所有2n 和 废牌
                    left_numCP[MJ.convert_hex2index(c)] = self.remainNum  # Q:把剩下的那些牌剩余牌数人工修改为最多？
                left_cards.append(card)
                left_cards_w.append(self.left_card_weight(card=card, left_num=left_numCP))  # 更新点：添加废牌权重
        if left_cards != []:
            # print(left_cards[left_cards_w.index(min(left_cards_w))])
            print ('state first')
            # 差胡牌将
            if jiangCards != []:
                if len(jiangCards) == 2:
                    if left_num[MJ.convert_hex2index(jiangCards[0])] > left_num[MJ.convert_hex2index(jiangCards[1])]:
                        return jiangCards[0]
                    else:
                        return jiangCards[1]
                else:
                    print('jiangCards Error', jiangCards)
            return left_cards[left_cards_w.index(min(left_cards_w))]

        # '''
        # 第二三阶段
        # 使用搜索树处理ｘｔｓ<=3的出牌情况
        #
        # 当unique_l不为空时，从所有废牌(unique_l)中出一张
        # 如果为空，从所有的t2Ｎ中出一张
        # '''
        # all = all
        ndcards = {}
        # for a in all:
        #     cards = []
        #     for t in a[2] + a[3]:
        #         cards.extend(t)
        #     cards.extend(a[-1])
        #     cards = list(set(cards))
        #     for card in cards:
        #         if card not in ndcards.keys():
        #             ndcards[card] = []
        #         for t2 in a[2] + a[3]:
        #             ndcards[card].extend(t2)
        #         ndcards[card].extend(a[-1])
        #         ndcards[card] = list(set(ndcards[card]))
        #         ndcards[card].remove(card)
        # print ('ndcards', ndcards)
        # if all[0][4] <= 3:
        #
        #     Tree = SearchTree(cards=self.cards, suits=self.suits, leftNum=self.leftNum, all=all,
        #                       all_fastWinning=all_fastWinning,
        #                       all_highFan=all_highFan,
        #                       remainNum=self.remainNum, T_selfmo=T_selfmo, RT1=RT1, RT2=RT2, RT3=RT3,
        #                       Txts_transpose=Txts_transpose, round=self.round)
        #     scoreDict = Tree.getCardScore()
        #     scoreDict = sorted(scoreDict.items(), key=lambda k: k[1], reverse=True)
        #     maxScoreCards = []
        #     for i in range(len(scoreDict)):
        #         if scoreDict[i][1] == scoreDict[0][1]:
        #             maxScoreCards.append(scoreDict[i][0])
        #     print ('maxScoreCards', maxScoreCards)
        #     if maxScoreCards != []:
        #         return self.discards_w(maxScoreCards, self.leftNum, ndcards)
        #
        # # 当unique_l不为空时(策略集)
        # # 重新生成胡牌信息还是使用以往的信息?
        # # 第一种测试方法：使用废牌数量(最简单的方法，局部最优，可能会出现遗漏的情况),之后改进
        # # v1 细化废牌权重
        unique_l = []
        all = all3
        print ("all_2", all)
        for a in all:
            unique_l += a[-1]
        if unique_l != []:
            discards_w = {}
            path_w = self.cost(all, suits=suits, left_num=left_num)
            for i in range(len(all)):
                for card in path_w[i][1]:
                    if card in discards_w.keys():
                        # todo 需要加上场面剩余牌信息
                        discards_w[card] += path_w[i][0]
                    else:
                        discards_w[card] = path_w[i][0]
            discards_w = sorted(discards_w.items(), key=lambda k: k[1], reverse=True)
            print ("discards_w", discards_w)
            return int(discards_w[0][0])

        else:  # 如果废牌区为空，使用搜索，出价值最低的２Ｎ
            path_w = self.cost(all, suits=suits, left_num=left_num)
            path_w.sort(key=lambda k: k[0], reverse=True)
            out_card = self.discards_w(discards=path_w[0][-1], left_num=left_num, ndcards=ndcards)
            print ("out_card", out_card)
            return out_card

    # 决策出牌
    def recommend_card(self):
        t1 = time.clock()
        all = self.sys_info_V3(cards=self.cards, suits=self.suits, left_num=self.leftNum)  # 所有拆牌组合的可能
        # all=[ [ [[1,1,1],[2,2,2]],  [[11,12,13],[14,15,16]],  [[21,21],[22,22]], [[26,27],[28,30]], 0, [] ], [...]  ]
        t2 = time.clock()
        print('sys_info_V3', t2 - t1, 'xts=', all[0][4])
        # print("",all)

        return self.defend_V2_2(all_combination=all)

    # 吃碰杠动作决策
    # 反向计算：将op_card 放到手牌中，如果能组成更好的３Ｎ，则返回该组合
    # 完全局部最优策略：选择拆分组合中期望最高的组合，
    def recommend_op(self, op_card, self_turn=False, isHu=False):
        if isHu:
            return True,[]
        # 2项比较：前项计算胡牌ｒａｔｅ，吃碰杠后计算胡牌ｒａｔｅ比较,杠牌在不过多影响条件下都进行，其他需增加胡牌概率
        cards = self.cards
        suits = self.suits
        left_num = self.leftNum
        all_former = self.sys_info_V3(cards=cards, suits=suits, left_num=left_num)
        print ("recommend_op,all_former", all_former)

        # 杠牌限制，只杠已成型，且没有被用到的牌（在废牌区），杠牌没有分数奖励，只有多摸一张牌的机会
        allSamexts = []
        for a in all_former:
            if a[4] == all_former[0][4]:
                allSamexts.append(a)
        if self_turn:  # 暗杠补杠
            for a in allSamexts:
                for card in a[-1]:
                    if [card, card, card] in a[0] or [card, card, card] in suits:
                        return [card, card, card, card]
            return False,[]
        # 明杠
        for a in allSamexts:
            if [op_card, op_card, op_card] in a[0]:
                return False, [op_card, op_card, op_card, op_card]

        '''
        # 向听数大于３则执行ｖ１版ｏｐ
        # 吃碰
        '''
        # if xts_min <= 3:
        #     opSet=[]
        #     print ('cards',cards,canchi)
        #     if canchi: # 上家出牌，可以吃，碰
        #         if op_card - 2 in cards and op_card - 1 in cards:
        #             opSet.append([op_card-2,op_card-1,op_card])
        #         if op_card-1 in cards and op_card+1 in cards:
        #             opSet.append([op_card-1 ,op_card,op_card+1])
        #         if op_card+1 in cards and op_card+2 in cards:
        #             opSet.append([op_card,op_card+1,op_card+2])
        #         if cards.count(op_card)>=2:
        #             opSet.append([op_card,op_card,op_card])
        #
        #     else:# 下对家出牌，只能碰
        #         if cards.count(op_card)>=2:
        #             opSet.append([op_card,op_card,op_card])
        #     opList = []
        #     print ('opSet',opSet)
        #     for op in opSet:
        #         print ('op=', op)
        #         suitsCP = copy.deepcopy(suits)
        #         suitsCP.append(op)
        #         opCP=copy.copy(op)
        #         opCP.remove(op_card)
        #         cardsCP = copy.copy(cards)
        #
        #         print ()
        #         # print ('opCP=', opCP)
        #         # print (cardsCP)
        #         for card in opCP:
        #             cardsCP.remove(card)
        #         print ('op cardsCP,suits', cardsCP, suitsCP)
        #         _, all_, _ = self.sys_info_V3(cards=cardsCP, suits=suitsCP)
        #
        #         samexts=[]
        #         for a in all_:
        #             if a[4]==all_[0][4]:
        #                 samexts.append(a)
        #
        #          # [[], [[22, 23, 24]], [[41, 41]], [[7, 9], [36, 38]], 4, [5, 9, 34, 40]],
        #          # [[], [[22, 23, 24]], [], [[7, 9], [34, 36], [40, 41]], 4, [5, 9, 38, 41]],
        #          # [[], [[22, 23, 24]], [], [[7, 9], [36, 38], [40, 41]], 4, [5, 9, 34, 41]]]
        #
        #         # [[], [[22, 23, 24]], [], [[5, 7], [34, 36], [40, 41]], 3, [38, 41]],
        #          # [[], [[22, 23, 24]], [], [[5, 7], [36, 38], [40, 41]], 3, [34, 41]]]
        #
        #
        #         print ('all',samexts)
        #         Tree = SearchTree(cards=cardsCP, suits=suitsCP, leftNum=self.leftNum, all=samexts,
        #                           remainNum=self.remainNum, dgtable=[1] * 34,type=2,op_card=op_card)
        #         # Tree.getCardScore()
        #         scoreDict = Tree.getCardScore()
        #         if op_card in scoreDict.keys():
        #             scoreDict[op_card]==0
        #
        #         scoreDict = sorted(scoreDict.items(), key=lambda k: k[1], reverse=True)
        #         # sumScore=0
        #         # for k in scoreDict.keys():
        #         #     sumScore+=scoreDict[k]
        #         if scoreDict!=[]:
        #             max_score = scoreDict[0][1]
        #             opList.append([op, max_score])
        #         else:
        #             opList.append([op,0])
        #         # opList.append([op,sumScore])
        #     opList.sort(key=lambda k: k[1], reverse=True)
        #     print ('opList',opList)
        #     Tree2 = SearchTree(cards=cards, suits=suits, leftNum=self.leftNum, all=allSamexts,
        #                       remainNum=self.remainNum, dgtable=[1] * 34,type=1)
        #     # Tree.getCardScore()
        #     # print (left_num[convert_hex2index(22)])
        #     # print (left_num[])
        #     scoreDict = Tree2.getCardScore()
        #     scoreDict = sorted(scoreDict.items(), key=lambda k: k[1], reverse=True)
        #     # if scoreDict!=[]:
        #     if scoreDict!=[]:
        #         print('op scoreDict',scoreDict)
        #         max_score = scoreDict[0][1]
        #     else:
        #         max_score=0
        #     # sumScore = 0
        #     # for k in scoreDict.keys():
        #     #     sumScore += scoreDict[k]
        #     print('opList',opList)
        #     print ('xts<3',opList[0][1],max_score)
        #     if opList[0][1]>max_score:
        #         return opList[0][0]
        #     else:
        #         return []


        # 计算前向胡牌概率 完全局部最优策略
        path_w_former = self.cost(all=all_former, suits=suits, left_num=left_num)
        path_w_former.sort(key=lambda k: (k[0]), reverse=True)
        print ("path_w_former", path_w_former)
        rate_former = path_w_former[0][0]  # 未执行动作的胡牌概率

        cards_add_op = copy.copy(cards)
        cards_add_op.append(op_card)
        all_later = self.sys_info_V3(cards=cards_add_op, suits=suits, left_num=left_num)
        val = []  # 记录满足条件的碰杠组合
        # if canchi:  # 四川麻将只能碰
        #     for a in all_later:
        #         t3N = a[0] + a[1]
        #         if op_card not in a[-1] and ([op_card - 2, op_card - 1, op_card] in t3N \
        #                                              or [op_card - 1, op_card, op_card + 1] in t3N \
        #                                              or [op_card, op_card + 1, op_card + 2] in t3N \
        #                                              or [op_card, op_card, op_card] in t3N):
        #             val.append(a)
        # else:  # 只能碰
        for a in all_later:
            if op_card not in a[-1] and [op_card, op_card, op_card] in a[0]:
                val.append(a)
        # print ("val", val)

        if val != []:
            path_w_later = self.cost(all=val, suits=suits, left_num=left_num)
            # print ("path_w_later", path_w_later)
            # index记录有效的吃碰杠组合索引
            index = []
            for i_p in range(len(path_w_later)):
                # if path_w_later[i_p][0] == 1 and xts_min == 1:  # 已胡牌,由于上饶麻将没有点炮胡，这里考虑下有效牌数量
                #     efc_cards = []
                #     max_remove_3N = 0  # 打掉一张３Ｎ的牌后，最大有效牌数量
                #     # aa+ab or aa+aa
                #     for a in all_former:
                #         if len(a[2]) == 1 and len(a[3]) == 1:
                #             efc_cards.extend(self.get_effective_cards(dz_set=a[3]))
                #         elif len(a[2]) == 2 and len(a[3]) == 0:
                #             efc_cards.extend(self.get_effective_cards(dz_set=a[2]))
                #         for t3 in a[0] + a[1]:
                #             # left_2N=copy.deepcopy(t3)
                #             # left_2N.remove(t3[0])
                #             # right_2N
                #             lc = self.get_effective_cards(dz_set=[[t3[1], t3[2]]])
                #             ln = sum([left_num[translate16_33(e)] for e in lc])
                #             # for card in lc:
                #             if ln > max_remove_3N:
                #                 max_remove_3N = ln
                #             rc = self.get_effective_cards(dz_set=[[t3[0], t3[1]]])
                #             rn = sum([left_num[translate16_33(e)] for e in rc])
                #             if rn > max_remove_3N:
                #                 max_remove_3N = rn
                #
                #     efc_num = 0  # 胡牌的有效牌数量
                #     efc_cards = set(efc_cards)
                #     for card in efc_cards:
                #         efc_num += left_num[translate16_33(card)]
                #     # print ("efc_num,max_remove_3N",efc_num,max_remove_3N)
                #     if max_remove_3N < 1.5 * efc_num:  # 如果有效牌数量翻倍，则执行此操作
                #         return []

                if path_w_later[i_p][0] > rate_former:
                    index.append([i_p, path_w_later[i_p][0]])
            index.sort(key=lambda k: k[1], reverse=True)
            # print ("index", index)
            if index != []:
                for t3 in val[index[0][0]][0] + val[index[0][0]][1]:  # 在最优吃碰杠组合中给出该３Ｎ,修正点，从all_later修正为ｖａｌ
                    print ("op_ t3", t3)
                    if op_card in t3 and t3[0] == t3[1]:
                        return False,t3
        return False,[]



class QiDui:
    def __init__(self,cards, suits=[], leftNum=[], discards=[], discards_real=[], discardsOp=[], round=0, remainNum=136,
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
        cards=self.cards
        suits=self.suits
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
        if len(duipai)>=3:
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
            effective_cards_num[i] = left_num[MJ.translate16_33(left_cards[i])]
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
        cards=self.cards
        suits=self.suits
        left_num=self.leftNum
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


def paixing_choose(hand_cards=[], suits=[], discards=[], discards_op=[], op_card=None,choose_color=[-1, -1, -1, -1],hu_cards=[[], [], [], []],hu_fan=[[],[],[],[]]):
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
    pinghu_info = pinghu(cards=hand_cards, suits=suits, leftNum=left_num, discards=[], discards_real=[], discardsOp=discards_op, round=0, remainNum=134,
                 fengWei=0, seat_id=0, choose_color=choose_color, hu_cards=hu_cards,
                 hu_fan=hu_fan).sys_info_V3(cards=cards_op, suits=suits, left_num=left_num)

    qidui_info = QiDui(cards=hand_cards, suits=suits, leftNum=left_num, discards=[], discards_real=[], discardsOp=discards_op, round=0, remainNum=134,
                 fengWei=0, seat_id=0, choose_color=choose_color, hu_cards=hu_cards,
                 hu_fan=hu_fan).qidui_info()
    print ("[pinghu_info[0], qidui_info[0]=",
           [pinghu_info[0][4], qidui_info["xts"]])

    min, index = MJ.get_min(
        list=[pinghu_info[0][4], qidui_info["xts"] + 1])
    return index


# 推荐出牌总接口
def recommend_card(cards=[], suits=[], round=0, remain_num=136, discards=[], discards_real=[], discards_op=[],
                   seat_id=0, choose_color=[], hu_cards=[[], [], [], []], hu_fan=[[], [], [], []]):
    # return cards[-1]
    start = time.time()
    left_num, discards_list = MJ.trandfer_discards(discards=discards, discards_op=discards_op, handcards=cards, type=27)
    paixing = paixing_choose(hand_cards=cards, suits=suits, discards=discards, discards_op=discards_op, op_card=None,choose_color=choose_color,hu_cards=hu_cards,hu_fan=hu_fan)
    if paixing == 0:
        # print 'choose_color',choose_color
        outCard = pinghu(cards, suits, leftNum=left_num, discards=discards, discards_real=discards_real,
                     discardsOp=discards_op, remainNum=remain_num, round=round, fengWei=0, seat_id=seat_id,
                     choose_color=choose_color, hu_cards=hu_cards, hu_fan=hu_fan).recommend_card(cards=cards,
                        suits=suits,left_num=left_num)
    elif paixing == 1:
        outCard = QiDui(cards, suits=suits, leftNum=left_num, discards=discards, discards_real=discards_real, discardsOp=discards_op, round=round, remainNum=remain_num,fengWei=0, seat_id=seat_id, choose_color=choose_color, hu_cards=hu_cards,
                 hu_fan=hu_fan).recommend_card()
    end = time.time()
    print('use time=', end - start)
    return outCard


# 推荐动作总接口
def recommend_op(op_card, cards=[], suits=[], round=0, remain_num=136, discards=[], discards_real=[], discards_op=[], self_turn=False, seat_id=0, choose_color=[], hu_cards=[[], [], [], []], hu_fan=[[], [], [], []], isHu = False):
    # return False,[]
    if isHu==True:
        return True, []
    left_num, discards_list = MJ.trandfer_discards(discards=discards, discards_op=discards_op, handcards=cards, type=27)
    paixing = paixing_choose(hand_cards=cards, suits=suits, discards=discards, discards_op=discards_op, op_card=None,choose_color=choose_color,hu_cards=hu_cards,hu_fan=hu_fan)
    if paixing==0:
        return pinghu(cards, suits, leftNum=left_num, discards=discards, discards_real=discards_real,
                     discardsOp=discards_op, remainNum=remain_num, round=round, fengWei=0, seat_id=seat_id,
                     choose_color=choose_color, hu_cards=hu_cards, hu_fan=hu_fan).recommend_op(op_card=op_card,self_turn=self_turn,isHu=isHu)
    elif paixing==1:
        # return QiDui().recommend_op()
        return False, [] #七对不考虑
    else:
        print ('recomend_op Error!')

def recommend_switch_cards(hand_cards=[],switch_n_cards=3):
    switch_cards=SwitchTiles(hand=hand_cards,n=switch_n_cards).switch_cards()
    return switch_cards

def recommend_choose_color(hand_cards=[],switch_n_cards=3):
    choose_color=SwitchTiles(hand=hand_cards,n=switch_n_cards).choose_color()
    return choose_color


