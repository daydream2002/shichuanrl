# ！/usr/bin/python3.8
# -*- coding: utf-8 -*-
# @Time     : 2022/2/28 14:18
# @Author   : Zou
# @Email    : 1091274580@qq.com
# @File     : isHu.py
# @Software : PyCharm

'''
1、是否胡牌
2、胡牌牌型：平胡、七对
'''

import copy
from  mah_tool import tool

def is_1l_list(l):
    for i in l:
        if type(i) == list:
            return False
    return True


def deepcopy(src):
    dst = []
    for i in src:
        if type(i) == list and not is_1l_list(i):
            i = deepcopy(i)
        dst.append(copy.copy(i))
    return dst

def cal_xts(all=[], suits=[]):
    """
     功能：计算组合的向听数
    思路：初始向听数为14，减去相应已成型的组合（kz,sz为３，aa/ab为２），当２Ｎ过剩时，只减去还需要的２Ｎ，对２Ｎ不足时，对还缺少的３Ｎ减去１，表示从孤张牌中选择一张作为３Ｎ的待选
    :param all: [[]]组合信息
    :param suits: 副露
    :return: all　计算向听数后的组合信息
    """
    for i in range(len(all)):
        t3N = all[i][0] + all[i][1]
        all[i][4] = 14 - (len(t3N) + len(suits)) * 3
        # 有将牌
        has_aa = False
        if len(all[i][2]) > 0:
            has_aa = True
        if has_aa:  # has do 当２Ｎ与３Ｎ数量小于4时，存在没有减去相应待填数，即废牌也会有１张作为２Ｎ或３Ｎ的待选位,
            if len(suits) + len(t3N) + len(all[i][2]) + len(all[i][3]) - 1 >= 4:
                all[i][4] -= (4 - (len(suits) + len(t3N))) * 2 + 2
            else:
                all[i][4] -= (len(all[i][2]) + len(all[i][3]) - 1) * 2 + 2 + 4 - (
                        len(suits) + len(t3N) + len(all[i][2]) + len(all[i][3]) - 1)
        # 无将牌
        else:
            if len(suits) + len(t3N) + len(all[i][2]) + len(all[i][3]) >= 4:
                all[i][4] -= (4 - (len(suits) + len(t3N))) * 2 + 1
            else:
                all[i][4] -= (len(all[i][2]) + len(all[i][3])) * 2 + 1 + 4 - (
                        len(suits) + len(t3N) + len(all[i][2]) + len(all[i][3]))
    all.sort(key=lambda k: (k[4], len(k[-1])))
    return all

class PingHu:
    def __init__(self, cards, suits):
        """
        类变量初始化
        :param cards: 手牌　
        :param suits:副露
        :return  平胡：[[kz], [sz], [aa], [t2], xts, [t1]]
        """
        cards.sort()
        self.cards = cards
        self.suits = suits

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
        for card in cards:
            if card & 0xF0 == 0x00:
                cards_wan.append(card)
            elif card & 0xF0 == 0x10:
                cards_tiao.append(card)
            elif card & 0xF0 == 0x20:
                cards_tong.append(card)
        return cards_wan, cards_tiao, cards_tong

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
        if len(cards) >= 8:
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
        while i < length_t32_set:
            t = t32_set[i]
            flag = True  # 本次划分是否合理
            if t != 0:
                if len(t) == 3:
                    if t[0] == t[1]:
                        kz.append(t)
                    else:
                        sz.append(t)
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
        allDeWeight = sorted(allDeWeight, key=lambda k: (len(k[0]), len(k[1]), len(k[2])), reverse=True)
        return allDeWeight

    def pinghu_CS(self, cards=[], suits=[]):
        """
        功能：综合计算手牌的组合信息
        思路：对手牌进行花色分离后，单独计算出每种花色的组合信息，再将其综合起来，计算每个组合向听数，最后输出最小向听数及其加一的组合
        :param cards: 手牌
        :param suits: 副露
        :param t1: 剩余牌
        :return: 组合信息 [[kz], [sz], [aa], [ab/ac], xts, [t1]]
        """
        if cards==[]:
            cards = copy.copy(self.cards)
            suits = deepcopy(self.suits)

        # 花色分离
        wan, tiao, tong = self.split_type_s(cards=cards)
        wan_expd = self.tree_expand(cards=wan)
        tiao_expd = self.tree_expand(cards=tiao)
        tong_expd = self.tree_expand(cards=tong)

        all = []
        for i in wan_expd:
            for j in tiao_expd:
                for k in tong_expd:
                        branch = []
                        # 将每种花色的4个字段合并成一个字段
                        for n in range(6):
                            branch.append(i[n] + j[n] + k[n])
                        all.append(branch)

        # 计算向听数
        # for i in range(len(all)):
        #     print(all[i])
        all = cal_xts(all, suits)

        # 获取向听数最小的all分支
        min_index = 0
        for i in range(len(all)):
            if all[i][4] > all[0][4]:  # xts+1以下的组合
                min_index = i
                break

        if min_index == 0:  # 如果全部都匹配，则min_index没有被赋值，将min_index赋予all长度
            min_index = len(all)

        all = all[:min_index]
        return all


class Mahjong:
    def __init__(self, hand_cards=None, suits=None, dingque=None):
        """
        麻将类，初始化部分信息
        :param hand_cards: 玩家手牌
        :param suits: 玩家副露
        :param dingque: 定缺花色：{0：万， 1：条， 2：筒}
        # 传入十进制数据
        """
        self.hand_cards = tool.list10_to_16(hand_cards)
        self.suits = tool.fulu_translate(suits)
        self.dingque = dingque

    def qiduiCS(self):
        CS = [[], [], 14]
        if self.suits:
            return CS
        for card in list(set(self.hand_cards)):
            n = self.hand_cards.count(card)
            if n == 1:
                CS[1].append(card)
            elif n == 2:
                CS[0].append([card, card])
            elif n == 3:
                CS[0].append([card, card])
                CS[1].append(card)
            elif n == 4:
                CS[0].append([card, card])
                CS[0].append([card, card])
        CS[-1] -= len(CS[0]) * 2 + (7 - len(CS[0]))
        return CS

    def isHu(self):
        dic = {0: 0x00, 1: 0x10, 2: 0x20}
        for card in self.hand_cards:
            if card & 0xF0 == dic[self.dingque]:
                return False, 14
        PH = PingHu(cards=self.hand_cards, suits=self.suits)
        PH_CS = PH.pinghu_CS()
        QD_CS = self.qiduiCS()
        if PH_CS[0][-2] == 0:
            return True, 0
        elif QD_CS[-1] == 0:
            return True, 1
        else:
            xts = min(PH_CS[0][-2], QD_CS[-1])
            return False, xts
        



