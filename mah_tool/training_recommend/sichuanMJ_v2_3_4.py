#  this is 2.34（v2.33优化唯一标识路径的set，和true_card+taking的分数集）
#  test1000=0.088
# set集合标识原：taking   新：true_card + taking    =搜索到的节点，如预期增加
# 分数集原：taking-1 新：true_card + taking-1      =存储的分数有小幅变化

# 220804 增加 将牌双需丢弃（可以去掉一些分支，很少） 不会影响最终性能

import copy
import time
# from interface.sichuanMJ_v1 import lib_MJ as MJ  # 使用的一些库函数
from mah_tool.training_recommend import lib_MJ as MJ  # 使用的一些库函数
import logging
# import opp_srmj as DFM  # 对手建模
import datetime
import itertools

# 需要调参的位置： line 40  418  443
# 碰牌和自摸牌的比重，番型倍数，有效牌路径计算

TIME_START = time.time()
w_type = 0  # lib_MJ的权重选择
ROUND = 0  # 轮数
# t3Set = MJ.get_t3info()
# t2Set, t2Efc, efc_t2index = MJ.get_t2info()
REMAIN_NUM = 108  # 剩余牌数
LEZZ = 160
const_sum = 0
max_mul = 0

my_mul = 6  # 番型放大倍数，用来鼓励走大番      //此时只影响，下面3中番型
qinyise = 0  # 清一色 大约3-6% 的额外番加成    ->10
duanyao = 0  # 断幺九 大约16%  的额外番加成    ->24
penpen = 0  # 碰碰胡  大约3%  的额外番加成     ->6.6

# 这下面的暂时不参与*运算，只有各自的+运算
daigen = 0  # 带根加成++
qidui = 0  # 七对 很少，不鼓励
yaojiu = 0  # 幺九 很少，不鼓励
jingougou = 0  # 金钩钩 很少，而且会破坏听多牌的牌型++

KING = None  # 宝牌
fei_king = 0  # 飞宝数

xts_num = 1  # xts的扩展范围
back_num = 1  # 最后几手进行 全扩展
ph_xts = -1  # 记录平胡向听数
qd_xts = -1  # 记录七对向听数

max_fan1 = 1  # 不进行全扩展的最大番（xts==阶段）
max_fan2 = 1  # 不进行全扩展的最大番（xts+1==阶段）
limit_fan2 = 0
limit_fan3 = 0

w_bb = 1  # 组成对子的权重，只有自摸所以=1

T_SELFMO = [0] * 34  # 自摸概率表，牌存在于牌墙中的概率表
LEFT_NUM = [0] * 34  # 未出现的牌的数量表
HANDCARD = []
RT1 = [[0] * 34, [0] * 34]  # 其他玩家的状态表[table1,table2] ，不需要的牌table1与需要的牌table2，
RT2 = [[0] * 34, [0] * 34]  # table1:计算吃碰的概率，table2：计算危险度
RT3 = [[0] * 34, [0] * 34]

# 生成t1,t2转化为t3的状态集合，便于搜索直接使用
# t1tot2_dict = MJ.t1tot2_info()
t1tot3_dict = {}  # MJ.t1tot3_info()  # t1转化为t3
t2tot3_dict = {}  # MJ.t2tot3_info()  # t2转化为t3


class SwitchTiles:
    def __init__(self, hand, n=3):
        """

        :param hand: 手牌
        :param n: 换牌张数，默认为换3张
        """
        self.hand = hand
        self.type = n
        self.color = MJ.splitColor(hand)

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
    #         pr int('SwitchTiles, choose_color ERROR! len(self.hand)=13 or 14, but =' + len(self.hand))
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
                # pri nt(i,cs,value_cs)
                one_max = max(one_max, value_cs)
            # pri nt(one_max)
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
                # pri nt(i,cs,value_cs)
                one_max = max(one_max, value_cs)
            # pr int(one_max)
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
            one_max_o = 0
            cards_other_cs = MJ.tree_expand_gang(cards_other)
            for cs in cards_other_cs:
                value_cs = MJ.judge_cs_value(cs) + 10 * len(cards_other)
                if value_cs >= one_max_o:
                    one_max_o = value_cs
                    max_cs_o = cs

            gap_tmp = one_max_o - one_max3
            if gap_tmp > gap:
                gap = gap_tmp
                choose_3cards = cards_3
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
    def __init__(self, take=None, AAA=[], ABC=[], jiang=[], T3=[], T2=[], T1=[], raw=[], taking_set=[], taking_set_w=[],
                 f_t3=True, s_t3=False, true_card=[]):
        self.take = take  # 缺的牌 1张
        self.AAA = AAA  # 刻子
        self.ABC = ABC  # 顺子
        self.jiang = jiang  # 将牌
        self.T3 = T3
        self.T2 = T2  # T2组合
        self.T1 = T1  # T1组合
        self.raw = raw  # 待扩展集合
        self.taking_set = taking_set  # 缺失牌
        self.taking_set_w = taking_set_w  # 数量，补齐类型？？权重
        self.score = 0
        self.f_t3 = f_t3
        self.s_t3 = s_t3
        self.true_card = true_card

    def node_info(self):
        print(self.AAA, self.ABC, self.jiang, "T1=", self.T1, "T2=", self.T2, self.raw, self.taking_set)


class SearchTree_PH():
    """
    平胡搜索模块
    """

    def __init__(self, hand, suits, combination_sets):
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
        self.node_hu = []
        self.taking_sets = set()

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
        # 先定将
        if node.jiang == []:  # 没有将牌
            #  对子必选，
            #  孤张-t3t2数量<5 or t2中无对子，
            #  搭子-t3t2数量>=5且无对子

            # ("扩展对子将牌---")
            # has_aa = False
            for t2 in node.T2:  # T2组合有将牌
                T2 = MJ.deepcopy(node.T2)
                # 从t2中找到对子作为将牌
                if t2[0] == t2[1]:
                    # has_aa = True
                    true_card = copy.copy(node.true_card)
                    true_card.extend(t2)
                    T2.remove(t2)
                    child = Node_PH(take=-1, AAA=node.AAA, ABC=node.ABC, jiang=t2, T3=node.T3, T2=T2, T1=node.T1,
                                    taking_set=node.taking_set, taking_set_w=node.taking_set_w,
                                    true_card=true_card)  # 非宝吊宝还原
                    # node.add_child(child=child)
                    self.expand_node(node=child)

            # ("扩展孤张将牌---")
            # if has_aa == False or len(node.ABC) + len(node.AAA) + len(node.T2) < 5:
            if True:
                jiangs = copy.copy(node.T1)  # 复制T1，尝试用T1扩展成将牌
                for t1 in jiangs:
                    if t1 == -1:  # op填充的-1不作扩展
                        continue
                    if HANDCARD.count(t1) > 1:  # 本来可以以对子的身份作为将牌，可惜你不做
                        pass
                    true_card = copy.copy(node.true_card)
                    true_card.append(t1)
                    taking_set = copy.copy(node.taking_set)
                    taking_set.append(t1)
                    taking_set_w = copy.copy(node.taking_set_w)
                    taking_set_w.append(w_bb)  # 要更改。。
                    T1 = copy.copy(jiangs)
                    T1.remove(t1)
                    child = Node_PH(take=t1, AAA=node.AAA, ABC=node.ABC, jiang=[t1, t1], T3=node.T3, T2=node.T2,
                                    T1=T1,
                                    taking_set=taking_set, taking_set_w=taking_set_w, true_card=true_card)
                    # node.add_child(child=child)
                    self.expand_node(node=child)

            # ("扩展搭子将牌---")    缺少的t3数量<现有的T2数量
            # if has_aa == False and len(node.ABC) + len(node.AAA) + len(node.T2) >= 5:
            if True:
                for t2 in node.T2:
                    if t2[0] == t2[-1]:
                        continue
                    jiangs = t2
                    T2 = MJ.deepcopy(node.T2)
                    T2.remove(t2)
                    for t1 in jiangs:  # jiangs总共2张，依次挑一张
                        if HANDCARD.count(t1) > 1:  # 本来可以以对子的身份作为将牌，可惜你不做
                            continue
                        true_card = copy.copy(node.true_card)
                        true_card.append(t1)
                        taking_set = copy.copy(node.taking_set)
                        taking_set.append(t1)
                        taking_set_w = copy.copy(node.taking_set_w)
                        taking_set_w.append(w_bb)
                        T1 = copy.copy(jiangs)
                        T1.remove(t1)  # 搭子剩余的那张牌，退回成T1
                        child = Node_PH(take=t1, AAA=node.AAA, ABC=node.ABC, jiang=[t1, t1], T3=node.T3, T2=T2, T1=T1,
                                        taking_set=taking_set, taking_set_w=taking_set_w, true_card=true_card)
                        # node.add_child(child=child)
                        self.expand_node(node=child)


        # 胡牌判断，此时有将牌
        elif len(node.AAA) + len(node.ABC) == 4:
            self.node_hu.append(Node_PH(take=None, AAA=node.AAA, ABC=node.ABC, jiang=node.jiang, T3=node.T3, T2=node.T2,
                                        T1=node.T1,
                                        taking_set=node.taking_set, taking_set_w=node.taking_set_w,
                                        true_card=node.true_card))
            return

        # T3扩展，此时有将牌
        else:
            # 第1次第2次，进来，必定使用全扩展

            # 当待扩展集合不为空时，使用该集合进行扩展
            if node.raw != []:
                tn = node.raw[-1]  # 取最后一个待扩展 todo 这里为什么最后一个 应该其实都一样
                raw = copy.copy(node.raw)  # 深度搜索后面的节点会改变raw，回退可能导致前面的节点raw不正确，这里需要copy
                raw.pop()
                if type(tn) == list:  # 使用t2扩展t3
                    if len(tn) == 2:
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

                            true_card = copy.copy(node.true_card)
                            true_card.extend(t2)
                            taking_set = copy.copy(node.taking_set)
                            taking_set_w = copy.copy(node.taking_set_w)
                            taking_set.append(item[-2])  # 补齐牌
                            taking_set_w.append(item[-1])  # 补齐类型 aa2 ab6
                            child = Node_PH(take=item[-2], AAA=AAA, ABC=ABC, jiang=node.jiang, T3=node.T3, T2=node.T2,
                                            T1=node.T1, raw=raw, taking_set=taking_set, taking_set_w=taking_set_w,
                                            f_t3=node.f_t3, s_t3=node.s_t3, true_card=true_card)
                            # node.add_child(child=child)
                            self.expand_node(node=child)
                    else:
                        t3 = tn
                        if t3[0] == t3[1]:
                            AAA = MJ.deepcopy(node.AAA)
                            AAA.append(t3)
                            ABC = node.ABC
                        else:
                            AAA = node.AAA
                            ABC = MJ.deepcopy(node.ABC)
                            ABC.append(t3)
                        true_card = copy.copy(node.true_card)
                        true_card.extend(t3)
                        child = Node_PH(take=None, AAA=AAA, ABC=ABC, jiang=node.jiang, T3=node.T3, T2=node.T2,
                                        T1=node.T1, raw=raw, taking_set=node.taking_set, taking_set_w=node.taking_set_w,
                                        f_t3=node.f_t3, s_t3=node.s_t3, true_card=true_card)
                        # node.add_child(child=child)
                        self.expand_node(node=child)
                # t1扩展为t3
                elif type(tn) == int:  # 单张牌，使用t1 扩展 t3
                    t1 = tn
                    for item in t1tot3_dict[str(t1)]:  # 同理 item:info:[[扩展成的t3],[缺失的2张牌],[补齐类型1，补齐类型2]]

                        if item[0][0] == item[0][1]:  # 扩展成刻子
                            AAA = MJ.deepcopy(node.AAA)
                            AAA.append(item[0])
                            ABC = node.ABC
                        else:  # 扩展成顺子
                            AAA = node.AAA
                            ABC = MJ.deepcopy(node.ABC)
                            ABC.append(item[0])

                        if True:
                            take = item[1]  # take = 缺失的牌   [1,1]
                            take_w = item[-1]  # 补齐类型

                            true_card = copy.copy(node.true_card)
                            true_card.append(t1)
                            taking_set = copy.copy(node.taking_set)
                            taking_set.extend(take)
                            taking_set_w = copy.copy(node.taking_set_w)
                            taking_set_w.extend(take_w)
                            child = Node_PH(take=take, AAA=AAA, ABC=ABC, jiang=node.jiang, T3=node.T3, T2=node.T2,
                                            T1=node.T1,
                                            raw=raw,
                                            taking_set=taking_set, taking_set_w=taking_set_w, f_t3=node.f_t3,
                                            s_t3=node.s_t3, true_card=true_card)
                            # node.add_child(child=child)
                            self.expand_node(node=child)

                else:  # 错误！！待扩展集合为list int 都if了
                    print("tn Error:519", tn)
                    # logger.error("tn Error")
            # 当待扩展集合为空时
            else:
                # 这里判断是不是第一次 第二次进来（如果是，当场进行全扩展（使用raw） todo 是否先t2后t1
                if node.f_t3 == True:
                    # raw 全部 先t2后t1
                    t1t2_sets = copy.copy(node.T3)
                    t1t2_sets.extend(copy.copy(node.T2))
                    for t1 in node.T1:
                        t1t2_sets.append(t1)
                    if -1 in t1t2_sets:
                        t1t2_sets.remove(-1)

                    T2 = copy.copy(node.T2)
                    T1 = copy.copy(node.T1)
                    T3 = copy.copy(node.T3)

                    # todo 这里可以改成 for in t12s
                    for t12_set in itertools.combinations(t1t2_sets, min(1, len(t1t2_sets))):
                        node.T1 = copy.copy(T1)
                        node.T2 = copy.copy(T2)
                        node.T3 = copy.copy(T3)
                        node.raw = list(t12_set)
                        node.f_t3 = False
                        for t12 in node.raw:
                            if type(t12) == int:
                                node.T1.remove(t12)
                            elif type(t12) == list:
                                if len(t12) == 2:
                                    node.T2.remove(t12)
                                else:
                                    node.T3.remove(t12)
                        self.expand_node(node=node)
                    # pass
                elif node.s_t3 == True:
                    # raw 全部 先t2后t1
                    t1t2_sets = copy.copy(node.T3)
                    t1t2_sets.extend(copy.copy(node.T2))
                    for t1 in node.T1:
                        t1t2_sets.append(t1)
                    if -1 in t1t2_sets:
                        t1t2_sets.remove(-1)

                    T2 = copy.copy(node.T2)
                    T1 = copy.copy(node.T1)
                    T3 = copy.copy(node.T3)

                    # todo 这里可以改成 for in t12s
                    for t12_set in itertools.combinations(t1t2_sets, min(1, len(t1t2_sets))):
                        node.T1 = copy.copy(T1)
                        node.T2 = copy.copy(T2)
                        node.T3 = copy.copy(T3)
                        node.raw = list(t12_set)
                        node.s_t3 = False
                        for t12 in node.raw:
                            if type(t12) == int:
                                node.T1.remove(t12)
                            elif type(t12) == list:
                                if len(t12) == 2:
                                    node.T2.remove(t12)
                                else:
                                    node.T3.remove(t12)
                        self.expand_node(node=node)
                    # pass

                else:  # 正常扩展
                    if node.T3 != [] or node.T2 != []:  # 1、先扩展T2为T3
                        t23_sets = node.T3 + node.T2
                        T2 = copy.copy(node.T2)
                        T3 = copy.copy(node.T3)
                        # 生成待扩展集合                       从t2_sets中选min(差几个T3，当前T2数量)个出来 的全组合
                        for t23_set in itertools.combinations(t23_sets,
                                                              min(4 - len(node.AAA) - len(node.ABC), len(t23_sets))):
                            node.T2 = copy.copy(T2)
                            node.T3 = copy.copy(T3)
                            node.raw = list(t23_set)  # 待扩展加入t2_set
                            for t23 in node.raw:
                                if len(t23) == 2:
                                    node.T2.remove(t23)  # T2移除t2
                                else:
                                    node.T3.remove(t23)
                            self.expand_node(node=node)


                    elif node.T1 != []:  # 生成T1扩展T3集合
                        t1_sets = copy.copy(node.T1)
                        if -1 in t1_sets:
                            t1_sets.remove(-1)
                        T1 = copy.copy(node.T1)
                        for t1_set in itertools.combinations(t1_sets,
                                                             min(4 - len(node.AAA) - len(node.ABC), len(t1_sets))):
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
        true_card = []
        # 将副露加入到节点的AAA和ABC状态中
        # todo 这里 true_card.append(t3)
        for t3 in self.suits:
            true_card.extend(t3)
            if t3[0] == t3[1]:
                kz.append(t3)
            else:
                sz.append(t3)
        # 使用拆分组合生成树
        for cs in self.combination_sets:
            root = Node_PH(take=None, AAA=kz, ABC=sz, jiang=[], T3=cs[0] + cs[1], T2=cs[2] + cs[3], T1=cs[-1],
                           taking_set=[], taking_set_w=[], true_card=true_card)  # 简单的传参，初始化类
            # 每一棵树都存储到树集合中
            # self.tree_dict.append(root)
            self.expand_node(node=root)  # 扩展树

    def cal_chance_2(self, node):

        value = 1.0
        taking_set = copy.deepcopy(node.taking_set)
        taking_setw = copy.deepcopy(node.taking_set_w)

        if taking_setw != []:
            for i in range(len(taking_set)):  # 遍历缺失牌
                card = taking_set[i]  # 取一张缺失牌
                taking_rate = T_SELFMO[MJ.convert_hex2index(card)]  # 自摸概率表，开始的时候，多走一步
                value *= taking_rate * taking_setw[i]  # todo 需要结合其他玩家打出这张牌的概率来计算，将获取权重具体化

        # 摸牌概率修正，当一张牌被重复获取时，T_selfmo修改为当前数量占未出现牌数量的比例
        taking_set2 = list(set(taking_set))  # 缺失牌集合(无重复)
        taking_set_num = [taking_set.count(i) for i in taking_set2]  # 缺失牌数量
        for i in range(len(taking_set_num)):
            n = taking_set_num[i]
            j = 0
            while n > 1:
                j += 1
                index = MJ.convert_hex2index(taking_set2[i])  # 返回 0-33 序数
                if LEFT_NUM[index] >= n:
                    # value * 3/4 * 2/4  eg:需要3张 value=4/136 * 4/136 * 4/136 * 3/4 * 2/4 修正概率为 4/136 3/136 2/136
                    value *= float(LEFT_NUM[index] - j) / LEFT_NUM[index]
                else:  # 摸牌数超过了剩余数，直接舍弃；；需要缺失牌的数量<可能获取的数量
                    value = 0.0
                    return value
                n -= 1

        return value

    def cal_score(self, node):
        """
        节点评估值计算模块-分数子模块
        :param node:
        :return: float 分数
        """

        # fan计算
        fan = Fan_PH(kz=node.AAA, sz=node.ABC, jiang=node.jiang).fanDetect()

        # 金钓钓 2
        if len(self.suits) == 4:
            fan *= 2 + jingougou

        cards_num = {}
        cards_list = []
        for aaa in node.AAA:
            for a in aaa:
                cards_list.append(a)
        for abc in node.ABC:
            for a in abc:
                cards_list.append(a)
        for j in node.jiang:
            cards_list.append(j)
        for card in cards_list:
            if card not in cards_num.keys():
                cards_num[card] = 1
            else:
                cards_num[card] += 1
        gen = 0
        for key in cards_num.keys():
            if cards_num[key] >= 4:
                gen += 1
        fan *= (2 + daigen) ** gen

        return float(fan)

    def calculate_path_expectation(self, node):
        """
        计算整条路径的上的评估值，并将其赋予为所有出牌的评估值
        :param node:
        :return:
        """
        # 深度搜索。搜索胡牌的叶子节点 todo 这里不变，依然搜索到胡牌节点，然后进行反选 n-1 + 1
        if len(node.AAA) + len(node.ABC) == 4 and node.jiang != []:

            if len(node.taking_set) > ph_xts + back_num + 1:
                return

            if node.taking_set.count(node.jiang[0]) >= 2:
                # print("触发将双需")
                # print(node.taking_set)
                # print(node.jiang)
                return

            # 790 1440
            taking_set_sorted = sorted(node.taking_set)  # 缺失的牌;有效牌
            true_card_sorted = sorted(node.true_card)  # 使用的手牌
            taking_set_str = str(taking_set_sorted)
            true_card_str = str(true_card_sorted)

            # if len(taking_set_sorted) + len(true_card_sorted) != 14:
            #     print(true_card_sorted)
            #     print(taking_set_sorted)
            #     print(true_card_str + taking_set_str)

            if true_card_str + taking_set_str in self.taking_sets:  # todo 这个一样就是一样吗？以后try T3+aa 相同
                return
            else:
                self.taking_sets.add(true_card_str + taking_set_str)  # 这里list改用set，可以直接节约一半的时间

            discard_set = []  # 出牌集合

            # 将没有使用的T2加入到出牌中
            for t3 in node.T3:
                discard_set.extend(t3)
            for t2 in node.T2:
                discard_set.extend(t2)
            discard_set.extend(node.T1)  # 多余的牌

            if discard_set != []:
                score = self.cal_score(node=node)  # 这里的分数就是fan
            else:
                return

            # if len(node.taking_set) - ph_xts > 1:
            # print(len(node.taking_set) - ph_xts)
            global max_fan1, max_fan2, limit_fan2, limit_fan3
            # todo 这里用新的
            # (xts==) 1的最大值迭代工作;重置2 3 的限制
            if len(node.taking_set) == ph_xts:
                if limit_fan2 != 0 or limit_fan3 != 0:
                    max_fan1 = 1
                    max_fan2 = 1
                    limit_fan2 = 0
                    limit_fan3 = 0
                max_fan1 = max(max_fan1, score)

            # (xts+1) 2的最大值迭代工作;记录max1
            elif len(node.taking_set) == ph_xts + 1:
                # 第一次进来全扩展位置
                if limit_fan2 == 0:
                    limit_fan2 = max_fan1
                    max_fan2 = max_fan1
                    limit_fan3 = 0
                max_fan2 = max(max_fan2, score)
                if score <= limit_fan2:
                    return

            # xts+2)  记录max2；
            elif len(node.taking_set) == ph_xts + 2:
                if limit_fan3 == 0:
                    limit_fan3 = max_fan2
                    limit_fan2 = 0
                if score <= max_fan2:  # +2采用严格大于?
                    return

            chance_all = self.cal_chance_2(node=node)
            if chance_all == 0:
                return

            self.node_num += 1  # 可胡牌节点  +1     40000次...上面截断之后->16000

            taking_set_ = list(set(node.taking_set))
            for i in range(len(taking_set_)):
                # 这个是路径概率+可胡张数，少一长度的
                # chance, hu_num = self.cal_chance(node=node, hu_card=taking_set_[i])
                # chance = float(chance)
                card_ = taking_set_[i]
                left_num = LEFT_NUM[MJ.convert_hex2index(card_)]
                need_num = node.taking_set.count(card_)
                hu_num = left_num - need_num + 1
                chance = chance_all / T_SELFMO[MJ.convert_hex2index(card_)]
                if node.taking_set.count(taking_set_[i]) > 1:
                    chance /= (left_num - need_num + 1) / left_num
                for j in range(len(node.taking_set)):
                    if node.taking_set[j] == card_ and node.taking_set_w[j] != 1:
                        chance /= node.taking_set_w[j]
                        break

                taking_set_sorted2 = copy.deepcopy(taking_set_sorted)
                taking_set_sorted2.remove(taking_set_[i])
                taking_set_lable = str(taking_set_sorted2)
                taking_set_lable += true_card_str

                for card in list(set(discard_set)):
                    if card not in self.discard_state.keys():  # 弃牌不在 状态里
                        self.discard_state[card] = {}
                    if taking_set_lable not in self.discard_state[card].keys():  # 有效牌集不在 状态[弃牌]里
                        self.discard_state[card][taking_set_lable] = [[], [], [], [], []]
                    if taking_set_[i] not in self.discard_state[card][taking_set_lable][0]:
                        self.discard_state[card][taking_set_lable][0].append(taking_set_[i])  # 加入有效牌
                        self.discard_state[card][taking_set_lable][1].append(chance)  # 加入路径概率
                        self.discard_state[card][taking_set_lable][2].append(hu_num)  # 加入可胡张数
                        self.discard_state[card][taking_set_lable][3].append(score)  # 加入胡一次分数
                        self.discard_state[card][taking_set_lable][-1].append(chance * score * hu_num)
                    else:
                        index = self.discard_state[card][taking_set_lable][0].index(taking_set_[i])
                        if chance * score * hu_num > self.discard_state[card][taking_set_lable][-1][index]:
                            self.discard_state[card][taking_set_lable][1][index] = chance  # 加入路径概率
                            self.discard_state[card][taking_set_lable][2][index] = hu_num  # 加入可胡张数
                            self.discard_state[card][taking_set_lable][3][index] = score  # 加入胡一次分数
                            self.discard_state[card][taking_set_lable][-1][index] = chance * score * hu_num

        # 当前节点不能胡牌，继续搜索其 子节点
        elif node.children != []:
            for child in node.children:
                self.calculate_path_expectation(node=child)

    def get_discard_score(self):
        """
        总接口。获取出牌的评估值
        :return: dict. 出牌的评估值集合
        """
        self.generate_tree()  # 生成树，将组合好的手牌，用生成树的方式计算所有扩展组合
        # print("生成树耗时", time.time() - TIME_START)

        for root in self.node_hu:  # 扩展树的过程集合
            self.calculate_path_expectation(root)  # 会更新 状态集： state[弃牌]=[[缺失牌集],[缺失牌分数集]]
        # print("nodenum:", self.node_num)  # 45627,9433,7232；    4290,3130
        # print("记录分数耗时耗时**", time.time() - TIME_START)

        # todo 这里的
        # discard_state={}
        # discard_state[弃牌]={}
        # discard_state[弃牌][有效牌集]=[[可胡牌],[可胡牌全胡的分数]]
        # 修正上面
        # discard_state[弃牌][有效牌集]=[[可胡牌],[路径概率],[可胡张数],[胡一次分数],[可胡牌全胡的分数]]

        for discard in self.discard_state.keys():
            if discard not in self.discard_score.keys():
                self.discard_score[discard] = 0
            all_score = 0
            max_score = 0
            for take in self.discard_state[discard]:
                all_score += sum(self.discard_state[discard][take][-1])
                max_score = max(max_score, sum(self.discard_state[discard][take][-1]))
            self.discard_score[discard] = all_score + max_score * max_mul

        # recommend_card = max(self.discard_score, key=lambda x: self.discard_score[x])
        # p rint("当前弃牌：", recommend_card)
        # for take in self.discard_state[recommend_card]:
        #     p rint("\n对应的有效牌：", take)
        #     p rint("对应的可胡牌(*)：", self.discard_state[recommend_card][take][0])
        #     p rint("对应的路径概率：", self.discard_state[recommend_card][take][1])
        #     p rint("对应的可胡张数：", self.discard_state[recommend_card][take][2])
        #     p rint("对应的胡一次分数：", self.discard_state[recommend_card][take][3])
        #     p rint("对应的总可能得分：", self.discard_state[recommend_card][take][-1])
        #
        # for discard in self.discard_state.keys():
        #     p rint("\n\n弃牌：", discard)
        #     for take in self.discard_state[discard]:
        #         p rint("\n对应的有效牌：", take)
        #         p rint("对应的可胡牌(*)：", self.discard_state[discard][take][0])
        #         p rint("对应的路径概率：", self.discard_state[discard][take][1])
        #         p rint("对应的可胡张数：", self.discard_state[discard][take][2])
        #         p rint("对应的胡一次分数：", self.discard_state[discard][take][3])
        #         p rint("对应的总可能得分：", self.discard_state[discard][take][-1])
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
        node = Node_Qidui(take=None, AA=CS[0], T1=CS[1], taking_set=[], king_num=self.king_num)
        self.tree_list.append(node)
        self.expand_node(node=node)

    def fan(self, node):
        """
        七对番型
        :param node:
        """

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
            fan *= 4 + yaojiu
        if flag_duanyijiu:
            fan *= 2 * my_mul + duanyao
        if flag_qinyise:
            fan *= 4 * my_mul + qinyise
        fan *= (2 + daigen) ** longqidui

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
                    if card == -1:  # -1代表 填充牌
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
            self.evaluate(tree)
        # t3=time.time()
        for discard in self.discard_state.keys():
            if discard not in self.discard_score:
                self.discard_score[discard] = 0
            self.discard_score[discard] = sum(self.discard_state[discard][-1])
        return self.discard_score


'''
番数计算类
'''


class Fan_PH():
    def __init__(self, kz, sz, jiang):
        """
        初始化类变量
        :param kz: 刻子
        :param sz: 顺子
        :param jiang: 将
        :param node: 待检测的结点
        """
        self.kz = kz
        self.sz = sz
        self.jiang = jiang

    # 碰碰胡
    def pengPengHu(self):
        return len(self.kz) == 4

    # 清一色 x2
    def qingYiSe(self):
        w = 0
        ti = 0
        to = 0
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

        for T in self.kz + self.sz + [self.jiang]:
            flag_yijiu = False
            for t in T:
                if t in yijiu:
                    flag_yijiu = True
            if flag_yijiu == False:
                return False
        return True

    # 断一九 2
    def duanYiJiu(self):
        yijiu = [1, 9, 0x11, 0x19, 0x21, 0x29]  # 一九牌 4   断一九 2

        for T in self.kz + self.sz + [self.jiang]:
            for t in T:
                if t in yijiu:
                    return False

        return True

    def fanDetect(self):
        """
        番数计算
        基础分４分，通过调用上述的番种检测来增加基础分
        :return: int 番数
        """
        fan = 1  # 平胡基础分 1
        # 碰碰胡，断一九，一九牌， 金钓钓（需要副露信息，在外面判定）， 清一色  5种
        # mul=1

        # 碰碰胡 2
        if self.pengPengHu():
            fan *= 2 * my_mul + penpen

        # 清一色 4
        if self.qingYiSe():
            fan *= 4 * my_mul + qinyise

        # 一九牌 4
        if self.yiJiuPai():
            fan *= 4 + yaojiu

        # 断一九 2
        if self.duanYiJiu():
            fan *= 2 * my_mul + duanyao

        # 258牌未写

        return fan


'''
平胡类，相关处理方法
分为手牌拆分模块sys_info，评估cost,出牌决策，吃碰杠决策等部分
'''


class PingHu:
    '''
    平胡类模块
    '''

    def __init__(self, cards, suits, kingCard=None, fei_king=0, padding=[]):
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
        effective_cards = set(effective_cards)
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
        # if True:
        if len(cards) >= 0:
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
        # print("t32N",t32N)
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
                        sz.append(t)  # prin t (sub)     -> 顺子

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

    # def pinghu_CS(self, cards=[], suits=[], t1=[]):
    #     if cards == []:
    #         cards = self.cards
    #         suits = self.suits
    #     cards.sort()
    #     # 加入一些特殊的处理 例如碰碰胡的CS生成
    #     CS = self.pinghu_CS2(cards, suits, t1)
    #     kingNum = 0
    #     RM_King = copy.copy(cards)
    #     if self.kingCard != None:
    #         kingNum = cards.count(self.kingCard)
    #         for i in range(kingNum):
    #             RM_King.remove(self.kingCard)
    #     pph = MJ.pengpengHu(outKingCards=RM_King, suits=suits, kingNum=kingNum)
    #     if pph[0] not in CS and pph[0][-2] <= CS[0][-2] + 1:
    #         CS += pph
    #     return CS

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

        # 花色分离
        wan, tiao, tong = self.split_type_s(RM_King)
        wan_expd = self.tree_expand(cards=wan)  # 返回的是 List[List[list or int]]：[kz,sz,aa,ab,xts,leftCards]: [[1,2,3],0]
        tiao_expd = self.tree_expand(cards=tiao)  # [  [[kz],[sz],[aa],[ab],xts,[leftCards]] , [同左边]  ]
        tong_expd = self.tree_expand(cards=tong)  # 这三个一样，都是平胡分支的 小分支

        all = []
        for i in wan_expd:  # 进行组合搭配
            for j in tiao_expd:
                for k in tong_expd:
                    branch = []
                    # 将每种花色的4个字段合并成一个字段
                    for n in range(6):
                        branch.append(i[n] + j[n] + k[n])

                    branch[-1] += self.padding + t1  # 2个都为空
                    all.append(branch)

        # 将获取概率为０的组合直接丢弃到废牌中 todo 由于有宝，这里也可能会被宝代替
        # 移到了出牌决策部分处理
        for a in all:  # 所有牌组合的一种 a: [kz + sz + aa + t2N + 0 + a]
            for i in range(len(a[3]) - 1, -1, -1):  # for(int i=len-1;i>=0;i--)
                ab = a[3][i]  # 找出 连续搭和中间搭    因为对子可以做将牌，可以不能成刻子
                efc = self.get_effective_cards([ab])  # 得到有效牌,有效牌是16进制，要转成0-33的连续数
                if sum([LEFT_NUM[MJ.convert_hex2index(e)] for e in efc]) <= 0:  # 如果剩余牌为0
                    a[3].remove(ab)  # 将t2N移出 T2
                    a[-1].extend(ab)  # 放入 T1

        # 计算向听数
        # 计算拆分组合的向听数
        all = MJ.cal_xts(all, suits)

        # 获取向听数最小的all分支
        min_index = 0
        for i in range(len(all)):  # all 中向听数已经按照 小->大 排序
            if all[i][4] > all[0][4]:  # 找到向听数变大的 index，舍弃后面的组合;
                min_index = i
                break

        if min_index == 0:  # 如果全部都匹配，则min_index没有被赋值，将min_index赋予ａｌｌ长度
            min_index = len(all)
        all = all[:min_index]  # 只保留 0-min_index 的组合

        # 处理向听数为0时的情况，需要从中依次选择一张牌作为t1. 可胡牌时，选择碰，xts=0
        if all[0][-2] == 0 and all[0][-1] == []:  # 向听数==0  and  孤张没有
            all = []  # 重新算过all组合
            for card in list(set(cards)):
                cards_ = copy.copy(cards)
                cards_.remove(card)
                all += self.pinghu_CS2(cards=cards_, suits=suits, t1=[card])  # 去掉一张牌的all

        # print("组合数量", len(all))
        # for a in all:
        #     print(a)
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
            # to do if card == fengwei:
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
            d_w = aa + left_num[i - 2] * left_num[i - 1] * 2 + left_num[i - 1] * left_num[i + 1] * 2 + left_num[i + 1] * \
                  left_num[
                      i + 2] * 2
        # if card<=0x31:
        #     if (card%0x0f==3 or card %0x0f==7): #给金3银7倍数
        #         d_w*=1.5
        #     elif card%0x0f==5:
        #         d_w*=1.2
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

    left_num = [4] * 34
    discards_list = [0] * 34
    for per in discards:  # 减去弃牌集-----------
        for item in per:
            discards_list[discards_map[item]] += 1
            left_num[discards_map[item]] -= 1
    for seat_op in discards_op:  # 减去副露集-----------
        for op in seat_op:
            for item in op:
                discards_list[discards_map[item]] += 1
                left_num[discards_map[item]] -= 1
    for item in handcards:  # 减去自己的手牌-------------
        left_num[discards_map[item]] -= 1

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


def get_score_dict(cards, suits, king_card=None, fei_king=0, padding=[], max_xts=14, get_xts=False):
    """
    功能：计算弃牌分数表
    思路：（待
    :param cards: 手牌
    :param suits: 副露
    :param king_card:无效
    :param fei_king: 无效
    :param padding: 填充牌
    :param max_xts: 最大向听数
    :param round: 轮次
    :param get_xts: 是否只计算向听数的flag
    :return: score_dict,min_xts ，各出牌的评估值与本轮计算的最小向听数（用于op中对比操作前后时）
    """

    # 寻找向听数在阈值内的牌型
    PH = PingHu(cards=cards, suits=suits, kingCard=king_card, fei_king=fei_king, padding=padding)
    QD = Qidui(cards=cards, suits=suits, king_card=king_card, fei_king=fei_king, padding=padding)

    # 组合信息
    CS_PH = PH.pinghu_CS2()  # 平胡的组合信息，最后只保留了向听数 最少的，[刻子，顺子，搭子，向听数，剩余牌]
    CS_QD = QD.qidui_CS()  # 七对的组合信息，[对子，剩余牌，向听数]

    # 向听数
    xts_list = [CS_PH[0][-2], CS_QD[-1]]
    global ph_xts, qd_xts
    ph_xts = CS_PH[0][-2]
    qd_xts = CS_QD[-1]
    # print("xts_list PH,QD", xts_list)
    min_xts = min(xts_list)

    if get_xts:
        return min_xts

    # op中吃碰后向听数增加的情况，特别是打非平胡的牌型
    # 这里的max_xts只有在动作判断才会传值，此时的 max值=动作前的最小xts  min值=动作后的最小xts
    if min_xts > max_xts + xts_num:  # 这里的1就是，碰杠允许的xts变化，虽然xts变大时 评估值必然变小。
        return {cards[-1]: 0}, min_xts  # 研究研究

    type_list = []  # 需搜索的牌型
    for i in range(2):
        if xts_list[i] <= min_xts + xts_num:  # 比最少向听数大1 的组合也考虑一下，因为按概率给路径，xts大的一般就只有影响作用，不能决定
            type_list.append(i)  # 比如平胡向听数=1，九幺向听数=2 .考虑到九幺容易胡，所以九幺也去计算

    score_list = []
    for i in type_list:  # 0123分别代表 4种胡牌类型。
        if i == 0:
            search_PH = SearchTree_PH(hand=cards, suits=suits, combination_sets=CS_PH)
            # print("扩展之前耗时", time.time() - TIME_START)
            score_list.append(search_PH.get_discard_score())  # 返回dict: 包含每一张弃牌的评估值
        if i == 1:
            score_list.append(QD.get_discard_score())  # 返回dict: 包含每一张弃牌的评估值==now

    # 计算总的评估值，有所有选中的牌型的评估值之和
    score_dict = {}
    # print("score_list", score_list)  # score_list:dict: 包含每一张弃牌的评估值
    # for i in score_list:
    #     print(i)
    for score in score_list:  # 取第一个分数集，所有牌都会有一个分数 （在上文，如果多种牌型向听数接近，就会叠加考虑）
        for key in score.keys():  # key ==牌ID  如果key未在dict中，存入一个震荡值；如果已经在，追加权值（叠加考虑？？）
            if key not in score_dict.keys():
                score_dict[key] = score[key] - float(value_t1(key)) / (10 ** (min_xts + 1) / 2)  # 用来区分相同权重的出牌，同价值浮动
            else:
                score_dict[key] += score[key]

    return score_dict, min_xts


def recommend_switch_cards(hand_cards=[], switch_n_cards=3):
    switch_cards = SwitchTiles(hand=hand_cards, n=switch_n_cards).choose_3card()
    return switch_cards


def recommend_choose_color(hand_cards=[], switch_n_cards=3):
    choose_color = SwitchTiles(hand=hand_cards, n=switch_n_cards).choose_color_final()
    return choose_color


def recommend_card(cards=[], suits=[], king_card=None, discards=[], discards_op=[], fei_king=0, remain_num=65,
                   round=0, seat_id=0, self_lack=0):
    """
    功能：推荐出牌接口
    思路：使用向听数作为牌型选择依据，对最小ｘｔｓ的牌型，再调用相应的牌型类出牌决策
    :param cards: 自己手牌
    :param suits: 自己副露
    :param king_card: 无效参数（待移除，历史遗留问题
    :param discards: 场上弃牌
    :param discards_op: 场上副露
    :param fei_king: 无效参数（待移除，历史遗留问题
    :param remain_num: 牌墙剩余牌数量
    :param round: 轮次
    :param seat_id: 自己座位id
    :param self_lack: 定缺牌花色
    :return: outCard 推荐出牌
    """

    # 更新全局变量
    global T_SELFMO, LEFT_NUM, TIME_START, ROUND, REMAIN_NUM, t2tot3_dict, t1tot3_dict, HANDCARD
    HANDCARD = cards
    # ROUND = round  # 轮数
    # MJ.KING = king_card  # 宝牌ID
    TIME_START = time.time()

    # 根据定缺数量，重新计算t1to todo
    t1tot3_dict = MJ.t1tot3_info()  # t1转化为t3
    t2tot3_dict = MJ.t2tot3_info()  # t2转化为t3

    # 计算获取概率
    LEFT_NUM, _ = trandfer_discards(discards=discards, discards_op=discards_op, handcards=cards)  # 利用场面所有已知信息，求出剩余牌
    REMAIN_NUM = max(1, min(LEZZ - sum(LEFT_NUM), 108))
    if const_sum != 0:
        REMAIN_NUM = const_sum
    T_SELFMO = [float(i) / REMAIN_NUM for i in LEFT_NUM]  # 求出每一张牌的概率  分子：这张牌的剩余数量 分母：剩余牌总数

    # 处理有定缺牌情况
    list_lack = []
    for i in range(len(cards)):
        if int(cards[i] // 16) == self_lack:
            list_lack.append(cards[i])
    # 先按照剩余牌数量，多->少 ；然后同剩余牌，按照19 28 的顺序
    if len(list_lack) > 0:
        remain_n = 0
        for c in list_lack:
            remain_n = max(remain_n, LEFT_NUM[translate16_33(c)])
        list_lack2 = []
        for c in list_lack:
            if LEFT_NUM[translate16_33(c)] == remain_n:
                list_lack2.append(c)
        list_lack2.sort(key=lambda x: abs(x % 16 - 5), reverse=True)  # 优先返回边张1 9，最后 5
        return list_lack2[0]

    # 计算所有可能出牌的评估值
    score_dict, _ = get_score_dict(cards, suits, king_card, fei_king)  # score_dict: 每种牌的评估值-----------now
    if score_dict != {}:
        recommend_card = max(score_dict, key=lambda x: score_dict[x])  # 输出score_dict里最大值的keys（用values比较）
    else:
        recommend_card = cards[-1]
    # print("score_dict", score_dict)
    if time.time() - TIME_START > 10:
        print(time.time() - TIME_START, "timeout  \"user_cards\": {\"hand_cards\": ", cards, ",\"operate_cards\": ",
              suits, "},\"discards\":", discards, ",\"discards_real\":[[],[],[],[]],\"discards_op\":",
              discards_op, "color", self_lack)

    return recommend_card


def recommend_op(op_card, cards=[], suits=[], discards=[], discards_op=[],
                 self_turn=True, isHu=False, round=0, remain_num=108, op_map=[]):
    """
    功能：动作决策接口
    思路：有胡必胡，有杠大概率杠，碰牌1.05阈值（待定：先判断是否胡牌；然后判断碰牌；最后杠牌。
    :param op_card: 别人回合的弃牌/自己回合的摸牌（摸牌也即是手牌的最后一张）
    :param cards: 自己手牌
    :param suits: 自己副露
    :param discards: 场上弃牌
    :param discards_op: 场上副露
    :param self_turn: 是否自己回合
    :param isHu: 是否到达可胡牌
    :param round: 轮次
    :param remain_num:场上剩余牌数量
    :param op_map: 可操作动作 0: '不操作', 4: '碰', 5: '明杠', 6: '暗杠', 7: '补杠'，8：‘胡’
    :return: [],isHu 动作组合牌，是否胡牌
    """
    # 胡牌
    if 8 in op_map:
        return [], True

    global T_SELFMO, LEFT_NUM, t2tot3_dict, t1tot3_dict, TIME_START, REMAIN_NUM, HANDCARD
    HANDCARD = cards

    # 根据定缺数量，重新计算t1to todo
    t1tot3_dict = MJ.t1tot3_info()  # t1转化为t3
    t2tot3_dict = MJ.t2tot3_info()  # t2转化为t3

    LEFT_NUM, discards_list = trandfer_discards(discards=discards, discards_op=discards_op, handcards=cards)
    min_xts_pre = get_score_dict(cards=cards, suits=suits, get_xts=True)
    # 补杠，副露+1张手牌（自己回合）
    if 7 in op_map:
        for suit in suits:
            if suit[0] == suit[1]:
                if suit[0] in cards:
                    # left不变
                    cards_ = copy.copy(cards)
                    cards_.remove(suit[0])
                    suits_ = MJ.deepcopy(suits)
                    suits_.remove(suit)
                    suits_.append([suit[0], suit[0], suit[0], suit[0]])
                    min_xts = get_score_dict(cards=cards_, suits=suits_, get_xts=True)
                    if min_xts - min_xts_pre <= 0:
                        return suit + [suit[0]], False
                    elif min_xts - min_xts_pre == 1:
                        return suit + [suit[0]], False
                    else:
                        pass
                    # return suit + [suit[0]]
        # return op_map[7] * 4, False
    # 暗杠，4张手牌（自己回合）
    if 6 in op_map:
        for card in cards:
            if cards.count(card) >= 4:
                # left不变
                cards_ = copy.copy(cards)
                cards_.remove(card)
                cards_.remove(card)
                cards_.remove(card)
                cards_.remove(card)
                suits_ = MJ.deepcopy(suits)
                suits_.append([card, card, card, card])
                min_xts = get_score_dict(cards=cards_, suits=suits_, get_xts=True)
                if min_xts - min_xts_pre <= 0:
                    return [card, card, card, card], False
                elif min_xts - min_xts_pre == 1:
                    return [card, card, card, card], False
                else:
                    pass
                # return [card, card, card, card]
        # return op_map[6] * 4, False
    # 明杠，3张手牌+别人出一张（同时存在碰牌）（别人回合）
    if 5 in op_map:
        if cards.count(op_card) >= 3:
            # 出牌已经再discard里面了，left不变
            cards_ = copy.copy(cards)
            cards_.remove(op_card)
            cards_.remove(op_card)
            cards_.remove(op_card)
            suits_ = MJ.deepcopy(suits)
            suits_.append([op_card, op_card, op_card, op_card])
            min_xts = get_score_dict(cards=cards_, suits=suits_, get_xts=True)
            if min_xts - min_xts_pre <= 0:
                return [op_card, op_card, op_card, op_card], False
            elif min_xts - min_xts_pre == 1:
                return [op_card, op_card, op_card, op_card], False
            else:
                pass
            # return [op_card, op_card, op_card, op_card]
        # return op_map[5] * 4, False

    # 碰牌，2手+别人 （别人回合）
    if 4 in op_map:
        # global T_SELFMO, LEFT_NUM, t2tot3_dict, t1tot3_dict, TIME_START, REMAIN_NUM
        LEFT_NUM, discards_list = trandfer_discards(discards=discards, discards_op=discards_op, handcards=cards)
        TIME_START = time.time()
        # REMAIN_NUM = sum(LEFT_NUM)
        REMAIN_NUM = max(1, min(LEZZ - sum(LEFT_NUM), 108))
        if const_sum != 0:
            REMAIN_NUM = const_sum
        T_SELFMO = [float(i) / REMAIN_NUM for i in LEFT_NUM]

        cards_pre = copy.copy(cards)
        score_dict_pre, min_xts_pre = get_score_dict(cards_pre, suits, padding=[-1])

        cards_ = copy.copy(cards)
        cards_.remove(op_card)  # 进行了吃碰，从手牌移除
        cards_.remove(op_card)
        suits_ = MJ.deepcopy(suits)  # 加入副露中
        suits_.append([op_card, op_card, op_card])
        score_dict, _ = get_score_dict(cards=cards_, suits=suits_, max_xts=min_xts_pre)
        if score_dict != {} and score_dict_pre != {}:
            score = max(score_dict.values())
            score_pre = max(score_dict_pre.values())
            if score > score_pre * 1.05:
                return [op_card, op_card, op_card], False
    return [], False


if __name__ == "__main__":
    import random

    walls = [1, 2, 3, 4, 5, 6, 7, 8, 9, 17, 18, 19, 20, 21, 22, 23, 24, 25, 33, 34, 35, 36, 37, 38, 39, 40, 41] * 4
    for i in range(1000):
        cards = random.sample(walls, 13)
        cards.sort()
        suits = []
        t32_set = []

        choose_3cards = recommend_switch_cards(cards)
        final_c = recommend_choose_color(cards)

        # pr int(cards)
        # pr int(choose_3cards)
        # pri nt(final_c)
