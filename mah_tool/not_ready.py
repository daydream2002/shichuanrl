#  this is 2.34 改 查大叫模块

import copy
from mah_tool.so_lib import lib_MJ as MJ  # 使用的一些库函数
import itertools
from mah_tool.tool import list10_to_16, fulu_translate

max_score = 1

ph_xts = -1  # 记录平胡向听数
qd_xts = -1  # 记录七对向听数

HANDCARD = []

t1tot3_dict = MJ.t1tot3_info()  # t1转化为t3
t2tot3_dict = MJ.t2tot3_info()  # t2转化为t3


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
                    taking_set_w.append(1)  # 要更改。。
                    T1 = copy.copy(jiangs)
                    T1.remove(t1)
                    child = Node_PH(take=t1, AAA=node.AAA, ABC=node.ABC, jiang=[t1, t1], T3=node.T3, T2=node.T2,
                                    T1=T1,
                                    taking_set=taking_set, taking_set_w=taking_set_w, true_card=true_card)
                    # node.add_child(child=child)
                    self.expand_node(node=child)

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
                        taking_set_w.append(1)
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
                    print("tn Error")
                    # logger.error("tn Error")
            # 当待扩展集合为空时
            else:
                # 这里判断是不是第一次 第二次进来（如果是，当场进行全扩展（使用raw） todo 是否先t2后t1
                # if node.f_t3 == True:
                #     # raw 全部 先t2后t1
                #     t1t2_sets = copy.copy(node.T3)
                #     t1t2_sets.extend(copy.copy(node.T2))
                #     for t1 in node.T1:
                #         t1t2_sets.append(t1)
                #     if -1 in t1t2_sets:
                #         t1t2_sets.remove(-1)
                #
                #     T2 = copy.copy(node.T2)
                #     T1 = copy.copy(node.T1)
                #     T3 = copy.copy(node.T3)
                #
                #     # todo 这里可以改成 for in t12s
                #     for t12_set in itertools.combinations(t1t2_sets, min(1, len(t1t2_sets))):
                #         node.T1 = copy.copy(T1)
                #         node.T2 = copy.copy(T2)
                #         node.T3 = copy.copy(T3)
                #         node.raw = list(t12_set)
                #         node.f_t3 = False
                #         for t12 in node.raw:
                #             if type(t12) == int:
                #                 node.T1.remove(t12)
                #             elif type(t12) == list:
                #                 if len(t12) == 2:
                #                     node.T2.remove(t12)
                #                 else:
                #                     node.T3.remove(t12)
                #         self.expand_node(node=node)
                #     # pass
                # elif node.s_t3 == True:
                #     # raw 全部 先t2后t1
                #     t1t2_sets = copy.copy(node.T3)
                #     t1t2_sets.extend(copy.copy(node.T2))
                #     for t1 in node.T1:
                #         t1t2_sets.append(t1)
                #     if -1 in t1t2_sets:
                #         t1t2_sets.remove(-1)
                #
                #     T2 = copy.copy(node.T2)
                #     T1 = copy.copy(node.T1)
                #     T3 = copy.copy(node.T3)
                #
                #     # todo 这里可以改成 for in t12s
                #     for t12_set in itertools.combinations(t1t2_sets, min(1, len(t1t2_sets))):
                #         node.T1 = copy.copy(T1)
                #         node.T2 = copy.copy(T2)
                #         node.T3 = copy.copy(T3)
                #         node.raw = list(t12_set)
                #         node.s_t3 = False
                #         for t12 in node.raw:
                #             if type(t12) == int:
                #                 node.T1.remove(t12)
                #             elif type(t12) == list:
                #                 if len(t12) == 2:
                #                     node.T2.remove(t12)
                #                 else:
                #                     node.T3.remove(t12)
                #         self.expand_node(node=node)
                #     # pass
                #
                # else:  # 正常扩展
                if True:
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

            self.expand_node(node=root)  # 扩展树

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
            fan *= 2

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
        fan *= 2 ** gen

        return int(fan)

    def calculate_path_expectation(self, node):
        if len(node.AAA) + len(node.ABC) == 4 and node.jiang != []:

            if len(node.taking_set) > ph_xts:
                # print("error")
                return

            score = self.cal_score(node=node)  # 这里的分数就是fan

            global max_score
            if score > max_score:
                max_score = score


        elif node.children != []:
            for child in node.children:
                self.calculate_path_expectation(node=child)

    def get_discard_score(self):
        self.generate_tree()

        for root in self.node_hu:  # 扩展树的过程集合
            self.calculate_path_expectation(root)  # 会更新 状态集： state[弃牌]=[[缺失牌集],[缺失牌分数集]]


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

        CS[-1] -= len(CS[0]) * 2 + (7 - len(CS[0]))  # 向听数= 7-已有对子数量
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
                # value = 1
                # for card in taking_set_sorted:
                #     if card == -1:  # -1代表 填充牌
                #         value = 1.0 / 34
                #     else:
                #         value *= T_SELFMO[MJ.convert_hex2index(card)]
                fan = self.fan(node=node)  # --=

                # score = value * fan  # 概率*番  （这里的概率分母都是一样的，分子就是当前剩余牌数，所以是等价的。。可以
                global max_score
                if fan > max_score:
                    max_score = fan
                # discards = node.T1 + self.padding
                # for discard in discards:
                #     if discard not in self.discard_state.keys():
                #         self.discard_state[discard] = [[], []]
                #         self.discard_state[discard][0].append(taking_set_sorted)
                #         self.discard_state[discard][-1].append(score)
                #     elif taking_set_sorted not in self.discard_state[discard][0]:
                #         self.discard_state[discard][0].append(taking_set_sorted)
                #         self.discard_state[discard][-1].append(score)  # 这里精髓，可以让当前弃牌产生的价值，全部汇聚于这里
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
        # for discard in self.discard_state.keys():
        #     if discard not in self.discard_score:
        #         self.discard_score[discard] = 0
        #     self.discard_score[discard] = sum(self.discard_state[discard][-1])
        # return self.discard_score


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

        # 258牌未写

        return fan


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

    def pinghu_CS2(self, cards=[], suits=[], t1=[]):

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

        return all


def get_score_dict(cards, suits, king_card=None, fei_king=0, padding=[], max_xts=14, get_xts=False):
    # 寻找向听数在阈值内的牌型
    PH = PingHu(cards=cards, suits=suits, kingCard=king_card, fei_king=fei_king, padding=padding)
    QD = Qidui(cards=cards, suits=suits, king_card=king_card, fei_king=fei_king, padding=padding)

    # 组合信息
    CS_PH = PH.pinghu_CS2()  # 平胡的组合信息，最后只保留了向听数 最少的，[刻子，顺子，搭子，向听数，剩余牌]
    CS_QD = QD.qidui_CS()  # 七对的组合信息，[对子，剩余牌，向听数]

    # 向听数
    xts_list = [CS_PH[0][-2], CS_QD[-1]]
    global ph_xts, qd_xts, max_score
    ph_xts = CS_PH[0][-2]
    qd_xts = CS_QD[-1]
    min_xts = min(xts_list)
    if min_xts > 1:
        max_score = -1
        return

    search_PH = SearchTree_PH(hand=cards, suits=suits, combination_sets=CS_PH)
    search_PH.get_discard_score()
    QD.get_discard_score()


def get_cadajiao_score(cards=[], suits=[]):
    global HANDCARD
    HANDCARD = cards

    get_score_dict(cards, suits)

    # print("max score", max_score)
    return max_score


if __name__ == "__main__":
    get_cadajiao_score(list10_to_16([4, 4, 4, 8]), fulu_translate([[5, 5, 5, 5], [2, 2, 2], [3, 3, 3]]))
