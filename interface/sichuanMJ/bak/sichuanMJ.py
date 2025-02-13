

import copy
# import numpy as np
import time

import numpy as np
import math

w_ways = 1
w_aa = (1 + 3 * w_ways + 4)
w_ab = (1 + w_ways)
w_type = 0

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
                        leftNum[convert_hex2index(catchCard)]) / remainNum * 1
            elif len(t2) == 2:
                if t2[0] == t2[1]:
                    self.rate = float(leftNum[convert_hex2index(catchCard)]) / remainNum * 8
                else:
                    # if
                    self.rate = float(leftNum[convert_hex2index(catchCard)]) / remainNum * 2
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

    def nodeInfo(self):
        print ['type', self.type, 'cards', self.cards, 'catchCard', self.catchCard, 'rate', self.rate, 't2', self.t2,
               'level', self.level, 'ocards', self.ocards, 't2N', self.t2N, 'kz', self.kz, 'sz', self.sz, 'jiang',
               self.jiang, 'formerCatchCards', self.formerCatchCards, 'formerOutCards', self.formerOutCards, 'kingCard',
               self.kingCard, 'baoHuanYuan', self.baoHuanYuan]


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
        self.rate = dgRate[convert_hex2index(outCard)]  # 危险概率
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

    def nodeInfo(self):
        """
        打印结点信息
        """
        print('type', self.type, 'cards', self.cards, 'outCard', self.outCard, 'rate', self.rate, 'level', self.level,
              'ocards', self.ocards, 't2N', self.t2N, 'kz', self.kz, 'sz', self.sz, 'jiang', self.jiang,
              'formerCatchCards', self.formerCatchCards, 'formerOutCards', self.formerOutCards, 'kingCard',
              self.kingCard, 'baoHuanYuan', self.baoHuanYuan)


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
        print('leftNum', leftNum)
        print  ('xts', all[0][4])
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
                w.append(float(left_num[translate16_33(dz[0])]) / cards_num)
            elif dz[1] == dz[0]:
                effective_cards.append(dz[0])
                w.append(float(
                    left_num[translate16_33(dz[0])]) / cards_num * 8.1)  # 修改缩进,发现致命错误panic 忘了写float,这里写６是因为评估函数计算的缺陷

            elif dz[1] == dz[0] + 1:
                if int(dz[0]) & 0x0F == 1:
                    effective_cards.append(dz[0] + 2)
                    w.append(float(left_num[translate16_33(dz[0] + 2)]) / cards_num * 2)
                elif int(dz[0]) & 0x0F == 8:
                    effective_cards.append((dz[0] - 1))
                    w.append(float(left_num[translate16_33(dz[0] - 1)]) / cards_num * 2)
                else:
                    effective_cards.append(dz[0] - 1)
                    effective_cards.append(dz[0] + 2)
                    w.append(float(left_num[translate16_33(int(dz[0]) - 1)] + left_num[
                        translate16_33(int(dz[0]) + 2)]) / cards_num * 2)
            elif dz[1] == dz[0] + 2:
                effective_cards.append(dz[0] + 1)
                w.append(float(left_num[translate16_33(int(dz[0]) + 1)]) / cards_num * 2)
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
            return

        # 宝吊多一层　
        if self.kingNum > 0 and node.feiKingNum + node.baoHuanYuan < self.kingNum + self.fei_king:

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

                    # 全遍历，将所有２Ｎ轮流加入到ocards中
                    for t2 in t2N:
                        t2NCP = copy.deepcopy(t2N)
                        t2NCP.remove(t2)
                        ocardsCP = copy.copy(ocards)
                        ocardsCP.extend(t2)
                        self.expandNode(node, ocardsCP, t2NCP, kingNum=kingNum, kz=kz, sz=sz, xts=xts)
                # 分支２　：kingCard加入到ocards中
                if kingNum != 0:
                    ocardsCPaddKing = [self.kingCard]
                    # print (ocardsCPaddKing)
                    self.expandNode(node, ocardsCPaddKing, t2N, kingNum=kingNum - 1, kz=kz, sz=sz, xts=xts)
                # 结束分支
                return

            else:
                ocardsTMP = copy.copy(ocards)
                # t2NCP = t2N

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

            for t2 in t2NCPTMP:
                # print ('t2NCPTMP',t2NCPTMP)
                t2NCP = copy.deepcopy(t2NCPTMP)
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
                            t2N_BHY = copy.deepcopy(t2N)
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

                        cNode.feiKingNum = node.feiKingNum
                        cNode.kz = copy.copy(node.kz)
                        # cNode.kz.extend(kz)
                        cNode.sz = copy.copy(node.sz)
                        # cNode.sz.extend(sz)
                        cNode.jiang = node.jiang

                        t2NCP2 = copy.deepcopy(t2NCP)
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
                        if len(cNode.kz) + len(cNode.sz) == 4:
                            if (len(t2NCP2) == 1 and t2NCP2[0][0] == t2NCP2[0][1]):  # 普通无宝胡牌，包括了宝吊（搜索时另一张牌也赋予了宝牌值）的情况

                                cNode.jiang = t2NCP2[0][0]

                            elif kingNum == 2:  # 宝还原　宝做将　胡牌
                                #     cNode.baoHuanYuan = True
                                cNode.jiang = self.kingCard

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

            print ('ocards', ocards, 't2NCP', t2NCP)

            self.expandNode(node=node, ocards=ocards, t2N=t2NCP, kingNum=self.kingNum, kz=kz, sz=sz, xts=a[4])

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

            if len(node.sz) + len(node.kz) == 4 and node.jiang != 0 and node.rate != 0 and node.type == 2:

                if node.type == 2 and len(node.t2) == 2:

                    node.rate = float(self.leftNum[convert_hex2index(node.catchCard)]) / self.remainNum
                    if node.t2[0] == node.t2[1]:
                        node.rate *= 1.5

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
                    fan = Fan(node.kz, node.sz, node.jiang, node, node.feiKingNum,
                              self.kingNum + self.fei_king).fanDetect()
                    if fan > self.maxScore[0]:
                        # self.maxScore[2]=self.maxScore[1]
                        self.maxScore[1] = self.maxScore[0]
                        self.maxScore[0] = fan

                    # print ('fan',fan)
                    # print (rate)
                    score = rate * fan
                    state.append(fan)

                    for card in outCards:

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

        print ('score', self.scoreDict)
        print ('stateSet', self.stateSet)
        print('nodeNum', nodeNum)
        print ('usetime', t2 - t1, t3 - t2)
        return self.scoreDict


'''
番数计算类
'''


class Fan():
    def __init__(self, kz, sz, jiang, suits, node=None, fei_king=0, kingNum=0):
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
        self.suits=suits

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

    def gen(self):
        """
        根：杠牌的数量，分数*2
        :return:
        """
        num_gen=0
        for suit in self.suits:
            if len(suit)==4:
                num_gen+=1
        return num_gen


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

    def __init__(self, cards, suits, leftNum=[], discards=[], discards_real=[], discardsOp=[], round=0, remainNum=134,
                 seat_id=0, kingCard=None, fei_king=0, op_card=0x00):
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
        self.remainNum = remainNum
        self.leftNum = leftNum
        self.round = round
        self.seat_id = seat_id
        self.fei_king = fei_king
        # print ('self.leftNum',self.leftNum)
        if self.leftNum == []:
            leftNum, discardsList = trandfer_discards(discards, discardsOp, cards)
            self.leftNum = leftNum

        # self.fengWei = fengWei
        self.kingCard = kingCard
        self.preKingCard = pre_king(kingCard)
        self.op_card = op_card
        if kingCard != None:
            self.kingNum = cards.count(kingCard)
        else:
            self.kingNum = 0  # print('kingNum111',leftNum[convert_hex2index(self.kingCard)],self.cards)

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
        cards_num = self.remainNum
        effective_cards = []
        w = []
        for dz in dz_set:
            if dz[1] == dz[0]:
                effective_cards.append(dz[0])
                w.append(float(
                    left_num[translate16_33(dz[0])]) / cards_num * w_aa)  # 修改缩进,发现致命错误panic 忘了写float,这里写６是因为评估函数计算的缺陷

            elif dz[1] == dz[0] + 1:
                if int(dz[0]) & 0x0F == 1:
                    effective_cards.append(dz[0] + 2)
                    w.append(float(left_num[translate16_33(dz[0] + 2)]) / cards_num * w_ab)
                elif int(dz[0]) & 0x0F == 8:
                    effective_cards.append((dz[0] - 1))
                    w.append(float(left_num[translate16_33(dz[0] - 1)]) / cards_num * w_ab)
                else:
                    effective_cards.append(dz[0] - 1)
                    effective_cards.append(dz[0] + 2)
                    w.append(float(left_num[translate16_33(dz[0] - 1)] + left_num[
                        translate16_33(dz[0] + 2)]) / cards_num * w_ab)
            elif dz[1] == dz[0] + 2:
                effective_cards.append(dz[0] + 1)
                w.append(float(left_num[translate16_33(int(dz[0]) + 1)]) / cards_num * w_ab)
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
        kz_suits = 0
        sz_suits = 0
        for suit in suits:
            if suit[0] == suit[1]:
                kz_suits += 0
            else:
                sz_suits += 0
        # 当副露中存在ｓｚ时，返回空
        if sz_suits != 0:
            return [all_PengPengHu]

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

    def sys_info_V3(self, cards, suits, left_num=[4] * 34, kingCard=None):
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
        if kingCard == None:
            kingCard = self.kingCard
        RM_King = copy.copy(cards)
        kingNum = 0
        if kingCard != None:
            kingNum = cards.count(kingCard)
            for i in range(kingNum):
                RM_King.remove(kingCard)

        # 特例，op操作计算胡牌概率时使用,在处理op_card是宝牌时，该宝牌只能作为宝还原使用
        if self.op_card == self.kingCard and len(cards) % 3 == 2:
            RM_King.append(self.kingCard)
            kingNum -= 1

        # 花色分离
        wan, tiao, tong, zi = self.split_type_s(RM_King)
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
                        all.append(branch)

        # 计算向听数
        # 计算拆分组合的向听数
        all = self.xts(all, suits, kingNum)

        # 获取向听数最小的all分支
        min_index = 0
        for i in range(len(all)):
            if all[i][4] > all[0][4] + 1:  # xts+１以下的组合
                min_index = i
                break

        if min_index == 0:  # 如果全部都匹配，则min_index没有被赋值，将min_index赋予ａｌｌ长度
            min_index = len(all)

        all = all[:min_index]
        print("all_terminal", all)
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
        i = convert_hex2index(card)
        # d_w = 0

        if left_num[i] == self.remainNum:
            sf = float(self.leftNum[i]) / remainNum * 6
        else:
            sf = float(left_num[i]) / remainNum * float((left_num[i] - 1)) / remainNum * 6
        if card >= 0x31:  # kz概率

            d_w = sf  # 7.22 １６:３５ 去除字牌
        elif card % 16 == 1:  # 11＋２3
            d_w = sf + float(left_num[i + 1]) / remainNum * float(left_num[i + 2]) / remainNum * 2
        elif card % 16 == 2:  # 22+13+3(14)+43   222 123 234
            d_w = sf + float(left_num[i - 1]) / remainNum * float(left_num[i + 1]) / remainNum * 2 + float(
                left_num[i + 1]) / remainNum * float(left_num[
                                                         i + 2]) / remainNum * 2
        elif card % 16 == 8:  # 888 678 789
            d_w = sf + float(left_num[i - 2]) / remainNum * float(left_num[i - 1]) / remainNum * 2 + float(
                left_num[i - 1]) / remainNum * float(left_num[
                                                         i + 1]) / remainNum * 2
        elif card % 16 == 9:  # 999 789
            d_w = sf + float(left_num[i - 2]) / remainNum * float(left_num[
                                                                      i - 1]) / remainNum * 2  # 删除多添加的×２
        else:  # 555 345 456 567
            # print (left_num)
            d_w = sf + float(left_num[i - 2]) / remainNum * float(left_num[i - 1]) / remainNum * 2 + float(
                left_num[i - 1]) / remainNum * float(left_num[i + 1]) / remainNum * 2 + float(
                left_num[i + 1]) / remainNum * float(left_num[
                                                         i + 2]) / remainNum * 2
        print("i=", i, d_w)
        return d_w

    def left_card_weight(self, card, left_num):
        """
        功能：对废牌组合中的每张废牌进行评估，计算其成为３Ｎ的概率
        思路：每张牌能成为３N的情况可以分为先成为搭子，在成为３Ｎ２步，成为搭子的牌必须自己摸到，而成为kz,sz可以通过吃碰。刻子为获取２张相同的牌，顺子为其邻近的２张牌

        :param card: 孤张牌
        :param left_num: 剩余牌
        :return: 评估值
        """
        i = convert_hex2index(card)
        if left_num[i] > 1:
            p_aa = 3
        else:
            p_aa = 1
        if card >= 0x31:  # kz概率

            d_w = left_num[i] * p_aa  # 7.22 １６:３５ 去除字牌
        elif card % 16 == 1:  # 11＋２3
            d_w = left_num[i] * p_aa + left_num[i + 1] + left_num[i + 2]
        elif card % 16 == 2:  # 22+13+3(14)+43   222 123 234
            d_w = left_num[i] * p_aa + left_num[i - 1] + left_num[i + 1] * 2 + left_num[i + 2]
        elif card % 16 == 8:  # 888 678 789
            d_w = left_num[i] * p_aa + left_num[i + 1] + left_num[i - 1] * 2 + left_num[i - 2]
        elif card % 16 == 9:  # 999 789
            d_w = left_num[i] * p_aa + left_num[i - 1] + left_num[i - 2]
        # 删除多添加的×２
        else:  # 555 345 456 567
            # print (left_num)
            d_w = left_num[i - 2] + left_num[i - 1] * 2 + left_num[i] * p_aa + left_num[i + 1] * 2 + left_num[i + 2]
        print("i=", i, d_w)
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
            t2NCopy = copy.deepcopy(t2N)
            t2NCopy.remove(t2)
            rate_ = copy.copy(rate)
            rate_ *= t2[2]
            if t2[0] != t2[1] or ab:
                ab_ = True
            else:
                ab_ = False

            self.choose_n(t2NCopy, n_, rate_, results, ab_, abSet)

    def calculate_path_w(self, a, king_num, feiKing=1):
        """
        一条拆分组合的胡牌概率评估
        分为有宝和无宝，无宝中又分为有将和无将情况进行计算
        :param a: 拆分组合
        :param king_num: 宝牌数量
        :param feiKing: 飞宝数
        :return: 胡牌概率和废牌
        """
        path_w = [feiKing, copy.deepcopy(a[-1])]
        t2N = copy.deepcopy(a[2] + a[3])
        efc_cards, t2_w = self.get_effective_cards_w(dz_set=t2N, left_num=self.leftNum)
        for i in range(len(t2N)):
            t2N[i].append(t2_w[i])
        bl = 4 - len(self.suits) - (len(a[0]) + len(a[1]))
        print ("cost t2N", t2N)
        results = []
        abSet = []
        if king_num == 0:  # 无宝
            # 对ａａ集合中选择一个作为将牌，在剩余的t2N中使用choose_n计算胡牌概率
            if a[2] != []:  # 定将
                t2N[:len(a[2])] = sorted(t2N[:len(a[2])], key=lambda k: k[2],
                                         reverse=True)  # 这里倒置会更好，如果aa的权重为０会导致整个评估为０
                t2N[len(a[2]):] = sorted(t2N[len(a[2]):], key=lambda k: k[2], reverse=True)
                # if len(a[2])-1 <= bl:
                #     self.jiang_rate(t2N[:len(a[2])])

                # has_ab = False
                print ('bl', bl)
                if bl <= len(t2N) - 1:  # t2N溢出,需要出一张２N
                    # aa_rate=self.jiang_rate(aa)
                    t2NCP = copy.deepcopy(t2N)
                    if a[-1] == [] and t2N != [] and a[4] != 0:  # 只添加最后的废牌: #只有当废牌区为空时，才将２Ｎ放入
                        path_w[1].append(t2N[-1][0])
                        path_w[1].append(t2N[-1][1])
                        t2NCP.remove(t2N[-1])
                    # yc=len(t2NCP)-1-bl

                    for aa in t2NCP[:len(a[2])]:
                        t2NCopy = copy.deepcopy(t2NCP)
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
                    print ("results", results)


                else:
                    for i in range(bl - len(t2N) + 1):
                        path_w[0] *= (80.0) / (self.remainNum * self.remainNum)
                    # rateSet=[]
                    for aa in t2N[:len(a[2])]:
                        t2NCopy = copy.deepcopy(t2N)
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
                    t2NCP = copy.deepcopy(t2N)
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
                w_jiang = [0] * len(left_cards)
                for k in range(len(left_cards)):
                    if translate16_33(left_cards[k]) == -1:
                        n = 3.0
                    else:
                        n = float(self.leftNum[translate16_33(left_cards[k])])
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

        else:
            feiKingHu = 0

            # 计算摸到其他２Ｎ有效牌的概率
            if a[2] != [] and a[4] == 1 and king_num == 1 and len(t2N) == 2:  # 向听数为1,一张宝牌，达到胡牌状态，但是一张宝没做宝吊，不能胡,xts=1:

                if len(a[2]) == 2:  # 计算摸到２张aa的概率
                    feiKingHu = float(t2N[0][2] + t2N[1][2]) / w_aa  # 飞宝胡牌概率,为可

                elif len(a[2]) == 1 and len(a[3]) == 1:  # 一张aa,一张ａｂ
                    feiKingHu = float(t2N[1][2]) / w_ab  # 飞宝胡牌概率
                    # if len(self.get_effective_cards(a[3])) == 1:  # 当胡牌是ac这种时，/2.5
                    #     feiKingHu /= 2

                # todo  aa过大会导致有ａａ就不飞宝
                t2N.sort(key=lambda k: k[2], reverse=True)
                if t2N[0][2] > t2N[1][2]:
                    max2Nindex = 0
                else:
                    max2Nindex = 1
                # if t2N[max2Nindex][0]==t2N[max2Nindex][1]:
                #     NoFeiKingHu=float(t2N[max2Nindex][2])/w_aa*6  #矫正了真实的ａａ权重
                # else:
                #     NoFeiKingHu=float(t2N[max2Nindex][2])/w_ab*2

                if feiKingHu * 5 < t2N[max2Nindex][2]:  # 不飞宝，下调权重，增加飞宝概率
                    # path_w[0] = t2N[max2Nindex][2]
                    path_w[0] = t2N[max2Nindex][2]
                    path_w[1].append(t2N[1 - max2Nindex][0])  # 添加废牌
                    path_w[1].append(t2N[1 - max2Nindex][1])
                else:  # 飞宝
                    path_w[0] = feiKingHu * 2
                    path_w[1].append(self.kingCard)
            #
            else:  # 正常打
                # if True:
                t2N.sort(key=lambda k: k[2], reverse=True)
                if len(t2N) >= bl:
                    if a[-1] == []:
                        if t2N != []:

                            path_w[1].append(t2N[-1][0])
                            path_w[1].append(t2N[-1][1])
                            t2NCP = copy.deepcopy(t2N)
                            t2NCP.remove(t2N[-1])

                            # 不飞宝打法

                            if bl - king_num + 1 >= 0:
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
                                # print ('t2NCP',t2NCP)
                                self.choose_n(t2N=t2NCP, n=bl - king_num + 1, rate=1, results=results)
                                nums = math.factorial(bl - king_num + 1)
                                path_w[0] *= float(sum(results)) / nums

                        # 飞宝打法

                        aCopy = copy.deepcopy(a)
                        aCopy[-1].append(self.kingCard)
                        # 宝－１后，再次计算胡牌概率，可能只有一个宝，所以会变为计算无宝的打法，还剩下宝会另计算
                        path_w_feiKing = self.calculate_path_w(aCopy, king_num - 1, 2)
                        print ('comparable', path_w, path_w_feiKing)
                        if path_w[0] <= path_w_feiKing[0] * feiKing:
                            path_w = path_w_feiKing

                    else:
                        if bl - king_num + 1 >= 0:
                            self.choose_n(t2N=t2N, n=bl - king_num + 1, rate=1, results=results)
                            nums = math.factorial(bl - king_num + 1)
                            path_w[0] *= float(sum(results)) / nums




                else:  # 未溢出
                    # t2NCP = copy.deepcopy(t2N)
                    # for i in range(use_king):
                    #     t2NCP.pop(-1)
                    # 不够的添加
                    for i in range(bl - len(t2N)):
                        path_w[0] *= (80.0) / (self.remainNum * self.remainNum)
                    if len(t2N) - king_num + 1 >= 0:
                        self.choose_n(t2N=t2N, n=len(t2N) - king_num + 1, rate=1, results=results)
                        nums = math.factorial(len(t2N) - king_num + 1)
                        path_w[0] *= float(sum(results)) / nums

                if king_num - 1 > len(t2N):
                    use_king = len(t2N)
                else:
                    use_king = king_num - 1
                if use_king > 0 and path_w[1] == []:  # 表示宝牌未用完，这里可以打出一张宝牌
                    for i in range(use_king):
                        path_w[1].append(self.kingCard)
        # print ('results', results, len(results),nums)

        return path_w

    def cost(self, all, suits, left_num=[], king_num=0, king_card=None):
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
            path_w.append([1.0, copy.deepcopy(all[i][-1])])

        # 全部搜索会导致搜索空间极大
        for index_all in range(len(all)):  # 选出最大期望概率胡牌路径，选择该路径，从剩余牌中再选择最佳出牌顺序，局部最优

            path_w[index_all] = self.calculate_path_w(all[index_all], king_num, 1)

        print("path_w_end", path_w)
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
                        left_numCP[convert_hex2index(ndcard)] = self.remainNum
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
            # print ()
            a = copy.deepcopy(all[i])
            if i in used_index:
                continue
            for j in range(i + 1, len(all)):
                if len(all[j][0]) + len(all[j][1]) == len(a[0]) + len(a[1]) and all[j][2] == a[2]:
                    if all[j][3] == a[3]:
                        used_index.append(j)
                        for card in all[j][-1]:
                            if card not in a[-1]:
                                a[-1].append(card)
            all3.append(a)

        return all3

    def defend_V2_2(self, all_combination):
        """
        功能：出牌策略
        思路：分为３阶段，第一阶段完全孤张牌出牌策略，计算出所有组合中都包含的孤张牌，出评估值最低的孤张牌，剩余牌与孤张牌的联系性最低
                    第二阶段：当ｘｔｓ<=3时，采用搜索树计算出最佳出牌
                    第三阶段：当xts>3时，采用快速评估的方法计算出最佳出牌
        :param all_combination: 组合信息
        :return: 决策出牌
        """

        '''
                           第一阶段：完全孤张牌出牌策略
                           原则：出相关性最低的孤张牌，剩余牌与孤张牌的联系性最低
                           现阶段只考虑ｘｔｓ最小的情况
        '''
        kz = []
        sz = []
        for s in self.suits:
            if s[0] == s[1]:
                kz.append(s[0])
            else:
                sz.append(s[0])

        allCP = copy.deepcopy(all_combination)

        all_same_xts_and_left = copy.deepcopy(all_combination)
        min_xts = all_combination[0][4]
        # 对手模型
        # todo 目前对手模型质量太差，移除
        # DFM=DefendModel(cards,suits,self.discards,self.discardsOp,self.discards_real,self.round,self.seat_id)
        # DFM.defendModel()
        # print('DFM')
        print ('all_combination', all_combination)
        for a in all_combination:  # 获取ｘｔｓ相同的组合
            if a[4] != min_xts:
                if not (sz == [] and a[1] == [] and a[3] == []):
                    allCP.remove(a)
            if a[4] != min_xts or len(a[-1]) != len(all_combination[0][-1]):
                all_same_xts_and_left.remove(a)

        # 去重处理
        # all_same_xts_and_left=self.mergeSameall(all_same_xts_and_left) #todo ３４５６　９不能解决将牌问题
        all_MG = self.mergeSameall(allCP)
        # all_MG=allCP

        print ('allCP', allCP)
        print ("all_same_xts_and_left", all_same_xts_and_left)
        left_all_cards = []  # 全部组合的废牌集合
        left_cards_ndCards = {}  # 废牌与之联系的次级牌（包括２Ｎ，与废牌，用来计算手牌权重）

        for branch in all_same_xts_and_left:
            left_all_cards += branch[-1]

            # 生成left_cards_ndCards
            for card in branch[-1]:
                if card not in left_cards_ndCards.keys():
                    left_cards_ndCards[card] = []
                for t2 in branch[2] + branch[3]:
                    left_cards_ndCards[card].extend(t2)
                left_cards_ndCards[card].extend(branch[-1])
                left_cards_ndCards[card].remove(card)
        unique_l = list(set(left_all_cards))
        left_cards = []  # 任何组合都包含的真正废牌
        left_cards_w = []
        for card in unique_l:
            if left_all_cards.count(card) == len(all_same_xts_and_left):
                left_numCP = copy.copy(self.leftNum)
                for c in set(left_cards_ndCards[card]):
                    left_numCP[convert_hex2index(c)] = self.remainNum
                left_cards.append(card)
                left_cards_w.append(self.left_card_weight(card=card, left_num=left_numCP))  # 更新点：添加废牌权重
        # left_cards=[]
        if left_cards != [] and len(all_same_xts_and_left) > 1:
            print ('state first')
            return left_cards[left_cards_w.index(min(left_cards_w))]

        '''
        第二阶段
        当unique_l不为空时，从所有废牌(unique_l)中出一张
        如果为空，从所有的t2Ｎ中出一张
        '''

        # 手牌影响　todo 可能需要改进
        ndcards = {}
        for branch in allCP:
            left_all_cards += branch[-1]

            # 生成left_cards_ndCards
            for card in branch[-1]:
                if card not in ndcards.keys():
                    ndcards[card] = []
                for t2 in branch[2] + branch[3]:
                    ndcards[card].extend(t2)
                    ndcards[card].extend(branch[-1])
                    ndcards[card].remove(card)

        # 在ｘｔｓ<3的情况下，使用搜索树
        if all_MG[0][4] <= 3:

            Tree = SearchTree(cards=self.cards, suits=self.suits, leftNum=self.leftNum, all=allCP,
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
            print ('maxScoreCards', maxScoreCards)

            if maxScoreCards != []:
                return self.discards_w(maxScoreCards, self.leftNum, ndcards={})

        # 当unique_l不为空时(策略集)
        # 重新生成胡牌信息还是使用以往的信息?
        # 第一种测试方法：使用废牌数量(最简单的方法，局部最优，可能会出现遗漏的情况),之后改进
        # v1 细化废牌权重
        unique_l = []
        print ("all_2", all_MG)
        for a in all_MG:
            unique_l += a[-1]
        ndcards = {}
        for branch in all_MG:
            # left_all_cards += branch[-1]

            # 生成left_cards_ndCards
            for card in branch[-1]:
                if card not in ndcards.keys():
                    ndcards[card] = []
                for t2 in branch[2] + branch[3]:
                    ndcards[card].extend(t2)
                    ndcards[card].extend(branch[-1])
                    ndcards[card].remove(card)
        # if unique_l!=[]:
        if True:
            discards_w = {}
            path_w = self.cost(all=all_MG, suits=self.suits, left_num=self.leftNum, king_num=self.kingNum,
                               king_card=self.kingCard)
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
                                ln = sum([self.leftNum[translate16_33(e)] for e in lc])

                                if ln >= max_remove_3N:
                                    max_remove_3N = ln
                                    remove_card = t3[0]
                                rc = self.get_effective_cards(dz_set=[[t3[0], t3[1]]])
                                rn = sum([self.leftNum[translate16_33(e)] for e in rc])
                                if rn > max_remove_3N:
                                    max_remove_3N = rn
                                    remove_card = t3[2]
                        elif len(a[2]) != []:  # 单吊
                            remove_card = a[2][0][0]
                print("defend_V2_2,has Hu,and out a highest rate card", 1 / remove_card, remove_card)
                return remove_card
            out_card = self.discards_w(discards=path_w[0][-1], left_num=self.leftNum, ndcards={})
            return out_card

    # 决策出牌
    def recommend_card(self):
        """
        推荐出牌接口
        :return: 返回最佳出牌
        """
        t1 = time.clock()

        all = self.sys_info_V3(cards=self.cards, suits=self.suits, left_num=self.leftNum, kingCard=self.kingCard)
        t2 = time.clock()
        print('sys_info_V3', t2 - t1, 'xts=', all[0][4])
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
                kz_suits.append(suit[0])
            else:
                sz_suits.append(suit[0])
        for a in all:
            kz = []
            kz.extend(kz_suits)
            sz = []
            sz.extend(sz_suits)

            jiang = 0x00

            if a[4] == 0:

                for kz_ in a[0] + a[2]:
                    kz.append(kz_[0])
                for sz_ in a[1] + a[3]:
                    if sz_[0] != 8:
                        sz.append(sz_[0])
                    else:
                        sz.append(sz_[0] - 1)

                if kingNum != 0:
                    jiang = self.kingCard
                else:
                    jiang = a[2][0][0]
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
        all_former = self.sys_info_V3(cards=cards_former, suits=suits, left_num=left_num, kingCard=self.kingCard)
        print ("recommend_op,all_former", all_former)
        # 计算前向胡牌概率 完全局部最优策略
        path_w_former = self.cost(all=all_former, suits=suits, left_num=left_num, king_num=self.kingNum,
                                  king_card=self.kingCard)
        path_w_former.sort(key=lambda k: (k[0]), reverse=True)
        print ("path_w_former", path_w_former)
        rate_former = path_w_former[0][0]  # 未执行动作的胡牌概率

        # 是否胡牌判断
        if isHu:
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
                    return [card, card, card, card], False

            for card in cards:
                if [card, card, card] in suits:
                    cards_BuGang = copy.copy(cards)
                    cards_BuGang.remove(card)
                    all_BuGang = self.sys_info_V3(cards=cards_BuGang, suits=suits, left_num=left_num,
                                                  kingCard=self.kingCard)
                    asset = self.cost(all_BuGang, suits=suits, left_num=left_num,
                                      king_num=cards_BuGang.count(self.kingCard), king_card=self.kingCard)
                    asset.sort(key=lambda k: (k[0]), reverse=True)
                    buGangHuRate = asset[0][0]
                    # 如果补杠后也能胡，则直接杠，否则算期望
                    if buGangHuRate == 1:
                        return [card, card, card, card], False
                    else:
                        kz, sz, jiang = self.hu_info(all_former, self.suits, kingNum=self.kingNum)
                        if jiang == 0:
                            return [], True
                        score = Fan(kz=kz, sz=sz, jiang=jiang, node=None, fei_king=self.fei_king)
                        # 胡牌分数高，则直接胡，否则，看几率
                        if score >= 8:
                            return [], True
                        else:
                            if buGangHuRate <= rate_former * 0.5:
                                return [], True
                            else:
                                return [card, card, card, card], False
            # return [],True
            # 手中有２张宝牌，先不胡，打掉一张宝牌后３手内的胡牌概率是否超过原有期望
            if self.kingNum >= 2:
                # 如果作为宝还原，宝吊则直接胡
                cards_FeiBao = copy.copy(cards)
                cards_FeiBao.remove(self.kingCard)
                path_w_out1King = self.cost(all=all_former, suits=suits, left_num=left_num, king_num=self.kingNum - 1,
                                            king_card=self.kingCard)
                path_w_out1King.sort(key=lambda k: (k[0]), reverse=True)

                if path_w_out1King[0][0] * 2 < 1:
                    return [], True
                else:
                    return [], False

            else:
                return [], True

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
        all_later = self.sys_info_V3(cards=cards_add_op, suits=suits, left_num=left_num, kingCard=self.kingCard)
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
        print ("val", val)

        if val != []:
            path_w_later = self.cost(all=val, suits=suits, left_num=left_num, king_num=self.kingNum,
                                     king_card=self.kingCard)
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
                                    ln = sum([left_num[translate16_33(e)] for e in lc])
                                    # for card in lc:
                                    if ln > max_remove_3N:
                                        max_remove_3N = ln
                                    rc = self.get_effective_cards(dz_set=[[t3[0], t3[1]]])
                                    rn = sum([left_num[translate16_33(e)] for e in rc])
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
                                if self.leftNum[translate16_33(t2Ns[0][0])] * 2 > max_remove_3N:
                                    max_remove_3N = self.leftNum[translate16_33(t2Ns[0][0])]

                    efc_num = 0  # 胡牌的有效牌数量
                    efc_cards = set(efc_cards)
                    for card in efc_cards:
                        efc_num += left_num[translate16_33(card)]
                    print ("efc_num,max_remove_3N", efc_num, max_remove_3N)
                    if max_remove_3N < efc_num * 1.2:  # or not (max_remove_3N==efc_num and len(cards)<=5):  # 如果有效牌数量增加，则执行此操作
                        return [], False  # continue

                # 有宝可以打宝吊，单吊
                print (path_w_later[i_p][0], rate_former)
                if path_w_later[i_p][0] >= 1:
                    path_w_later[i_p][0] = 1
                if path_w_later[i_p][0] > rate_former:  # or (self.kingNum != 0 and len(cards) <= 4): #单吊
                    index.append([i_p, path_w_later[i_p][0]])
            index.sort(key=lambda k: k[1], reverse=True)
            if index != []:
                for t3 in val[index[0][0]][0] + val[index[0][0]][1]:  # 在最优吃碰杠组合中给出该３Ｎ,修正点，从all_later修正为ｖａｌ
                    print ("op_ t3", t3)
                    if op_card in t3:
                        if canchi:
                            return t3, False
                        elif t3[0] == t3[1]:
                            return t3, False
        return [], False


# 九幺牌型类
class jiuyao():
    def __init__(self):
        """
        类变量初始化
        存储幺九牌
        """
        self.yaojiu_cards = [0x01, 0x09, 0x11, 0x19, 0x21, 0x29, 0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37]
        pass

    def get_yaojiu_num(self, cards=[], suits=[]):
        """
        计算手牌中幺九牌的数量
        副露中若存在非幺九牌，则直接返回[-1]*34
        :param cards: 手牌
        :param suits: 副露
        :return: yaojiu_num, yaojiu_num_hand　手牌与副露中的幺九牌总数列表　手牌中的幺九牌数量
        """
        yaojiu_num = [0] * (2 * 3 + 7)  # 按位存储幺九牌的个数
        yaojiu_num_hand = [0] * (2 * 3 + 7)
        for i in suits:
            if i[0] not in self.yaojiu_cards or i[1] != i[0]:
                return [-1] * (2 * 3 + 7), yaojiu_num_hand
            else:
                yaojiu_num[self.yaojiu_cards.index(i[0])] += 3
        for i in range(13):
            yaojiu_num_hand[i] = cards.count(self.yaojiu_cards[i])
            yaojiu_num[i] += cards.count(self.yaojiu_cards[i])

        return yaojiu_num, yaojiu_num_hand

    def jiuyao_info(self, cards=[], suits=[], left_num=[]):
        """
        综合计算幺九牌型类相关信息
        :param cards: 手牌
        :param suits: 副露
        :param left_num: 剩余牌
        :return: {} 字典格式存储的幺九牌信息
        """
        jiuyao_info = {}
        left_cards = []

        yaojiu_num, yaojiu_num_hand = self.get_yaojiu_num(cards=cards, suits=suits)
        jiuyao_info["yaojiu_num"] = yaojiu_num
        jiuyao_info["yaojiu_num_hand"] = yaojiu_num_hand
        # 副露中有非幺九牌，无法胡九幺
        if yaojiu_num[0] == -1:
            jiuyao_info["xts"] = 14
            return jiuyao_info

        cards_copy = copy.deepcopy(cards)
        for i in range(len(yaojiu_num_hand)):
            for j in range(yaojiu_num_hand[i]):
                cards_copy.remove(self.yaojiu_cards[i])
        # discards 算剩余幺九牌的数目

        jiuyao_info["left_cards"] = cards_copy
        jiuyao_info["xts"] = 14 - sum(yaojiu_num)
        return jiuyao_info

    @staticmethod
    def defend_V1(left_cards=[], king_card=None):
        """
        幺九牌出牌决策
        按出牌次序discards_order，直接出非幺九牌
        :param left_cards: 剩余牌
        :param king_card: 宝牌
        :return:
        """
        discards_order = [0x02, 0x08, 0x12, 0x18, 0x22, 0x28, 0x03, 0x07, 0x13, 0x17, 0x23, 0x27, 0x04, 0x06, 0x14,
                          0x16, 0x24, 0x26, 0x05, 0x15, 0x25]

        # print("jiuyao,defend_V1,left_cards=",left_cards)
        for card in discards_order:
            if card in left_cards:
                # 宝牌处理：有宝的情况,宝牌后出
                if card == king_card:
                    continue
                return card

        if king_card in left_cards:
            return king_card

        return None

        #

    def one_of(self, array1, array2):
        """
        判断一个数组是否存在于另一个数组中
        :param array1: 数组１
        :param array2: 数组２
        :return:
        """
        for e in array2:
            if (array1 == e).all():
                return True
        return False

    def recommend_op(self, op_card, cards=[], suits=[], left_num=[], king_card=None, canchi=False, self_turn=False):
        """
        幺九牌型的动作决策
        有幺九牌的碰杠直接碰杠，非幺九不碰杠
        :param op_card: 操作牌
        :param cards: 手牌
        :param suits: 副露
        :param left_num: 剩余牌
        :param king_card: 宝牌
        :param canchi: 吃牌权限
        :param self_turn: 是否是自己回合
        :return: [] 动作组合牌
        """
        jiuyao_info = self.jiuyao_info(cards=cards, suits=suits, left_num=left_num)
        # 本场次只考虑九幺的情况
        # 处理补杠和暗杠的情况
        yaojiu_num_hand = jiuyao_info["yaojiu_num_hand"]
        # yaojiu_num = jiuyao_info["yaojiu_num"]
        if self_turn:  # 补杠　暗杠
            for i in range(len(yaojiu_num_hand)):
                if yaojiu_num_hand[i] == 1 and suits != [] and self.one_of(
                        np.array([self.yaojiu_cards[i], self.yaojiu_cards[i], self.yaojiu_cards[i]]), np.array(suits)):
                    return [self.yaojiu_cards[i], self.yaojiu_cards[i], self.yaojiu_cards[i], self.yaojiu_cards[i]]
                elif yaojiu_num_hand[i] == 4:
                    return [self.yaojiu_cards[i], self.yaojiu_cards[i], self.yaojiu_cards[i], self.yaojiu_cards[i]]
            return []
        else:
            if op_card in self.yaojiu_cards:
                if yaojiu_num_hand[self.yaojiu_cards.index(op_card)] == 3:  # 明杠
                    return [op_card, op_card, op_card, op_card]
                elif yaojiu_num_hand[self.yaojiu_cards.index(op_card)] == 2 and jiuyao_info["xts"] > 1:  # 碰 增加填胡不再碰的情况
                    return [op_card, op_card, op_card]
                else:
                    return []
            else:
                return []

    def recommend_card(self, cards=[], suits=[], left_num=[], king_card=None):
        """
        出牌接口
        :param cards:手牌
        :param suits: 副露
        :param left_num: 剩余牌
        :param king_card: 宝牌
        :return: card 出牌
        """
        jiuyao_info = self.jiuyao_info(cards=cards, suits=suits, left_num=left_num)
        # jiuyao_info["yaojiu_num"]==-1:
        left_cards = jiuyao_info["left_cards"]
        return self.defend_V1(left_cards=left_cards, king_card=king_card)


# 七对牌型类
class qidui:
    def __init__(self):
        pass

    def get_cards_num(self, cards=[]):
        """
        获取手牌中每张牌的数量
        :param cards: 手牌
        :return: cards_unique, cards_num　去重后的手牌及其数量
        """
        # if len(suits)!=0:
        #     return

        cards_unique = np.unique(cards)
        cards_num = [0] * len(cards_unique)
        for i in range(len(cards_unique)):
            cards_num[i] = cards.count(cards_unique[i])

        return cards_unique, cards_num

    def qidui_info(self, cards=[], suits=[], left_num=[], king_num=0):
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
        qidui_info["xts"] = 14 - (len(duipai) * 2 + 7 - len(duipai)) - king_num

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
            effective_cards_num[i] = left_num[translate16_33(left_cards[i])]
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

    def recommend_card(self, cards=[], suits=[], left_num=[], king_card=None):
        """
        七对出牌接口
        :param cards:手牌
        :param suits: 副露
        :param left_num: 剩余牌
        :param king_card: 宝牌
        :return: 最佳出牌
        """
        cards_copy = copy.deepcopy(cards)
        if king_card != None:
            # cards_copy=copy.deepcopy(cards)
            king_num = cards.count(king_card)
            for i in range(king_num):
                cards_copy.remove(king_card)

        qidui_info = self.qidui_info(cards=cards_copy, suits=suits, left_num=left_num, king_num=king_num)
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


# 十三烂牌型类
class ssl:
    # todo　宝牌翻倍的
    def __init__(self, handcards, suits, discards):
        """
        十三烂类变量初始化
        :param handcards:手牌
        :param suits: 副露
        :param discards: 弃牌
        """
        self.handcards = handcards
        self.suits = suits

        self.type = None

        self.discards = discards

    def wait_types_13(self):
        """
        十三浪的向听数判断，手中十四张牌中，序数牌间隔大于等于3，字牌没有重复所组成的牌形
        先计算0x0,0x1,0x2中的牌，起始位a，则a+3最多有几个，在wait上减，0x3计算不重复最多的数
        :return: wait_num, handcardsapart, effectiveCards, entire_discards　向听数　万条筒具体拆分情况　万条筒的具体有效牌　完全废牌
        """

        # numCanUsejing = 0
        tile_list = copy.deepcopy(self.handcards)
        # numJing = 0
        wait_num = 14  # 表示向听数
        numZi = 0  # 记录字牌个数
        handcardsapart = []  # 万条筒具体拆分情况
        effectiveCards = []  # 万条筒的具体有效牌
        entire_discards = []  # 完全废牌

        if self.suits != []:
            wait_num = 14
            return wait_num, [], [], []
        else:
            L = set(tile_list)  # 去除重复手牌
            L_num0 = []  # 万数牌
            L_num1 = []  # 条数牌
            L_num2 = []  # 筒数牌
            L_num3 = []  # 字牌数
            for i in L:
                if i & 0xf0 == 0x30:
                    # 计算字牌的向听数
                    numZi += 1
                    wait_num -= 1
                if i & 0xf0 == 0x00:
                    L_num0.append(i & 0x0f)
                if i & 0xf0 == 0x10:
                    L_num1.append(i & 0x0f)
                if i & 0xf0 == 0x20:
                    L_num2.append(i & 0x0f)
                if i & 0xf0 == 0x30:
                    L_num3.append(i & 0x0f)

            if L_num0 != []:
                self.type = 0
                # print ("L_num0=",L_num0)
                a, b, c, d = self.calculate_13(L_num0)  # 减去万数牌的向听数
                # print (a,b,c)
                wait_num -= a
                handcardsapart.append(b)
                effectiveCards.append(c)
                entire_discards.extend(d)
            else:
                handcardsapart.append([])
                effectiveCards.append([])
            if L_num1 != []:
                self.type = 1
                a, b, c, d = self.calculate_13(L_num1)  # 减去条数牌的向听数
                wait_num -= a
                handcardsapart.append(b)
                effectiveCards.append(c)
                entire_discards.extend([e + 16 for e in d])
            else:
                handcardsapart.append([])
                effectiveCards.append([])
            if L_num2 != []:
                self.type = 2
                a, b, c, d = self.calculate_13(L_num2)  # 减去筒数牌的向听数
                wait_num -= a
                handcardsapart.append(b)
                effectiveCards.append(c)
                entire_discards.extend([e + 32 for e in d])
            else:
                handcardsapart.append([])
                effectiveCards.append([])
            if L_num3 != []:
                self.type = 3
                c = self.getzieffectiveCards(L_num3)
                handcardsapart.append(L_num3)
                effectiveCards.append(c)
            else:
                handcardsapart.append([])
                effectiveCards.append([])

        return wait_num, handcardsapart, effectiveCards, entire_discards

    def calculate_13(self, tiles):  # 返回 向听数，手牌情况，有效牌
        """
        计算十三烂中各花色的向听数，拆分情况，有效牌，完全废牌
        :param tiles: 某一花色的手牌
        :return: wait_num, handcardsapart, effectiveCards, entire_discards　向听数　万条筒具体拆分情况　万条筒的具体有效牌　完全废牌
        """
        # 计算十三浪的数牌最大向听数
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
        waitnumMax2 = max((tiles.count(2) + tiles.count(7)), (tiles.count(3) + tiles.count(7)),
                          (tiles.count(3) + tiles.count(8)), )
        ssl_table = [[1, 6, 9], [1, 4, 9], [1, 4, 7], [1, 4, 8], [1, 5, 9], [1, 5, 8], [2, 5, 9], [2, 5, 8], [3, 6, 9],
                     [2, 6, 9]]
        if max(waitnumMax1, waitnumMax2) == 3:  # 当向听数为3 的时候 直接返回3，无有效牌
            handcardapart = []  # 手牌拆分情况
            entire_discards = []  # 完全废牌
            for i in ssl_table:
                if i[0] in tiles and i[1] in tiles and i[2] in tiles:
                    # print("i=",i)
                    handcardapart = i
                    tmp = copy.deepcopy(tiles)
                    tmp.remove(i[0])
                    tmp.remove(i[1])
                    tmp.remove(i[2])
                    entire_discards = tmp
                    break
            return 3, handcardapart, [], entire_discards
        elif max(waitnumMax1, waitnumMax2) == 2:  # 当向听数为2 的时候 返回向听数
            youxiao, entire_discards = self.geteffectiveCards(tiles)
            return 2, youxiao[0], youxiao[1], entire_discards
        elif max(waitnumMax1, waitnumMax2) == 1:  # 当向听数只有1 的时候
            # 20190411.12.18 修正向听数为１的情况
            effective_cards = []
            ssl_one_list = [1, 9, 2, 8, 3, 7, 4, 6, 5]
            ssl_one_efc = [[4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 5, 6], [5, 6, 7, 8, 9], [1, 2, 3, 4, 5], [6, 7, 8, 9],
                           [1, 2, 3, 4], [1, 7, 8, 9], [1, 2, 3, 9], [1, 2, 8, 9]]
            for i in range(len(ssl_one_list)):
                if ssl_one_list[i] in tiles:
                    return 1, [ssl_one_list[i]], ssl_one_efc[i], []
        # 没有这种花色时的情况
        return 0, [], [1, 2, 3, 4, 5, 6, 7, 8, 9], []

    @staticmethod
    def getTable():
        """
        建十三烂的表（每个元素：【【牌型】，【有效牌集合】，有效牌个数】)
        :return: 十三烂的表
        """
        table = [[[1, 4], [7, 8, 9], 3], [[1, 5], [8, 9], 2], [[1, 6], [9], 1], [[1, 7], [4], 1], [[1, 8], [4, 5], 2],
                 [[1, 9], [4, 5, 6], 3], [[2, 5], [8, 9], 2], [[2, 6], [9], 1], [[2, 7], [], 0], [[2, 8], [5], 1],
                 [[2, 9], [5, 6], 2], [[3, 6], [9], 1], [[3, 7], [], 0], [[3, 8], [], 0], [[3, 9], [6], 1],
                 [[4, 7], [1], 1], [[4, 8], [1], 1], [[4, 9], [1], 1], [[5, 8], [1, 2], 2], [[5, 9], [1, 2], 2],
                 [[6, 9], [1, 2, 3], 3]]
        return table

    @staticmethod
    def get2N(cards=[]):
        """
        计算十三烂万条筒花色的搭子集合及剩余牌
        :param cards: 某一类非字牌
        :return: 所有可能的搭子
        """
        all2N = []  # ２Ｎ
        left_cards = []  # 废牌
        for i in cards:
            for j in range(i + 3, 10, 1):  # 20190411修改，9改10
                if j in cards:
                    all2N.append([i, j])
                    # 添加废牌
                    tmp = copy.deepcopy(cards)
                    tmp.remove(i)
                    tmp.remove(j)
                    left_cards.extend(tmp)  # all2N[1].append()
        return all2N, left_cards

    def geteffectiveCards(self, cards):
        """
        当某一花色的十三烂牌的有用牌为２张时，获取有效牌数量最多的拆分组合
        获取万条筒的手牌分布情况，有效牌分布，有效牌的实际个数(非字牌)
        :param cards: 某一花色的手牌
        :return: effective　手牌情况，有效牌,有效牌实际个数，　entire_discards　完全废牌
        """
        effective = [[], [], 0]  # 手牌情况，有效牌,有效牌实际个数
        entire_discards = []  # 完全废牌
        all2N, left_cards = self.get2N(cards)  # 获取当前手牌所有2N
        for card in left_cards:
            if left_cards.count(card) == len(left_cards):
                entire_discards.append(card)
        ssl2NTable = self.getTable()  # 获取十三烂2N表
        for two2N in ssl2NTable:
            if two2N[0] in all2N:  # 如果手牌里的2N与库里的相同
                if self.getEffectiveNum(two2N[1]) >= effective[2]:  # 如果当前有效牌多于有效牌
                    effective[1] = two2N[1]  # 当前有效牌赋给有效牌
                    effective[0] = two2N[0]  # 把当前手牌情况放进去
                    effective[2] = self.getEffectiveNum(two2N[1])
        # print ("ssl.geteffectiveCards=",effective)
        return effective, entire_discards

    def getzieffectiveCards(self, cards):
        """
        获取字牌的有效牌
        :param cards:字牌
        :return: 字牌的有效牌
        """
        effectivecards = []
        allZi = [1, 2, 3, 4, 5, 6, 7]
        for i in allZi:
            if i not in cards:
                effectivecards.append(i)
        return effectivecards

    def translate(self, op_card, type):
        """
        个位数和type转换到 0-33 ／34转换
        :param op_card: 操作牌
        :param type:花色
        :return: ０－３３索引
        """
        if type == 0:  # 万字1-9对应 0-8
            return op_card - 1
        elif type == 1:  # 条字1-9对应 9-17
            return op_card - 1 + 9
        elif type == 2:  # 筒字1-9 对应18-26
            return op_card - 1 + 18
        elif type == 3:  # 字牌1-7  对应 27 - 33
            return op_card - 1 + 27

    def translate2(self, i):  # 1-34转换到16进制的card
        """
        将１－３４转化为牌值
        :param i:
        :return:
        """
        if i >= 10 and i <= 18:
            i = i + 7
        elif i >= 19 and i <= 27:
            i = i + 14
        elif i >= 28 and i <= 34:
            i = i + 21
        return i

    def getEffectiveNum(self, effectiveCards):
        """
        获取有效牌的数量
        输入effectiveCards有效牌集合，返回有效牌数量
        :param effectiveCards: 有效牌集合
        :return: 有效牌数量
        """
        Numeffective = len(effectiveCards) * 4
        for eC in effectiveCards:
            Numeffective -= self.discards[
                self.translate(eC, self.type)]  # 减去弃牌表中的有效牌  # Numeffective -= self.handcards.count(eC)  # 减去手牌中的有效牌
        return Numeffective

    def ssl_info(self, left_num=[]):
        """
        十三烂信息

        :param left_num:有效牌数量
        :return: ｛｝　字典格式存储的十三烂信息
        """
        ssl_info = {}
        # print("ssl,wait_types_13=",self.wait_types_13())
        # effctive_cards=[[],[],[],[]]每种花色的有效牌，万条筒取值为1－9,字为1－7
        xts, split_cards_, effective_cards, entire_discards = self.wait_types_13()
        # print("ssl,split_cards_=",split_cards_)
        if xts == 14:
            ssl_info["xts"] = xts
            return ssl_info
        split_cards = []
        # print (split_cards)
        split_cards.append(split_cards_[0])
        split_cards.append([e + 16 for e in split_cards_[1]])
        split_cards.append([e + 32 for e in split_cards_[2]])
        split_cards.append([e + 48 for e in split_cards_[3]])
        left_cards = copy.deepcopy(self.handcards)
        # print("ssl_info,split_cards=",split_cards)
        for s_cards in split_cards:
            for card in s_cards:
                left_cards.remove(card)

        ssl_info["xts"] = xts  # 向听数
        ssl_info["split_cards"] = split_cards  # 有用的十三烂牌
        ssl_info["effective_cards"] = effective_cards  # 有效牌
        ssl_info["left_cards"] = left_cards  # 去除十三烂后的剩余牌
        ssl_info["entire_discards"] = entire_discards  # 完全废牌

        return ssl_info

    def defend_V1(self, left_cards, entire_discards):
        """
        十三烂出牌策略
        按出牌次序discards_order　直接出废牌
        :param left_cards: 剩余牌
        :param entire_discards: 完全废牌
        :return: 最佳出牌
        """
        # print("ssl,defend_V1,left_cards=",left_cards)
        discards_order = [0x03, 0x07, 0x13, 0x17, 0x23, 0x27, 0x04, 0x06, 0x14, 0x16, 0x24, 0x26, 0x02, 0x08, 0x12,
                          0x18, 0x22, 0x28, 0x05, 0x15, 0x25, 0x01, 0x09, 0x11, 0x19, 0x21, 0x29,  # 留91,可能会转牌型
                          0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37]
        # 先出完全废牌
        for card in discards_order:
            if card in entire_discards:
                return card
        # 先出ａａ
        for card in left_cards:
            if self.handcards.count(card) >= 2:
                return card
        for card in discards_order:
            if card in left_cards:
                return card

    def recommend_card(self, ):
        """
        十三烂出牌接口
        :return: 最佳出牌
        """
        ssl_info = self.ssl_info()
        return self.defend_V1(ssl_info["left_cards"], ssl_info["entire_discards"])

    def recommend_op(self):
        """
        十三烂动作接口
        十三烂不考虑动作操作
        :return: []
        """
        return []


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
        print ("translate16_33 is error,i=%d" % i)
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
    for per in discards:
        for item in per:
            discards_list[discards_map[item]] += 1
            left_num[discards_map[item]] -= 1
    for seat_op in discards_op:
        for op in seat_op:
            for item in op:
                discards_list[discards_map[item]] += 1
                left_num[discards_map[item]] -= 1
    for item in handcards:
        left_num[discards_map[item]] -= 1

    # print ("trandfer_discards,left_num=",left_num)
    return left_num, discards_list


# 获取ｌｉｓｔ中的最小值和下标
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


def paixing_choose(cards=[], suits=[], king_card=None, discards=[], discards_op=[], op_card=None, fei_king=0):
    """
    牌型选择
    通过计算向听数来判断
    :param cards: 手牌
    :param suits: 副露
    :param king_card:宝牌
    :param discards: 弃牌
    :param discards_op: 场面副露
    :param op_card: 操作牌
    :param fei_king: 飞宝数
    :return: 牌型序号　０为平胡　１　为九幺　２七对　３十三烂
    """
    left_num, discards_list = trandfer_discards(discards=discards, discards_op=discards_op, handcards=cards)

    # king_num = 0
    if king_card is not None:
        # left_num[translate16_33(king_card)] = 0
        left_num[translate16_33(pre_king(king_card))] -= 1
        out_king_cards = copy.deepcopy(cards)
        king_num = cards.count(king_card)
        for i in range(king_num):
            out_king_cards.remove(king_card)
    # xts, t3N, suits, t2N, [aa, ab, ac], left_cards, effctive_cards, split_cards, w,split_len=pinghu().sys_info()
    # 0,     1  ,  2,   3,      4,            5,            6,             7    ,  8      9
    cards_op = copy.deepcopy(cards)
    out_king_cards_op = copy.deepcopy(out_king_cards)
    if op_card != None:
        cards_op.append(op_card)
        out_king_cards_op.append(op_card)
    pinghu_info = pinghu(cards_op, suits, leftNum=left_num, discards=discards, discards_real=[], discardsOp=discards_op,
                         round=0, remainNum=sum(left_num), seat_id=0, kingCard=king_card, fei_king=fei_king,
                         op_card=op_card).sys_info_V3(cards=cards_op, suits=suits, left_num=left_num,
                                                      kingCard=king_card)
    jiuyao_info = jiuyao().jiuyao_info(cards=cards, suits=suits, left_num=left_num)
    qidui_info = qidui().qidui_info(cards=out_king_cards, suits=suits, left_num=left_num, king_num=king_num)
    ssl_info = ssl(cards, suits, discards_list).ssl_info()
    print ("[pinghu_info[0],jiuyao_info[0],qidui_info[0],ssl_info[0]]=",
           [pinghu_info[0][4], jiuyao_info["xts"], qidui_info["xts"], ssl_info["xts"]])

    min, index = get_min(list=[pinghu_info[0][4], jiuyao_info["xts"] - 2, qidui_info["xts"] + 1, ssl_info["xts"] + 1])
    return index


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


def recommend_card(cards=[], suits=[], king_card=None, discards=[], discards_op=[], fei_king=0, remain_num=136,
                   round=0):
    """
    功能：推荐出牌接口
    思路：使用向听数作为牌型选择依据，对最小ｘｔｓ的牌型，再调用相应的牌型类出牌决策

    :param cards: 手牌
    :param suits: 副露
    :param king_card: 宝牌
    :param discards: 弃牌
    :param discards_op: 场面副露
    :param fei_king: 飞宝数
    :param remain_num: 剩余牌
    :return: outCard 推荐出牌
    """
    if remain_num == 0:
        remain_num = 1
    # cards.sort()
    # suits=[sorted(e) for e in suits]
    left_num, discards_list = trandfer_discards(discards=discards, discards_op=discards_op, handcards=cards)
    # left_num[translate16_33(king_card)] = 0
    left_num[translate16_33(pre_king(king_card))] -= 1
    paixing = paixing_choose(cards=cards, suits=suits, king_card=king_card, discards=discards, discards_op=discards_op,
                             fei_king=fei_king)
    if remain_num == 136:
        remain_num = sum(left_num)
    if paixing == 0:
        print("choose pinghu")
        return pinghu(cards, suits, leftNum=left_num, discards=discards, discards_real=[], discardsOp=discards_op,
                      round=round, remainNum=remain_num, seat_id=0, kingCard=king_card,
                      fei_king=fei_king).recommend_card()
    elif paixing == 1:
        print ("choose jiuyao")
        return jiuyao().recommend_card(cards=cards, suits=suits, left_num=left_num)
    elif paixing == 2:
        print("choose qidui")
        return qidui().recommend_card(cards=cards, suits=suits, left_num=left_num, king_card=king_card)
    elif paixing == 3:
        print("choose ssl")
        return ssl(cards, suits, discards_list).recommend_card()


def recommend_op(op_card, cards=[], suits=[], king_card=None, discards=[], discards_op=[], canchi=False,
                 self_turn=False, fei_king=0, isHu=False, round=0):
    """
    功能：动作决策接口
    思路：使用向听数作为牌型选择依据，对最小ｘｔｓ的牌型，再调用相应的牌型类动作决策

    :param op_card: 操作牌
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

    left_num, discards_list = trandfer_discards(discards=discards, discards_op=discards_op, handcards=cards)
    # left_num[translate16_33(king_card)] = 0
    left_num[translate16_33(pre_king(king_card))] -= 1  # 宝牌前一张减一
    remain_num = sum(left_num)
    if remain_num == 0:
        remain_num = 1
    paixing = paixing_choose(cards=cards, suits=suits, king_card=king_card, discards=discards, discards_op=discards_op,
                             op_card=op_card, fei_king=fei_king)
    if paixing == 0:
        print("choose pinghu", cards)
        return pinghu(cards, suits, leftNum=left_num, discards=discards, discards_real=[], discardsOp=discards_op,
                      round=round, remainNum=sum(left_num), seat_id=0, kingCard=king_card, fei_king=fei_king,
                      op_card=op_card).recommend_op(op_card=op_card, canchi=canchi, self_turn=self_turn, isHu=isHu)
    elif paixing == 1:
        print ("choose jiuyao")
        if isHu:
            return [], isHu
        return jiuyao().recommend_op(op_card=op_card, cards=cards, suits=suits, left_num=left_num, king_card=king_card,
                                     canchi=canchi, self_turn=self_turn), False
    elif paixing == 2:
        if isHu:
            return [], isHu
        print("choose qidui")
        return qidui().recommend_op(op_card=op_card, cards=cards, suits=suits), False
    elif paixing == 3:
        print("choose ssl")
        if isHu:
            return [], isHu
        return ssl(cards, suits, discards_list).recommend_op(), False
