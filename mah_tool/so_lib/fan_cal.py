# -*- coding:utf-8 -*-
from mah_tool.so_lib.sr_xt_ph import pinghu


# from sr_xt_ph import pinghu
# 计算可能的番型
# # [清一色、断幺九、碰碰胡、自摸、杠上开花、金钩钩、幺九、一根、两根、三根、四根]
# 这里传入的牌应当都是十六进制的！

class FanList(object):
    def __init__(self, choosePaiXing=0, handcards=[], suits=[], jingCard=0, gangNum=0, isHuJudge=False,
                 isSelfTurn=True, preActionIsGang=False):
        '''
        番型检测功能
        :param choosePaiXing:  选择的胡牌类型 [0,1,2,3] -> 平胡， 九幺 ， 七对， 十三烂 这里只需要平胡和七对
        :param handcards:  手牌
        :param suits:  副露
        :param jingCard: 宝牌（精牌）此处全为0
        :param gangNum: 杠子数
        :param isHuJudge: 是否胡牌时的番型检测
        :param isSelfTurn: 是否为当前出牌
        '''
        self.choosePaiXing = choosePaiXing
        self.handcards = handcards
        self.suits = suits
        self.jingCard = jingCard
        self.gang_suit_n = 0
        self.all_suits_cards = self.__merge_suits()
        self.gangNum = gangNum
        self.jingCount = self.handcards.count(self.jingCard)
        self.isHuJudge = isHuJudge
        self.yaojiu = [1, 9, 0x11, 0x19, 0x21, 0x29]  # 九幺牌型所能包含的牌
        self.isSelfTurn = isSelfTurn
        self.preActionIsGang = preActionIsGang

    def __merge_suits(self):
        '''
        将所有的副露融合成手牌的形式 内部函数
        :return:
        '''
        all_suits_cards = []
        for suit in self.suits:
            if len(suit) == 4:
                self.gang_suit_n += 1
            all_suits_cards.extend(suit)
        return all_suits_cards

    def __getQiDuiXt(self):
        '''
        去除宝牌的七对向听数
        :return:
        '''
        qdXt = 7
        L = set(self.handcards)
        for i in L:
            if i != self.jingCard and self.handcards.count(i) >= 2:
                qdXt -= 1
        return qdXt

    def isQingYiSe(self):
        '''
        判断是否为清一色
        :return:
        '''
        cards = self.handcards + self.all_suits_cards
        w = 0
        s = 0
        t = 0
        for card in cards:
            if w == 0 and card & 0xf0 == 0x00:
                w = 1
            if s == 0 and card & 0xf0 == 0x10:
                s = 1
            if t == 0 and card & 0xf0 == 0x20:
                t = 1
        if w + s + t == 1:
            return True
        else:
            return False

    def isDuanJiuYao(self):
        '''
        求是否是断91
        :return:
        '''
        for card in self.handcards:
            if card in self.yaojiu:
                return False
        for card in self.all_suits_cards:
            if card in self.yaojiu:
                return False
        return True

    def isPengPengHu(self):
        '''
        判断是否为碰碰胡
        :return:
        '''

        def getSzKzInSuits(suits):  # 判断碰碰胡
            kz = 0
            sz = 0
            for suit in suits:
                if suit[0] == suit[1]:
                    kz += 1
                else:
                    sz += 1
            return kz, sz

        # 九幺同样适用该函数
        pinghu_info = pinghu(self.handcards, self.suits, self.jingCard).get_xts_info()

        kz, sz = getSzKzInSuits(self.suits)
        if sz == 0:
            if self.isHuJudge:  # 判胡
                if (len(pinghu_info[0]) + kz) == 4 and len(pinghu_info[1]) == 0:  # 碰碰胡，并且已经胡牌
                    return True
            else:  # 判番型方向 只有手牌刻子数跟副露刻子数大于三时才会引导往碰碰胡番上走
                if (len(pinghu_info[0]) + kz) >= 3 and len(pinghu_info[1]) == 0:  # 可以往碰碰胡方向走
                    return True
        return False

    def isZiMo(self):
        if self.isSelfTurn and self.isHuJudge:
            return True
        else:
            return False

    def isGangShangKaiHua(self):
        if self.isHuJudge and self.preActionIsGang:
            return True
        else:
            return False

    def isJinGouGou(self):
        return len(self.suits) == 4

    def isYaoJiu(self):
        # sub[0] kz
        # sub[1] sz
        # sub[2] aa
        # sub[3] ２N
        # sub[4] 得分
        # sub[5] 废牌

        def is91List2d(list2d):  # 判断2维list是否符合91
            if not list2d or len(list2d) == 0: return True
            for list in list2d:
                isContain91 = False
                for card in list:
                    if (card in self.yaojiu):
                        isContain91 = True
                        break
                if not isContain91:
                    return False
            return True

        if not is91List2d(self.suits): return False  # 先判断副露中的是否满足幺九
        pinghu_info = pinghu(self.handcards, self.suits, self.jingCard).get_xts_info()
        return is91List2d(pinghu_info[0]) and is91List2d(pinghu_info[1]) and is91List2d(pinghu_info[2])

    def getFanList(self):
        '''
        return the may fans base on the choosePaiXing
        :return:
        '''
        # # [清一色、断幺九、碰碰胡、自摸、杠上开花、金钩钩、幺九、一根、两根、三根、四根]
        # choosePaiXing [平胡  七对]
        fanList = [0] * 11

        if self.isQingYiSe():
            fanList[0] = 1
        if self.isDuanJiuYao():
            fanList[1] = 1
        if self.isPengPengHu():
            fanList[2] = 1
        if self.isZiMo():
            fanList[3] = 1  # baodiao
        if self.isGangShangKaiHua():
            fanList[4] = 1
        if self.isJinGouGou():
            fanList[5] = 1
        if self.isYaoJiu():
            fanList[6] = 1
        l = []
        l.extend(self.handcards)
        for suit in self.suits:
            l.extend(suit)
        geng_Num = 0
        for i in set(l):
            if l.count(i) == 4:
                geng_Num += 1
        if geng_Num > 0:
            for i in range(geng_Num):
                fanList[7 + i] = 1
        return fanList

# if __name__ == '__main__':
#     # test
#     # fan_list= FanList(1, [1, 2, 3, 6, 35],[[36, 36, 36], [29, 29, 29], [9, 9, 9]], 35, 2, isHuJudge=False).getFanList()
#     fan_list= FanList(1, [1, 2, 3, 1, 1],[[17, 18, 19], [23, 24, 25], [9, 9, 9]], 0, 0, isHuJudge=False).isYaoJiu()
#     # fan_list= FanList(0, [6, 35], [[1, 2, ],[36, 36, 36], [29, 29, 29], [9, 9, 9]], 35, 2, isHuJudge=False).isYaoJiu()
#     print(fan_list)
