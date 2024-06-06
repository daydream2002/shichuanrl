# -*- coding: utf-8 -*-
import random
import requests
import time

#上饶
import shangraoMJ.shangraoMJ_v1 as shangraoMJ_v1 #规则
import shangraoMJ.shangraoMJ_v3 as shangraoMJ_v2 #搜索树
import shangraoMJ.shangraoMJ_v3 as shangraoMJ_v3 #只摸牌搜索
import shangraoMJ.shangraoMJ_v4 as shangraoMJ_v4 #

#台湾
import taiwanMJ.twmj_KF_v1 as KF_v1 #规则
import taiwanMJ.twmj_KF_v2 as KF_v2 #搜索

#四川
import sichuanMJ.sichuanMJ_v1 as sichuanMJ_v1 #
import sichuanMJ.sichuanMJ_v1 as sichuanMJ_v2



MOP_NONE = -1
MOP_PASS = 0
MOP_LCHI = 1
MOP_MCHI = 2
MOP_RCHI = 3
MOP_PENG = 4
MOP_MGANG = 5
MOP_AGANG = 6
MOP_BGANG = 7

input_pre = []  # 记录本手之前的输入特征
output_pre = []  # 记录本手之前的输出结果


def trans10to16(i):
    """0-33 to
    0x01-0x09
    0x11-0x19
    0x21-0x29
    0x31-0x37
    10 to 16 in mahjong
    :param i:
    :return:
    """
    if i >= 0 and i <= 8:
        i += 1
    elif i >= 9 and i <= 17:
        i = i + 8
    elif i >= 18 and i <= 26:
        i = i + 15
    elif i >= 27 and i <= 33:
        i = i + 22
    return i


def translate1_37to0_33(i):
    if i >= 1 and i <= 9:
        i = i - 1
    elif i >= 11 and i <= 19:
        i = i - 2
    elif i >= 21 and i <= 29:
        i = i - 2
    elif i >= 31 and i <= 37:
        i = i - 3
    else:
        print ("translate1_37to0_33 is error,i=%d" % i)
        i = 34
    return i


def RecommendCard_shangraoMJ_v1(input={}):
    user_cards = input.get('user_cards', {})
    catch_card = input.get('catch_card', 0)
    king_card = input.get('king_card', 0)
    fei_king = input.get('fei_king', 0)
    seat_id = input.get('seat_id', 0)
    discards = input.get('discards', [])
    discards_op = input.get('discards_op', [])

    remain_num = input.get('remain_num', 0)
    round = input.get('round', [])
    hand_cards = user_cards.get('hand_cards', [])

    operate_cards = user_cards.get('operate_cards', [])
    hand_cards.sort()
    [e.sort() for e in operate_cards]
    recommendCard = shangraoMJ_v1.recommend_card(cards=hand_cards, suits=operate_cards, king_card=king_card,
                                                 discards=discards, discards_op=discards_op)
    return recommendCard


def RecommendOprate_shangraoMJ_v1(input={}):
    user_cards = input.get('user_cards', {})
    out_card = input.get('out_card', 0)
    king_card = input.get('king_card', 0)
    fei_king = input.get('fei_king', 0)
    # discards = input.get('discards',[])
    discards_op = input.get('discards_op', [])
    seat_id = input.get('seat_id', 0)
    out_seat_id = input.get('out_seat_id', 0)
    discards = input.get('discards', [])
    remain_num = input.get('remain_num', 0)
    isHu=input.get('isHu',False)
    hand_cards = user_cards.get('hand_cards', [])
    operate_cards = user_cards.get('operate_cards', [])
    # discards = trandfer_discards(discards, hand_cards)  # 转化弃牌表
    hand_cards.sort()
    [e.sort() for e in operate_cards]
    recommend_op,isHu = shangraoMJ_v1.recommend_op(op_card=out_card, cards=hand_cards, suits=operate_cards,
                                              king_card=king_card, discards=discards, discards_op=discards_op,
                                              canchi=canchi(seat_id, out_seat_id), self_turn=(len(hand_cards) % 3 == 2),isHu=isHu)
    return recommend_op,isHu


def RecommendCard_shangraoMJ_v2(input={}):
    user_cards = input.get('user_cards', {})
    catch_card = input.get('catch_card', 0)
    king_card = input.get('king_card', 0)
    fei_king = input.get('fei_king', 0)
    seat_id = input.get('seat_id', 0)
    discards = input.get('discards', [])
    discards_op = input.get('discards_op', [])
    remain_num = input.get('remain_num', 136)

    round = input.get('round', [])
    hand_cards = user_cards.get('hand_cards', [])
    operate_cards = user_cards.get('operate_cards', [])
    hand_cards.sort()
    [e.sort() for e in operate_cards]
    recommendCard = shangraoMJ_v2.recommend_card(cards=hand_cards, suits=operate_cards, king_card=king_card,
                                                 discards=discards, discards_op=discards_op, fei_king=fei_king,remain_num=remain_num,round=round)
    return recommendCard




def RecommendOprate_shangraoMJ_v2(input={}):
    user_cards = input.get('user_cards', {})
    out_card = input.get('out_card', 0)
    king_card = input.get('king_card', 0)
    fei_king = input.get('fei_king', 0)
    # discards = input.get('discards',[])
    discards_op = input.get('discards_op', [])
    remain_num = input.get('remain_num', 0)
    isHu=input.get('isHu',False)
    seat_id = input.get('seat_id', 0)
    out_seat_id = input.get('out_seat_id', 0)
    discards = input.get('discards', [])
    round = input.get('round',0)
    hand_cards = user_cards.get('hand_cards', [])
    operate_cards = user_cards.get('operate_cards', [])
    # discards = trandfer_discards(discards, hand_cards)  # 转化弃牌表
    hand_cards.sort()
    [e.sort() for e in operate_cards]
    recommend_op, isHu = shangraoMJ_v2.recommend_op(op_card=out_card, cards=hand_cards, suits=operate_cards,
                                              king_card=king_card, discards=discards, discards_op=discards_op,
                                              canchi=canchi(seat_id, out_seat_id), self_turn=(len(hand_cards) % 3 == 2),
                                              fei_king=fei_king,isHu=isHu,round=round)
    return recommend_op,isHu

def RecommendCard_shangraoMJ_v2_thread(input={}):
    user_cards = input.get('user_cards', {})
    catch_card = input.get('catch_card', 0)
    king_card = input.get('king_card', 0)
    fei_king = input.get('fei_king', 0)
    seat_id = input.get('seat_id', 0)
    discards = input.get('discards', [])
    discards_op = input.get('discards_op', [])
    remain_num = input.get('remain_num', 136)

    round = input.get('round', [])
    hand_cards = user_cards.get('hand_cards', [])
    operate_cards = user_cards.get('operate_cards', [])
    hand_cards.sort()
    [e.sort() for e in operate_cards]
    recommendCard = shangraoMJ_v3.recommend_card(cards=hand_cards, suits=operate_cards, king_card=king_card,
                                                 discards=discards, discards_op=discards_op, fei_king=fei_king,remain_num=remain_num,round=round,seat_id=seat_id)
    return recommendCard

def RecommendOprate_shangraoMJ_v2_thread(input={}):
    user_cards = input.get('user_cards', {})
    out_card = input.get('out_card', 0)
    king_card = input.get('king_card', 0)
    fei_king = input.get('fei_king', 0)
    # discards = input.get('discards',[])
    discards_op = input.get('discards_op', [])
    remain_num = input.get('remain_num', 0)
    isHu=input.get('isHu',False)
    seat_id = input.get('seat_id', 0)
    out_seat_id = input.get('out_seat_id', 0)
    discards = input.get('discards', [])
    round = input.get('round',0)
    hand_cards = user_cards.get('hand_cards', [])
    operate_cards = user_cards.get('operate_cards', [])
    # discards = trandfer_discards(discards, hand_cards)  # 转化弃牌表
    hand_cards.sort()
    [e.sort() for e in operate_cards]
    recommend_op, isHu = shangraoMJ_v3.recommend_op(op_card=out_card, cards=hand_cards, suits=operate_cards,
                                              king_card=king_card, discards=discards, discards_op=discards_op,
                                              canchi=canchi(seat_id, out_seat_id), self_turn=(len(hand_cards) % 3 == 2),
                                              fei_king=fei_king,isHu=isHu,round=round)
    return recommend_op,isHu



def RecommendCard_shangraoMJ_v4(input={}):
    user_cards = input.get('user_cards', {})
    catch_card = input.get('catch_card', 0)
    king_card = input.get('king_card', 0)
    fei_king = input.get('fei_king', 0)
    seat_id = input.get('seat_id', 0)
    discards = input.get('discards', [])
    discards_real = input.get('discards_real',[])
    discards_op = input.get('discards_op', [])
    remain_num = input.get('remain_num', 136)

    hands = input.get('hands',[])
    wall = input.get('wall',[])

    round = input.get('round', [])
    hand_cards = user_cards.get('hand_cards', [])
    operate_cards = user_cards.get('operate_cards', [])
    hand_cards.sort()
    [e.sort() for e in operate_cards]
    recommendCard = shangraoMJ_v4.recommend_card(cards=hand_cards, suits=operate_cards, king_card=king_card,
                                                 discards=discards, discards_real=discards_real, discards_op=discards_op, fei_king=fei_king,
                                                 remain_num=remain_num,round=round,seat_id=seat_id,hands=hands,wall=wall)
    return recommendCard

def RecommendOprate_shangraoMJ_v4(input={}):
    user_cards = input.get('user_cards', {})
    out_card = input.get('out_card', 0)
    king_card = input.get('king_card', 0)
    fei_king = input.get('fei_king', 0)
    discards_op = input.get('discards_op', [])
    discards_real = input.get('discards_real', [])
    remain_num = input.get('remain_num', 0)
    isHu=input.get('isHu',False)
    seat_id = input.get('seat_id', 0)
    out_seat_id = input.get('out_seat_id', 0)
    discards = input.get('discards', [])
    round = input.get('round',0)

    hands = input.get('hands', [])
    wall = input.get('wall', [])

    hand_cards = user_cards.get('hand_cards', [])
    operate_cards = user_cards.get('operate_cards', [])
    # discards = trandfer_discards(discards, hand_cards)  # 转化弃牌表
    hand_cards.sort()
    [e.sort() for e in operate_cards]
    recommend_op, isHu = shangraoMJ_v4.recommend_op(op_card=out_card, cards=hand_cards, suits=operate_cards,
                                              king_card=king_card, discards=discards, discards_op=discards_op,
                                              canchi=canchi(seat_id, out_seat_id), self_turn=(len(hand_cards) % 3 == 2),
                                              fei_king=fei_king,isHu=isHu,round=round)
    return recommend_op,isHu



# twmj_kf　outcard
def RecommendCard_KF_v1(hand_cards, operate_cards, round, remain_num, discards, discards_op):
    # return hand_cards[0]
    recommendCard = KF_v1.recommend_card(cards=hand_cards, suits=operate_cards, round=round, remain_num=remain_num,
                                         discards=discards,
                                         discards_op=discards_op)
    return recommendCard


# twmj KF op
def RecommendOprate_KF_v1(hand_cards, operate_cards, out_card, round, remain_num, discards, discards_op, canchi,
                          self_turn):
    # return []
    recommend_op = KF_v1.recommend_op(op_card=out_card, cards=hand_cards, suits=operate_cards,
                                      round=round, remain_num=remain_num, discards=discards, discards_op=discards_op,
                                      canchi=canchi, self_turn=self_turn)
    return recommend_op


def RecommendCard_KF_v2(hand_cards, operate_cards, round, remain_num, discards, discards_real, discards_op, seat_id):
    # return hand_cards[0]
    recommendCard = KF_v2.recommend_card(cards=hand_cards, suits=operate_cards, round=round, remain_num=remain_num,
                                         discards=discards, discards_real=discards_real,
                                         discards_op=discards_op, seat_id=seat_id)
    return recommendCard


def RecommendOprate_KF_v2(hand_cards, operate_cards, out_card, round, remain_num, discards, discards_real, discards_op,
                          canchi, self_turn, seat_id):
    # return []
    recommend_op = KF_v2.recommend_op(op_card=out_card, cards=hand_cards, suits=operate_cards, round=round,
                                      remain_num=remain_num, discards=discards, discards_real=discards_real,
                                      discards_op=discards_op, canchi=canchi, self_turn=self_turn, seat_id=seat_id)
    return recommend_op



def RecommendCard_sichuanMJ_v1(input={}):
    user_cards = input.get('user_cards', {})
    catch_card = input.get('catch_card', 0)
    # king_card = input.get('king_card', 0)
    # fei_king = input.get('fei_king', 0)
    seat_id = input.get('seat_id', 0)

    discards = input.get('discards', [])
    # print "1",discards_old["0"]
    # discards = [discards_old["0"], discards_old["1"], discards_old["2"], discards_old["3"]]

    discards_real=input.get('discards_real',[])
    # discards_real = list([discards_real_old["0"], discards_real_old["1"], discards_real_old["2"], discards_real_old["3"]])

    discards_op = input.get('discards_op', [])

    remain_num = input.get('remain_num', 0)

    round = input.get('round', [])
    hand_cards = user_cards.get('hand_cards', [])
    operate_cards = user_cards.get('operate_cards', [])
    choose_color=input.get('choose_color')

    hu_cards = input.get('hu_cards', [])
    # hu_cards = [hu_cards_old["0"], hu_cards_old["1"], hu_cards_old["2"], hu_cards_old["3"]]

    hu_fan=input.get('hu_fan', {})
    # hu_fan = hu_fan_old["hu_fan"]

    hand_cards.sort()
    [e.sort() for e in operate_cards]
    recommendCard = sichuanMJ_v1.recommend_card(cards=hand_cards, suits=operate_cards, round=round, remain_num=remain_num, discards=discards, discards_real=discards_real, discards_op=discards_op,
                                                seat_id=seat_id, choose_color=choose_color, hu_cards=hu_cards, hu_fan=hu_fan)
    return recommendCard


def RecommendOperate_sichuanMJ_v1(input={}):
    user_cards = input.get('user_cards', {})
    out_card = input.get('out_card', 0)
    # king_card = input.get('king_card', 0)
    # fei_king = input.get('fei_king', 0)
    # discards = input.get('discards',[])
    round = input.get('round', 0)


    remain_num = input.get('remain_num', 0)
    isHu=input.get('isHu',False)
    seat_id = input.get('seat_id', 0)
    out_seat_id = input.get('out_seat_id', 0)

    discards = input.get('discards', [])
    # discards = [discards_old["0"], discards_old["1"], discards_old["2"], discards_old["3"]]

    discards_real = input.get('discards_real', [])
    # discards_real = [discards_real_old["0"], discards_real_old["1"], discards_real_old["2"], discards_real_old["3"]]

    discards_op = input.get('discards_op', [])
    discards_real = input.get('discards_real', [])

    choose_color = input.get('choose_color', [])

    hu_cards = input.get('hu_cards',[])
    # hu_cards = [hu_cards_old["0"], hu_cards_old["1"], hu_cards_old["2"], hu_cards_old["3"]]

    hu_fan = input.get('hu_fan',[])
    # hu_fan = hu_fan_old["hu_fan"]

    hand_cards = user_cards.get('hand_cards', [])
    operate_cards = user_cards.get('operate_cards', [])
    # discards = trandfer_discards(discards, hand_cards)  # 转化弃牌表
    hand_cards.sort()
    [e.sort() for e in operate_cards]
    recommend_op, isHu = sichuanMJ_v1.recommend_op(op_card=out_card, cards=hand_cards, suits=operate_cards, round=round, remain_num=remain_num, discards=discards, discards_real=discards_real, discards_op=discards_op, self_turn=len(hand_cards) % 3 == 2, seat_id=seat_id, choose_color=choose_color, hu_cards=hu_cards, hu_fan=hu_fan, isHu = isHu)
    return recommend_op, isHu

def Recommend_switch_cards_sichuanMJ_v1(input={}):
    user_cards = input.get('user_cards', {})
    hand_cards = user_cards.get('hand_cards', [])
    operate_cards = user_cards.get('operate_cards', [])
    switch_n_cards=input.get('switch_n_cards',3)
    hand_cards.sort()
    recommend_switch_cards= sichuanMJ_v1.recommend_switch_cards(hand_cards=hand_cards, switch_n_cards=switch_n_cards)
    return recommend_switch_cards

def Recommend_choose_color_sichuanMJ_v1(input={}):
    user_cards = input.get('user_cards', {})
    hand_cards = user_cards.get('hand_cards', [])
    operate_cards = user_cards.get('operate_cards', [])
    switch_n_cards=input.get('switch_n_cards',3)
    hand_cards.sort()
    recommend_choose_color= sichuanMJ_v1.recommend_choose_color(hand_cards=hand_cards, switch_n_cards=switch_n_cards)
    return recommend_choose_color


def RecommendCard_sichuanMJ_v2(input={}):
    user_cards = input.get('user_cards', {})
    catch_card = input.get('catch_card', 0)
    # king_card = input.get('king_card', 0)
    # fei_king = input.get('fei_king', 0)
    seat_id = input.get('seat_id', 0)

    discards = input.get('discards', [])
    # print "1",discards_old["0"]
    # discards = [discards_old["0"], discards_old["1"], discards_old["2"], discards_old["3"]]

    discards_real=input.get('discards_real',[])
    # discards_real = list([discards_real_old["0"], discards_real_old["1"], discards_real_old["2"], discards_real_old["3"]])

    discards_op = input.get('discards_op', [])

    remain_num = input.get('remain_num', 0)

    round = input.get('round', [])
    hand_cards = user_cards.get('hand_cards', [])
    operate_cards = user_cards.get('operate_cards', [])
    choose_color=input.get('choose_color')

    hu_cards = input.get('hu_cards', [])
    # hu_cards = [hu_cards_old["0"], hu_cards_old["1"], hu_cards_old["2"], hu_cards_old["3"]]

    hu_fan=input.get('hu_fan', {})
    # hu_fan = hu_fan_old["hu_fan"]

    hand_cards.sort()
    [e.sort() for e in operate_cards]
    recommendCard = sichuanMJ_v2.recommend_card(cards=hand_cards, suits=operate_cards, round=round, remain_num=remain_num, discards=discards, discards_real=discards_real, discards_op=discards_op,
                   seat_id=seat_id,choose_color=choose_color,hu_cards=hu_cards,hu_fan=hu_fan)
    return recommendCard


def RecommendOperate_sichuanMJ_v2(input={}):
    user_cards = input.get('user_cards', {})
    out_card = input.get('out_card', 0)
    # king_card = input.get('king_card', 0)
    # fei_king = input.get('fei_king', 0)
    # discards = input.get('discards',[])
    round = input.get('round', 0)


    remain_num = input.get('remain_num', 0)
    isHu=input.get('isHu',False)
    seat_id = input.get('seat_id', 0)
    out_seat_id = input.get('out_seat_id', 0)

    discards = input.get('discards', [])
    # discards = [discards_old["0"], discards_old["1"], discards_old["2"], discards_old["3"]]

    discards_real = input.get('discards_real', [])
    # discards_real = [discards_real_old["0"], discards_real_old["1"], discards_real_old["2"], discards_real_old["3"]]

    discards_op = input.get('discards_op', [])
    discards_real = input.get('discards_real', [])

    choose_color = input.get('choose_color', [])

    hu_cards = input.get('hu_cards',[])
    # hu_cards = [hu_cards_old["0"], hu_cards_old["1"], hu_cards_old["2"], hu_cards_old["3"]]

    hu_fan = input.get('hu_fan',[])
    # hu_fan = hu_fan_old["hu_fan"]

    hand_cards = user_cards.get('hand_cards', [])
    operate_cards = user_cards.get('operate_cards', [])
    # discards = trandfer_discards(discards, hand_cards)  # 转化弃牌表
    hand_cards.sort()
    [e.sort() for e in operate_cards]
    recommend_op, isHu = sichuanMJ_v2.recommend_op(op_card=out_card, cards=hand_cards, suits=operate_cards, round=round, remain_num=remain_num, discards=discards, discards_real=discards_real, discards_op=discards_op, self_turn=len(hand_cards)%3==2, seat_id=seat_id, choose_color=choose_color, hu_cards=hu_cards, hu_fan=hu_fan, isHu = isHu)
    return recommend_op, isHu

def Recommend_switch_cards_sichuanMJ_v2(input={}):
    user_cards = input.get('user_cards', {})
    hand_cards = user_cards.get('hand_cards', [])
    operate_cards = user_cards.get('operate_cards', [])
    switch_n_cards=input.get('switch_n_cards',3)
    hand_cards.sort()
    recommend_switch_cards=sichuanMJ_v2.recommend_switch_cards(hand_cards=hand_cards,switch_n_cards=switch_n_cards)
    return recommend_switch_cards

def Recommend_choose_color_sichuanMJ_v2(input={}):
    user_cards = input.get('user_cards', {})
    hand_cards = user_cards.get('hand_cards', [])
    operate_cards = user_cards.get('operate_cards', [])
    switch_n_cards=input.get('switch_n_cards',3)
    hand_cards.sort()
    recommend_choose_color=sichuanMJ_v2.recommend_choose_color(hand_cards=hand_cards,switch_n_cards=switch_n_cards)
    return recommend_choose_color


def canchi(seat_id, out_seat_id):
    if seat_id == 0:
        if out_seat_id == 3:
            return True
        else:
            return False
    else:
        if seat_id == out_seat_id + 1:
            return True
        else:
            return False


def translate_index_to_hex( i):  # 1-34转换到16进制的card
    """
    将１－３４转化为牌值
    :param i:
    :return:
    """
    if i>=0 and i<=8:
        i=i+1
    if i >= 9 and i <= 17:
        i = i + 8
    elif i >= 18 and i <= 26:
        i = i + 15
    elif i >= 27 and i <= 33:
        i = i + 22
    return i


def test_time():
    # 耗时测试
    handCardsSet = []
    # 随机生成１００手手牌信息
    for i in range(100):
        handCards = []
        for j in random.sample(range(1, 137), 14):
            handCards.append(translate_index_to_hex(j / 4))
        handCardsSet.append(handCards)
    f = open('log.txt', 'w')

    i = 0  # 几率手数
    start = time.time()
    f.write('总开始时间:' + str(start) + '\n')
    for handCards in handCardsSet:
        request['user_cards']['hand_cards'] = handCards
        s = time.time()
        out_card = RecommendCard_shangraoMJ_v1(request)
        e = time.time()
        u = e - s
        i = i + 1
        f.write('第' + str(i) + '手:\n')
        f.write('开始时间:' + str(s) + '\n')
        f.write('手牌:' + str(handCards) + '\n')
        f.write('出牌:' + str(out_card) + '\n')
        f.write('结束时间：' + str(e) + '\n')
        f.write('耗时：' + str(u) + '\n\n')

    end = time.time()
    use_time = end - start
    avg_time = float(use_time) / 100
    f.write('总结束时间：' + str(end) + '\n')
    f.write('总耗时：' + str(use_time) + '\n')
    f.write('平均每手决策时间：' + str(avg_time) + '\n')
    f.write('平均每分钟决策次数：' + str(60.0 / avg_time) + '\n')


def request_interface():
    #
    import json

    # s = requests.session()
    # s.keep_alive = False
    url = "http://http://172.81.238.92:8085/shangraoMJ/v2/outcard"
    headers = {'Connection': 'close','Content-Type': 'application/json;charset=UTF-8'}
    requests.adapters.DEFAULT_RETRIES = 5
    # headers = {'Connection': 'close' }
    request_param = {"discards":[[52,54,50,41,7,39,2,17,9,8,3,34,23,7,6,37,54,6,18],[41,25,35,53,8,21,19,4,53,41,6,1,51,39,20,36,33,22],[50,54,51,9,33,25,20,36,18,52,40,21,24,2,39,4,55,55,33,54,7],[50,52,51,17,55,5,21,52,41,39,3,40,17,9,36,22,23,50,49,55]],"discards_op":[[],[[33,34,35]],[[49,49,49],[25,24,23]],[]],"fei_king":0,"isHu":False,"king_card":18,"out_card":7,"out_seat_id":2,"remain_num":5,"round":20,"seat_id":3,"user_cards":{"hand_cards":[1,2,5,7,8,9,23,24,38,19,5,25,36],"operate_cards":[]}}
    ti=time.time()
    response = requests.post(url, data=json.dumps(request_param), headers=headers)
    tj=time.time()
    print('time=',tj-ti)


if __name__ == '__main__':
    request = {"catch_card":2,"discards":[[2]],"discards_op":[],"fei_king":0,"king_card":0x31,"round":13,"seat_id":1,"user_cards":{"hand_cards":[1,3,6,6,0x31],"operate_cards":[[1,1,1],[1,1,1],[1,2,3]]}}
    start = time.time()
    print ("out_card=", RecommendCard_shangraoMJ_v4(request))
    end = time.time()
    print()
    print("time=", end - start)

    request_op = {"discards":[[7,51,54,2,7,25,25,19,18,40,38,22,5],[55,54,9,51,21,17,49,23,7,6,54,17,4],[55,33,17,8,25,2,23,8,37,18,5,2,37,41],[55,50,24,49,40,25,22,36,49,9,2,33,50,21]],"discards_op":[[[23,22,21],[3,3,3]],[[35,36,37],[19,19,19],[40,39,38]],[[53,53,53],[24,23,22]],[[20,20,20],[39,39,39]]],"fei_king":0,"isHu":True,"king_card":52,"out_card":3,"out_seat_id":0,"remain_num":28,"round":15,"seat_id":0,"user_cards":{"hand_cards":[4,5,6,24,24,7,5,3],"operate_cards":[[23,22,21],[3,3,3]]}}


    startop = time.time()
    # print ("op=",RecommendOprate_shangraoMJ_v2(request_op))
    endop = time.time()
    print()
    print("time=", endop - startop)
