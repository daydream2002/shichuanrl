# -*- coding: utf-8 -*-
# import interface_v7.feature_extract_v7 as feature_extract
# import interface_v7.model as model
# import interface_v9.model_res_tf as model2
# import interface_v9.model_lstm_keras as model_lstm

# import interface_v9.num_waiting as nw
# import interface_v9.feature_extract_v9 as feature_extract_v9
# import interface_v7.model_new_res as model3
# import interface_v7.feature_extract_sr_pic as fesp
# import ren.pinghu as ren


# import twmj_KF.twmj_KF_v1 as KF_v1
# import twmj_KF.twmj_KF_v2 as KF_v2

# import shangraoMJ.shangraoMJ_v1 as shangraoMJ_v1
# import shangraoMJ.shangraoMJ_v2 as shangraoMJ_v2
from sichuanMJ.bak import SCMJ as sichuanMJ_v1

# ph =pinghu.pinghu()
# import interface_v9.feature_extract_v9 as feature_extract9
# import interface_v9.model_res_tf as model9
"""
20180823 modify the example, add the discarded information
20180906 replace model_res_tf to interface_v7/model_res_tf which was trained with V1 data
20181006 fix waiting chow/pong/gong problem
"""

import time

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


# 处理推荐操作
def RecommandOprate(handCards, actions, kingCard, outCard,
                    fei_king, canchi=False, canpeng=False, isSelfTurn=False):
    if kingCard not in handCards:
        feature_op = feature_extract.calculate3(handCards, actions, outCard)
        feature_op.append(8)
        op = model.model_choose(ismyturn=isSelfTurn, isking=False, list=feature_op, hand_cards=handCards,
                                op_card=outCard, canchi=canchi,
                                sess=model.sess3, L=model.L3)
    else:
        feature_op = feature_extract.calculate4(handCards, actions, outCard, kingCard, fei_king,
                                                handCards.count(kingCard))
        feature_op.append(8)
        op = model.model_choose(ismyturn=isSelfTurn, isking=True, list=feature_op, hand_cards=handCards,
                                op_card=outCard, canchi=canchi,
                                sess=model.sess4, L=model.L4)

    return op


# 根据手牌、副露推荐出牌
def RecommandCard(handCards, kingCard, actions, fei_king, isSelfTurn=True):
    opCard = 0x0
    if kingCard in handCards:
        king_num = handCards.count(kingCard)
        feature_king = feature_extract.calculate6(handCards, actions, kingCard, fei_king, king_num, 0)
        feature_king.append(35)
        temp = model.model_choose(ismyturn=isSelfTurn, isking=True, list=feature_king, hand_cards=handCards,
                                  op_card=255, canchi=None,
                                  sess=model.sess2, L=model.L2)
        opCard = (int(temp / 10)) * 16 + (temp % 10)
        # opCard = trans10to16(temp)
    else:
        feature_noking = feature_extract.calculate5(handCards, actions)
        feature_noking.append(35)
        temp = model.model_choose(ismyturn=isSelfTurn, isking=False, list=feature_noking, hand_cards=handCards,
                                  op_card=255, canchi=None,
                                  sess=model.sess1, L=model.L1)
        opCard = (int(temp / 10)) * 16 + (temp % 10)
        # opCard = trans10to16(temp)
    return opCard


# 处理推荐操作v2
def RecommandOprateV2(handCards, actions, kingCard, outCard,
                      fei_king, canchi=False, canpeng=False, isSelfTurn=False):
    if kingCard not in handCards:
        full_handCards = handCards + [outCard]
        feature_op = feature_extract.calculate3(handCards, actions, outCard)
        feature_op.append(8)
        op = model2.model_choose(ismyturn=isSelfTurn, isking=False, list=feature_op, hand_cards=handCards,
                                 op_card=outCard, canchi=canchi,
                                 sess=model2.sess3)

    #       wait_num = [nw.num_common_types(full_handCards, actions), nw.num_wait_types_7(full_handCards, actions),
    #                   nw.num_wait_types_13(full_handCards, actions), nw.num_wait_types_91(full_handCards, actions)]
    #       if wait_num[0] == 1 or wait_num[1] or wait_num[2] == 1:
    #           op = []

    else:
        full_handCards = handCards + [outCard]
        feature_op = feature_extract.calculate4(handCards, actions, outCard, kingCard, fei_king,
                                                handCards.count(kingCard))
        feature_op.append(8)
        op = model2.model_choose(ismyturn=isSelfTurn, isking=True, list=feature_op, hand_cards=handCards,
                                 op_card=outCard, canchi=canchi,
                                 sess=model2.sess4)
    #      wait_num = [nw.num_common_types(full_handCards, actions), nw.num_wait_types_7(full_handCards, actions),
    #                  nw.num_wait_types_13(full_handCards, actions), nw.num_wait_types_91(full_handCards, actions)]
    #      if wait_num[0] == 1 or wait_num[1] or wait_num[2] == 1:
    #          op = []
    return op


# 根据手牌、副露推荐出牌
def RecommandCardV2(handCards, kingCard, actions, fei_king, isSelfTurn=True):
    """
    canchi is invalid
    :param handCards:
    :param kingCard:
    :param actions:
    :param fei_king:
    :param discards:
    :param round:
    :param isSelfTurn:
    :return:
    """
    opCard = 0x0
    if kingCard in handCards:
        king_num = handCards.count(kingCard)
        feature_king = feature_extract.calculate6(handCards, actions, kingCard, fei_king, king_num, 0)
        feature_king.append(35)
        temp = model2.model_choose(ismyturn=isSelfTurn, isking=True, list=feature_king, hand_cards=handCards,
                                   op_card=255, canchi=None,
                                   sess=model2.sess2)
        opCard = (int(temp / 10)) * 16 + (temp % 10)
        # opCard = trans10to16(temp)
    else:
        feature_noking = feature_extract.calculate5(handCards, actions)
        feature_noking.append(35)
        temp = model2.model_choose(ismyturn=isSelfTurn, isking=False, list=feature_noking, hand_cards=handCards,
                                   op_card=255, canchi=None,
                                   sess=model2.sess1)
        opCard = (int(temp / 10)) * 16 + (temp % 10)
        # opCard = trans10to16(temp)
    return opCard


def RecommandCardV3(dict, isSelfTurn=True, handCards=None):
    print("orgin message")
    print(dict)
    feature_pic = fesp.targetCal(dict)
    temp = model3.model_choose(ismyturn=isSelfTurn, hand_cards=handCards, list=feature_pic,
                               sess=model3.sess1)
    opCard = (int(temp / 10)) * 16 + (temp % 10)

    # opCard = trans10to16(temp)
    return opCard


def RecommandCard_lstm(handCards, actions, round, discards, king_card, fei_king, input_pre, output_pre):
    king_num = handCards.count(king_card)
    feature_king = feature_extract_v9.calculate_king_sys(handCards, actions, round, discards, king_card, fei_king,
                                                         king_num)
    # feature_king.append(35)
    # temp = 1-9 11-19 21-29 31-37
    if round == 0:
        input_pre = []
        output_pre = []

    temp = model_lstm.model_choose(ismyturn=True, isking=True, list=feature_king, list_pre=input_pre,
                                   output_lstm=output_pre,
                                   hand_cards=handCards, op_card=255, canchi=None, sess=model_lstm.sess_lstm)
    opCard = (int(temp / 10)) * 16 + (temp % 10)

    output_pre.append(translate1_37to0_33(temp))
    input_pre.append(feature_king)
    return opCard


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
    return recommend_op, isHu


def RecommendCard_shangraoMJ_v2(input={}):
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
    recommendCard = shangraoMJ_v2.recommend_card(cards=hand_cards, suits=operate_cards, king_card=king_card,
                                                 discards=discards, discards_op=discards_op, fei_king=fei_king)
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

    hand_cards = user_cards.get('hand_cards', [])
    operate_cards = user_cards.get('operate_cards', [])
    # discards = trandfer_discards(discards, hand_cards)  # 转化弃牌表
    hand_cards.sort()
    [e.sort() for e in operate_cards]
    recommend_op, isHu = shangraoMJ_v2.recommend_op(op_card=out_card, cards=hand_cards, suits=operate_cards,
                                              king_card=king_card, discards=discards, discards_op=discards_op,
                                              canchi=canchi(seat_id, out_seat_id), self_turn=(len(hand_cards) % 3 == 2),
                                              fei_king=fei_king,isHu=isHu)
    return recommend_op, isHu


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
    discards_real=input.get('discards_real',[])
    discards_op = input.get('discards_op', [])
    remain_num = input.get('remain_num', 0)

    round = input.get('round', [])
    hand_cards = user_cards.get('hand_cards', [])
    operate_cards = user_cards.get('operate_cards', [])
    choose_color=input.get('choose_color')
    hu_cards=input.get('hu_cards')
    hu_fan=input.get('hu_fan')
    hand_cards.sort()
    [e.sort() for e in operate_cards]
    recommendCard = sichuanMJ_v1.recommend_card(cards=hand_cards, suits=operate_cards, round=round, remain_num=remain_num, discards=discards, discards_real=discards_real, discards_op=discards_op,
                   seat_id=seat_id,choose_color=choose_color,hu_cards=hu_cards,hu_fan=hu_fan)
    return recommendCard


def RecommendOprate_sichuanMJ_v1(input={}):
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

    hand_cards = user_cards.get('hand_cards', [])
    operate_cards = user_cards.get('operate_cards', [])
    # discards = trandfer_discards(discards, hand_cards)  # 转化弃牌表
    hand_cards.sort()
    [e.sort() for e in operate_cards]
    recommend_op, isHu = sichuanMJ_v1.recommend_op(op_card=out_card, cards=hand_cards, suits=operate_cards,
                                              king_card=king_card, discards=discards, discards_op=discards_op,
                                              canchi=canchi(seat_id, out_seat_id), self_turn=(len(hand_cards) % 3 == 2),
                                              fei_king=fei_king,isHu=isHu)
    return recommend_op, isHu


def trandfer_discards(discards, handcards):
    discards_map = {
        0x01: 0,
        0x02: 1,
        0x03: 2,
        0x04: 3,
        0x05: 4,
        0x06: 5,
        0x07: 6,
        0x08: 7,
        0x09: 8,
        0x11: 9,
        0x12: 10,
        0x13: 11,
        0x14: 12,
        0x15: 13,
        0x16: 14,
        0x17: 15,
        0x18: 16,
        0x19: 17,
        0x21: 18,
        0x22: 19,
        0x23: 20,
        0x24: 21,
        0x25: 22,
        0x26: 23,
        0x27: 24,
        0x28: 25,
        0x29: 26,
        0x31: 27,
        0x32: 28,
        0x33: 29,
        0x34: 30,
        0x35: 31,
        0x36: 32,
        0x37: 33,
    }

    discards_list = [0] * 34

    for item in discards:
        discards_list[discards_map[item]] += 1
    for item in handcards:
        discards_list[discards_map[item]] += 1

    return discards_list


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


def canpeng(out_card, hand_cards):
    return hand_cards.count(out_card) >= 2

('all_terminal', [[[], [[7, 8, 9], [21, 22, 23]], [], [[18, 20]], 0, [1]],
                  [[], [[7, 8, 9]], [], [[20, 21], [22, 23]], 1, [1, 18]], [[], [[7, 8, 9]], [], [[20, 22], [21, 23]], 1, [1, 18]], [[], [[7, 8, 9], [20, 21, 22]], [], [], 1, [1, 18, 23]]])


('all_terminal', [[[], [[7, 8, 9], [18, 19, 20], [21, 22, 23]], [], [], 0, []],
                  [[], [[7, 8, 9], [19, 20, 21]], [], [[22, 23]], 1, [18]],
                  [[], [[7, 8, 9], [20, 21, 22]], [], [[18, 19]], 1, [23]]])

def recommendCard(request={}):

        user_cards = request.get('user_cards', {})
        catch_card = request.get('catch_card', 0)
        # king_card = request.get_json().get('king_card', 0)
        # fei_king = request.get_json().get('fei_king', 0)
        remain_num = request.get('remain_num', 0)
        seat_id = request.get('seat_id', 0)
        discards_real = request.get('discards_real', [])
        discards = request.get('discards', [])
        discards_op = request.get('discards_op', [])
        round = request.get('round', 0)
        hand_cards = user_cards.get('hand_cards', [])
        operate_cards = user_cards.get('operate_cards', [])

        # if catch_card != 0:
        #     hand_cards.append(catch_card)

        # 转化弃牌表
        # discards = trandfer_discards(discards, hand_cards)
        hand_cards.sort()
        [e.sort() for e in operate_cards]
        out_card = RecommendCard_KF_v1(hand_cards, operate_cards, round, remain_num, discards,
                                       discards_op)
        return out_card


if __name__ == '__main__':
    request = {"catch_card":1,"discards":[],"discards_op":[[],[[23,24,25]],[[7,8,9]],[]],"discards_real":[],"remain_num":49,"round":6,"seat_id":0,"user_cards":{"hand_cards":[1,1,1,5,5,5,7,7,7,0x11,0x12,0x14,0x16,0x18],"operate_cards":[]},"choose_color":[],"hu_cards":[[],[],[],[]],"hu_fan":[[],[],[],[]]}
    start = time.time()
    print ("out_card=", RecommendCard_sichuanMJ_v1(request))
    end = time.time()
    print()
    print("time=", end - start)

    request_op = {"discards":[[7,51,54,2,7,25,25,19,18,40,38,22,5],[55,54,9,51,21,17,49,23,7,6,54,17,4],[55,33,17,8,25,2,23,8,37,18,5,2,37,41],[55,50,24,49,40,25,22,36,49,9,2,33,50,21]],"discards_op":[[[23,22,21],[3,3,3]],[[35,36,37],[19,19,19],[40,39,38]],[[53,53,53],[24,23,22]],[[20,20,20],[39,39,39]]],"fei_king":0,"isHu":True,"king_card":52,"out_card":3,"out_seat_id":0,"remain_num":28,"round":15,"seat_id":0,"user_cards":{"hand_cards":[4,5,6,24,24,7,5,3],"operate_cards":[[23,22,21],[3,3,3]]}}


    startop = time.time()
    # print ("op=",RecommendOprate_shangraoMJ_v2(request_op))
    endop = time.time()
    print()
    print("time=", endop - startop)
