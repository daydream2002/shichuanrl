# -*- coding: utf-8 -*-
import time

START = time.time()
import random
# import requests
from mah_tool.training_recommend import sichuanMJ_v2_3_4 as sichuanMJ_v1
import mah_tool.tool2 as tool2
# from mah_tool.url_recommend import trans_discards, trans_discards_op


# from interface.sichuanMJ_v1 import test1 as t1
# from interface.sichuanMJ_v1 import test2 as t2
# from interface.sichuanMJ_v1 import test3 as t3
# from interface.sichuanMJ_v1 import test4 as t4
# from interface.sichuanMJ_v1 import test5 as t5
# from interface.sichuanMJ_v1 import test6 as t6
# from interface.sichuanMJ_v1 import test7 as t7

# 把discards转换为url json格式
def trans_discards(player_discards_display):
    discards = [[], [], [], []]
    if not player_discards_display or len(player_discards_display) == 0:
        return discards
    discards[0] = tool2.list10_to_16_2(player_discards_display[0])
    discards[1] = tool2.list10_to_16_2(player_discards_display[1])
    discards[2] = tool2.list10_to_16_2(player_discards_display[2])
    discards[3] = tool2.list10_to_16_2(player_discards_display[3])
    return discards

# 把player_fulu 转换为 discard_op
def trans_discards_op(player_fulu):
    discards_op = [[], [], [], []]
    discards_op[0] = tool2.fulu_translate(player_fulu[0])
    discards_op[1] = tool2.fulu_translate(player_fulu[1])
    discards_op[2] = tool2.fulu_translate(player_fulu[2])
    discards_op[3] = tool2.fulu_translate(player_fulu[3])
    return discards_op

def Recommend_switch_cards_sichuanMJ_v2(input={}):
    user_cards = input.get("user_cards", {})
    hand_cards = user_cards.get("hand_cards", [])
    switch_n_cards = input.get("switch_n_cards", 3)
    start = time.time()
    recommend_switch_cards = sichuanMJ_v1.recommend_switch_cards(hand_cards=hand_cards, switch_n_cards=switch_n_cards)
    end = time.time()
    if end - start >= 2:
        print("switch_cards_overtime:%s", input)
    return recommend_switch_cards


def Recommend_choose_color_sichuanMJ_v2(input={}):
    user_cards = input.get("user_cards", {})
    hand_cards = user_cards.get("hand_cards", [])
    switch_n_cards = input.get("switch_n_cards", 3)
    hand_cards.sort()
    start = time.time()
    recommend_choose_color = sichuanMJ_v1.recommend_choose_color(hand_cards=hand_cards, switch_n_cards=switch_n_cards)
    end = time.time()
    if end - start >= 2:
        print("choose_color_overtime:%s", input)
    return recommend_choose_color


# def Test(input={}, index=1):
#     seat_id = input.get('seat_id', 0)
#     dealer_id = input.get('dealer_id', 0)
#     catch_card = input.get('catch_card', 0)
#     user_cards = input.get('user_cards', {})
#     hand_cards = user_cards.get('hand_cards', [])
#     operate_cards = user_cards.get('operate_cards', [])
#     discards = input.get('discards', [])
#     discards_real = input.get('discards_real', [])
#     discards_op = input.get('discards_op', [])
#     eff_cards = input.get('eff_cards', [])  # 玩家有效牌？
#     hu_types = input.get('hu_types', [])
#     hu_cards = input.get('hu_cards', [])
#     colors = input.get('colors', [])
#     max_fans = input.get('max_fans', [])
#     round = input.get('round', [])
#     remain_num = input.get('remain_num', 55)
#     wall = input.get('wall', [])
#     hands = input.get('hands', [])
#
#     # colors_type_dict = {'0': 0x00, '1': 0x10, '2': 0x20}
#     # c_type = colors_type_dict[str(input.get('colors')[seat_id])]
#
#     # if hu_cards != [[], [], [], []]:
#     #     print("discards", discards)
#     #     print("discardsreal", discards_real)
#     #     print("discardsop", discards_op)
#     #     print("hu_card",hu_cards)
#
#     hand_cards.sort()  # 对手牌排序
#     [e.sort() for e in operate_cards]  # 对自己的副露排序
#
#     # 2.6版本之后，新增colors 和 hu_card    ->使用1234
#     # 原版使用 5
#     # waa特别版暂时无位置
#     recommendCard = hand_cards[-1]
#     if index == 1:
#         recommendCard = t1.recommend_card(cards=hand_cards, suits=operate_cards, discards=discards,
#                                           discards_op=discards_op, round=round, seat_id=seat_id, c_type=colors[seat_id],
#                                           colors=colors, hu_card=hu_cards, remain_num=remain_num)
#     elif index == 2:
#         recommendCard = t2.recommend_card(cards=hand_cards, suits=operate_cards, discards=discards,
#                                           discards_op=discards_op, round=round, seat_id=seat_id, c_type=colors[seat_id],
#                                           colors=colors, hu_card=hu_cards, remain_num=remain_num)
#     elif index == 3:
#         recommendCard = t3.recommend_card(cards=hand_cards, suits=operate_cards, discards=discards,
#                                           discards_op=discards_op, round=round, seat_id=seat_id, c_type=colors[seat_id],
#                                           colors=colors, hu_card=hu_cards, remain_num=remain_num)
#     elif index == 4:
#         recommendCard = t4.recommend_card(cards=hand_cards, suits=operate_cards, discards=discards,
#                                           discards_op=discards_op, round=round, seat_id=seat_id, c_type=colors[seat_id],
#                                           colors=colors, hu_card=hu_cards, remain_num=remain_num)
#     elif index == 5:
#         recommendCard = t5.recommend_card(cards=hand_cards, suits=operate_cards, discards=discards,
#                                           discards_op=discards_op, round=round, seat_id=seat_id, c_type=colors[seat_id],
#                                           colors=colors, hu_card=hu_cards, remain_num=remain_num)
#     elif index == 6:
#         recommendCard = t6.recommend_card(cards=hand_cards, suits=operate_cards, discards=discards,
#                                           discards_op=discards_op, round=round, seat_id=seat_id, c_type=colors[seat_id],
#                                           colors=colors, hu_card=hu_cards, remain_num=remain_num)
#     # elif index == 7:
#     #     recommendCard = t7.recommend_card(cards=hand_cards, suits=operate_cards, discards=discards,
#     #                                       discards_op=discards_op, round=round, seat_id=seat_id, c_type=colors[seat_id],
#     #                                       colors=colors, hu_card=hu_cards, remain_num=remain_num)
#     elif index == 7:
#         recommendCard = t7.recommend_card(cards=hand_cards, suits=operate_cards, king_card=None,
#                                           discards=discards, discards_op=discards_op, fei_king=0,
#                                           remain_num=remain_num, round=round, seat_id=seat_id,
#                                           self_lack=colors[seat_id])
#     return recommendCard


def json_url(seat_id, dealer_id, catch_card, handcards, fulu, player_discards_display, player_fulu, eff_cards,
             hu_types, hu_cards, colors, max_fans, round, remain_num, card_library=[], hands=[[], [], [], []],
             discards_real=[[], [], [], []]):
    json_data = {}
    json_data["seat_id"] = seat_id
    json_data["dealer_id"] = dealer_id
    json_data["catch_card"] = tool2.f10_to_16(catch_card)

    user_cards = {}
    user_cards["hand_cards"] = tool2.list10_to_16(handcards)
    user_cards["operate_cards"] = tool2.fulu_translate(fulu)
    json_data["user_cards"] = user_cards

    json_data["discards"] = trans_discards(player_discards_display)
    json_data["discards_real"] = trans_discards(discards_real)

    json_data["discards_op"] = trans_discards_op(player_fulu)
    json_data["eff_cards"] = tool2.list10_to_16(eff_cards)

    json_data["hu_types"] = hu_types
    json_data["hu_cards"] = trans_discards(hu_cards)
    json_data["colors"] = colors
    json_data["max_fans"] = max_fans

    json_data["round"] = round
    json_data["remain_num"] = remain_num

    json_data["wall"] = tool2.list10_to_16_2(card_library)
    json_data["hands"] = trans_discards(hands)
    return json_data


##根据场上形式，推荐出牌
def RecommendCard_sichuanMJ_v1(input={}):
    seat_id = input.get('seat_id', 0)
    dealer_id = input.get('dealer_id', 0)
    catch_card = input.get('catch_card', 0)
    user_cards = input.get('user_cards', {})
    hand_cards = user_cards.get('hand_cards', [])
    operate_cards = user_cards.get('operate_cards', [])
    discards = input.get('discards', [])
    discards_real = input.get('discards_real', [])
    discards_op = input.get('discards_op', [])
    eff_cards = input.get('eff_cards', [])  # 玩家有效牌？
    hu_types = input.get('hu_types', [])
    hu_cards = input.get('hu_cards', [])
    colors = input.get('colors', [])
    max_fans = input.get('max_fans', [])
    round = input.get('round', [])
    remain_num = input.get('remain_num', 55)
    wall = input.get('wall', [])
    hands = input.get('hands', [])

    hand_cards.sort()  # 对手牌排序
    [e.sort() for e in operate_cards]  # 对自己的副露排序
    # print("my",time.time()-START)
    recommendCard = sichuanMJ_v1.recommend_card(cards=hand_cards, suits=operate_cards, king_card=None,
                                                discards=discards, discards_op=discards_op, fei_king=0,
                                                remain_num=remain_num, round=round, seat_id=seat_id,
                                                self_lack=colors[seat_id])
    return recommendCard


# def Test_op(input={}, index=1):
#     seat_id = input.get('seat_id', 0)
#     dealer_id = input.get('dealer_id', 0)
#     out_seat_id = input.get('out_seat_id', 0)
#     out_card = input.get('out_card', 0)
#     allow_op = input.get('allow_op', [])
#     user_cards = input.get('user_cards', {})
#     hand_cards = user_cards.get('hand_cards', [])
#     operate_cards = user_cards.get('operate_cards', [])
#     discards = input.get('discards', [])
#     discards_real = input.get('discards_real', [])
#     discards_op = input.get('discards_op', [])
#     eff_cards = input.get('eff_cards', [])  # 有效牌？
#     hu_types = input.get('hu_types', [])
#     hu_cards = input.get('hu_cards', [])
#     colors = input.get('colors', [])
#     max_fans = input.get('max_fans', [])
#     round = input.get('round', 0)
#     remain_num = input.get('remain_num', 55)
#     wall = input.get('wall', [])
#     hands = input.get('hands', [])  # 所有玩家手牌
#     isHu = input.get('isHu', False)
#
#     # colors_type_dict = {"0": 0x00, "1": 0x10, "2": 0x20}
#     # c_type = colors_type_dict[str(input.get("colors")[seat_id])]
#     # win_player_num = 0
#     # for index_p in range(len(hu_cards)):
#     #     if index_p == seat_id:
#     #         continue
#     #     if hu_cards[index_p]:
#     #         win_player_num += 1
#
#     hand_cards.sort()
#     [e.sort() for e in operate_cards]
#     recommend_op = []
#     if index == 1:
#         recommend_op, isHu = t1.recommend_op(op_card=out_card, cards=hand_cards, suits=operate_cards,
#                                              discards=discards, discards_op=discards_op,
#                                              self_turn=(len(hand_cards) % 3 == 2), isHu=isHu, round=round,
#                                              remain_num=remain_num, op_map=allow_op, c_type=colors[seat_id],
#                                              colors=colors, hu_card=hu_cards)
#     elif index == 2:
#         recommend_op, isHu = t2.recommend_op(op_card=out_card, cards=hand_cards, suits=operate_cards,
#                                              discards=discards, discards_op=discards_op,
#                                              self_turn=(len(hand_cards) % 3 == 2), isHu=isHu, round=round,
#                                              remain_num=remain_num, op_map=allow_op, c_type=colors[seat_id],
#                                              colors=colors, hu_card=hu_cards)
#     elif index == 3:
#         recommend_op, isHu = t3.recommend_op(op_card=out_card, cards=hand_cards, suits=operate_cards,
#                                              discards=discards, discards_op=discards_op,
#                                              self_turn=(len(hand_cards) % 3 == 2), isHu=isHu, round=round,
#                                              remain_num=remain_num, op_map=allow_op, c_type=colors[seat_id],
#                                              colors=colors, hu_card=hu_cards)
#     elif index == 4:
#         recommend_op, isHu = t4.recommend_op(op_card=out_card, cards=hand_cards, suits=operate_cards,
#                                              discards=discards, discards_op=discards_op,
#                                              self_turn=(len(hand_cards) % 3 == 2), isHu=isHu, round=round,
#                                              remain_num=remain_num, op_map=allow_op, c_type=colors[seat_id],
#                                              colors=colors, hu_card=hu_cards)
#     elif index == 5:
#         recommend_op, isHu = t5.recommend_op(op_card=out_card, cards=hand_cards, suits=operate_cards,
#                                              discards=discards, discards_op=discards_op,
#                                              self_turn=(len(hand_cards) % 3 == 2), isHu=isHu, round=round,
#                                              remain_num=remain_num, op_map=allow_op, c_type=colors[seat_id],
#                                              colors=colors, hu_card=hu_cards)
#     elif index == 6:
#         recommend_op, isHu = t6.recommend_op(op_card=out_card, cards=hand_cards, suits=operate_cards,
#                                              discards=discards, discards_op=discards_op,
#                                              self_turn=(len(hand_cards) % 3 == 2), isHu=isHu, round=round,
#                                              remain_num=remain_num, op_map=allow_op, c_type=colors[seat_id],
#                                              colors=colors, hu_card=hu_cards)
#     # elif index == 7:
#     #     recommend_op, isHu = t7.recommend_op(op_card=out_card, cards=hand_cards, suits=operate_cards,
#     #                                          discards=discards, discards_op=discards_op,
#     #                                          self_turn=(len(hand_cards) % 3 == 2), isHu=isHu, round=round,
#     #                                          remain_num=remain_num, op_map=allow_op, c_type=colors[seat_id],
#     #                                          colors=colors, hu_card=hu_cards)
#     elif index == 7:
#         recommend_op, isHu = t7.recommend_op(op_card=out_card, cards=hand_cards, suits=operate_cards,
#                                              discards=discards, discards_op=discards_op,
#                                              self_turn=(len(hand_cards) % 3 == 2), isHu=isHu, round=round,
#                                              remain_num=remain_num, op_map=allow_op)
#
#     return recommend_op, isHu


def RecommendOperate_sichuanMJ_v1(input={}):
    seat_id = input.get('seat_id', 0)
    dealer_id = input.get('dealer_id', 0)
    out_seat_id = input.get('out_seat_id', 0)
    out_card = input.get('out_card', 0)
    allow_op = input.get('allow_op', [])
    user_cards = input.get('user_cards', {})
    hand_cards = user_cards.get('hand_cards', [])
    operate_cards = user_cards.get('operate_cards', [])
    discards = input.get('discards', [])
    discards_real = input.get('discards_real', [])
    discards_op = input.get('discards_op', [])
    eff_cards = input.get('eff_cards', [])  # 有效牌？
    hu_types = input.get('hu_types', [])
    hu_cards = input.get('hu_cards', [])
    colors = input.get('colors', [])
    max_fans = input.get('max_fans', [])
    round = input.get('round', 0)
    remain_num = input.get('remain_num', 55)
    wall = input.get('wall', [])
    hands = input.get('hands', [])  # 所有玩家手牌
    isHu = input.get('isHu', False)

    hand_cards.sort()
    [e.sort() for e in operate_cards]
    recommend_op, isHu = sichuanMJ_v1.recommend_op(op_card=out_card, cards=hand_cards, suits=operate_cards,
                                                   discards=discards, discards_op=discards_op,
                                                   self_turn=(len(hand_cards) % 3 == 2), isHu=isHu, round=round,
                                                   remain_num=remain_num, op_map=allow_op)
    return recommend_op, isHu


##得到 弃牌集合
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


# 1-33转换到16进制的card
def translate_index_to_hex(i):
    """
    将１－３４转化为牌值
    :param i:
    :return:
    """
    if i >= 0 and i <= 8:
        i = i + 1
    elif i >= 9 and i <= 17:
        i = i + 8
    elif i >= 18 and i <= 26:
        i = i + 15
    elif i >= 27 and i <= 33:
        i = i + 22
    return i


# def test_time():
#     # 耗时测试
#     handCardsSet = []
#     # 随机生成１００手手牌信息
#     times = 1000
#     for i in range(times):
#         handCards = []
#         for j in random.sample(range(72), 14):
#             handCards.append(translate_index_to_hex(j // 4))
#         handCardsSet.append(handCards)
#     f = open('log.txt', 'w', encoding='utf-8')
#
#     i = 0  # 几率手数
#     start = time.time()
#     f.write('总开始时间:' + str(start) + '\n')
#
#     for handCards in handCardsSet:
#         request['user_cards']['hand_cards'] = handCards
#         s = time.time()
#         out_card = Test(request, 2)
#         # print("out_card:", out_card, "\n")
#         e = time.time()
#         u = e - s
#         if u > 1:
#             print(request)
#         i = i + 1
#         f.write('第' + str(i) + '手:\n')
#         f.write('开始时间:' + str(s) + '\n')
#         f.write('手牌:' + str(handCards) + '\n')
#         f.write('出牌:' + str(out_card) + '\n')
#         f.write('结束时间：' + str(e) + '\n')
#         f.write('耗时：' + str(u) + '\n\n')
#
#     end = time.time()
#     use_time = end - start
#     avg_time = float(use_time) / times
#     f.write('总结束时间：' + str(end) + '\n')
#     f.write('总耗时：' + str(use_time) + '\n')
#     f.write('平均每手决策时间：' + str(avg_time) + '\n')
#     f.write('平均每分钟决策次数：' + str(60.0 / avg_time) + '\n')


# def request_interface():
#     #
#     import json
#
#     # s = requests.session()
#     # s.keep_alive = False
#     url = "http://http://172.81.238.92:8085/shangraoMJ/v2/outcard"
#     headers = {'Connection': 'close', 'Content-Type': 'application/json;charset=UTF-8'}
#     requests.adapters.DEFAULT_RETRIES = 5
#     # headers = {'Connection': 'close' }
#     request_param = {"discards": [[52, 54, 50, 41, 7, 39, 2, 17, 9, 8, 3, 34, 23, 7, 6, 37, 54, 6, 18],
#                                   [41, 25, 35, 53, 8, 21, 19, 4, 53, 41, 6, 1, 51, 39, 20, 36, 33, 22],
#                                   [50, 54, 51, 9, 33, 25, 20, 36, 18, 52, 40, 21, 24, 2, 39, 4, 55, 55, 33, 54, 7],
#                                   [50, 52, 51, 17, 55, 5, 21, 52, 41, 39, 3, 40, 17, 9, 36, 22, 23, 50, 49, 55]],
#                      "discards_op": [[], [[33, 34, 35]], [[49, 49, 49], [25, 24, 23]], []], "fei_king": 0,
#                      "isHu": False, "king_card": 18, "out_card": 7, "out_seat_id": 2, "remain_num": 5, "round": 20,
#                      "seat_id": 3,
#                      "user_cards": {"hand_cards": [1, 2, 5, 7, 8, 9, 23, 24, 38, 19, 5, 25, 36], "operate_cards": []}}
#     ti = time.time()
#     response = requests.post(url, data=json.dumps(request_param), headers=headers)
#     tj = time.time()
#     print('time=', tj - ti)


if __name__ == '__main__':
    request = {'seat_id': 1, 'dealer_id': 0, 'catch_card': 24,
               "user_cards": {"hand_cards": [5, 17, 17, 18, 20, 20, 21, 21, 23, 24, 25],
                              "operate_cards": [[19, 19, 19]]},
               "discards": [[3, 33, 4, 25, 8, 24, 21, 34, 4, 35, 40, 18],
                            [39, 25, 35, 40, 23, 35, 17, 19, 20, 25, 33, 18],
                            [9, 35, 41, 24, 17, 41, 23, 41, 20, 9, 4, 24, 21], [33, 40, 38, 9, 3, 41, 5, 5, 9, 23]],
               "discards_real": [[], [], [], []],
               "discards_op": [[[22, 22, 22, 22], [36, 36, 36, 36], [37, 37, 37]], [[8, 8, 8], [1, 1, 1, 1]],
                               [[34, 34, 34]], [[19, 19, 19]]],
               'eff_cards': [], 'hu_types': [-1, -1, -1, -1], 'hu_cards': [[], [], [], []], 'colors': [2, 2, 2, 2],
               'max_fans': [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], 'round': 2,
               'remain_num': 60,
               'wall': [2, 1, 41, 1, 17, 19, 33, 1, 17, 24, 41, 9, 17, 24, 36, 23, 40, 33, 20, 34, 25, 9, 5, 38, 38, 19,
                        8, 39, 8, 4, 18, 23, 21, 35, 37, 19, 38, 9, 36, 2, 7, 8, 7, 33, 21, 22, 18, 17, 20, 37, 18, 25,
                        39, 6], 'hands': [[1, 3, 4, 6, 8, 34, 37, 39, 40, 41, 35, 21, 22],
                                          [4, 5, 6, 7, 7, 18, 20, 20, 21, 23, 23, 24, 25, 24],
                                          [3, 3, 19, 22, 33, 35, 36, 37, 39, 40, 34, 36, 41],
                                          [2, 4, 5, 5, 9, 22, 34, 35, 38, 40, 2, 3, 6]]}

    start = time.time()
    print("out_card=", RecommendCard_sichuanMJ_v1(request))
    # print("out_card=", Test(request,2))
    end = time.time()
    print()
    print("time=", end - start)

    # test_time()

    request_op = {"seat_id": 0, "dealer_id": 1, "out_seat_id": 2, "out_card": 22, "allow_op": [0, 4],
                  "user_cards": {"hand_cards": [1, 3, 2, 5, 5, 6, 6, 18, 17, 17, 18, 19, 22, 22],
                                 "operate_cards": []},
                  "discards": [[], [], [], []], "discards_real": [[37], [1, 1], [25, 20], [3]],
                  "discards_op": [[], [], [], []], "eff_cards": [3, 3, 21, 22, 23], "hu_types": [-1, -1, -1, -1],
                  "hu_cards": [[], [], [], []], "colors": [2, 2, 2, 2],
                  "max_fans": [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], "round": 2,
                  "remain_num": 50,
                  "wall": [7, 1, 22, 12, 23, 29, 11, 26, 16, 11, 18, 26, 9, 29, 23, 2, 4, 29, 26, 15, 17, 6, 8, 25, 18,
                           5, 18, 13, 9, 21, 29, 8, 28, 19, 1, 6, 23, 18, 27, 27, 17, 7, 21, 15, 25, 28, 7, 9, 2, 28],
                  "hands": [[3, 5, 5, 18, 18, 20, 20, 21, 22, 23, 25, 25, 38],
                            [2, 2, 3, 4, 4, 5, 8, 9, 17, 19, 21, 22, 23],
                            [3, 4, 6, 6, 7, 8, 22, 33, 34, 36, 36, 36, 39],
                            [17, 18, 19, 19, 20, 33, 34, 34, 35, 36, 37, 39, 40]], "isHu": False}

    startop = time.time()
    # print("op=", RecommendOperate_sichuanMJ_v1(request_op))
    # print("op=", Test_op(request_op, 3))
    endop = time.time()
    # print()
    # print("time=", endop - startop)
