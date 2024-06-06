from mah_tool.so_lib import lib_MJ as MJ
from mah_tool import tool2


# import lib_MJ as MJ

def trandfer_discards(discards, discards_op, handcards, wall):
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
    for item in wall:
        left_num[discards_map[item]] -= 1
    return left_num, discards_list


def get_e_cards_map(handCards, suits, discards, discards_op, wall, c_color):
    ph_cs = MJ.pinghu_CS2(cards=handCards, suits=suits)
    qd_cs = MJ.qidui_CS(cards=handCards, suits=suits)
    left_num, _ = trandfer_discards(discards, discards_op, handCards, wall)
    # print(left_num)
    # print(ph_cs)
    # print(qd_cs)
    all_t2 = []
    eff_map = {}
    if ph_cs[0][-2] < qd_cs[-1]:
        for cs in ph_cs:
            for t2 in cs[2] + cs[3]:
                all_t2.append(t2)

        ecards = MJ.get_effective_cards(all_t2)

        for card in ecards:
            if int(card / 16) == c_color:
                continue
            index = MJ.translate16_33(card)
            eff_map[card] = left_num[index]

        return 0, eff_map
    else:
        ecards = qd_cs[1]

        for card in ecards:
            if int(card / 16) == c_color:
                continue
            index = MJ.translate16_33(card)
            eff_map[card] = left_num[index]
        # print(ecards)
        return 1, eff_map


def get_e_cards_list(handCards, suits, discards, discards_op, wall, c_color):
    '''
    获取1维的有效牌 以及牌型 16进制的牌
    :param handCards:
    :param suits:
    :param discards:
    :param discards_op:
    :param wall:
    :param c_color:
    :return: 返回int牌型 0：平胡  1：七对
    '''
    px, eff_map = get_e_cards_map(handCards, suits, discards, discards_op, wall, c_color)
    eff_list = []
    for card, num in eff_map.items():
        eff_list.extend([card] * num)
    return px, eff_list


def get_e_cards_list_10(handCards, suits, discards, discards_op, wall, c_color):
    '''
    获取1维的有效牌 以及牌型 10进制的牌
    :param handCards:
    :param suits:
    :param discards:
    :param discards_op:
    :param wall:
    :param c_color:
    :return: 返回int牌型 0：平胡  1：七对
    '''
    handCards_hex = tool2.list10_to_16(handCards)  # 转成16进制
    suits_hex = []
    for suit in suits:
        suits_hex.append(tool2.list10_to_16(suit))
    discards_hex = []
    discards_op_hex = []
    for p_idx in range(4):
        discards_hex.append(tool2.list10_to_16(discards[p_idx]))
        cur_play_fulu = []
        for suit in discards_op[p_idx]:
            cur_play_fulu.append(tool2.list10_to_16(suit))
        discards_op_hex.append(cur_play_fulu)
    wall_hex = tool2.list10_to_16(wall)

    px, eff_list = get_e_cards_list(handCards_hex, suits_hex, discards_hex, discards_op_hex, wall_hex, c_color)

    return px, tool2.list16_to_10(eff_list)


if __name__ == '__main__':
    # a = [1, 2, 3, 5, 5, 5, 17, 18, 18, 19, 20]
    a = [1, 2, 3, 5, 5, 5, 17, 18, 18, 19, 20]
    discard_op = [[[53, 54, 55], [34, 34, 34], [23, 23, 23], [21, 20, 19]], [[54, 54, 54]],
                  [[3, 4, 5], [25, 25, 25, 25], [35, 36, 37]], []]
    discards = [[52, 54, 50, 41, 7, 39, 2, 17, 9, 8, 3, 34, 23, 7, 6, 37, 54, 6, 18],
                [41, 25, 35, 53, 8, 21, 20, 4, 53, 41, 6, 1, 51, 39, 20, 36, 33, 22],
                [50, 54, 51, 9, 33, 25, 20, 36, 18, 52, 40, 21, 24, 2, 39, 4, 55, 55, 33, 54, 7],
                [50, 52, 51, 17, 55, 5, 21, 52, 41, 39, 3, 40, 17, 9, 36, 22, 23, 50, 49, 55]]
    choose_c = 2
    wall = []
    b = []
    ab, ac = get_e_cards_list(a, b, discards, discard_op, wall, choose_c)
    print(ab, ac)
