#!sftp://tonnn!10.0.10.204:22/home/tonnn/xiu/venvs/scmj_ppo2/bin/python3.6
# -*- coding: utf-8 -*-
# @Time    : 2023/3/3
# @Author  : xiu
# @File    : url_recommend.py
# @Description: 推荐出牌、动作、换三张、定缺的接口封装
import logging

import requests
import json
import mah_tool.tool2 as tool2
from mah_tool.training_recommend.recommond import RecommendCard_sichuanMJ_v1, json_url
# from mah_tool.so_lib.shangraoMJ_v5 import recommend_op
from interface.sichuanMJ.sichuanMJ_v2 import recommend_op
from mah_tool.suphx_extract_features.search_tree_ren import SearchInfo
import time
import datetime

# logger = logging.getLogger("sichuanMJ_log_v2")
# logger.setLevel(level=logging.DEBUG)
# time_now = datetime.datetime.now()
# handler = logging.FileHandler("./log/sichuanMJ_log_v2_1_%i%i%i.txt" % (time_now.year, time_now.month, time_now.day))
# handler.setLevel(logging.INFO)
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# handler.setFormatter(formatter)
# logger.addHandler(handler)
# logger.info("sichuanMJ_v2_1 compile finished...")

# 日志输出
logger = logging.getLogger("SCMJURL请求_log")
logger.setLevel(level=logging.DEBUG)
time_now = datetime.datetime.now()
handler = logging.FileHandler("./log/SCMJURL_log_%i%i%i.txt" % (time_now.year, time_now.month, time_now.day))
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.info("环境启动成功！开始记录URL日志")

headers = {
    "Content-Type": "application/json; charset=UTF-8",
    # "Referer": "http://jinbao.pinduoduo.com/index?page=5",
    # "User-Agent": "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/67.0.3396.87 Safari/537.36",
}
# 默认的地址
outcard_sichuan_url = "http://10.0.10.202:8089/sichuanMJ/RLV2/outcard"
operate_sichuan_url = "http://10.0.10.202:8089/sichuanMJ/RLV2/operate"
switch_cards_sichuan_url = "http://10.0.10.203:8989/sichuan/v2/switch_cards"
choose_color_sichuan_url = "http://10.0.10.203:8989/sichuan/v2/choose_color"


def trans_discards(player_discards_display):
    """
    弃牌信息的格式转化，10进制->16进制
    @param player_discards_display:四个玩家的弃牌（不包括被吃、碰、杠的牌）
    @return: 16进制表示的弃牌列表
    """
    discards = [[], [], [], []]
    if not player_discards_display or len(player_discards_display) == 0:
        return discards
    discards[0] = tool2.list10_to_16_2(player_discards_display[0])
    discards[1] = tool2.list10_to_16_2(player_discards_display[1])
    discards[2] = tool2.list10_to_16_2(player_discards_display[2])
    discards[3] = tool2.list10_to_16_2(player_discards_display[3])
    return discards


def trans_discards_op(player_fulu):
    """
    副露的格式转换，10进制->16进制
    @param player_fulu:玩家副露
    @return: 16进制的玩家副露
    """
    discards_op = [[], [], [], []]
    discards_op[0] = tool2.fulu_translate(player_fulu[0])
    discards_op[1] = tool2.fulu_translate(player_fulu[1])
    discards_op[2] = tool2.fulu_translate(player_fulu[2])
    discards_op[3] = tool2.fulu_translate(player_fulu[3])
    return discards_op


def state_transfer_sc_for_url(state, seat_id, type="outcards"):
    """
    将state集成的信息转换成url需要的形式 默认是出牌模式下的参数获取
    :param state:状态
    :param seat_id:玩家座位号
    :param type:出牌 or 动作
    :return:url出牌（动作）接口所需要的参数
    """
    seat_id = seat_id  # 玩家座位
    dealer_id = state.dealer_seat_id
    player = state.players[seat_id]

    handcards = player.handcards  # 当前玩家手牌

    fulu = player.fulu  # 玩家副露
    player_discards_display = state.player_discards_display
    player_fulu = state.player_fulu  # 所有玩家副露
    colors = state.players_choose_color  # 花色
    round_ = state.round  # 轮数
    remain_num = len(state.card_library)  # 牌库剩余牌数

    eff_cards = state.eff_cards_list  # 当前玩家有效牌
    hu_types = state.players_already_hu  # 所有玩家的胡牌状态
    hu_cards = state.players_already_hu_cards  # 所有玩家的胡牌牌集
    max_fans = state.players_max_fan_list  # 所有玩家的胡牌最大番型
    card_library = state.card_library  # 牌库
    hands = [state.players[0].handcards, state.players[1].handcards, state.players[2].handcards,
             state.players[3].handcards]  # 四个玩家的手牌

    # 出牌接口需要的参数
    if type == "outcards":
        catch_card = player.catch_card  # 抓牌
        return seat_id, dealer_id, catch_card, handcards, fulu, player_discards_display, player_fulu, eff_cards, hu_types, \
               hu_cards, colors, max_fans, round_, remain_num, card_library, hands
    # 动作接口需要的参数
    else:
        out_seat_id = state.out_seat_id
        out_card = state.outcard
        return seat_id, dealer_id, out_seat_id, out_card, handcards, fulu, player_discards_display, player_fulu, \
               eff_cards, hu_types, hu_cards, colors, max_fans, round_, remain_num, card_library, hands


def trans_result2Op(operate_result, handcards, isHu, out_card):
    '''
    允许操作0: '不操作', 1: '吃左', 2: '吃中', 3: '吃右', 4: '碰', 5: '明杠', 6: '暗杠', 7: '补杠'，8：‘胡’
    将模型推荐的result—op结果转换成op，与动作对应
    进入该模块的牌，进制都采用十进制
    :param operate_result: 推荐结果
    :param handcards:
    :param isHu: 胡牌标志
    :param out_card: 操作牌
    :return: 对应操作的index
    '''
    if isHu:
        return 8
    if not operate_result:
        operate = 0
    else:
        if len(handcards) % 3 == 2:  # 自己的回合
            # 暗杠或者补杠,或者胡牌
            if isHu:
                operate = 8
            else:
                operate = 6 if handcards.count(operate_result[0]) == 4 else 7
        else:
            # 左、中、右吃、碰、明杠
            if operate_result[0] != operate_result[1]:  # 吃
                operate = 1 + operate_result.index(out_card)
            else:  # 碰或者明杠
                operate = 4 if len(operate_result) == 3 else 5
    return operate


def outcard_sc_url(seat_id, dealer_id, catch_card, handcards, fulu, player_discards_display, player_fulu, eff_cards,
                   hu_types, hu_cards, colors, max_fans, round, remain_num, card_library=[], hands=[[], [], [], []],
                   discards_real=[[], [], [], []], outcard_url=outcard_sichuan_url):
    '''
    seat_id = request_json.get("seat_id", 0)  # 获取需要推荐出牌者ID
    dealer_id = request_json.get("dealer_id", seat_id)  # 庄家的ID 默认为seat_id
    dealer_flag = [0] * 4
    dealer_flag[dealer_id] = 1

    catch_card = request_json.get('catch_card', 0)  # 摸牌
    user_cards = request_json.get('user_cards', {})
    hand_cards = user_cards.get('hand_cards', [])   #手牌
    operate_cards = user_cards.get('operate_cards', []) #副露

    if (len(hand_cards) + len(operate_cards)*3) == 13:
        hand_cards.append(catch_card)
        log.info('current hand_cards:{} len is 13, need add catch card:{}'.format(str(hand_cards)+str(operate_cards),
                                                                                  catch_card))
    # 弃牌，副露信息 有效牌
    all_player_discards = request_json.get('discards', [[], [], [], []])
    discards_num = sum([len(all_player_discards[sid]) for sid in range(4)])
    fulus = request_json.get('discards_op', [[], [], [], []])
    eff_cards = request_json.get('eff_cards', [])

    # 有效牌 所有玩家已胡标识 和 胡牌牌集 选择定缺花色 胡牌时最大番型
    all_player_alread_hu = request_json.get('hu_types', [-1, -1, -1, -1])
    all_player_alread_hu_cards = request_json.get('hu_cards', [[], [], [], []])
    all_player_choose_color = request_json.get('colors', [-1, -1, -1, -1])
    all_player_max_fan_list = request_json.get('max_fans', [[], [], [], []])
    # 轮数 剩余牌数信息
    round_ = request_json.get('round', len(all_player_discards[seat_id]))
    remain_card_num = request_json.get('remain_num', 136-14-13*3 - discards_num)

    # 牌库信息 所有玩家手牌信息
    card_library = request_json.get('wall', [])
    all_player_handcards = request_json.get('hands', [[], [], [], []])
    @return:推荐打出的牌
    '''
    pre_time = time.time()
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

    if not outcard_url or outcard_url == "":
        outcard_url = outcard_sichuan_url
    data = json.dumps(json_data)
    # print("处理数据花费时间：{}， json：{}".format(datetime.datetime.now() - pre_time, data))
    # pre_time = datetime.datetime.now()

    try_time = 0  # 尝试次数
    if len(user_cards["hand_cards"]) + len(user_cards["operate_cards"]) * 3 < 13:
        print(json_data)
        input("error stop!!!!!!!")

    while True:
        if try_time >= 3 or (time.time() - pre_time) > 10:
            break
        try:
            pre_post = time.time()
            response = requests.post(outcard_url, data=data, headers=headers, timeout=(10, 15))
            if (time.time() - pre_post) >= 3:
                logger.info("OutCard Timeout:{},URL:{}, json:{}".format(str(time.time() - pre_post),
                                                                        outcard_url, data))
            try_time += 1
        except:
            print('outcard_sc_url， Timeout, try again')
            # logger.info("time out! json:", data)
            print(data)
            logger.info("OutCard Timeout,URL:{}, json:{}".format(outcard_url, data))
            time.sleep(1)
            try_time += 1
            # continue
        else:
            # 成功获取
            # print('ok')
            break
    # print(response.text)
    try:
        result = json.loads(response.text)["out_card"]
    except:
        result = user_cards["hand_cards"][-1]
        print("the json data：", data)
        # logger.info("error json:", data)
        print("error result:", result)
        print("get url recommend error, default out first index of handcards", result)

    if result == None:
        print("result == None!!!!!!!!!")
        result = user_cards["hand_cards"][-1]
    if result not in user_cards["hand_cards"]:
        # print(handcards, result)
        result = user_cards["hand_cards"][-1]
    # print("获取结果花费时间：{}， result：{}".format(datetime.datetime.now() - pre_time, result))
    return result


def outcard_sc_v2(cards=[], suits=[], discards=[], discards_op=[], remain_num=136, round=0, seat_id=0, choose_color=[]):
    """
    四川麻将搜索树v2版接口的推荐出牌
    @param cards: 手牌
    @param suits: 副露
    @param discards: 弃牌
    @param discards_op: 副露
    @param remain_num: 牌墙中的麻将牌数
    @param round: 回合数
    @param seat_id: 玩家的座位号
    @param choose_color: 定缺花色
    @return: 推荐打出的牌
    """
    discards = trans_discards(discards)
    discards_op = trans_discards_op(discards_op)

    cards = tool2.list10_to_16_2(cards)
    suits = tool2.fulu_translate(suits)

    recommend_card, _, _ = SearchInfo.getSearchInfo_sc(cards, suits, discards, discards_op, remain_num, round, seat_id,
                                                       choose_color)
    return recommend_card


def operate_sc_url(seat_id, dealer_id, out_seat_id, out_card, handcards, fulu, player_discards_display, player_fulu,
                   eff_cards, hu_types,
                   hu_cards, colors, max_fans, round, remain_num, card_library=[], hands=[[], [], [], []],
                   discards_real=[[], [], [], []], allow_op=[], operate_url=operate_sichuan_url):
    """
    动作推荐的接口封装
    @param seat_id: 玩家座位号
    @param dealer_id: 庄家座位号
    @param out_seat_id: 出牌玩家座位号
    @param out_card: 打出的牌
    @param handcards: 当前玩家的手牌
    @param fulu: 当前玩家的副露
    @param player_discards_display: 所有玩家的弃牌信息（不包括被碰、杠的牌）
    @param player_fulu: 四个玩家的副露
    @param eff_cards:有效牌集
    @param hu_types:胡牌类型
    @param hu_cards:胡牌获取的牌
    @param colors:定缺花色
    @param max_fans:最大胡牌番型
    @param round:回合数
    @param remain_num:牌墙中的麻将牌数
    @param card_library: 牌墙
    @param hands:四个玩家的手牌
    @param discards_real:四个玩家的真实弃牌
    @param allow_op:允许的动作
    @param operate_url:动作url接口
    @return:推荐的动作
    """
    # pre_time = datetime.datetime.now()
    json_data = {}
    json_data["seat_id"] = int(seat_id)
    json_data["dealer_id"] = int(dealer_id)
    json_data["out_seat_id"] = int(out_seat_id)
    json_data["out_card"] = int(tool2.f10_to_16(out_card))
    json_data["allow_op"] = allow_op

    user_cards = {}
    user_cards["hand_cards"] = tool2.list10_to_16_2(handcards)
    user_cards["operate_cards"] = tool2.fulu_translate(fulu)
    json_data["user_cards"] = user_cards

    json_data["discards"] = trans_discards(player_discards_display)
    json_data["discards_real"] = trans_discards(discards_real)
    json_data["discards_op"] = trans_discards_op(player_fulu)
    json_data["eff_cards"] = tool2.list10_to_16_2(eff_cards)

    json_data["hu_types"] = hu_types
    json_data["hu_cards"] = trans_discards(hu_cards)
    json_data["colors"] = colors
    json_data["max_fans"] = max_fans

    json_data["round"] = round
    json_data["remain_num"] = remain_num

    json_data["wall"] = card_library
    json_data["hands"] = trans_discards(hands)
    json_data["isHu"] = 8 in allow_op

    if len(user_cards["hand_cards"]) + len(user_cards["operate_cards"]) * 3 < 13:
        print(json_data)
        input("error stop!!!!!!!")

    if not operate_url or operate_url == "":
        operate_url = operate_sichuan_url
    try:
        data = json.dumps(json_data)
    except:
        print(json_data)
        data = json.dumps(json_data)
    # print("处理数据花费时间：{}， json：{}".format(datetime.datetime.now() - pre_time, data))
    pre_time = time.time()
    try_time = 0  # 尝试次数
    while True:
        if try_time >= 3 or (time.time() - pre_time) > 10:
            break
        try:
            pre_post = time.time()
            response = requests.post(operate_url, data=data, headers=headers, timeout=(10, 15))
            if (time.time() - pre_post) >= 3:
                logger.info("operate Timeout:{},URL:{}, json:{}".format(str(time.time() - pre_post),
                                                                        operate_url, data))
            try_time += 1
        except:  # requests.exceptions.ConnectionError

            print('operate_sc_url， Timeout, try again')
            # logger.info("time out! json:", data)
            print(data)
            logger.info("operate Timeout,URL:{}, json:{}".format(operate_url, data))
            time.sleep(1)
            try_time += 1
            # continue
        else:
            # 成功获取
            # print('ok')
            break
    # print(response.text)
    operate_result = []
    isHu = False
    try:
        operate_result = json.loads(response.text)["operate_cards"]
        isHu = json.loads(response.text)["isHu"]
    except:
        print("[INFO]: url_recommend operate appear except")
        print(operate_result)
    # print("获取结果花费时间：{}， result：{}".format(datetime.datetime.now() - pre_time, operate_result))
    # operate = trans_result2Op(operate_result, user_cards["hand_cards"], isHu, json_data["out_card"])
    return operate_result, isHu


def operate_sc_v2(op_card, cards=[], suits=[], round=0, remain_num=136, discards=[], discards_op=[],
                  seat_id=0, choose_color=[], hu_cards=[[], [], [], []], hu_fan=[[], [], [], []]):
    """
    四川麻将搜索树v2版的推荐动作
    @param op_card: 操作牌
    @param cards: 当前玩家的手牌
    @param suits: 当前玩家的副露
    @param round: 回合数
    @param remain_num: 牌墙中的麻将牌数
    @param discards: 四个玩家的弃牌
    @param discards_op: 四个玩家的副露
    @param seat_id: 当前玩家座位号
    @param choose_color: 定缺花色
    @param hu_cards: 四个玩家已经胡的牌
    @param hu_fan: 四个玩家的已胡牌番型
    @return: 推荐动作
    """
    op_card = tool2.f10_to_16(op_card)
    discards = trans_discards(discards)
    discards_real = discards
    discards_op = trans_discards_op(discards_op)

    cards = tool2.list10_to_16_2(cards)
    suits = tool2.fulu_translate(suits)

    self_turn = (len(cards) % 3 == 2)  # 判断是否是自己的轮次

    isHu = False  # 吃和胡牌默认为False

    op_result, isHu = recommend_op(op_card, cards, suits, round, remain_num, discards, discards_real, discards_op,
                                   self_turn, seat_id, choose_color, hu_cards, hu_fan, isHu)
    # operate = trans_result2Op(op_result, cards, isHu, op_card)
    return op_result, isHu


def get_url_recommend(state, seat_id, local_sc=True, outcard_url=None):
    """
    推荐出牌的最外层接口
    @param state: 状态
    @param seat_id: 玩家座位号
    @param local_sc: 是否开启全局代理
    @param outcard_url: 出牌url接口
    @return: 推荐打出的牌
    """
    # pre_time = datetime.datetime.now()
    if local_sc:
        recommend_card = outcard_sc_v2(state.handcards, state.fulu, state.player_discards_display,
                                       state.player_fulu, state.remain_card_num, state.round, seat_id,
                                       state.players_choose_color)
    else:
        params = state_transfer_sc_for_url(state, seat_id)
        # seat_id,dealer_id,catch_card,handcards,fulu,player_discards_display,player_fulu,eff_cards,hu_types,
        # hu_cards,colors,max_fans,round,remain_num,card_library=[], hands=[[],[],[],[]]
        recommend_card = outcard_sc_url(*params, state.player_discards, outcard_url)
    result = tool2.f16_to_10(recommend_card)  # 十六进制转换成十进制
    # cur_time = datetime.datetime.now()
    # print("get url card recommend spend time:{}, result is:{}".format(cur_time - pre_time, result))
    return result


def get_url_recommend_new(state, player):
    """
    推荐出牌的最外层接口（本地调用）
    @param state:状态
    @param player:玩家
    @return:推荐打出的牌
    """
    if player.episode % 10 < 8:
        # print("::", player.episode)
        params = state_transfer_sc_for_url(state, player.seat_id)
        params1 = json_url(*params, state.player_discards)
        recommend_card = RecommendCard_sichuanMJ_v1(params1)
        result = tool2.f16_to_10(recommend_card)  # 十六进制转换成十进制
    else:
        features = tool2.card_preprocess_suphx_sc(state, True, global_state=True)
        action, _ = player.brain1.predict(features)
        result = tool2.index_to_card(action)
    return result


def get_url_recommend_op(state, seat_id, allow_op, local_v5=True, operate_url=None):
    '''
    允许操作0: '不操作', 1: '吃左', 2: '吃中', 3: '吃右', 4: '碰', 5: '明杠', 6: '暗杠', 7: '补杠'，8：‘胡’
    获取URL推荐动作，当local_v5 为True时，调用本地接口
    :param state: state状态
    :param seat_id:当前操作者座位id2
    :param allow_op:允许的所有操作
    :param local_v5:是否启用本地的v5
    :param operate_url:动作的url接口
    :return:推荐的动作
    '''
    # pre_time = datetime.datetime.now()
    if local_v5:
        op_cards, hu_flag = operate_sc_v2(state.outcard, state.handcards, state.fulu, state.round,
                                          state.remain_card_num,
                                          state.player_discards_display, state.player_fulu, state.seat_id,
                                          state.players_choose_color,
                                          state.players_already_hu_cards, state.players_max_fan_list)
    else:
        params = state_transfer_sc_for_url(state, seat_id, "operate")
        op_cards, hu_flag = operate_sc_url(*params, state.player_discards, allow_op, operate_url)
    # cur_time = datetime.datetime.now()
    # print("get url op recommend spend time:{}, result is:{},isHu{}".format(cur_time - pre_time, op_cards, hu_flag))
    # if recommend_op in allow_op:
    #     return recommend_op
    # else:
    #     print("推荐动作{}不在允许操作中，请检查。".format(recommend_op, allow_op))
    #     return 0
    op_cards = tool2.list16_to_10(op_cards)  # 十六进制转换成十进制
    return op_cards, hu_flag


def get_url_recommend_switch_cards(handcards, switch_n_cards=3, switch_cards_url=None):
    """
    推荐换三张，url接口
    @param handcards: 手牌
    @param switch_n_cards: 换张数
    @param switch_cards_url: 换三张url接口
    @return:玩家换三张推荐舍弃的牌
    """
    # print("handcards:",handcards)
    # print("switch_n_cards:",switch_n_cards)
    # print("switch_cards_url:",switch_cards_url)
    json_data = {}
    user_cards = {}
    user_cards["hand_cards"] = handcards
    json_data["user_cards"] = user_cards
    json_data["switch_n_cards"] = switch_n_cards
    if not switch_cards_url or switch_cards_url == "":
        switch_cards_url = switch_cards_sichuan_url
    data = json.dumps(json_data)
    # print("处理数据花费时间：{}， json：{}".format(datetime.datetime.now() - pre_time, data))
    pre_time = time.time()
    try_time = 0  # 尝试次数
    while True:
        if try_time >= 3 or (time.time() - pre_time) > 10:
            break
        try:
            pre_post = time.time()
            response = requests.post(switch_cards_url, data=data, headers=headers, timeout=(10, 15))
            if (time.time() - pre_post) >= 3:
                logger.info("switch_cards Timeout:{},URL:{}, json:{}".format(str(time.time() - pre_post),
                                                                             switch_cards_url, data))
            try_time += 1
        except:  # requests.exceptions.ConnectionError

            print('get_url_recommend_switch_cards， Timeout, try again')
            # logger.info("time out! json:", data)
            print(data)
            logger.info("switch_cards Timeout,URL:{}, json:{}".format(switch_cards_url, data))
            time.sleep(1)
            try_time += 1
            # continue
        else:
            # 成功获取
            # print('ok')
            break
    switch_cards = []
    try:
        switch_cards = json.loads(response.text)["switch_cards"]
    except:
        print("[INFO]: get_url_recommend_switch_cards appear except")
    switch_cards = tool2.list16_to_10(switch_cards)
    return switch_cards


def get_url_recommend_choose_color(handcards, switch_n_cards=3, choose_color_url=None):
    """
    定缺花色推荐（url接口）
    @param handcards: 手牌
    @param switch_n_cards: 换张数
    @param choose_color_url: 定缺花色推荐的url接口
    @return: 推荐的定缺花色
    """
    json_data = {}
    user_cards = {}
    user_cards["hand_cards"] = handcards
    json_data["user_cards"] = user_cards
    json_data["switch_n_cards"] = switch_n_cards
    if not choose_color_url or choose_color_url == "":
        choose_color_url = choose_color_sichuan_url
    data = json.dumps(json_data)
    # print("处理数据花费时间：{}， json：{}".format(datetime.datetime.now() - pre_time, data))
    pre_time = time.time()
    try_time = 0  # 尝试次数
    while True:
        if try_time >= 3 or (time.time() - pre_time) > 10:
            break
        try:
            pre_post = time.time()
            response = requests.post(choose_color_url, data=data, headers=headers, timeout=(10, 15))
            if (time.time() - pre_post) >= 3:
                logger.info("choose_color Timeout:{},URL:{}, json:{}".format(str(time.time() - pre_post),
                                                                             choose_color_url, data))
            try_time += 1
        except:  # requests.exceptions.ConnectionError

            print('get_url_recommend_choose_color， Timeout, try again')
            # logger.info("time out! json:", data)
            print(data)
            logger.info("choose_color Timeout,URL:{}, json:{}".format(choose_color_url, data))
            time.sleep(1)
            try_time += 1
            # continue
        else:
            # 成功获取
            # print('ok')
            break
    choose_color = -1
    try:
        choose_color = json.loads(response.text)["choose_color"]
    except:
        print("[INFO]: get_url_recommend_choose_color appear except")
    return choose_color
