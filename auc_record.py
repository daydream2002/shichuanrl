#!sftp://tonnn!10.0.10.204:22/home/tonnn/xiu/venvs/scmj_ppo2/bin/python3.6
# -*- coding: utf-8 -*-
# @Time    : 2023/3/3
# @Author  : xiu
# @File    : auc_record.py
# @Description: 用于经验回放中存储数据时的局id生成和对战信息保存
import time
import random


def get_round_id():
    """
    生成某局麻将游戏的局id
    @return: 局id
    """
    return str(round((time.time() % 1000) * 1000)) + str(random.randint(1000, 9999))


def dict_to_list4(discards):
    """
    字典转列表：{0: [], 1: [], 2: [], 3: []}->[[],[],[],[]]
    @param discards: 字典
    @return: 列表
    """
    l = []
    for i in range(4):
        l.append(discards[i])
    return l


def battle_info_record(data, seat_id, discards, discards_real, discards_op, handcards, action_type, operate_card,
                       passivity_action_site, combine_cards, round):
    """
    记录麻将对战中每一手动作的场面信息
    @param data: 对战信息存储列表
    @param seat_id: 高手玩家座位号
    @param discards: 弃牌列表（不包括被碰、杠的牌）
    @param discards_real: 真实弃牌列表（包括被碰、杠的牌）
    @param discards_op: 副露
    @param handcards: 高手玩家手牌
    @param action_type: 动作类型（摸牌G、弃牌d、碰牌N、杠牌Kkt、胡牌A）
    @param operate_card: 被操作牌
    @param passivity_action_site: 被碰杠的玩家座位号
    @param combine_cards: 碰、杠形成的副露
    @param round: 回合数
    """
    item = {}
    item["seat_id"] = seat_id
    item["discards"] = discards
    item["discards_real"] = discards_real
    item["discards_op"] = discards_op
    item["handcards"] = handcards
    item["action_type"] = action_type
    item["operate_card"] = operate_card
    item["passivity_action_site"] = passivity_action_site
    item["combine_cards"] = combine_cards
    item["round"] = round
    data.append(item)
