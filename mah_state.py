#!sftp://tonnn!10.0.10.204:22/home/tonnn/xiu/venvs/scmj_ppo2/bin/python3.6
# -*- coding: utf-8 -*-
# @Time    : 2023/3/3
# @Author  : xiu
# @File    : mah_state.py
# @Description: 状态信息，即麻将对战的某一时刻某个玩家的视野
import copy
from mah_tool import tool


class RL_state(object):
    '''
        记录游戏之前的所有状态
        # 游戏状态信息类
        # 为了加速计算，对list类型采用自定义deepcopy操作，其他类型用copy.deepcopy
        className:RL_state
        fileName:mah_state.py
    '''

    def __init__(self, player, game):
        """
        构造器
        @param player:玩家座位号
        @param game: 游戏类型
        """
        self.jing_card = 0  # 精牌 作废
        # game信息
        # self.game = game
        self.round = game.round  # 游戏轮数
        self.discards = tool.deepcopy(game.discards)  # 弃牌表
        self.player_discards = game.player_discards  # 所有玩家的真实弃牌（包括吃碰杠的牌） #{0:[0,0,0],1:[1,0,2]，2:[],3:[]}
        self.player_discards_display = game.player_discards_display  # 所有玩家的弃牌（不包括吃碰杠的牌）
        self.player_fulu = game.player_fulu  # 所有玩家的副露
        self.players = game.players  # 所有玩家

        # 胡牌的信息
        self.players_already_hu = game.players_already_hu  # 玩家是否胡牌类型标志位 未胡牌:-1  平胡:0 七对:1
        self.players_already_hu_cards = game.players_already_hu_cards  # 已胡牌牌集

        # 换三张信息
        self.players_switch_cards = game.players_switch_cards  # 各玩家初始换张牌，这是个隐藏信息
        self.players_choose_color = game.players_choose_color  # 各玩家
        self.switch_type = game.switch_type  # 1->上家换给你 2->对家换你  3->下家换你

        self.players_max_fan_list = [[], [], [], []]  # 四个玩家最大胡牌番型
        self.player_handcard_num = [0, 0, 0, 0]  # 每位玩家手牌长度
        for p in self.players:
            self.player_handcard_num[p.seat_id] = len(p.handcards)
            self.players_max_fan_list[p.seat_id] = p.max_fan_list

        self.card_library = tool.deepcopy(game.card_library)  # 所有牌

        # self.remain_cards = self.card_library + self.players[(player.seat_id + 1)%2] # 可能的剩余牌

        self.remain_card_num = game.remain_card_num  # 牌墙中的麻将牌数目

        # 两个参数绑定
        self.outcard = game.outcard  # 被打出的牌
        self.out_seat_id = game.out_seat_id  # 打出某张牌的玩家座位号
        self.dealer_seat_id = game.dealer_seat_id  # 庄家座位号
        self.win_result = copy.deepcopy(game.win_result)  # 赢家信息
        # self.data = []  # 对局数据
        self.competition_op = tool.deepcopy(game.competition_op)  # 竞争性op
        self.terminal = game.terminal  # 游戏是否中止

        # 私人信息
        self.model = player.model  # 玩家使用的模型
        self.catch_card = player.catch_card  # 刚摸到的牌
        self.seat_id = player.seat_id  # 座位号
        self.name = player.name  # 玩家名称
        self.handcards = tool.deepcopy(player.handcards)  # 玩家手牌s
        self.fulu = tool.deepcopy(player.fulu)  # 玩家副露
        self.allow_op = tool.deepcopy(player.allow_op)  # 允许操作
        self.allow_handcards = tool.deepcopy(player.allow_handcards)  # 允许操作的手牌
        self.compe_op = player.compe_op  # 竞争性操作(左吃，中吃，右吃，碰，明杠，胡)
        self.eff_cards_list = player.eff_cards_list  # 有效牌列表
        self.dingQue_cards = player.dingQue_cards  # 定缺牌集

        # 私有的换三张信息
        self.choose_color = player.choose_color  # 定缺的花色下标   0 - 万 1-条 2-筒
        self.already_hu_cards = player.already_hu_cards  # 已经胡牌的列表
        self.switch_cards = player.switch_cards  # 玩家初始换牌列表
        self.already_hu = player.already_hu  # 玩家

        self.to_do = game.to_do  # 下一步动作
