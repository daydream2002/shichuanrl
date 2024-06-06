#!sftp://tonnn!10.0.10.204:22/home/tonnn/xiu/venvs/scmj_ppo2/bin/python3.6
# -*- coding: utf-8 -*-
# @Time    : 2023/3/3
# @Author  : xiu
# @File    : mahjong.py
# @Description: 四人麻将对战环境
import sys
import os

sys.path.append('/home/tonnn/.nas/.xiu/works/node6-sichuang_mj_rl_v3_suphx-master')

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import datetime
import random
import json

# from mahjongEnv.mah_tool import hu
# from mahjongEnv.mah_state import RL_state
# from  mahjongEnv.mah_player import Player_RL
from mah_tool import tool2
from mah_tool import sc_hu
from mah_state import RL_state
from mah_player import Player_RL
from mah_tool.suphx_extract_features.search_tree_ren import SearchInfo
from mah_tool.so_lib.effec_C import get_e_cards_list_10
import logging
import math
import time
from auc_record import get_round_id, battle_info_record, dict_to_list4
from mah_tool.NpEncoder import NpEncoder
from mah_tool.not_ready import get_cadajiao_score
from mah_tool.tool import list10_to_16, fulu_translate

# 全局变量：开局序号
i = 0
import numpy as np
import json

# 日志输出
logger = logging.getLogger("SCMJEnv_log")
logger.setLevel(level=logging.DEBUG)
time_now = datetime.datetime.now()
handler = logging.FileHandler("./log/SCMJEnv_log_%i%i%i.txt" % (time_now.year, time_now.month, time_now.day))
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.info("环境启动成功！开始记录日志")


class Game2(object):
    '''
        两人四川麻将，四川麻将-血流成河换三张
        className:Game2
        fileName:mahjong.py
    '''

    # 四川麻将-血流成河换三张
    # hu_cands_table={}#各个类共享参数
    def __init__(self, select_jing_model="random"):
        """
        构造器
        @param select_jing_model: 宝牌选择方法
        """
        self.round = 0  # 游戏轮数
        self.discards = []  # 弃牌表
        self.player_discards = {0: [], 1: [], 2: [], 3: []}  # 特定某玩家的弃牌 #{0:[0,0,0],1:[1,0,2]}
        self.card_library = [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                             21, 22, 23, 24, 25, 26, 27, 28, 29] * 4  # 所有牌 以十进制表示  没有字牌
        # 打乱牌库
        random.shuffle(self.card_library)

        self.remain_card_num = len(self.card_library)
        # 两个参数绑定
        self.outcard = -1
        self.out_seat_id = -1
        self.episode = 0
        # 这里的番可以写成[[类型(番型)， 倍数， [玩家ID]]]
        self.win_result = {0: {"win": 0, "score": 0, "fan": []}, 1: {"win": 0, "score": 0, "fan": [], "info": []},
                           2: {"win": 0, "score": 0, "fan": []},
                           3: {"win": 0, "score": 0, "fan": [], "info": []}}  # 赢家信息
        self.data = {}  # 对局数据
        self.competition_op = [-1, -1, -1, -1]  # 竞争性op
        self.baseM = 1  # 底金
        self.json_data_record = {}  # 保存对局数据（局表+对打）到json文件中
        self.battle_info = []  # 保存每一手动作的对局信息（对打）

    def reset(self):
        """
        参数重置
        """
        self.round = 0  # 游戏轮数
        self.discards = []  # 弃牌表
        self.player_discards = {0: [], 1: [], 2: [], 3: []}  # 特定某玩家的弃牌 #{0:[0,0,0],1:[1,0,2]}
        self.card_library = [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23, 24, 25, 26, 27,
                             28, 29] * 4  # 所有牌
        # 打乱牌库
        random.shuffle(self.card_library)

        # self.select_jing()  # 宝牌
        self.remain_card_num = len(self.card_library)
        # 两个参数绑定
        self.outcard = -1
        self.out_seat_id = -1
        # self.episode = 0
        # info 存放刮风、下雨、退税、查花猪、大叫等信息
        self.win_result = {0: {"win": 0, "score": 0, "fan": [], "info": []},
                           1: {"win": 0, "score": 0, "fan": [], "info": []},
                           2: {"win": 0, "score": 0, "fan": [], "info": []},
                           3: {"win": 0, "score": 0, "fan": [], "info": []}}  # 赢家信息
        self.data = {}  # 对局数据
        self.competition_op = [-1, -1, -1, -1]  # 竞争性op
        self.json_data_record = {}  # 保存对局数据到json文件中
        self.battle_info = []  # 保存每一手动作的对局信息（对打）

    def cardsToKey(self, cardlist):
        """
        转换牌为key，用在胡牌算法里
        @param cardlist: 手牌
        @return: 胡牌算法中的key
        """
        key1 = ''
        for k in cardlist:
            if 1 <= k <= 9:
                key1 = key1 + '0' + str(k)
            else:
                key1 = key1 + str(k)
        return key1

    # 判断某位玩家是否胡牌
    # def is_hu(self, handcards, fulu, catch_card):
    #     return hu.is_hu(handcards, fulu, self.jing_card, catch_card)[0]
    #     #return  False

    def deal_cards(self, num):
        """
        给某位玩家发指定数量的牌
        @param num:发牌数目
        @return:发的牌
        """
        cards_list = []
        for _ in range(num):
            cards_list.append(self.card_library.pop(0))  # 发牌
        cards_list.sort()
        self.remain_card_num -= num  # 计算牌库剩余牌数
        return cards_list

    # def select_jing(self):
    #     if self.select_jing_model == "random":
    #         self.jing_card = random.choice([1,2,3,4,5,6,7,8,9,11,12,13,14,15,16,17,18,19,
    #                        21,22,23,24,25,26,27,28,29,31,32,33,34,35,36,37])
    #     else:
    #         jing_card = input("请输入精牌：")
    #         self.jing_card = int(jing_card)


class Mcts_game(Game2):
    '''
        二人四处血流成河四川麻将mcts类
        className:Mcts_game
        fileName:mahjong.py
    '''

    def __init__(self):
        """
        构造器，获取玩家当前可见的状态信息
        """
        super(Mcts_game, self).__init__()
        self.terminal = False  # 游戏是否中止
        self.hu_type = ""  # 胡牌类型
        self.players = []  # 所有的玩家
        self.player_fulu = {0: [], 1: [], 2: [], 3: []}  # 四个玩家副露
        self.player_discards_display = {0: [], 1: [], 2: [], 3: []}  # 四个玩家的弃牌

    def reset(self):
        """
        参数重置
        """
        super(Mcts_game, self).reset()
        self.terminal = False
        self.hu_type = ""  # 胡牌类型
        self.players = []
        self.player_fulu = {0: [], 1: [], 2: [], 3: []}
        self.player_discards_display = {0: [], 1: [], 2: [], 3: []}

    def perform(self, player, player0, player1, player2, player3):
        """
        输出信息
        @param player: 四个玩家
        @param player0: 玩家0
        @param player1: 玩家1
        @param player2: 玩家2
        @param player3: 玩家3
        """
        print("player" + str(player.seat_id) + ":")
        print("玩家", self.out_seat_id, "   outcard: " + str(self.outcard) + "       余牌：" + str(self.remain_card_num))
        print("幅露：" + str(player.fulu) + "   " + "手牌：" + str(player.handcards) + "    " + "抓牌" + str(player.catch_card))
        print("player0:  " + "幅露：" + str(player0.fulu) + "   " + "手牌：" + str(player0.handcards) + "   手牌长度：" + str(
            len(player0.handcards)))
        print("player1:  " + "幅露：" + str(player1.fulu) + "   " + "手牌：" + str(player1.handcards) + "   手牌长度：" + str(
            len(player1.handcards)))
        print("player2:  " + "幅露：" + str(player2.fulu) + "   " + "手牌：" + str(player2.handcards) + "   " + str(
            len(player2.handcards)))
        print("player3:  " + "幅露：" + str(player3.fulu) + "   " + "手牌：" + str(player3.handcards) + "   " + str(
            len(player3.handcards)))
        print("\n")

    def is_hu(self, handcards, fulu, dingque):
        """
        判断某位玩家是否胡牌
        @param handcards: 手牌
        @param fulu: 副露
        @param dingque: 定缺花色
        @return: 是否胡牌，bool值
        """
        hu_map = {0: "平胡", 1: "七对"}
        flag, res = sc_hu.Mahjong(handcards, fulu, dingque).isHu()  # 传入十进制数据
        if flag:
            self.hu_type = hu_map[res]
        return flag
        # return  False

    def final_reward(self, player, state):
        """
        返回最终奖励
        @param player: 玩家
        @param state: 状态
        """
        pass
        # for pp in player:
        #     #调用相关函数，让函数自己获得最后奖励
        #     card=pp.recommend_card(state)
        #     op = pp.recommend_op(state)
        # return


class Game_RL(Mcts_game):
    '''
        四人血流成河四川麻将类-强化学习
        className:Game_RL
        fileName:mahjong.py
    '''

    def __init__(self, is_render=False):
        """
        构造器，用在mcts模拟牌局的时候   根据state设置牌局信息，包括游戏信息game与玩家信息player
        @param is_render: 是否打印相关信息
        """
        super(Game_RL, self).__init__()
        self.to_do = "catch_card"
        self.is_render = is_render
        self.other_seat = [[1, 2, 3], [2, 3, 0], [3, 0, 1], [0, 1, 2]]  # 当seat_id = 0 时，其他玩家的id为[1,2,3]
        self.players_switch_cards = [[], [], [], []]  # 所有玩家初始换的牌
        self.players_choose_color = [-1, -1, -1, -1]
        self.dealer_seat_id = random.choice([0, 1, 2, 3])  # 随机选择庄家座位ID
        self.switch_type = random.choice([1, 2, 3])  # 随机选择换牌类型 1：换给下家、2：换给对家、3：换给上家
        self.players_already_hu = [-1, -1, -1, -1]  # 玩家是否胡牌标志位 未胡牌:-1 平胡:0 七对:1
        self.players_already_hu_cards = [[], [], [], []]  # 已胡牌牌集
        self.not_ready = [False] * 4  # 是否被查大叫(未听牌次数，流局除外) add by xiu 2022.5.30

        self.players_isGang = [False] * 4  # 每个玩家此步是否杠牌的标志位，用于杠上开花的判断。
        self.max_fan_nums = [0, 0, 0, 0]  # 每个玩家的最大番型

        self.players_angang_num = [0, 0, 0, 0]
        self.players_bugang_num = [0, 0, 0, 0]
        self.players_minggang_num = [0, 0, 0, 0]
        self.players_gang_num = [0, 0, 0, 0]
        self.players_minggang_list = [[], [], [], []]  # 存储被明杠的玩家id

        self.first_win_round = [-1, -1, -1, -1]  # 每个玩家第一次胡牌的轮数
        self.hu_type_list = [[0, 0], [0, 0], [0, 0], [0, 0]]  # 每个玩家的胡牌类型[平胡/七对]
        self.fan_list = [[0] * 11, [0] * 11, [0] * 11, [0] * 11]  # 每个玩家的胡牌番型[清一色、断幺九、碰碰胡、自摸、杠上开花、金钩钩、幺九、一根、两根、三根、四根]

    def get_state(self, player, game, addition_info=False):
        """
        获取当前状态下指定玩家可见的状态信息
        @param player: 玩家
        @param game: 游戏类别
        @param addition_info: 额外信息
        @return: state
        """
        if addition_info:
            # 获取有效牌
            px, eff_cards_list = get_e_cards_list_10(player.handcards, player.fulu, game.player_discards,
                                                     game.player_fulu,
                                                     game.card_library, game.players_choose_color[player.seat_id])
            player.eff_cards_list = eff_cards_list
            player.dingQue_cards = tool2.get_dingQue_cards(player.handcards, player.choose_color)
        state = RL_state(player, game)
        return state

    def reset(self):
        """
        参数重置
        """
        super(Game_RL, self).reset()
        self.to_do = "catch_card"
        # self.is_render = False
        self.other_seat = [[1, 2, 3], [2, 3, 0], [3, 0, 1], [0, 1, 2]]  # 当seat_id = 0 时，其他玩家的id为[1,2,3]
        self.players_switch_cards = [[], [], [], []]  # 所有玩家初始换的牌
        self.players_choose_color = [-1, -1, -1, -1]
        self.dealer_seat_id = random.choice([0, 1, 2, 3])  # 随机选择庄家座位ID
        self.switch_type = random.choice([1, 2, 3])  # 随机选择换牌类型 1：换给下家、2：换给对家、3：换给上家
        self.players_already_hu = [-1, -1, -1, -1]  # 玩家是否胡牌标志位 -1-》 未胡牌  0-》平胡 1-》七对
        self.players_already_hu_cards = [[], [], [], []]  # 已胡牌牌集
        self.not_ready = [False] * 4  # 是否被查大叫(未听牌次数，流局除外)
        self.players_isGang = [False] * 4  # 每个玩家此步是否杠牌的标志位，用于杠上开花的判断。
        self.max_fan_nums = [0, 0, 0, 0]  # 每个玩家的最大番型

        self.players_angang_num = [0, 0, 0, 0]
        self.players_bugang_num = [0, 0, 0, 0]
        self.players_minggang_num = [0, 0, 0, 0]
        self.players_gang_num = [0, 0, 0, 0]  # 总的杠数
        self.players_minggang_list = [[], [], [], []]  # 存放明杠的ID， 用于退税

        self.first_win_round = [-1, -1, -1, -1]  # 每个玩家第一次胡牌的轮数
        self.hu_type_list = [[0, 0], [0, 0], [0, 0], [0, 0]]
        self.fan_list = [[0] * 11, [0] * 11, [0] * 11, [0] * 11]

    def mah_step(self, player0, player1, player2, player3, action_id=0):
        """
        与gym相对应的函数，此处修改，应该只走单步，用单步更新，不直接进行一场游戏
        @param player0: 玩家0
        @param player1: 玩家1
        @param player2: 玩家2
        @param player3: 玩家3
        @param action_id: 当前玩家座位号
        @return:
        """
        self.round += 1  # 游戏轮数+1
        current_p_index = action_id  # 当前玩家座位号
        player = [player0, player1, player2, player3]
        # 计算可能的番型
        # #
        # [平胡 七对]
        idx2px_dict = {0: "平胡", 1: "七对"}
        px2idx_dict = {v: k for k, v in idx2px_dict.items()}

        #  [清一色、断幺九、碰碰胡、自摸、杠上开花、金钩钩、幺九、一根、两根、三根、四根]
        idx2fan_dict = {0: "清一色", 1: "断幺九", 2: "碰碰胡", 3: "自摸", 4: "杠上开花", 5: "金钩钩",
                        6: "幺九", 7: "一根", 8: "两根", 9: "三根", 10: "四根"}
        self.players = player

        def get_outcard():  ############################
            """
            丢牌
            """
            # card2 = input("player"+str(i % 2)+": "+"请输出牌：")
            if player[current_p_index % 4].already_hu:  # 已经胡牌了
                card2 = player[current_p_index % 4].catch_card  # 出上把抓的牌
            else:
                if player[current_p_index % 4].model == "PPOModel":
                    state = self.get_state(player[current_p_index % 4], self, isGetEffCard=True)  ###################
                else:
                    state = self.get_state(player[current_p_index % 4], self)
                t44 = datetime.datetime.now()
                card2 = player[current_p_index % 4].recommend_card2(state)  ######################
                t444 = datetime.datetime.now() - t44

            try:
                player[current_p_index % 4].handcards.remove(card2)
            except:
                # 不在手里出最后一张牌
                # print("[INFO]出牌错误，默认出最后一张牌。", player[current_p_index % 4].handcards, "/t/t", current_p_index % 4, card2)
                card2 = player[current_p_index % 4].handcards[-1]
                player[current_p_index % 4].handcards.remove(card2)

            self.outcard = card2
            self.out_seat_id = player[current_p_index % 4].seat_id
            self.player_discards[current_p_index % 4].append(card2)
            self.player_discards_display[current_p_index % 4].append(card2)
            self.discards.append(card2)

            # 记录对局信息-丢牌
            discards = dict_to_list4(self.player_discards_display)  # 四个玩家的弃牌
            discards_real = dict_to_list4(self.player_discards)  # 四个玩家的真实弃牌
            discards_op = dict_to_list4(self.player_fulu)  # 四个玩家的副露
            handcards = [player0.handcards, player1.handcards, player2.handcards,
                         player3.handcards]  # 四个玩家的手牌
            battle_info_record(self.battle_info, player[current_p_index % 4].seat_id, discards, discards_real,
                               discards_op, handcards, "d", card2, 255, [], self.round)

        def score_op(add_score_idx, add_score, subtract_idx=-1):
            """
            对分数进行操作
            @param add_score_idx: 加分玩家的下标
            @param add_score: 加的分数
            @param subtract_idx: 减分玩家的下标
            """

            # 计算杠牌的加分
            self.win_result[add_score_idx]["score"] += add_score
            if subtract_idx == -1:
                for idx in self.other_seat[add_score_idx]:
                    self.win_result[idx]["score"] -= (add_score // 3)
            else:
                assert add_score_idx != subtract_idx
                self.win_result[subtract_idx]["score"] -= add_score

        while self.card_library:  # 打四手牌
            # 当前状态为换张状态
            if self.to_do == "switch_cards":
                # 换张代码逻辑
                pre_switch_time = time.time()
                # 每个玩家确定要换的牌
                for player_ in self.players:
                    myidx = player_.seat_id
                    switch_cards = player_.recommend_switch_cards(player_.handcards)
                    self.players_switch_cards[myidx] = switch_cards
                    player_.switch_cards = switch_cards
                    # 将要换的牌从手牌中移除
                    for card in switch_cards:
                        player_.handcards.remove(card)
                # 随机互换牌
                for player_ in self.players:
                    # 当前玩家的座位号
                    my_idx = player_.seat_id
                    # 提供给当前玩家换牌的座位号
                    offer_swith_cards_idx = (my_idx + self.switch_type) % 4
                    player_.handcards.extend(self.players_switch_cards[offer_swith_cards_idx])
                # 换完之后跳转到选花色状态
                self.to_do = "choose_color"

                # 超时检查
                spend_switch_time = time.time() - pre_switch_time
                if spend_switch_time > 2.5:
                    print("换三张状态超时,为", str(spend_switch_time))
                    logger.info("换三张 overtime, need:{}".format(str(spend_switch_time)))

            # 定缺
            if self.to_do == "choose_color":
                pre_color_time = time.time()
                for player_ in self.players:
                    my_idx = player_.seat_id
                    color = player_.recommend_choose_color(self.get_state(player_, self))
                    self.players_choose_color[my_idx] = color
                    player_.choose_color = color
                # 选花色代码逻辑
                self.to_do = "check_allow_op"
                # 超时检查
                spend_color_time = time.time() - pre_color_time
                if spend_color_time > 2.5:
                    print("选花色状态超时,为", str(spend_color_time))
                    logger.info("选花色 overtime, need:{}".format(str(spend_color_time)))

            # 摸牌
            if self.to_do == "catch_card":  # 0:00:00.000016
                pre_catch_time = time.time()
                t21 = datetime.datetime.now()
                card1 = self.deal_cards(1)  # 发牌
                player[current_p_index % 4].catch_card = card1[0]  # 抓牌  #########################
                player[current_p_index % 4].handcards.sort()  # 最后一张不排序
                player[current_p_index % 4].handcards.append(player[current_p_index % 4].catch_card)  # 加入手牌

                # 超时检查
                spend_catch_time = time.time() - pre_catch_time
                if spend_catch_time > 1:
                    print("抓牌状态超时,为", str(spend_catch_time))
                    logger.info("抓牌 overtime, need:{}, 玩家ID：{}，手牌：{}， 抓牌：{}".format(str(spend_catch_time),
                                                                                    current_p_index,
                                                                                    player[
                                                                                        current_p_index % 4].handcards,
                                                                                    card1))
                # 记录对局信息-摸牌
                discards = dict_to_list4(self.player_discards_display)  # 四个玩家的弃牌
                discards_real = dict_to_list4(self.player_discards)  # 四个玩家的真实弃牌
                discards_op = dict_to_list4(self.player_fulu)  # 四个玩家的副露
                handcards = [player0.handcards, player1.handcards, player2.handcards,
                             player3.handcards]  # 四个玩家的手牌
                battle_info_record(self.battle_info, player[current_p_index % 4].seat_id, discards, discards_real,
                                   discards_op, handcards, "G", card1[0], 255, [], self.round)

                # if self.is_render:
                #     print("玩家：{}抓牌中..... 手牌：{}，  副露：{}，抓牌：{}".format(current_p_index % 4,
                #                                                      player[current_p_index % 4].handcards,
                #                                                      player[current_p_index % 4].fulu,
                #                                                      player[current_p_index % 4].catch_card))
                # self.perform(player[i % 2], player0, player1)  # 打印
                self.to_do = "check_allow_op"  #
                pass

            if self.to_do == "check_allow_op":  # 0:00:00.000363
                pre_check_allow = time.time()
                op_map = {0: []}  # 操作的map
                player[current_p_index % 4].allow_op = []
                # 重置 #允许操作0: '不操作', 1: '吃左', 2: '吃中', 3: '吃右', 4: '碰', 5: '明杠', 6: '暗杠', 7: '补杠'，8：‘胡’

                # anGang与buGang的opList是指在一手牌中会存在多个可操作的组合
                anGang_flag, anGang_opList = player[current_p_index % 4].canAnGang()
                buGang_flag, buGang_opList = player[current_p_index % 4].canBuGang()
                if self.is_render:
                    print("玩家：{}抓牌完成，正在检查是否有678操作 手牌：{}，  副露：{}，抓牌：{}".format(current_p_index % 4,
                                                                              player[current_p_index % 4].handcards,
                                                                              player[current_p_index % 4].fulu,
                                                                              player[current_p_index % 4].catch_card))
                    assert len(player[current_p_index % 4].handcards) % 3 == 2
                if anGang_flag:  # 暗杠
                    if not player[current_p_index % 4].already_hu:
                        player[current_p_index % 4].allow_op.append(6)
                        op_map[6] = anGang_opList
                    else:
                        new_anGang_opList = []
                        for op_angang in anGang_opList:
                            tmp_handcards = tool2.deepcopy(player[current_p_index % 4].handcards)
                            tmp_fulus = tool2.deepcopy(player[current_p_index % 4].fulu)
                            for _ in range(4):
                                tmp_handcards.remove(op_angang)
                            tmp_fulus.append([op_angang, op_angang, op_angang, op_angang])

                            can_add_gang = True  # 是否可以加杠的flag
                            for hu_card in set(player[current_p_index % 4].already_hu_cards):  # 需要逐一判断杠前后已胡牌集是否都能胡牌
                                gang_after_hu = self.is_hu(tmp_handcards + [hu_card], tmp_fulus,
                                                           player[current_p_index % 4].choose_color)
                                can_add_gang = gang_after_hu
                                if not can_add_gang:  # 不满足条件，跳出循环
                                    break

                            if can_add_gang:
                                new_anGang_opList.append(op_angang)

                            del tmp_handcards, tmp_fulus
                        if new_anGang_opList:
                            player[current_p_index % 4].allow_op.append(6)
                            op_map[6] = new_anGang_opList
                if buGang_flag:  # 补杠
                    if not player[current_p_index % 4].already_hu:
                        player[current_p_index % 4].allow_op.append(7)
                        op_map[7] = buGang_opList
                    else:
                        new_buGang_opList = []
                        for op_bugang in buGang_opList:
                            tmp_handcards = tool2.deepcopy(player[current_p_index % 4].handcards)
                            tmp_fulus = tool2.deepcopy(player[current_p_index % 4].fulu)
                            for tmp_fulu in tmp_fulus:
                                if tmp_fulu[0] == op_bugang:
                                    tmp_fulu.append(op_bugang)
                                    break
                            tmp_handcards.remove(op_bugang)  # 这里不处理副露是因为杠和碰的副露效果一样

                            can_add_gang = True  # 是否可以加杠的flag
                            for hu_card in set(player[current_p_index % 4].already_hu_cards):  # 需要逐一判断杠前后已胡牌集是否都能胡牌
                                gang_after_hu = self.is_hu(tmp_handcards + [hu_card], tmp_fulus,
                                                           player[current_p_index % 4].choose_color)
                                can_add_gang = gang_after_hu
                                if not can_add_gang:  # 不满足条件，跳出循环
                                    break

                            if can_add_gang:
                                new_buGang_opList.append(op_bugang)
                            del tmp_handcards, tmp_fulus
                        if new_buGang_opList:
                            player[current_p_index % 4].allow_op.append(7)
                            op_map[7] = new_buGang_opList

                if self.is_hu(player[current_p_index % 4].handcards, player[current_p_index % 4].fulu,
                              player[current_p_index % 4].choose_color):  # 胡    ##jjjjj
                    player[current_p_index % 4].allow_op.append(8)
                    op_map[8] = [None]
                player[current_p_index % 4].allow_op.append(0)
                player[current_p_index % 4].allow_op.sort()

                '''(2)执行操作'''
                if len(player[current_p_index % 4].allow_op) >= 2:  # 除“过”以外，还有其它操作
                    if self.is_render:
                        print("player" + str(player[current_p_index % 4].seat_id) + "   outcard:" + str(
                            self.outcard) + "   catch_card:" + str(
                            player[current_p_index % 4].catch_card) + "    允许操作" + str(
                            player[current_p_index % 4].allow_op))
                    # op=input("请输入操作：")
                    state = self.get_state(player[current_p_index % 4], self)
                    op, op_card = player[current_p_index % 4].recommend_op(state, op_map)  ##########################

                    # 执行
                    if op not in op_map.keys():
                        print(op, op_map)

                    # 过
                    if op == 0 or op not in op_map.keys():
                        player[current_p_index % 4].guo()
                        self.to_do = "output_card"

                    # 暗杠
                    elif op == 6:
                        self.players_isGang[current_p_index % 4] = True
                        self.players_angang_num[current_p_index % 4] += 1  # 暗杠数+1
                        self.players_gang_num[current_p_index % 4] += 1  # 总杠数+1

                        # print("执行暗杠操作者ID：{}".format(current_p_index % 4))
                        player[current_p_index % 4].anGang(op_card)
                        # print("执行暗杠后的副露：{}，{}".format(player[current_p_index % 4].fulu, self.player_fulu))
                        self.player_fulu[player[current_p_index % 4].seat_id] = player[current_p_index % 4].fulu
                        cur_round_score_base = int(math.pow(2, 1)) * self.baseM  # 本轮单玩家的基础分
                        score_op(current_p_index % 4, cur_round_score_base * 3)  # 刮风下雨

                        other_player_id = [0, 1, 2, 3]
                        other_player_id.remove(current_p_index % 4)

                        score_detail = ["下雨(暗杠)", cur_round_score_base, other_player_id]
                        self.win_result[current_p_index % 4]["info"].append(score_detail)
                        score_detail = ["被下雨(暗杠)", - cur_round_score_base, [current_p_index % 4]]
                        for i in other_player_id:
                            self.win_result[i]["info"].append(score_detail)
                        self.to_do = "catch_card"

                        # 记录对局信息-暗杠
                        discards = dict_to_list4(self.player_discards_display)  # 四个玩家的弃牌
                        discards_real = dict_to_list4(self.player_discards)  # 四个玩家的真实弃牌
                        discards_op = dict_to_list4(self.player_fulu)  # 四个玩家的副露
                        handcards = [player0.handcards, player1.handcards, player2.handcards,
                                     player3.handcards]  # 四个玩家的手牌
                        battle_info_record(self.battle_info, player[current_p_index % 4].seat_id, discards,
                                           discards_real, discards_op, handcards, "K", op_card, 255,
                                           [op_card, op_card, op_card, op_card], self.round)

                    # 补杠
                    elif op == 7:
                        self.players_isGang[current_p_index] = True
                        self.players_bugang_num[current_p_index % 4] += 1
                        self.players_gang_num[current_p_index % 4] += 1  # 总杠数+1

                        player[current_p_index % 4].buGang(op_card)
                        self.player_fulu[player[current_p_index % 4].seat_id] = player[current_p_index % 4].fulu
                        cur_round_score_base = int(math.pow(2, 0)) * self.baseM  # 本轮单玩家的基础分
                        score_op(current_p_index % 4, cur_round_score_base * 3, -1)  # 刮风下雨

                        other_player_id = [0, 1, 2, 3]
                        other_player_id.remove(current_p_index % 4)

                        score_detail = ["刮风(补杠)", self.baseM, other_player_id]
                        self.win_result[current_p_index % 4]["info"].append(score_detail)
                        score_detail = ["被刮风(补杠)", -cur_round_score_base, [current_p_index % 4]]
                        for i in other_player_id:
                            self.win_result[i]["info"].append(score_detail)
                        self.to_do = "catch_card"

                        # 记录对局信息-补杠
                        discards = dict_to_list4(self.player_discards_display)  # 四个玩家的弃牌
                        discards_real = dict_to_list4(self.player_discards)  # 四个玩家的真实弃牌
                        discards_op = dict_to_list4(self.player_fulu)  # 四个玩家的副露
                        handcards = [player0.handcards, player1.handcards, player2.handcards,
                                     player3.handcards]  # 四个玩家的手牌
                        battle_info_record(self.battle_info, player[current_p_index % 4].seat_id, discards,
                                           discards_real, discards_op, handcards, "t", op_card, 255,
                                           [op_card, op_card, op_card, op_card], self.round)

                    # 胡牌
                    elif op == 8:
                        assert player[current_p_index % 4].seat_id == current_p_index % 4

                        # 统计每个玩家第一次胡牌的轮数
                        if self.first_win_round[current_p_index % 4] == -1:
                            self.first_win_round[current_p_index % 4] = self.round

                        # 胡牌标志分配
                        # self.win_result[player[current_p_index % 4].seat_id]["win"] = 1
                        # for myidx in self.other_seat[current_p_index % 4]:
                        #     self.win_result[myidx]["win"] = -1

                        # cal_fan.fan(player[i % 2], self, player[i % 2].handcards, player[i % 2].fulu, self.win_result)

                        # score_op(current_p_index % 4, self.baseM * 3)
                        # [清一色、断幺九、碰碰胡、自摸、杠上开花、金钩钩、幺九、一根、两根、三根、四根]
                        cur_player_fulu = []  # 当前玩家副露表 用hex表示
                        for fulu_ in player[current_p_index % 4].fulu:
                            cur_player_fulu.append(tool2.list10_to_16(fulu_))

                        fanList = SearchInfo.getFanList(
                            paixing=px2idx_dict.get(self.hu_type, 0),
                            cards=tool2.list10_to_16(player[current_p_index % 4].handcards),
                            suits=cur_player_fulu,
                            jingCard=0,
                            gangNum=self.players_gang_num[current_p_index % 4],
                            isHuJudge=True,
                            isSelfTurn=True,
                            preActionIsGang=self.players_isGang[current_p_index % 4])

                        fanType = self.hu_type  # 基础番型

                        # 统计这一局的胡牌类型和胡牌番型数(去重)
                        self.hu_type_list[current_p_index % 4][px2idx_dict[fanType]] = 1
                        for i, e in enumerate(fanList):
                            if e == 1:
                                self.fan_list[current_p_index % 4][i] = 1

                        # self.win_result[current_p_index%4]["fan"].append("自摸:(" + self.hu_type + ") ")
                        fan_num = 1  # 番的个数
                        if fanType == "七对":
                            fan_num += 2
                        # [清一色、断幺九、碰碰胡、自摸、杠上开花、金钩钩、幺九、一根、两根、三根、四根]
                        fan_map = [2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1]
                        # 番型分
                        for fan_index in range(len(fanList)):
                            if fanList[fan_index] == 1:
                                fanType += "," + idx2fan_dict[fan_index]
                                # self.win_result[current_p_index % 4]["fan"].append()
                                fan_num += fan_map[fan_index]

                        if self.max_fan_nums[current_p_index] < fan_num:
                            self.max_fan_nums[current_p_index] = fan_num
                            player[current_p_index].max_fan_list = fanList

                        fan = int(math.pow(2, fan_num - 1))
                        if fan > 256:
                            fan = 256  # 最高256倍
                        cur_round_score_base = fan * self.baseM  # 本轮单玩家的基础分
                        # 根据番型算分
                        score_op(current_p_index % 4, cur_round_score_base * 3)

                        other_player_id = [0, 1, 2, 3]
                        other_player_id.remove(current_p_index % 4)

                        score_detail = ["自摸（" + fanType + ")", cur_round_score_base, other_player_id]  # 番型累加制
                        self.win_result[current_p_index % 4]["fan"].append(score_detail)
                        # 计算其他玩家的得分情况
                        score_detail = ["被自摸（" + fanType + ")", -cur_round_score_base, [current_p_index % 4]]
                        for idx in other_player_id:
                            self.win_result[idx % 4]["fan"].append(score_detail)

                        # 胡牌玩家胡牌位置为True
                        player[current_p_index % 4].already_hu = True
                        self.players_already_hu[current_p_index % 4] = px2idx_dict[self.hu_type]

                        # print("\n")
                        # 当前玩家的本轮抓牌
                        cur_p_catch_card = player[current_p_index % 4].catch_card
                        if cur_p_catch_card not in player[current_p_index % 4].handcards:
                            # 不在手牌中的原因，换三张完之后就胡牌了，且抓的牌被换出去了
                            player[current_p_index % 4].handcards.remove(
                                random.choice(player[current_p_index % 4].handcards))
                        else:
                            player[current_p_index % 4].handcards.remove(cur_p_catch_card)
                        player[current_p_index % 4].already_hu_cards.append(cur_p_catch_card)
                        self.players_already_hu_cards[current_p_index % 4].append(cur_p_catch_card)

                        # 记录对局信息-胡牌
                        discards = dict_to_list4(self.player_discards_display)  # 四个玩家的弃牌
                        discards_real = dict_to_list4(self.player_discards)  # 四个玩家的真实弃牌
                        discards_op = dict_to_list4(self.player_fulu)  # 四个玩家的副露
                        handcards = [player0.handcards, player1.handcards, player2.handcards,
                                     player3.handcards]  # 四个玩家的手牌
                        battle_info_record(self.battle_info, player[current_p_index % 4].seat_id, discards,
                                           discards_real, discards_op, handcards, "A", 255, 255, [], self.round)

                        current_p_index = (current_p_index + 1) % 4  # 切换到下一个玩家摸牌
                        self.to_do = "catch_card"
                        # return self.win_result  # 自摸胡牌结束
                        # return

                else:
                    # get_outcard()
                    self.to_do = "output_card"
                    # self.perform(player[i % 2], player0, player1)

                    # print("######################################################################################")
                # 超时检查
                spend_check_allow_time = time.time() - pre_check_allow
                if spend_check_allow_time > 3:
                    print("检查自己的允许操作状态超时,为", str(spend_check_allow_time))
                    logger.info("检查自己的允许操作 overtime, need:{}".format(str(spend_check_allow_time)))

            if self.to_do == "output_card":  # 0:00:00.088973
                # t23 = datetime.datetime.now()
                pre_outcard = time.time()
                # 出牌后，杠的状态置为false
                self.players_isGang[current_p_index % 4] = False
                if current_p_index % 4 == 0:  # 如果轮到玩家0在此出牌，则进行中断
                    return
                else:
                    get_outcard()
                    self.to_do = "check_others_allow_op"

                # 超时检查
                speed_outcard_time = time.time() - pre_outcard
                if speed_outcard_time > 2.5:
                    print("出牌状态超时,为", str(speed_outcard_time))
                    logger.info("出牌 overtime, need:{}".format(str(speed_outcard_time)))
                # t33 = datetime.datetime.now() - t23
                # self.discards    ######################################

            if self.to_do == "check_others_allow_op":  # 0:00:00.000430
                pre_check_other = time.time()
                t24 = datetime.datetime.now()
                other_player = [player0, player1, player2, player3]
                other_player[current_p_index % 4].compe_op = 0
                other_hu_type = ["", "", "", ""]  # 如判断点炮胡，所有玩家可能的胡牌情况
                ming_gang_ID = -1  # 明杠者的ID， -1 -》 无人杠  否则对应其他的id
                del other_player[current_p_index % 4]
                if self.is_render:
                    print("玩家：{}出牌完成，正在检查其他玩家是否有45操作 手牌：{}，  副露：{}，出牌：{}， 庄家ID：{}\n".format(current_p_index % 4,
                                                                                            player[
                                                                                                current_p_index % 4].handcards,
                                                                                            player[
                                                                                                current_p_index % 4].fulu,
                                                                                            self.outcard,
                                                                                            self.dealer_seat_id))
                    assert len(player[current_p_index % 4].handcards) % 3 == 1
                '''(1)检测所有玩家执行op'''
                while other_player:
                    p = other_player.pop(0)
                    # print(p.seat_id)
                    # 检测
                    # 允许操作0: '不操作', 1: '吃左', 2: '吃中', 3: '吃右', 4: '碰', 5: '明杠', 6: '暗杠', 7: '补杠'，8：‘胡’
                    '''(1)检测allow_op'''
                    p.allow_op = []  # 重置

                    op_map = {0: []}  # 操作的map

                    if not p.already_hu and p.canPeng(self.outcard):
                        op_map[4] = [self.outcard]
                        p.allow_op.append(4)
                    if p.canMingGang(self.outcard):
                        # todo:胡牌的明杠需要在不改变牌型的情况下 -》 杠前杠后已胡过的牌也能胡
                        if not p.already_hu:
                            op_map[5] = [self.outcard]
                            p.allow_op.append(5)
                            ming_gang_ID = p.seat_id
                        else:
                            tmp_handcards = tool2.deepcopy(p.handcards)
                            tmp_fulus = tool2.deepcopy(p.fulu)
                            for _ in range(3):
                                tmp_handcards.remove(self.outcard)
                            tmp_fulus.append([self.outcard, self.outcard, self.outcard, self.outcard])

                            can_add_gang = True  # 是否可以加杠的flag
                            for hu_card in set(p.already_hu_cards):  # 需要逐一判断杠前后已胡牌集是否都能胡牌
                                gang_after_hu = self.is_hu(tmp_handcards + [hu_card], tmp_fulus, p.choose_color)
                                can_add_gang = gang_after_hu
                                if not can_add_gang:  # 不满足条件，跳出循环
                                    break

                            if can_add_gang:
                                op_map[5] = [self.outcard]
                                p.allow_op.append(5)
                                ming_gang_ID = p.seat_id
                            del tmp_handcards, tmp_fulus  # 销毁

                    # 放炮胡
                    handcards_copy = tool2.deepcopy(p.handcards)
                    handcards_copy.append(self.outcard)
                    handcards_copy.sort()
                    if self.is_hu(handcards_copy, p.fulu, p.choose_color):  # 胡
                        # todo: 这里只是判断符合胡牌的条件
                        # 这里默认选择有胡就胡
                        p.allow_op.append(8)
                        other_hu_type[p.seat_id] = self.hu_type
                        op_map[8] = [self.outcard]
                    p.allow_op.append(0)
                    p.allow_op.sort()

                    '''(2)汇集所有玩家操作'''
                    if len(p.allow_op) == 1:  # 只有“过”
                        p.compe_op = 0
                    if len(p.allow_op) >= 2:  # 除“过”以外，还有其它操作
                        # print("player"+str(p.seat_id)+"  outcard:" + str(self.outcard) + "    允许操作" + str(p.allow_op))
                        # op = input("请输入操作：")
                        state = self.get_state(p, self)
                        op, _ = p.recommend_op(state, op_map)  ######################################################
                        if op in op_map.keys():
                            p.compe_op = op
                        else:
                            print("定缺牌，不能操作")

                '''(2)执行优先级高的op'''
                self.competition_op = [player0.compe_op, player1.compe_op, player2.compe_op, player3.compe_op]

                self.competition_op[current_p_index % 4] = -1  # 不纳入考虑，非本手玩家
                run_op = max(self.competition_op)

                index1 = []  # 可操作的玩家ID
                for j in range(4):
                    if self.competition_op[j] == run_op:
                        index1.append(j)
                if run_op != 8:  # 非胡牌情况下，只能选一个玩家进行操作
                    player_index = random.sample(index1, 1)[0]  # 随机选取
                # player_index = self.competition_op.index(run_op)
                if run_op > 0:  # 如果推荐run_op 大于0， 且在op_map中
                    if run_op == 4:  # 碰
                        player[player_index].peng(self.outcard)
                        self.player_fulu[player[player_index].seat_id] = player[player_index].fulu
                        self.player_discards_display[current_p_index].pop()  # 展示弃牌出栈

                        # 记录对局信息-碰牌
                        discards = dict_to_list4(self.player_discards_display)  # 四个玩家的弃牌
                        discards_real = dict_to_list4(self.player_discards)  # 四个玩家的真实弃牌
                        discards_op = dict_to_list4(self.player_fulu)  # 四个玩家的副露
                        handcards = [player0.handcards, player1.handcards, player2.handcards,
                                     player3.handcards]  # 四个玩家的手牌
                        battle_info_record(self.battle_info, player[player_index % 4].seat_id, discards,
                                           discards_real, discards_op, handcards, "N", self.outcard,
                                           current_p_index % 4,
                                           [self.outcard, self.outcard, self.outcard], self.round)

                        current_p_index = player_index
                        self.to_do = "output_card"


                    elif run_op == 5:  # 明杠
                        self.players_isGang[player_index] = True
                        self.players_minggang_num[player_index] += 1
                        self.players_gang_num[player_index % 4] += 1  # 总杠数+1
                        player[player_index].mingGang(self.outcard)
                        self.player_fulu[player[player_index].seat_id] = player[player_index].fulu
                        cur_round_score_base = int(math.pow(2, 0)) * self.baseM  # 本轮单玩家的基础分
                        score_op(player_index, cur_round_score_base, current_p_index % 4)  # 刮风
                        score_detail = ["刮风(明杠)", cur_round_score_base, [current_p_index]]
                        self.win_result[player_index % 4]["info"].append(score_detail)

                        score_detail = ["被刮风(明杠)", -cur_round_score_base, [player_index]]
                        self.win_result[current_p_index % 4]["info"].append(score_detail)

                        self.players_minggang_list[player_index].append(current_p_index % 4)
                        self.player_discards_display[current_p_index].pop()  # 展示弃牌出栈

                        # 记录对局信息-明杠
                        discards = dict_to_list4(self.player_discards_display)  # 四个玩家的弃牌
                        discards_real = dict_to_list4(self.player_discards)  # 四个玩家的真实弃牌
                        discards_op = dict_to_list4(self.player_fulu)  # 四个玩家的副露
                        handcards = [player0.handcards, player1.handcards, player2.handcards,
                                     player3.handcards]  # 四个玩家的手牌
                        battle_info_record(self.battle_info, player[player_index % 4].seat_id, discards,
                                           discards_real, discards_op, handcards, "k", self.outcard,
                                           current_p_index % 4,
                                           [self.outcard, self.outcard, self.outcard, self.outcard], self.round)

                        current_p_index = player_index
                        self.to_do = "catch_card"


                    elif run_op == 8:  # 胡了
                        self.discards.pop()  # 打出的牌给别人胡
                        self.player_discards_display[current_p_index].pop()  # 展示弃牌出栈

                        for player_index in index1:
                            assert player[player_index % 4].seat_id == player_index % 4

                            # 胡牌标志分配
                            # self.win_result[player[player_index % 4].seat_id]["win"] = 1

                            # 统计每个玩家第一次胡牌的轮数
                            if self.first_win_round[player_index % 4] == -1:
                                self.first_win_round[player_index % 4] = self.round

                            hu_type = other_hu_type[player_index]  # 胡牌类型

                            handcards_copy = player[player_index % 4].handcards + [self.outcard]
                            handcards_copy.sort()

                            cur_player_fulu = []  # 当前玩家副露表 用hex表示
                            for fulu_ in player[player_index % 4].fulu:
                                cur_player_fulu.append(tool2.list10_to_16(fulu_))

                            fanList = SearchInfo.getFanList(
                                paixing=px2idx_dict.get(hu_type, 0),
                                cards=tool2.list10_to_16(handcards_copy),
                                suits=cur_player_fulu,
                                jingCard=0,
                                gangNum=self.players_gang_num[player_index % 4],
                                isHuJudge=True,
                                isSelfTurn=False,
                                preActionIsGang=self.players_isGang[player_index % 4])

                            fanType = hu_type  # 基础番型

                            # 统计这一局的胡牌类型和胡牌番型数(去重)
                            self.hu_type_list[player_index % 4][px2idx_dict[fanType]] = 1
                            for i, e in enumerate(fanList):
                                if e == 1:
                                    self.fan_list[player_index % 4][i] = 1

                            # self.win_result[current_p_index%4]["fan"].append("自摸:(" + self.hu_type + ") ")
                            fan_num = 1  # 番的个数
                            # [清一色、断幺九、碰碰胡、自摸、杠上开花、金钩钩、幺九、一根、两根、三根、四根]
                            fan_map = [2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1]
                            # 番型分
                            for fan_index in range(len(fanList)):
                                if fanList[fan_index] == 1:
                                    fanType += "," + idx2fan_dict[fan_index]
                                    # self.win_result[current_p_index % 4]["fan"].append()
                                    fan_num += fan_map[fan_index]

                            # 设置当前玩家为已经胡牌状态， 并把胡牌加入
                            player[player_index].already_hu = True
                            self.players_already_hu[player_index] = px2idx_dict[hu_type]  # 标记为已胡

                            player[player_index].already_hu_cards.append(self.outcard)
                            self.players_already_hu_cards[player_index].append(self.outcard)  # 添加到胡牌集

                            # 获取最大番
                            if self.max_fan_nums[player_index] < fan_num:
                                self.max_fan_nums[player_index] = fan_num
                                player[player_index].max_fan_list = fanList
                            fan = int(math.pow(2, fan_num - 1))
                            if fan > 256:
                                fan = 256
                            cur_round_score_base = fan * self.baseM  # 本轮单玩家的基础分
                            # 根据番型算分
                            score_op(player_index % 4, cur_round_score_base, current_p_index % 4)

                            score_detail = ["点炮（" + fanType + ")", -cur_round_score_base, [player_index % 4]]
                            self.win_result[current_p_index % 4]["fan"].append(score_detail)

                            score_detail = ["点炮胡（" + fanType + ")", cur_round_score_base, [current_p_index % 4]]
                            self.win_result[player_index % 4]["fan"].append(score_detail)

                            # 判断呼叫转移 当有明杠的玩家操作并且胡牌的玩家为一个时 且可明杠玩家与胡牌玩家不相同时
                            if ming_gang_ID != -1 and len(index1) == 1 and ming_gang_ID != player_index:
                                score_op(player_index, self.baseM, current_p_index % 4)  # 刮风
                                score_detail = ["呼叫转移（原杠ID：" + str(ming_gang_ID) + ")", self.baseM, [current_p_index]]
                                self.win_result[player_index % 4]["info"].append(score_detail)
                                score_detail = ["呼叫转移（原杠ID：" + str(ming_gang_ID) + ")", -self.baseM, [player_index]]
                                self.win_result[current_p_index % 4]["info"].append(score_detail)

                            # 记录对局信息-胡牌
                            discards = dict_to_list4(self.player_discards_display)  # 四个玩家的弃牌
                            discards_real = dict_to_list4(self.player_discards)  # 四个玩家的真实弃牌
                            discards_op = dict_to_list4(self.player_fulu)  # 四个玩家的副露
                            handcards = [player0.handcards, player1.handcards, player2.handcards,
                                         player3.handcards]  # 四个玩家的手牌
                            battle_info_record(self.battle_info, player[player_index % 4].seat_id, discards,
                                               discards_real, discards_op, handcards, "A", self.outcard,
                                               current_p_index % 4, [], self.round)

                        current_p_index = (player_index + 1) % 4
                        self.to_do = "catch_card"
                    else:
                        run_op = 0

                    # 超时检查
                    spend_check_other_time = time.time() - pre_check_other
                    if spend_check_other_time > 3:
                        print("检查其他玩家状态超时,为", str(spend_check_other_time))
                        logger.info("检查其他玩家状态 overtime, need:{}".format(str(spend_check_other_time)))

                if run_op == 0:  # 所有其他玩家都选择“过”,或者不可碰杠
                    player[player_index].guo()
                    current_p_index = (current_p_index + 1) % 4
                    self.to_do = "catch_card"
                t34 = datetime.datetime.now() - t24
                pass

        if not self.card_library:

            # todo：这个时候需要进行查大叫，查花猪，退税等操作
            end_status = {0: {'isHu': False, 'xts': -1}, 1: {'isHu': False, 'xts': -1},
                          2: {'isHu': False, 'xts': -1}, 3: {'isHu': False, 'xts': -1}}
            for player_index in range(4):
                if self.players_already_hu[player_index] > -1:
                    end_status[player_index]['isHu'] = True
                    end_status[player_index]['xts'] = 0
                else:
                    huFlag, res = sc_hu.Mahjong(self.players[player_index].handcards, self.players[player_index].fulu,
                                                self.players[player_index].choose_color).isHu()
                    end_status[player_index]['isHu'] = huFlag
                    end_status[player_index]['xts'] = res

            for player_index in range(4):
                if (self.players_already_hu[player_index] > -1) or end_status[player_index]['xts'] == 1:
                    # 胡了 或者向听数为1都可以不用判断
                    continue
                else:  # 没胡
                    hu_flag, xts = end_status[player_index]['isHu'], end_status[player_index]['xts']

                    if not hu_flag and xts == 14:  # 查花猪
                        for add_score_id in self.other_seat[player_index]:
                            score_op(add_score_id, 16 * self.baseM, player_index)
                            self.win_result[add_score_id]["info"].append(['查花猪', 16 * self.baseM, player_index])
                        self.win_result[player_index]["info"].append(
                            ['花猪', -16 * self.baseM, self.other_seat[player_index]])

                    # 被查花猪不需要查大叫
                    if 1 < xts < 14:  # res 为14代表花猪
                        for check_hu_id in self.other_seat[player_index]:
                            # 未胡且向听数为1的玩家
                            if not end_status[check_hu_id]["isHu"] and end_status[check_hu_id]["xts"] == 1:
                                max_score = get_cadajiao_score(list10_to_16(player[check_hu_id].handcards),
                                                               fulu_translate(player[check_hu_id].fulu))
                                # print("查大叫分数：", max_score)
                                score_op(check_hu_id, max_score * self.baseM, player_index)
                                self.not_ready[player_index] = True
                                self.win_result[check_hu_id]["info"].append(
                                    ["查大叫", max_score * self.baseM, player_index])
                                self.win_result[player_index]["info"].append(
                                    ["被查大叫", -max_score * self.baseM, check_hu_id])
                    # 退税操作
                    if self.players_gang_num[player_index] > 0 and xts > 1:
                        # 暗杠:
                        if self.players_angang_num[player_index]:
                            cur_round_score_base = self.players_angang_num[player_index] * 2 * self.baseM  # 本轮单玩家的基础分
                            score_op(player_index, -cur_round_score_base * 3)
                            tmp = []
                            for i in range(4):
                                if i != player_index:
                                    tmp.append(i)
                                    self.win_result[i]["info"].append(
                                        ["被退税(暗杠)", cur_round_score_base, player_index])
                            self.win_result[player_index]["info"].append(["退税(暗杠)", -cur_round_score_base, tmp])

                        # 补杠:
                        if self.players_bugang_num[player_index]:
                            cur_round_score_base = self.players_bugang_num[player_index] * self.baseM  # 本轮单玩家的基础分
                            score_op(player_index, -3 * cur_round_score_base)
                            tmp = []
                            for i in range(4):
                                if i != player_index:
                                    tmp.append(i)
                                    self.win_result[i]["info"].append(
                                        ["被退税(补杠)", cur_round_score_base, player_index])
                            self.win_result[player_index]["info"].append(["退税(补杠)", -cur_round_score_base, tmp])

                        # 明杠:
                        if self.players_minggang_num[player_index]:
                            cur_round_score_base = self.baseM  # 本轮单玩家的基础分
                            for i in self.players_minggang_list[player_index]:
                                score_op(i, cur_round_score_base, player_index)
                                self.win_result[i]["info"].append(["被退税(明杠)", cur_round_score_base, player_index])
                            self.win_result[player_index]["info"].append(
                                ["退税(明杠)", -cur_round_score_base, self.players_minggang_list[player_index]])
            # for player_index in range(4):
            #     if self.win_result[player_index]["score"] > 0:
            #         self.win_result[player_index]["win"] = 1

            # 获胜规则 -- 本场最高分为获胜者，可以同分
            max_score = 0
            max_idx = []
            for i in range(4):
                if self.win_result[i]["score"] >= max_score:
                    if self.win_result[i]["score"] > max_score:
                        max_idx = [i]
                    else:  # ==
                        max_idx.append(i)
                    max_score = self.win_result[i]["score"]
            for win_idx in max_idx:
                self.win_result[win_idx]["win"] = 1
            self.terminal = True
            # print("play0:", self.win_result[0])
            # print("play1:", self.win_result[1])
            # print("play2:", self.win_result[2])
            # print("play3:", self.win_result[3])
            return


class MahjongEnv(object):
    '''
        四人血流成河四川麻将环境，与gym对应
        className:MahjongEnv
        fileName:mahjong.py
    '''

    def __init__(self, four_player_model_name):
        """
        构造器，初始化玩家实例与游戏实例
        @param four_player_model_name: 四个玩家的模型
        """
        self.action_set = [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23, 24, 25, 26, 27,
                           28, 29]
        self.four_player_model_name = four_player_model_name
        self.player0 = Player_RL("a", 0, self.four_player_model_name[0])
        self.player1 = Player_RL("b", 1, self.four_player_model_name[1])
        self.player2 = Player_RL("c", 2, self.four_player_model_name[2])
        self.player3 = Player_RL("d", 3, self.four_player_model_name[3])
        self.game = Game_RL(is_render=False)

        # self.reset()

    def reset(self):
        """
        参数重置
        @return: state
        """
        while True:
            global i
            i = random.choice([0, 1])
            # 重置玩家实例与游戏实例
            self.game.reset()
            self.player0.reset()
            self.player1.reset()
            self.player2.reset()
            self.player3.reset()
            self.player0.episode = self.game.episode
            self.player1.episode = self.game.episode
            self.player2.episode = self.game.episode
            self.player3.episode = self.game.episode
            # #打乱牌库
            # random.shuffle(self.game.card_library)
            # 给玩家发牌
            self.player0.handcards = self.game.deal_cards(13)  # 训练模型
            self.player1.handcards = self.game.deal_cards(13)  # 陪打模型
            self.player2.handcards = self.game.deal_cards(13)  # 陪打模型
            self.player3.handcards = self.game.deal_cards(13)  # 陪打模型

            # 进行游戏
            # self.game.mah_step(self.player0, self.player1)
            players = [self.player0, self.player1, self.player2, self.player3]

            self.game.players = players  # 初始化对手
            self.game.players[self.game.dealer_seat_id].catch_card = self.game.deal_cards(1)[0]  # 庄家先发牌

            self.player0.handcards.sort()

            self.game.players[self.game.dealer_seat_id].handcards.append(
                self.game.players[self.game.dealer_seat_id].catch_card)

            self.game.to_do = "switch_cards"  # 换牌阶段
            self.game.mah_step(self.player0, self.player1, self.player2, self.player3, self.game.dealer_seat_id)

            if not self.game.terminal:  # 避免开局是天胡等情况
                break
                # 获取状态信息

        state = self.game.get_state(self.player0, self.game, addition_info=True)
        self.state = state
        self.start_flag = True
        return state

    def step(self, action):
        """
        gym中的step函数，出一张牌
        @param action:执行的动作
        @return:[state,reward,done,info]
        """
        # action指示要打出张牌
        if action not in self.player0.handcards:  # 对无效动作设置负面奖励，并提前终止对局
            # if self.game.is_render:
            #     # logging.error("player0.handcards:{}, 出牌：{}动作并不在手牌中，请检查！".format(self.player0.handcards, action))
            #     print("player0.handcards:{}, 出牌：{}动作并不在手牌中，请检查！".format(self.player0.handcards, action))
            print("player0.handcards:{}, 出牌：{}动作并不在手牌中，请检查！".format(self.player0.handcards, action))
            state = self.game.get_state(self.player0, self.game, addition_info=True)
            return state, -16, True, {}
        else:
            global i
            agent_out_card = action  # 仅限Agent出牌，另一个陪打模型由智一出牌
            #  player = [self.player0, self.player1, self.player2, self.player3]
            self.game.outcard = agent_out_card
            self.game.out_seat_id = self.player0.seat_id
            self.game.player_discards[0].append(agent_out_card)
            self.game.player_discards_display[0].append(agent_out_card)
            self.game.discards.append(agent_out_card)
            # print("player0.catch_card:{},  player0.handcards:{}, player0.fulu:{}, player0.outcard:{})".
            #       format(self.player0.catch_card, self.player0.handcards, self.player0.fulu, agent_out_card))
            # # print("\n")
            self.player0.handcards.remove(agent_out_card)

            # # 记录对局信息
            # discards = dict_to_list4(self.game.player_discards_display)  # 四个玩家的弃牌
            # discards_real = dict_to_list4(self.game.player_discards)  # 四个玩家的真实弃牌
            # discards_op = dict_to_list4(self.game.player_fulu)  # 四个玩家的副露
            # handcards = [self.player0.handcards, self.player1.handcards, self.player2.handcards,
            #              self.player3.handcards]  # 四个玩家的手牌
            # battle_info_record(self.game.battle_info, 0, discards, discards_real, discards_op,
            #                    handcards, "d", agent_out_card, 255, [], self.game.round)

            self.game.to_do = "check_others_allow_op"
            # 进行一步游戏
            self.game.mah_step(self.player0, self.player1, self.player2, self.player3)  # player0为训练的模型

            # 获取状态信息
            state = self.game.get_state(self.player0, self.game, addition_info=True)

            # 奖励函数设计  ###考虑用向听数

            '''考虑向听数'''
            reward = 0
            done = False
            if not self.game.terminal:  # 游戏未结束  奖励函数用7-向听数 表示
                # if self.player0.dingQue_cards != [] and (action // 10 != self.player0.choose_color):
                #     reward = -1
                # elif action // 10 == self.player0.choose_color:
                #     reward = 0.1
                pass
                # cur_xt_min = hu.min_xt_add_weight(self.player0.handcards, self.player0.fulu, 0)
                # if self.start_flag:  # 第一手
                #     self.temp_xt = cur_xt_min
                #     self.start_flag = False
                # else:
                #     # reward = (self.temp_xt - cur_xt_min) / 10  # 除10，reward求出sqrt，防止agent过于贪婪
                #     reward = (self.temp_xt - cur_xt_min) / 100  # 除100，防止agent过于贪婪追求中间步奖励
                #     self.temp_xt = cur_xt_min
                # pass
            else:  # 游戏结束
                reward = self.game.win_result[0]["score"]
                reward = math.sqrt(reward) if reward > 0 else -math.sqrt(-reward)
                done = True
                self.start_flag = True
                # print(self.game.win_result) #打印win_result
            #  mah_state = tool2.card_preprocess_suphx_sc(state, search=True, global_state=True)
        self.state = state
        return state, reward, done, {}

    def get_zhiyi_recommend_action(self, state):
        """
        智一版推荐出牌
        @param state: 状态
        @return: 推荐打出的牌
        """
        return self.player0.recommend_card2(state)  # 因为player0为智一模型，直接使用play0的推荐出牌即可


def run():
    """
    测试函数
    """
    # four_player_model_name = [
    #     "scmj_jump",
    #     "scmj_new",
    #     "scmj_jump",
    #     "scmj_new",
    # ]  # 四个玩家类配置
    four_player_model_name = [
        "PPOModel_new",
        "zlc_v3",
        "PPOModel_new",
        "zlc_v3",
    ]  # 四个玩家类配置
    # four_player_model_name = [
    #     "zlc_v3",
    #     "zlc_v3",
    #     "zlc_v3",
    #     "zlc_v3",
    # ]  # 四个玩家类配置
    # four_player_model_name = [
    #     "PPOModel_new",
    #     "PPOModel_local",
    #     "PPOModel_new",
    #     "PPOModel_local",
    # ]  # 四个玩家类配置
    # four_player_model_name = [
    #     "PPOModel_local",
    #     "scmj_zlc_v2_local",
    #     "scmj_zlc_v2_local",
    #     "scmj_zlc_v2_local",
    # ]  # 四个玩家类配置
    # four_player_model_name = [
    #     "scmj_zlc_v1_node3",
    #     "scmj_zlc_v1_node3",
    #     "scmj_zlc_v1_node3",
    #     "scmj_zlc_v1_node3",
    # ]  # 四个玩家类配置
    # four_player_model_name = [
    #     "scmj_zlc_v2",
    #     "scmj_zlc_v2",
    #     "scmj_zlc_v2",
    #     "scmj_zlc_v2",
    # ]  # 四个玩家类配置
    env = MahjongEnv(four_player_model_name)
    model_name = "sc_Test_zlcV2_ai8989"
    logger = logging.getLogger("mahjong.py")
    logger.setLevel(level=logging.DEBUG)
    time_now = datetime.datetime.now()
    handler = logging.FileHandler(
        "./indicator/%s_%i%i%i_2v2.txt" % (model_name, time_now.year, time_now.month, time_now.day))  # 将日志写入对应的文件中
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.info("对打开始...")
    logger.info("四个玩家模型分别是[%s， %s， %s， %s]" % (env.player0.model, env.player1.model,
                                               env.player2.model, env.player3.model))

    not_ready = [0] * 4  # 被查大叫次数(未听牌次数，流局除外)
    win_count = [0] * 4  # 赢的次数
    win_reward = [0] * 4  # 赢时所有分累加
    all_reward = [0] * 4  # 所有分的累加
    dealer_count = [0] * 4  # 当庄次数
    hu_count_distribution = [0] * 12  # 下标为胡的次数，值为场次
    abortCount = 0  # 流局次数
    illegalN = 0  # 出非法牌的次数
    episodes = 10000  # 总的对打局数
    obv_episode = 100  # 观测结果的间隔

    rounds = [0, 0, 0, 0]  # 每个玩家的每局首次胡牌轮数求和
    hu_episodes = [0, 0, 0, 0]  # 每个玩家的胡牌局数
    first_win_round = [0, 0, 0, 0]  # 每个玩家的平均的首次胡牌轮数

    # 统计episodes局中各个胡牌类型番型的局数
    hu_type_list = [[0, 0], [0, 0], [0, 0], [0, 0]]  # 胡牌类型
    fan_list = [[0] * 11, [0] * 11, [0] * 11, [0] * 11]  # 胡牌番型
    pinghu = [0, 0, 0, 0]  # 普通平胡
    qidui = [0, 0, 0, 0]  # 普通七对
    #  [清一色、断幺九、碰碰胡、自摸、杠上开花、金钩钩、幺九、一根、两根、三根、四根]
    idx2fan_dict = {0: "清一色", 1: "断幺九", 2: "碰碰胡", 3: "自摸", 4: "杠上开花", 5: "金钩钩",
                    6: "幺九", 7: "一根", 8: "两根", 9: "三根", 10: "四根"}

    # model_name = "reward0.9359512007854218_3pv5_vs_ppo_sameOp_noPerfectInfo60000"

    # with open("./indicator/win_rate_" + model_name + ".txt", 'w', encoding='utf-8') as f:
    #     f.write("当前EPOCh \t 当前赢次 \t\t\t 赢场均分 \t\t\t 累计分 \t\t 当前胜率  \t\t 流局率 \t 非法牌次数 \t 当庄次数")
    #     f.close()
    pre_time = datetime.datetime.now()

    # 每一局
    for i_episode in range(episodes):
        print(i_episode)
        # 获取回合 i_episode 第一个 observation
        env.game.episode = i_episode
        observation = env.reset()

        # 存储每局游戏的数据
        round_id = get_round_id()
        data = {}
        # with open('record/' + round_id + '.json', 'w', encoding='utf-8') as fw:
        data["round_id"] = round_id
        data["players_id"] = [0, 1, 2, 3]
        data["high_score_id"] = data["players_id"][0]
        data["zhuang_id"] = env.game.dealer_seat_id
        data["hu_type_0"] = ""
        data["hu_type_1"] = ""
        data["hu_type_2"] = ""
        data["hu_type_3"] = ""
        data["grade"] = [0, 0, 0, 0]
        # fw.write(json.dumps(data))
        env.game.json_data_record = data

        while True:
            if env.game.players[0].already_hu == True:
                action = env.game.players[0].catch_card
            else:
                action = env.player0.recommend_card2(observation)

            observation, reward, done, info = env.step(action)  # 获取下一个 state

            # 游戏结束
            if done:

                # 统计每个玩家该局的首次胡牌轮数
                for i, e in enumerate(env.game.first_win_round):
                    if e != -1:
                        hu_episodes[i] += 1
                        rounds[i] += e

                # 统计episodes局中各个胡牌类型番型的局数
                for i in range(4):
                    l = env.game.fan_list[i]
                    ld = env.game.hu_type_list[i]
                    hu_type_list[i] = list(np.add(hu_type_list[i], ld))
                    fan_list[i] = list(np.add(fan_list[i], l))
                    data["grade"][i] = env.game.win_result[i]["score"]

                    # 统计普通平胡和普通七对的局数
                    # [清一色、断幺九、碰碰胡、自摸、杠上开花、金钩钩、幺九、一根、两根、三根、四根]
                    if l[0] == 0 and l[1] == 0 and l[2] == 0 and l[5] == 0 and l[6] == 0:
                        # [平胡,七对]
                        if ld[0] == 1:
                            pinghu[i] += 1
                        elif ld[1] == 1:
                            qidui[i] += 1

                    # 统计胡牌类型和番型
                    if ld[0] == 1:
                        data["hu_type_" + str(i)] += "平胡 "
                    if ld[1] == 1:
                        data["hu_type_" + str(i)] += "七对 "
                    for idx, e in enumerate(l):
                        if e == 1:
                            data["hu_type_" + str(i)] += idx2fan_dict[idx] + " "

                # 记录对打数据
                with open('record/' + round_id + '.json', 'w', encoding='utf-8') as fw:
                    data["battle_info"] = env.game.battle_info
                    fw.write(json.dumps(data, cls=NpEncoder, ensure_ascii=False))

                dealer_count[env.game.dealer_seat_id] += 1
                allow_flag = False  # 判断此局是否正常结束

                for i in range(4):
                    if env.game.win_result[i]["win"] == 1:
                        win_count[i] += 1
                        allow_flag = True  # 正常结束， 达到胡牌条件
                        win_reward[i] += env.game.win_result[i]["score"]
                    all_reward[i] += env.game.win_result[i]["score"]
                    try:
                        hu_count_distribution[len(env.game.players_already_hu_cards[i])] += 1
                    except:
                        hu_count_distribution[-1] += 1
                if allow_flag:  # 游戏正常结束，统计被查大叫次数
                    l = env.game.not_ready
                    for i in range(4):
                        if l[i]:
                            not_ready[i] += 1
                else:  # 符合非法动作的reward并且不因达到胡牌而结束
                    if reward == -8:  # 非法动作
                        illegalN += 1
                    else:  # 流局
                        abortCount += 1
                # print(env.game.win_result)
                break

        if (i_episode + 1) % obv_episode == 0:
            # 计算平均首次胡牌round
            for i, e in enumerate(hu_episodes):
                if e > 0:
                    first_win_round[i] = round(rounds[i] / hu_episodes[i], 1)

            l0 = {"清一色": 0, "断幺九": 0, "碰碰胡": 0, "自摸": 0, "杠上开花": 0, "金钩钩": 0,
                  "幺九": 0, "一根": 0, "两根": 0, "三根": 0, "四根": 0}
            l1 = {"清一色": 0, "断幺九": 0, "碰碰胡": 0, "自摸": 0, "杠上开花": 0, "金钩钩": 0,
                  "幺九": 0, "一根": 0, "两根": 0, "三根": 0, "四根": 0}
            l2 = {"清一色": 0, "断幺九": 0, "碰碰胡": 0, "自摸": 0, "杠上开花": 0, "金钩钩": 0,
                  "幺九": 0, "一根": 0, "两根": 0, "三根": 0, "四根": 0}
            l3 = {"清一色": 0, "断幺九": 0, "碰碰胡": 0, "自摸": 0, "杠上开花": 0, "金钩钩": 0,
                  "幺九": 0, "一根": 0, "两根": 0, "三根": 0, "四根": 0}
            fan = {0: l0, 1: l1, 2: l2, 3: l3}
            for i in range(4):
                j = -1
                for key, val in fan[i].items():
                    j += 1
                    fan[i][key] = fan_list[i][j]

            hu = {0: {"普通平胡/平胡": "0", "普通七对/七对": "0"}, 1: {"普通平胡/平胡": "0", "普通七对/七对": "0"},
                  2: {"普通平胡/平胡": "0", "普通七对/七对": "0"}, 3: {"普通平胡/平胡": "0", "普通七对/七对": "0"}}
            for i in range(4):
                j = -1
                for key, val in hu[i].items():
                    j += 1
                    hu[i][key] = str(hu_type_list[i][j])
                s = hu[i]["普通平胡/平胡"]
                hu[i]["普通平胡/平胡"] = str(pinghu[i]) + '/' + s
                s = hu[i]["普通七对/七对"]
                hu[i]["普通七对/七对"] = str(qidui[i]) + '/' + s

            # print("当前局数：", (i_episode + 1))
            # print("首次胡牌round：", first_win_round)
            # print("胡牌类型：", hu)
            # print("胡牌番型：", fan)
            logger.info("当前局数：%s" % (i_episode + 1))
            logger.info("首次胡牌round：%s" % first_win_round)
            logger.info("胡牌类型：%s" % hu)
            logger.info("胡牌番型：%s" % fan)
            avgWinScore = []
            for i, e in enumerate(win_count):
                if e == 0:
                    avgWinScore.append(0)
                else:
                    avgWinScore.append(win_reward[i] / win_count[i])
            # avgWinScore = np.asarray(win_reward) / np.asarray(win_count)
            winRate = np.asarray(win_count) / (i_episode + 1)
            abortRate = abortCount / (i_episode + 1)  # 流局率
            cur_time = datetime.datetime.now()
            logger.info("当前统计数据：[%s-%s)； 赢总次:%s; 流局率:%s； 当庄次数:%s;胡牌次数分布：%s" % (
                1, i_episode + 1, win_count, abortRate, dealer_count, hu_count_distribution))
            logger.info("被查大叫次数:%s;  胜率:%s; 赢场均分:%s; 累计得分:%s" % (not_ready, winRate, avgWinScore, all_reward))
            logger.info("运行花费时长:%s" % (cur_time - pre_time))
            pre_time = cur_time


# 测试代码
if __name__ == '__main__':
    # game = Mcts_game()
    # res = game.is_hu(tool2.list16_to_10([38, 38]), tool2.fulu_translate([[22, 22, 22], [33, 33, 33], [35, 35, 35], [39, 39, 39]]), tool2.f16_to_10(38))
    # print(res, game.hu_type)
    run()
