#!sftp://tonnn!10.0.10.204:22/home/tonnn/xiu/venvs/scmj_ppo2/bin/python3.6
# -*- coding: utf-8 -*-
# @Time    : 2023/3/3
# @Author  : xiu
# @File    : mah_player.py
# @Description: 玩家信息，包括推荐出牌、推荐动作、换三张推荐、定缺推荐、执行碰杠动作等

import random
# from interface.interface_v1.recommand import recommand_card as recommand_card1 #智一
# from interface.interface_v1.recommand import recommand_op as recommand_op1
# 四川麻将搜索树版本推荐出牌
from interface.sichuanMJ import sichuanMJ_v2  # 拉胯版本
from interface.sichuanMJ import sichuanMJ_v1
from interface.sichuanMJ import sichuanMJ_zlc_v2
# from interface.sichuanMJ.sichuanMJ_v3 import recommend_card
# from interface.sichuanMJ.sichuanMJ_v3 import recommend_op
# from interface.sichuanMJ.sichuanMJ_v3 import recommend_switch_cards
# from interface.sichuanMJ.sichuanMJ_v3 import recommend_choose_color

from mah_tool import url_recommend
from stable_baselines.ppo2 import PPO2
from mah_tool import tool2


class Player_RL(object):
    '''
        玩家信息，包括推荐出牌、推荐动作、换三张推荐、定缺推荐、执行碰杠动作等
        className:Player_RL
        fileName:mah_player.py
    '''

    def __init__(self, name, seat_id, model_name, brain=None):
        """
        构造器
        @param name: 玩家名称
        @param seat_id: 玩家座位号
        @param model_name: 模型名称
        @param brain: DQN模型
        """
        self.model = model_name  # 模型名称
        self.catch_card = 0  # 摸的牌
        self.seat_id = seat_id  # 座位号
        self.name = name  # 玩家名称
        self.handcards = []  # 玩家手牌
        self.fulu = []  # 玩家副露
        self.allow_op = []  # 允许操作
        self.allow_handcards = self.handcards  # 允许操作的手牌
        self.compe_op = -1  # 竞争性操作(左吃，中吃，右吃，碰，明杠，胡)

        self.brain1 = brain  # BrainDQN(34) #先不改
        self.isInitstate = True  # 用于为DQN获取初始状态的标志
        self.action = None  # 存储DQN上一次的动作
        self.reward = None  # 存储DQN与上一次的动作对应的奖励

        # 血流成河-换三张玩家额外信息
        self.choose_color = -1  # 定缺的花色下标   0 - 万 1-条 2-筒
        self.switch_cards = []  # 初始换牌
        self.already_hu = False  # 是否已经胡牌标志位
        self.already_hu_cards = []  # 已胡牌列表
        # [清一色、断幺九、碰碰胡、自摸、杠上开花、金钩钩、幺九、一根、两根、三根、四根]
        self.max_fan_list = [0] * 11  # 最大番列表
        self.eff_cards_list = []  # 有效牌列表
        self.dingQue_cards = []  # 手中定缺牌列表
        self.episode = 0  # 轮次数

        self.PPOMODEL_FLAG = False  # 是否启用PPOModel进行决策
        # if model_name == "PPOModel" and self.PPOMODEL_FLAG:
        if model_name in ["PPOModel_local", "PPOModel_new"]:
            # model_path = "ppo2_policyResnet101_tf_Nfea1274_updateStep2048_lr0.00028860349439999996_gamma0.925_entCoef-0.03278114010654392_vfCoef0.5_reward0.9359512007854218_GangIsFalse_MidReward0.01_illegaA-3_abort_reward-1_nsteps_4222976"
            path = "Node5_A4_ppo2_scResnet101_tf_Nfea848_updateStep256_lr0.0022618484736_gamma0.925_entCoef0.0005_vfCoef0.5_reward3.3032884406347343_GangIsTrue_MidReward0_illegaA-16_abort_reward0_opponentraw_nsteps_6397952"
            model_path = "sbln_learning/training_files/train_from_saveModel/saveModel/" + path
            self.brain1 = PPO2.load(model_path)

    def reset(self):
        """
        玩家信息重置
        """
        self.catch_card = 0
        self.handcards = []  # 玩家手牌
        self.fulu = []  # 玩家副露
        self.allow_op = []  # 允许操作
        self.allow_handcards = self.handcards  # 允许操作的手牌
        self.compe_op = -1  # 竞争性操作(左吃，中吃，右吃，碰，明杠，胡)

        # 血流成河-换三张玩家额外信息
        self.choose_color = -1  # 定缺的花色下标   0 - 万 1-条 2-筒
        self.switch_cards = []  # 初始换牌
        self.already_hu = False  # 是否已经胡牌标志位
        self.already_hu_cards = []  # 已胡牌列表
        # [清一色、断幺九、碰碰胡、自摸、杠上开花、金钩钩、幺九、一根、两根、三根、四根]
        self.max_fan_list = [0] * 11  # 最大番列表
        self.eff_cards_list = []  # 有效牌列表
        self.dingQue_cards = []  # 手中定缺牌列表

        self.isInitstate = True
        self.action = None
        self.reward = None

    def recommend_card2(self, state):
        """
        推荐出牌
        @param state: 状态信息
        @return: 推荐打出的牌
        """
        # import reward
        # 转换格式
        if self.model == "random":  # 完全随机
            return random.sample(self.handcards, 1)[0]
        elif self.model == "random_partly":  # 部分随机，将手牌分成定缺牌与非定缺再随机
            dingQue_cards = self.get_dingQue_cards(state.handcards, self.choose_color)
            if dingQue_cards:
                return random.sample(dingQue_cards, 1)[0]
            else:
                return random.sample(self.handcards, 1)[0]

        elif self.model == "scmjV1":  # 四川麻将v1
            handCards = tool2.list10_to_16(state.handcards)
            allow_hand_cards = handCards

            actions = []
            for i in state.fulu:
                actions.append(tool2.list10_to_16(i))

            discards_op = [[], [], [], []]
            discards = [[], [], [], []]
            for i in range(4):
                discards_op[i] = tool2.fulu_translate(state.player_fulu[i])
                discards[i] = tool2.list10_to_16(state.player_discards[i])

            _, card = sichuanMJ_v1.recommend_card_rf(allow_hand_cards, actions, state.round,
                                                     remain_num=state.remain_card_num,
                                                     discards=discards,
                                                     discards_real=discards, discards_op=discards_op,
                                                     seat_id=self.seat_id,
                                                     choose_color=state.players_choose_color, hu_cards=[[], [], [], []],
                                                     hu_fan=[[], [], [], []])
            card = tool2.f16_to_10(card)
            return card
        elif self.model == "scmjV2":  # 四川麻将v2 拉胯版本
            dingQue_cards = self.get_dingQue_cards(state.handcards, self.choose_color)  # 先做预处理
            if dingQue_cards:
                return random.sample(dingQue_cards, 1)[0]
            else:
                handCards = tool2.list10_to_16(state.handcards)
                allow_hand_cards = handCards

                actions = []
                for i in state.fulu:
                    actions.append(tool2.list10_to_16(i))

                discards_op = [[], [], [], []]
                discards = [[], [], [], []]
                for i in range(4):
                    discards_op[i] = tool2.fulu_translate(state.player_fulu[i])
                    discards[i] = tool2.list10_to_16(state.player_discards[i])

                card = sichuanMJ_v2.recommend_card(allow_hand_cards, actions, state.round,
                                                   remain_num=state.remain_card_num, discards=discards,
                                                   discards_real=discards, discards_op=discards_op,
                                                   seat_id=self.seat_id,
                                                   choose_color=state.players_choose_color, hu_cards=[[], [], [], []],
                                                   hu_fan=[[], [], [], []])
                card = tool2.f16_to_10(card)
                return card
        elif self.model == "scmj_zlc_v2_local":
            handCards = tool2.list10_to_16(state.handcards)
            allow_hand_cards = handCards

            actions = []
            for i in state.fulu:
                actions.append(tool2.list10_to_16(i))

            discards_op = [[], [], [], []]
            discards = [[], [], [], []]
            discards_real = [[], [], [], []]
            for i in range(4):
                discards_op[i] = tool2.fulu_translate(state.player_fulu[i])
                discards[i] = tool2.list10_to_16(state.player_discards_display[i])
                discards_real[i] = tool2.list10_to_16(state.player_discards[i])

            _, card = sichuanMJ_zlc_v2.recommend_card_rf(allow_hand_cards, actions, state.round,
                                                         remain_num=state.remain_card_num,
                                                         discards=discards,
                                                         discards_real=discards_real, discards_op=discards_op,
                                                         seat_id=self.seat_id,
                                                         choose_color=state.players_choose_color,
                                                         hu_cards=[[], [], [], []],
                                                         hu_fan=[[], [], [], []])
            card = tool2.f16_to_10(card)
            return card
        elif self.model == "scmj_zlc_v1_1":
            return url_recommend.get_url_recommend(state, self.seat_id, False,
                                                   "http://172.81.238.92:8888/sichuan/v1/outcard")
        elif self.model == "scmj_zlc_v2":
            return url_recommend.get_url_recommend(state, self.seat_id, False,
                                                   "http://172.81.238.92:8989/sichuan/v2/outcard")
        elif self.model == "scmj_zlc_v1_node3":
            return url_recommend.get_url_recommend(state, self.seat_id, False,
                                                   "http://10.0.10.203:8989/sichuan/v2/outcard")
        elif self.model == "scmj_zlc_v1_1":
            return url_recommend.get_url_recommend(state, self.seat_id, False,
                                                   "http://172.81.238.92:8888/sichuan/zlc_v1_1/outcard")
        elif self.model == "scmj_jump":
            return url_recommend.get_url_recommend(state, self.seat_id, False,
                                                   "http://172.81.238.92:8999/sichuan/v2/outcard")
        elif self.model == "scmj_new":
            return url_recommend.get_url_recommend(state, self.seat_id, False,
                                                   "http://172.81.238.92:8999/sichuan/v2/outcard")
        elif self.model == "PPOModel_local":
            features = tool2.card_preprocess_suphx_sc(state, True, global_state=True)
            action, _ = self.brain1.predict(features)
            result = tool2.index_to_card(action)
            return result
        elif self.model == "PPOModel":
            if self.PPOMODEL_FLAG:
                features = tool2.card_preprocess_suphx_sc(state, True, global_state=True)
                action, _ = self.brain1.predict(features)
                result = tool2.index_to_card(action)
            else:
                result = url_recommend.get_url_recommend(state, self.seat_id, False)
            return result
        elif self.model == "PPOModel_new":
            # features = tool2.card_preprocess_suphx_sc(state, True, global_state=True)
            # action, _ = self.brain1.predict(features)
            # result = tool2.index_to_card(action)
            # return url_recommend.get_url_recommend_new(state, self.seat_id, self.episode, result)
            return url_recommend.get_url_recommend_new(state, self)
        elif self.model == "zlc_v3":
            return url_recommend.get_url_recommend(state, self.seat_id, False,
                                                   "http://172.81.238.92:8999/sichuan/v3_1/outcard")
        elif self.model == "PPOModel_8089":
            return url_recommend.get_url_recommend(state, self.seat_id, False,
                                                   "http://10.0.10.202:8089/sichuanMJ/RLV2/outcard")
        elif self.model == "PPOModel_8090":
            return url_recommend.get_url_recommend(state, self.seat_id, False,
                                                   "http://10.0.10.202:8090/sichuanMJ/RLV2/outcard")
        elif self.model == "PPOModelV3_8030":
            return url_recommend.get_url_recommend(state, self.seat_id, False,
                                                   "http://10.0.10.202:8030/sichuanMJ/RLV3/outcard")
        elif self.model == "PPOModelV3_8031":
            return url_recommend.get_url_recommend(state, self.seat_id, False,
                                                   "http://10.0.10.202:8031/sichuanMJ/RLV3/outcard")
        else:  # ppo
            print("--------模型选择错误， 默认随机--------")
            return random.sample(self.handcards, 1)[0]

    def recommend_op(self, state, op_map=None):
        '''
        推荐动作
        :param state: 状态
        :param op_map: 待请求决策的map表， key为动作决策下标 value为动作决策对应的牌集
        :return: 操作动作，操作了哪张牌  比如，有两个暗杠，需要知道是对哪个暗杠进行操作的
        '''
        if self.model == "scmjV1":
            handCards = tool2.list10_to_16(state.handcards)

            actions = []
            for i in state.fulu:
                actions.append(tool2.list10_to_16(i))

            discards_op = [[], [], [], []]
            discards = [[], [], [], []]
            for i in range(4):
                discards_op[i] = tool2.fulu_translate(state.player_fulu[i])
                discards[i] = tool2.list10_to_16(state.player_discards[i])

            for key in sorted(op_map.keys(), reverse=True):
                if key == 8:
                    # todo: 具体是否选择胡牌，需要重新编写是否选择胡牌的AI 默认有胡必胡
                    # recommend_opCards, hu_flag = recommend_op(None, handCards, actions, state.round,
                    #                                           remain_num=state.remain_card_num
                    #                                           , discards=discards, discards_real=discards,
                    #                                           discards_op=discards_op,
                    #                                           seat_id=self.seat_id, choose_color=state.players_choose_color,
                    #                                           hu_cards=[[], [], [], []], hu_fan=[[], [], [], []])
                    # if hu_flag: return key
                    return key, None
                else:
                    for card in op_map[key]:
                        card = tool2.f10_to_16(card)
                        self_turn = True if len(handCards) % 3 == 2 else False  # 判断是不是自己的轮次 14张牌时为自己轮次
                        # if self.model == "scmjV1":
                        recommend_opCards, hu_flag = sichuanMJ_v1.recommend_op_rf(card, handCards, actions, state.round,
                                                                                  remain_num=state.remain_card_num,
                                                                                  discards=discards,
                                                                                  discards_real=discards,
                                                                                  discards_op=discards_op,
                                                                                  seat_id=self.seat_id,
                                                                                  choose_color=state.players_choose_color,
                                                                                  hu_cards=[[], [], [], []],
                                                                                  hu_fan=[[], [], [], []],
                                                                                  self_turn=self_turn)

                        # recommend_opCards, hu_flag = recommend_op(card, handCards, actions, state.round, remain_num=state.remain_card_num,
                        #                                           discards=discards,discards_real=discards, discards_op=discards_op,
                        #                                           seat_id=self.seat_id,choose_color=state.players_choose_color,
                        #                                           hu_cards=[[], [], [], []], hu_fan=[[], [], [], []], self_turn=self_turn)
                        if card in recommend_opCards:
                            return key, tool2.f16_to_10(card)
            return 0, None

        elif self.model == "scmjV1_URL":
            op_cards, hu_flag = url_recommend.get_url_recommend_op(state, self.seat_id, list(op_map.keys()), False,
                                                                   "http://172.81.238.92:8888/sichuan/v1/operate")
            op = url_recommend.trans_result2Op(op_cards, state.handcards, hu_flag, state.outcard)
            op_card = None if not op_cards else op_cards[0]
            # print("获取动作决策：{}， 返回操作牌：{}".format(op, op_card))
            return op, op_card
        elif self.model == "scmj_zlc_v2_ai":
            op_cards, hu_flag = url_recommend.get_url_recommend_op(state, self.seat_id, list(op_map.keys()), False,
                                                                   "http://172.81.238.92:8888/sichuan/v1/operate")
            op = url_recommend.trans_result2Op(op_cards, state.handcards, hu_flag, state.outcard)
            op_card = None if not op_cards else op_cards[0]
            return op, op_card
        elif self.model == "scmj_zlc_v2_1":
            op_cards, hu_flag = url_recommend.get_url_recommend_op(state, self.seat_id, list(op_map.keys()), False,
                                                                   "http://172.81.238.92:8888/sichuan/v1/operate")
            op = url_recommend.trans_result2Op(op_cards, state.handcards, hu_flag, state.outcard)
            op_card = None if not op_cards else op_cards[0]
            return op, op_card
        elif self.model == "scmj_zlc_v1_node3":
            op_cards, hu_flag = url_recommend.get_url_recommend_op(state, self.seat_id, list(op_map.keys()), False,
                                                                   "http://10.0.10.203:8989/sichuan/v2/operate")
            op = url_recommend.trans_result2Op(op_cards, state.handcards, hu_flag, state.outcard)
            op_card = None if not op_cards else op_cards[0]
            return op, op_card
        elif self.model == "scmj_jump" or self.model == "scmj_new":
            op_cards, hu_flag = url_recommend.get_url_recommend_op(state, self.seat_id, list(op_map.keys()), False,
                                                                   "http://172.81.238.92:8999/sichuan/v2/operate")
            op = url_recommend.trans_result2Op(op_cards, state.handcards, hu_flag, state.outcard)
            op_card = None if not op_cards else op_cards[0]
            return op, op_card
        elif self.model == "zlc_v3" or self.model == "PPOModel_new":
            op_cards, hu_flag = url_recommend.get_url_recommend_op(state, self.seat_id, list(op_map.keys()), False,
                                                                   "http://172.81.238.92:8999/sichuan/v3/operate")
            op = url_recommend.trans_result2Op(op_cards, state.handcards, hu_flag, state.outcard)
            op_card = None if not op_cards else op_cards[0]
            return op, op_card
        else:  # 默认使用 sichuanMJ_zlc_v2.recommend_op_rf
            handCards = tool2.list10_to_16(state.handcards)
            actions = []
            for i in state.fulu:
                actions.append(tool2.list10_to_16(i))
            discards_op = [[], [], [], []]
            discards = [[], [], [], []]
            discards_real = [[], [], [], []]
            for i in range(4):
                discards_op[i] = tool2.fulu_translate(state.player_fulu[i])
                discards[i] = tool2.list10_to_16(state.player_discards_display[i])
                discards_real[i] = tool2.list10_to_16(state.player_discards[i])
            self_turn = True if len(handCards) % 3 == 2 else False  # 判断是不是自己的轮次 14张牌时为自己轮次
            card = tool2.f10_to_16(state.outcard)
            is_hu = 8 in op_map.keys()
            op_cards, hu_flag = sichuanMJ_zlc_v2.recommend_op_rf(card, handCards, actions, state.round,
                                                                 remain_num=state.remain_card_num,
                                                                 discards=discards,
                                                                 discards_real=discards_real,
                                                                 discards_op=discards_op,
                                                                 seat_id=self.seat_id,
                                                                 choose_color=state.players_choose_color,
                                                                 hu_cards=[[], [], [], []],
                                                                 hu_fan=[[], [], [], []],
                                                                 self_turn=self_turn,
                                                                 isHu=is_hu)
            op_cards = tool2.list16_to_10(op_cards)  # 十六进制转换成十进制
            op = url_recommend.trans_result2Op(op_cards, state.handcards, hu_flag, state.outcard)
            op_card = None if not op_cards else op_cards[0]
            return op, op_card

    def recommend_switch_cards(self, handcards, switch_n=3):
        """
        换三张推荐
        @param handcards: 玩家手牌
        @param switch_n: 换张数目，四川麻将血流成河默认换3张
        @return: 推荐玩家舍弃的3张牌
        """
        handcards_hex = tool2.list10_to_16(handcards)
        # if self.model == "scmj_zlc_v1_node3":
        #     return url_recommend.get_url_recommend_switch_cards(handcards_hex, 3,
        #                                                         "http://10.0.10.203:8989/sichuan/v2/switch_cards")
        # elif self.model == "scmj_zlc_v2":
        #     return url_recommend.get_url_recommend_switch_cards(handcards_hex, 3,
        #                                                         "http://172.81.238.92:8989/sichuan/v2/switch_cards")
        # elif self.model == "scmj_jump" or self.model == "scmj_new":
        #     return url_recommend.get_url_recommend_switch_cards(handcards_hex, 3,
        #                                                         "http://172.81.238.92:8999/sichuan/v2/switch_cards")
        # if self.model == "zlc_v3":
        #     return url_recommend.get_url_recommend_switch_cards(handcards_hex, 3,
        #                                                         "http://10.0.10.203:9800/sichuan/v2/switch_cards")
        # else:
        switch_cards = sichuanMJ_zlc_v2.recommend_switch_cards(handcards_hex, switch_n)
        switch_cards = tool2.list16_to_10(switch_cards)
        return switch_cards

    def recommend_choose_color(self, state, switch_n=3):
        """
        定缺花色推荐
        @param state: 状态信息
        @param switch_n: 换张数目
        @return: 推荐的定缺花色
        """
        handcards = state.handcards
        handcards_hex = tool2.list10_to_16(handcards)
        # choose_color = -1

        # if self.model == "scmj_zlc_v1_node3":
        #     return url_recommend.get_url_recommend_choose_color(handcards_hex, 3,
        #                                                         "http://10.0.10.203:8989/sichuan/v2/choose_color")
        # elif self.model == "scmj_jump" or self.model == "scmj_new":
        #     return url_recommend.get_url_recommend_choose_color(handcards_hex, 3,
        #                                                         "http://172.81.238.92:8999/sichuan/v2/choose_color")
        # if self.model == "zlc_v3":
        #     return url_recommend.get_url_recommend_choose_color(handcards_hex, 3,
        #                                                         "http://10.0.10.203:9800/sichuan/v2/choose_color")
        # else:
        return sichuanMJ_zlc_v2.recommend_choose_color(handcards_hex, switch_n)

    def get_dingQue_cards(self, handcards, choose_color, Dec=True):
        """
        获取玩家手牌中的定缺牌列表（十进制）
        @param handcards: 手牌
        @param choose_color: 定缺花色
        @param Dec:是否十进制表示
        @return: 手牌中的定缺牌列表
        """
        DingQue_cards = []
        if Dec:  # 默认十进制
            for card in handcards:
                if card // 10 == choose_color:
                    DingQue_cards.append(card)
        else:  # 十六进制
            for card in handcards:
                if card // 16 == choose_color:
                    DingQue_cards.append(card)
        return DingQue_cards

    def peng(self, op_card):
        """
        执行碰牌动作
        @param op_card: 被碰的牌
        @return:
        """
        assert op_card in self.handcards
        self.handcards.remove(op_card)
        self.handcards.remove(op_card)
        self.fulu.append([op_card, op_card, op_card])
        return

    def buGang(self, op_card):
        """
        执行补杠动作
        @param op_card: 被补杠的牌
        @return:
        """
        assert op_card in self.handcards
        for fulu in self.fulu:
            if fulu.count(op_card) == 3:
                self.handcards.remove(op_card)
                fulu.append(op_card)
                break
        return

    def mingGang(self, op_card):
        """
        执行明杠动作
        @param op_card: 被明杠的牌
        @return:
        """
        assert op_card in self.handcards
        if self.handcards.count(op_card) == 3:
            self.handcards.remove(op_card)
            self.handcards.remove(op_card)
            self.handcards.remove(op_card)
            self.fulu.append([op_card, op_card, op_card, op_card])
        return

    def anGang(self, op_card):
        """
        执行暗杠动作
        @param op_card: 被暗杠的牌
        @return:
        """
        assert op_card in self.handcards
        self.handcards.remove(op_card)
        self.handcards.remove(op_card)
        self.handcards.remove(op_card)
        self.handcards.remove(op_card)
        self.fulu.append([op_card, op_card, op_card, op_card])
        return

    def guo(self):
        """
        执行过牌动作
        @return:
        """
        # reset允许操作的动作
        self.allow_op = []
        return

    def canPeng(self, outcard):
        """
        是否可以碰牌
        @param outcard: 其他玩家刚打出的牌
        @return: bool值
        """
        # if  self.handcards.count(outcard)>=2 and self.choose_color ==  (outcard//10):
        #     print(self.choose_color, outcard)
        if self.choose_color != (outcard // 10) and self.handcards.count(outcard) >= 2:
            return True
        return False

    def canMingGang(self, outcard):
        """
        是否可以明杠
        @param outcard: 其他玩家丢弃的牌
        @return: bool值
        """
        if self.choose_color != (outcard // 10) and self.handcards.count(outcard) == 3:
            return True
        return False

    def canAnGang(self):
        """
        是否可以暗杠
        @return: bool值
        """
        flag = False
        op_list = []
        for i in set(self.handcards):
            if self.choose_color != (i // 10) and self.handcards.count(i) == 4:
                flag = True
                op_list.append(i)
            # if self.handcards.count(i) == 4 and self.choose_color == (i // 10):
            #     print("判断暗杠出错！", self.handcards, self.choose_color, i)
        # 这里添加op_list是因为可以有多个暗杠的情况出现
        return flag, op_list

    def canBuGang(self):
        """
        是否可以补杠
        @return: bool值
        """
        flag = False
        op_list = []
        for i in self.fulu:
            if i.count(i[0]) == 3:
                if i[0] in self.handcards:
                    flag = True
                    op_list.append(i[0])
        # 补杠也是会出现多个情况
        return flag, op_list
