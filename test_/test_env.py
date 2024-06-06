#!sftp://tonnn!10.0.10.204:22/home/tonnn/xiu/venvs/scmj_ppo2/bin/python3.6
# -*- coding: utf-8 -*-
# @Time    : 2023/3/3
# @Author  : xiu
# @File    : test_env.py
# @Description: 测试gym环境
from mahEnv import MahEnv
from mahjong import MahjongEnv, Game_RL
from mah_player import Player_RL
import logging
from mah_tool import tool2

class TestMahjong(MahjongEnv):
    '''
        测试麻将对战环境
        className:TestMahjong
        fileName:test_env.py
    '''

    def __init__(self, model):
        """
        构造器
        @param model: 使用的模型
        """
        self.model = model
        self.player0 = Player_RL("a", 0, "zhiyi_last")
        self.player1 = Player_RL("b", 1, "ppo", model)
        self.game = Game_RL()
        self.reset()


def index_to_card(index):
    """
    下标转换成十进制的card
    @param index: 下表
    @return: 十进制麻将牌表示
    """
    card = 0
    if 0 <= index <= 8:
        card = index + 1
    elif 8 < index <= 17:
        card = index + 2
    elif 17 < index <= 26:
        card = index + 3
    elif 26 < index <= 33:
        card = index + 4
    else:
        logging.error("index:", index, "输入错误，请检查")
    return card

class TestEnv(MahEnv):
    '''
        测试麻将对战环境
        className:TestEnv
        fileName:test_env.py
    '''

    def __init__(self, model):
        """
        构造器
        @param model: 模型名称
        """
        self.mahjong = TestMahjong(model)  # 创建麻将环境
        self.states = self.reset()

    def step(self):
        """
        与gym的step方法一致，走一步
        @return: state,reward,done,info
        """
        obs = tool2.get_obs_(self.states)
        action = self.mahjong.model.predict(obs)
        action = index_to_card(action[0])
        state, reward, done, info = self.mahjong.step(action)
        self.states = state
        return state, reward, done, info
