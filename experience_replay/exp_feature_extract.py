#!sftp://tonnn!10.0.10.204:22/home/tonnn/xiu/venvs/scmj_ppo2/bin/python3.6
# -*- coding: utf-8 -*-
# @Time    : 2023/3/3
# @Author  : xiu
# @File    : exp_feature_extract.py
# @Description: 用于麻将博弈数据的经验回放，将保存的json对局数据编码给模型训练
import json
import numpy as np
from mah_tool.feature_extract_v10 import calculate_exp_reply_sys_suphx_sc


def get_param(item, high_player_id):
    """
    从JSON文件提取信息，获取特征编码所需要的参数
    @param item: 某一回合的场面信息
    @param high_player_id: 高手玩家编号
    @return: 特征编码所需要的参数
    """
    handCards0 = item["handcards"][high_player_id]
    fulus = item["discards_op"]
    eff_cards = item["eff_card"]  # 有效牌列表
    dingQue_cards = item["dingQue_cards"]  # 手中定缺牌列表
    all_player_alread_hu_cards = item["all_player_alread_hu_cards"]  # 已胡牌集
    all_player_discards_seq = item["all_player_discards_seq"]  # 弃牌集合
    remain_card_num = item["remain_card_num"]  # 剩余牌数
    round = item["round"]  # 回合数
    dealer_flag = item["dealer_flag"]  # 庄家座位号
    all_player_choose_color = ["all_player_choose_color"]  # 选的花色
    all_player_alread_hu = ["all_player_alread_hu"]  # 是否胡牌
    all_player_max_fan_list = ["all_player_max_fan_list"]  # 最大番
    all_player_handcards = ["all_player_handcards"]  # 手牌
    card_library = ["card_library"]  # 牌墙
    return handCards0, fulus, eff_cards, dingQue_cards, all_player_alread_hu_cards, \
           all_player_discards_seq, remain_card_num, round, \
           dealer_flag, all_player_choose_color, all_player_alread_hu, all_player_max_fan_list, \
           all_player_handcards, card_library


def exp_card_preprocess_suphx_sc(item, high_player_id, search=False, global_state=False, dropout_prob=0):
    '''
    四川麻将特征提取并格式化,获取某一个动作的特征表示
    @param item: 某一回合的场面信息
    @param high_player_id: 高手玩家编号
    @param search: 开启前瞻搜索特征
    @param global_state: 是否编码隐藏信息特征
    @param dropout_prob: 对隐藏信息特征的dropout的概率
    @return: 模型输入所需要的特征
    '''
    features = calculate_exp_reply_sys_suphx_sc(item, high_player_id, search, global_state, dropout_prob)
    features = np.array(features)
    features = features.T
    features = np.expand_dims(features, 0)
    features = features.transpose([2, 1, 0])  # 更换位置  转换成c × 34 × 1的格式

    return features


def json_feature_calculate_sys_suphx_sc(i, search=False, global_state=False, dropout_prob=0):
    '''
    获取整局游戏所有手动作的特征表示
    牌都是用16进制进行表示，参数需要预先处理好
    @param i:第i个json文件
    @param search:开启前瞻搜索特征
    @param global_state:是否编码隐藏信息特征
    @param dropout_prob:对隐藏信息特征的dropout的概率
    @return:整局游戏的所有手动作的特征表示
    '''
    # 读json文件
    with open("../sbln_learning/training_files/exp_files" + i + ".json", 'r', encoding='utf-8')as f:
        data = json.load(f)
    # 高手玩家id
    high_player_id = data["players_id"].index(data["high_score_id"])

    # 每局游戏所有动作的feature
    features_list = []

    # 每一手动作
    for item in data["battle_info"]:
        features = exp_card_preprocess_suphx_sc(item, high_player_id, search, global_state, dropout_prob)
        features_list.append(features)

    # 返回整局游戏的所有手动作的feature列表
    return features_list
