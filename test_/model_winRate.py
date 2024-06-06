#!sftp://tonnn!10.0.10.204:22/home/tonnn/xiu/venvs/scmj_ppo2/bin/python3.6
# -*- coding: utf-8 -*-
# @Time    : 2023/3/3
# @Author  : xiu
# @File    : model_winRate.py
# @Description: 评测模型效果
from stable_baselines import PPO2, PPO1
from test_.test_env import TestEnv
from sbln_learning.resnetPolicy import ResNetPolicy
from sbln_learning.self_policy import selfPolicy
from mahEnv import MahEnv
import matplotlib.pyplot as plt
import numpy as np


EPOCH = 1000000
RATE_WIN_DISPLAY = 1000

env = MahEnv()
# model = PPO2(ResNetPolicy, env, learning_rate=2.5e-6, verbose=1, n_cpu_tf_sess=8)
model = PPO1(selfPolicy, env, verbose=1, optim_epochs=4, optim_stepsize=2.5e-5, optim_batchsize=64, lam=0.95, adam_epsilon=1e-6,n_cpu_tf_sess=8)
model = model.load("/media/lei/0EAA16130EAA1613/zengw_code/Zonst/shangrao_mahjong_rl_v3/sbln_learning/save_model/0.8924800000000005_reward", model.get_env())
test_env = TestEnv(model)
model_name = "0_8924reward_reward_zhiyi_deep_vs_last"


all_ep_r = []  #
with open("./indicator/win_rate_"+model_name+".txt", 'w', encoding='utf-8') as f:
    f.write("当前EPOCh \t 当前胜率  \t 当前败率 \t 流局率")
    f.close()
win_count = 0
fail_count = 0
win_rate = []
fail_rate = 0
for ep in range(EPOCH):
    done = False
    obs_ = test_env.reset()
    ep_r = 0
    while not done:
        state, reward, done, info = test_env.step()
        ep_r += reward
        if done:
            if state.game.win_result[1]["win"] == 1:
                win_count += 1
            if state.game.win_result[0]["win"] == 1:
                fail_count += 1
            print("本局完成，重新恢复状态")

    if ep % RATE_WIN_DISPLAY == 0:
        win_rate_ = win_count / RATE_WIN_DISPLAY  # 当前胜率

        fail_rate_ = fail_count / RATE_WIN_DISPLAY # 当前败率

        flow_rate_ = 1 - win_rate_ - fail_rate_

        with open("indicator/win_rate_"+model_name+".txt", 'a', encoding='utf-8') as f:
            f.write("\n{} \t {} \t {} \t {}".format(ep, win_rate_, fail_rate_, flow_rate_))
            f.close()

        print("ep: {}/{}当前胜率：{}, 败率：{}， 流局率{}".format(ep, EPOCH, win_rate_, fail_rate_, flow_rate_))
        win_rate.append(win_rate_)
        win_count = 0
        fail_count = 0

    if ep == 0:
        all_ep_r.append(ep_r)
    else:
        all_ep_r.append(all_ep_r[-1] * 0.9 + ep_r * 0.1)

f.close()
# plot reward change and test
plt.plot(np.arange(len(all_ep_r)), all_ep_r)
plt.xlabel('Episode')
plt.ylabel('Moving averaged episode reward')
plt.savefig("indicator/"+ model_name +"fig.png")
plt.show()

plt.plot(np.arange(len(win_rate)), win_rate)
plt.xlabel("Episode*1000")
plt.ylabel("The win_rate")
plt.savefig("indicator/"+model_name+".png")
plt.show()




