import sys
sys.path.append('../../../')
sys.path.append('../../../../')
sys.path.append("../")
import os
import tensorflow as tf
GPU = "1"  # 使用哪块gpu
os.environ["CUDA_VISIBLE_DEVICES"] = GPU  # Or 2, 3, etc. other than 0

# On CPU/GPU placement
config = tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=True)
# config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.24  # 设置显存占用率大小
tf.compat.v1.Session(config=config)

# from sbln_learning.resnet50_policy_new import Resnet50_policy_byZengw
import mahEnv
from sbln_learning.custom_policy.tf.resnet101_policy_tf import Resnet101_policy
from stable_baselines import PPO2
from stable_baselines.common.schedules import LinearSchedule
'''
  train model from the pretrainModel
'''
# from stable_baselines.common.callbacks import CheckpointCallback
# random_abort-3_from_best_reward=2.74

VERBOSE = 1  # (int) the verbosity level: 0 none, 1 training information, 2 tensorflow debug
# LEARNING_RATE = 3e-4  # (float or callable) The learning rate, it can be a function

N_STEPS = 2048  # (int) The number of steps to run for each environment per update # 每个minibatches的步数
NMINIBATCHES = N_STEPS // 256  # (int) Number of training minibatches per update. For recurrent policies,
                # thINIT_CLIPRANGE/2ee number of environments run in parallel should be a multiple of nminibatches.
NOPTEPOCHS = 8   # (int) Number of epoch when optimizing the surrogate
TOTAL_TIMESTEPS = 100000000  # 总共训练的步数
GAMMA = 0.925  # # reward折扣因子  设计原则  （1 - 1/平均步长） 微调
ENT_COEF = 0.0005  # 策略熵的折扣系数，默认0.01  微调
VF_COEF = 0.5  # loss的熵函数系数 默认0.5  不变
TENSORBOARD_LOG = "./tensorboardLog"  # tensorboard的存放路径
FULL_LOG = False  # 是否记录所有数据记录

NUMB = "C46_" # 编号
DynEnt = "_dynEnt_0.001"    # _dynEnt_0.5"  # 是否动态调节策略熵  数字必须放到后面，并且用_隔开
DynAppkl = ""  # 是否动态调整kl散度
MidReward = str(0.01)  # 是否有中间步奖励
IllegaA = -8   # 非法动作的奖励
POLICY = "Resnet101_tf"  # 使用的策略
DATACLIP=""  # 是否采用数据截断，默认False

INIT_CLIPRANGE = 0.2 # clip阈值 不变
CLIPRANGE = INIT_CLIPRANGE # LinearSchedule(TOTAL_TIMESTEPS, initial_p=INIT_CLIPRANGE, final_p=INIT_CLIPRANGE/2).value

INIT_LR = 2.898e-3
END_LR = 2.898e-4  # end LR率
LEARNING_RATE = LinearSchedule(TOTAL_TIMESTEPS, initial_p=INIT_LR, final_p=END_LR).value  # (float or callable) The learning rate, it can be a function

START_MODEL = "EP0.407_C40"
OPPONENT = "EP0.407_C40"

TENSORBOARD_LOG_NAME = NUMB+"PPO2_fea848_"+DATACLIP+"lr_start"+str(INIT_LR)+"end"+str(END_LR)+"_"+POLICY+"_gm"+str(GAMMA)+"_clip"+str(INIT_CLIPRANGE)+"_entCoef"+\
                       str(ENT_COEF)+"_sqrtR_3p" + DynEnt+"_start"+START_MODEL
# MODEL_PATH = "../pretrain/save_premodel/suphx_features455_premodel_lr0.0002_ep100_ppo2_lr0.0003_" \
#              "gamma0.95_enc_coef0.01_vfCoef0.6_nsteps2048_Resnet18_tfnoGang_noMid_abort-1_1pzhiyi" # 从哪个模型开始恢复训练
MODEL_TO_SAVE_PATH = "./save_model"  # 保存模型的为位置
SAVEMODEL_REWARD = 0.0  # 从多少reward开始保存模型


# MODEL_PATH = "./save_model/ppo2_scResnet101_tf_Nfea848_updateStep2048_lr0.0029411355739136_gamma0.925_entCoef0.0005009243374461306_vfCoef0.5_reward1.2990591371975218_GangIsTrue_MidReward0.01_illegaA-8_abort_reward0_opponentEP1.029_nsteps_57344" # 从哪个模型开始恢复训练
# MODEL_PATH = "./save_model/ppo2_scResnet101_tf_Nfea848_updateStep2048_lr0.0029225861768192_gamma0.925_entCoef0.0005750973137206183_vfCoef0.5_reward0.9885356163645124_GangIsTrue_MidReward0.01_illegaA-8_abort_reward0_opponentEP1.029_nsteps_757760" # 从哪个模型开始恢复训练
# MODEL_PATH = "./save_model/ppo2_scResnet101_tf_Nfea848_updateStep2048_lr0.0029429744945663997_gamma0.925_entCoef0.0005047952010204539_vfCoef0.5_reward0.9933905624784529_GangIsTrue_MidReward0.01_illegaA-8_abort_reward0_opponentEP0.936_nsteps_14336" # 从哪个模型开始恢复训练
# MODEL_PATH = "./save_model/ppo2_scResnet101_tf_Nfea848_updateStep2048_lr0.0029377664076287997_gamma0.925_entCoef0.0004943218694391182_vfCoef0.5_reward0.9856053783417973_GangIsTrue_MidReward0.01_illegaA-8_abort_reward0_opponentEP0.936_nsteps_210944" # 从哪个模型开始恢复训练
# MODEL_PATH = "./save_model/C14_ppo2_policyResnet101_tf_Nfea848_updateStep2048_lr0.002934280280576_gamma0.925_entCoef0.0004968545020875975_vfCoef0.5_reward0.932080404377067_GangIsTrue_MidReward0.01_illegaA-8_abort_reward0_nsteps_1351683pv5" # 从哪个模型开始恢复训练
# MODEL_PATH = "./save_model/B2_ppo2_scResnet101_tf_Nfea848_updateStep2048_lr0.0029415353748479998_gamma0.925_entCoef0.0005047952010204539_vfCoef0.5_reward0.978237145438066_GangIsTrue_MidReward0.01_illegaA-8_abort_reward0_opponentEP0.993_nsteps_57344" # 从哪个模型开始恢复训练
# MODEL_PATH = "./save_model/C15_ppo2_scResnet101_tf_Nfea848_updateStep2048_lr0.0029312712390144_gamma0.925_entCoef0.0004931023671001271_vfCoef0.5_reward0.9999854394053627_GangIsTrue_MidReward0.01_illegaA-8_abort_reward0_opponentEP0.932_C14_nsteps_116736" # 从哪个模型开始恢复训练
# MODEL_PATH = "./save_model/C18_ppo2_policyResnet101_tf_Nfea848_updateStep2048_lr0.002941066258176_gamma0.925_entCoef0.0005036946957750234_vfCoef0.5_reward1.1198106356379058_GangIsTrue_MidReward0.01_illegaA-8_abort_reward0_nsteps_184323pv5" # 从哪个模型开始恢复训练
# MODEL_PATH = "./save_model/B6_ppo2_scResnet101_tf_Nfea848_updateStep2048_lr0.0029279501572607998_gamma0.925_entCoef0.0004931023671001271_vfCoef0.5_reward0.9977211328354519_GangIsTrue_MidReward0.01_illegaA-8_abort_reward0_opponentEP0.9999_C15_nsteps_129024" # 从哪个模型开始恢复训练
# MODEL_PATH = "./save_model/C22_ppo2_scResnet101_tf_Nfea848_updateStep2048_lr0.002939698995712_gamma0.925_entCoef0.0005008419313835436_vfCoef0.5_reward1.2081752480431036_GangIsTrue_MidReward0.01_illegaA-8_abort_reward0_opponentEP1.12_C18_nsteps_51200" # 从哪个模型开始恢复训练
# MODEL_PATH = "./save_model/B9_ppo2_scResnet101_tf_Nfea848_updateStep2048_lr0.0029393207081471998_gamma0.925_entCoef0.0005008419313835436_vfCoef0.5_reward1.0435221634123106_GangIsTrue_MidReward0.01_illegaA-8_abort_reward0_opponentEP1.21_C22_nsteps_16384" # 从哪个模型开始恢复训练
# MODEL_PATH = "./save_model/C23_ppo2_scResnet101_tf_Nfea848_updateStep2048_lr0.00292611108864_gamma0.925_entCoef0.0004904702598775582_vfCoef0.5_reward0.8459850173158985_GangIsTrue_MidReward0.01_illegaA-8_abort_reward0_opponentEP0.998_B6_nsteps_73728" # 从哪个模型开始恢复训练
# MODEL_PATH = "./save_model/B11_ppo2_scResnet101_tf_Nfea848_updateStep2048_lr0.0029219550976_gamma0.925_entCoef0.0004904702598775582_vfCoef0.5_reward0.05146094399722333_GangIsTrue_MidReward0.01_illegaA-8_abort_reward0_opponentEP0.846_C23_nsteps_155648" # 从哪个模型开始恢复训练
# MODEL_PATH = "./save_model/B13_ppo2_scResnet101_tf_Nfea848_updateStep2048_lr0.002920276534272_gamma0.925_entCoef0.0004904702598775582_vfCoef0.5_reward0.011062286702365103_GangIsTrue_MidReward0.01_illegaA-8_abort_reward0_opponentEP0.0515_B11_nsteps_67584" # 从哪个模型开始恢复训练
# MODEL_PATH = "./save_model/B14_ppo2_scResnet101_tf_Nfea848_updateStep2048_lr0.002898683640832_gamma0.925_entCoef0.0004904702598775582_vfCoef0.5_reward0.1457500642847637_GangIsTrue_MidReward0.01_illegaA-8_abort_reward0_opponentEP0.0111_B13_nsteps_129024" # 从哪个模型开始恢复训练
# MODEL_PATH = "./save_model/C37_ppo2_scResnet101_tf_Nfea848_updateStep2048_lr0.0029261618746368_gamma0.925_entCoef0.00048177112355942287_vfCoef0.5_reward0.03467432103893706_GangIsTrue_MidReward0.01_illegaA-8_abort_reward0_opponentEP0.0886_C31_nsteps_321536" # 从哪个模型开始恢复训练
# MODEL_PATH = "./save_model/B17_ppo2_scResnet101_tf_Nfea848_updateStep2048_lr0.0029153753896960003_gamma0.925_entCoef0.00048177112355942287_vfCoef0.5_reward0.44279525740301773_GangIsTrue_MidReward0.01_illegaA-8_abort_reward0_opponentEP0.0347_C37_nsteps_405504" # 从哪个模型开始恢复训练
MODEL_PATH = "./save_model/C40_ppo2_scResnet101_tf_Nfea848_updateStep2048_lr0.002898038181376_gamma0.925_entCoef0.0004808815694338486_vfCoef0.5_reward0.4073617056616255_GangIsTrue_MidReward0.01_illegaA-8_abort_reward0_opponentEP0.146_B14_nsteps_38912" # 从哪个模型开始恢复训练

# 从哪个模型开始恢复训练


INFO_ON_LOG = {"numb": NUMB, "lr": INIT_LR, "lr_info": "LinearSchedule; end_lr:"+str(END_LR),"n_fea": 848,
               "clip":INIT_CLIPRANGE, "clip_info":"LinearSchedule; end_clip:"+str(INIT_CLIPRANGE),
               "nminibatches": NMINIBATCHES, "noptepochs": NOPTEPOCHS, "n_steps": N_STEPS,
               "total_timesteps": TOTAL_TIMESTEPS, "gamma": GAMMA, "ent_coef": ENT_COEF, "vf_coef": VF_COEF,
               "policy": POLICY, "gpu": GPU, "dynEnt": DynEnt, "dynAppkl": DynAppkl, "model_path": MODEL_PATH,
               "GameInfo": {"opponent": [OPPONENT, OPPONENT, OPPONENT], "midReward": MidReward, "sqrtReward": True,
                            "isGang_score": True, "abort_reward": 0, "illegaA": IllegaA},
               "model_save_path": MODEL_TO_SAVE_PATH, "best_model": SAVEMODEL_REWARD}

TENSORBOARD_LOG = TENSORBOARD_LOG + "/n_fea" + str(INFO_ON_LOG["n_fea"])+"_FromBestModel"

if not os.path.exists(TENSORBOARD_LOG):
    os.mkdir(TENSORBOARD_LOG)

env = mahEnv.MahEnv()

model = PPO2(Resnet101_policy, env, gamma=GAMMA, ent_coef=ENT_COEF, verbose=VERBOSE, vf_coef=VF_COEF,
             learning_rate=LEARNING_RATE, cliprange=INIT_CLIPRANGE,
             nminibatches=NMINIBATCHES, noptepochs=NOPTEPOCHS, n_steps=N_STEPS, tensorboard_log=TENSORBOARD_LOG,
             full_tensorboard_log=FULL_LOG)

model = PPO2.load(MODEL_PATH, model.get_env())
# 重新加载配置  如果重新加载，会使预训练模型出问题
model.learning_rate = LEARNING_RATE
# model.cliprange = CLIPRANGE
model.tensorboard_log = TENSORBOARD_LOG

model.learn(total_timesteps=TOTAL_TIMESTEPS, tb_log_name=TENSORBOARD_LOG_NAME, info=INFO_ON_LOG)

model.save("./save_model/ppo2_"+POLICY+"_lr"+str(LEARNING_RATE)+"_gamma"+str(GAMMA)+"_entCoef"+str(ENT_COEF) +
           "_vfCoef"+str(VF_COEF)+"_end_steps_"+str(TOTAL_TIMESTEPS)+"_1pv5_sqrtR_3pv5_mid"+MidReward+DynEnt, True)
