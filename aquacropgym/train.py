from aquacrop import AquaCropModel, Crop, FieldMngt, GroundWater, InitialWaterContent, IrrigationManagement, Soil

import pandas as pd
import matplotlib.pyplot as plt
#import seaborn as sns
import time
import os

from utils import calc_eto_faopm
from aquacropenv import Aquaenv, nebraska_maize_config
from utils import evaluate_agent

import copy
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

#CP  Lat.(deg)= 40.40  Long.(deg)=101.73  Elev.(m)=1072.
gendf = calc_eto_faopm('data/CPWG.dat', 1995, 40.4, 1072, True, ["simyear", "jday", "minTemp", "maxTemp", "precip", "rad"])
gendf.head()

IRR_CAP = 400 # max amount of irrigation (mm/ha) that can be applied in a single season
ACTION_SET = 'depth_discreet' # action sets, alternatives are: 'depth', 'binary', 'smt4', 'depth_discreet'
DAYS_TO_IRR = 5 # 'number of days between irrigation decisions (e.g., 1, 3, 5, 7)
envconfig = nebraska_maize_config.copy() # get default config dictionary
envconfig['gendf'] = gendf # set weather data
envconfig['year2'] = 70 # end of the train/test split
envconfig['normalize_obs'] = True # normalize input observation (with a pre-calculated mean and standard deviation)
envconfig['include_rain'] = True # include rainfall within weather data
envconfig['observation_set'] = 'default' # set of variables to pass to agent
envconfig['max_irr'] = 25 # max irrigation that can be applied in a single irrigation event
envconfig['action_set'] = ACTION_SET # action sets, alternatives are: 'depth', 'binary', 'smt4'
envconfig['days_to_irr'] = DAYS_TO_IRR # 'number of days between irrigation decisions (e.g., 1, 3, 5, 7)
envconfig['max_irr_season'] = IRR_CAP # max amount of irrigation (mm/ha) that can be applied in a single season
envconfig['planting_date'] = '05/01'
InitWC = InitialWaterContent(value=['WP']) #(wc_type='Pct',value=[70])
envconfig['init_wc'] = InitWC
env = Aquaenv(envconfig)
#check_env(env)

models_dir = f"models/PPO-{int(time.time())}"
logdir = f"logs/PPO-{int(time.time())}"

# Create and configure the PPO model
#model = PPO("MlpPolicy", env, verbose=1, n_steps=160, batch_size=512, gamma=1.0, tensorboard_log=logdir)
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=logdir)

proftrain = []
proftest = []
timesteps = []
caps = []

TIMESTEPS = 10000
for i in range(30000000):
    model.learn(total_timesteps=TIMESTEPS,reset_num_timesteps=False, tb_log_name="PPO")
    model.save(f"{models_dir}/{TIMESTEPS*i}")


# for i in range(1, 30000000):
#     model.learn(total_timesteps=1)
#     ts = model.num_timesteps

#     if i % 1000 == 0: # evaluate agent on train and test years
#         print('eval')
#         for irr_cap in [IRR_CAP]:
#             test_env_config = copy.deepcopy(envconfig) # make a copy of the training env
#             test_env_config['evaluation_run'] = True # sets env to evaluation mode

#             train_rew, test_rew = evaluate_agent(model, Aquaenv, test_env_config) # evaluate agent
            
#             proftrain.append(train_rew)
#             proftest.append(test_rew)
#             timesteps.append(ts)
#             caps.append(irr_cap)

#             print(irr_cap, f'Train: {round(train_rew, 3)}')
#             print(irr_cap, f'Test: {round(test_rew, 3)}')

#     if i % 1000 == 0: # save results
#         checkpoint_path = model.save(f"models/ppo_crop_model_{i}.zip")
#         print(checkpoint_path)

#         result_df = pd.DataFrame([timesteps, proftrain, proftest, caps]).T
#         result_df.to_csv(f'outputs/neb_corn_ppo_day_{DAYS_TO_IRR}_act_{ACTION_SET}_cap_{IRR_CAP}.csv')
#         plt.plot(timesteps, proftrain, label='Train Reward')
#         plt.plot(timesteps, proftest, label='Test Reward')
#         plt.legend()
#         plt.show()
