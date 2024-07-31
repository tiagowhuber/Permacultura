from aquacrop import AquaCropModel, Crop, FieldMngt, GroundWater, InitialWaterContent, IrrigationManagement, Soil
from stable_baselines3.common.vec_env import DummyVecEnv

from utils import calc_eto_faopm
from aquacropenv import Aquaenv, nebraska_maize_config
from stable_baselines3 import PPO
from utils import *
import matplotlib.pyplot as plt
import seaborn as sns
import copy



# Ensure you have the Aquaenv class defined with the render function

# Create the environment
#CP  Lat.(deg)= 40.40  Long.(deg)=101.73  Elev.(m)=1072.
gendf = calc_eto_faopm('data/CPWG.dat', 1995, 40.4, 1072, True, ["simyear", "jday", "minTemp", "maxTemp", "precip", "rad"])
gendf.head()

IRR_CAP = 10_000 # max amount of irrigation (mm/ha) that can be applied in a single season
ACTION_SET = 'smt4' # action sets, alternatives are: 'depth', 'binary', 'smt4'
DAYS_TO_IRR = 7 # 'number of days between irrigation decisions (e.g., 1, 3, 5, 7)

envconfig = nebraska_maize_config.copy() # get default config dictionary
envconfig['gendf'] = gendf # set weather data
envconfig['year2'] = 70 # end of the train/test split
envconfig['normalize_obs'] = True # normalize input observation (with a pre-calculated mean and standard deviation)
envconfig['include_rain'] = False # include rainfall within weather data
envconfig['observation_set'] = 'default' # set of variables to pass to agent
envconfig['max_irr'] = 25 # max irrigation that can be applied in a single irrigation event

envconfig['action_set'] = ACTION_SET # action sets, alternatives are: 'depth', 'binary', 'smt4'
envconfig['days_to_irr'] = DAYS_TO_IRR # 'number of days between irrigation decisions (e.g., 1, 3, 5, 7)
envconfig['max_irr_season'] = IRR_CAP # max amount of irrigation (mm/ha) that can be applied in a single season
envconfig['planting_date'] = '05/01'
InitWC = InitialWaterContent(value=['WP']) #(wc_type='Pct',value=[70])
envconfig['init_wc'] = InitWC
env = Aquaenv(envconfig)

proftrain = []
proftest = []
timesteps = []
caps = []

print('eval')

model = PPO.load("models/PPO-1717969013/490000.zip")

for irr_cap in [IRR_CAP]:
    test_env_config = copy.deepcopy(envconfig) # make a copy of the training env
    test_env_config['evaluation_run'] = True # sets env to evaluation mode

    train_rew, test_rew = evaluate_agent(model, Aquaenv, test_env_config) # evaluate agent
    
    proftrain.append(train_rew)
    proftest.append(test_rew)
    timesteps.append(10000)
    caps.append(irr_cap)

    print(irr_cap, f'Train: {round(train_rew, 3)}')
    print(irr_cap, f'Test: {round(test_rew, 3)}')


result_df = pd.DataFrame([timesteps, proftrain, proftest, caps]).T
result_df.to_csv(f'outputs/neb_corn_ppo_day_{DAYS_TO_IRR}_act_{ACTION_SET}_cap_{IRR_CAP}.csv')
plt.plot(timesteps, proftrain, label='Train Reward')
plt.plot(timesteps, proftest, label='Test Reward')
plt.legend()
plt.show()