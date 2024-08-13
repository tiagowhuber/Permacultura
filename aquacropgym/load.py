from aquacrop import AquaCropModel, Crop, FieldMngt, GroundWater, InitialWaterContent, IrrigationManagement, Soil
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import csv
from utils import calc_eto_faopm
from aquacropenv import Aquaenv, nebraska_maize_config

# Create a single Gym environment
# Create the environment
#CP  Lat.(deg)= 40.40  Long.(deg)=101.73  Elev.(m)=1072.
gendf = calc_eto_faopm('data/CPWG.dat', 1995, 40.4, 1072, True, ["simyear", "jday", "minTemp", "maxTemp", "precip", "rad"])
gendf.head()

IRR_CAP = 10_000 # max amount of irrigation (mm/ha) that can be applied in a single season
ACTION_SET = 'smt4' # action sets, alternatives are: 'depth', 'binary', 'smt4', 'depth_discreet'
DAYS_TO_IRR = 5 # 'number of days between irrigation decisions (e.g., 1, 3, 5, 7)

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
envconfig['evaluation_run'] = True
env = Aquaenv(envconfig)

# Wrap the single environment with DummyVecEnv to convert it into a VecEnv
vec_env = DummyVecEnv([lambda: env])

models_dir = "models\smt4-norain-defaultenvconfig-rtx"
model_path = f"{models_dir}/9890000.zip"

# Load the model with the VecEnv
model = PPO.load(model_path, env=vec_env)

episodes = 10

with open("trainedoutput.csv", "w", newline='') as file:
    pass

# Open the CSV file in append mode
with open("rewards.csv", "a", newline='') as file:
    writer = csv.writer(file)
    for ep in range(episodes):
        obs = vec_env.reset()  # Use vec_env.reset() instead of env.reset()
        done = False
        while not done:
            #vec_env.render()  # Use vec_env.render() instead of env.render()
            action, _ = model.predict(obs)
            #print("Action: ", action)
            #file.write(f"Action: {action}\n")  # Write the action to the text file
            obs, reward, done, info = vec_env.step(action)  # Use vec_env.step() instead of env.step()
            if reward != 0:
                writer.writerow(reward)  # Write the reward to the CSV file


# Close the VecEnv
vec_env.close()
