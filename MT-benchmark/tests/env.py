from aquacrop import AquaCropModel, Soil, Crop, InitialWaterContent, IrrigationManagement
from aquacrop.utils import prepare_weather, get_filepath
from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
import random
import pandas as pd

class IrrEnv(Env):

    filepath=get_filepath('tunis_climate.txt')
    weather_data = prepare_weather(filepath)
    weather_data

    sandy_loam = Soil(soil_type='SandyLoam')

    wheat = Crop('Wheat', planting_date='10/01')

    InitWC = InitialWaterContent(value=['WP'])

    dates = pd.DatetimeIndex(['10/10/1979','10/11/1979','10/12/1979'])
    depths = [25,25,25]
    irr=pd.DataFrame([dates,depths]).T
    irr.columns=['Date','Depth']

    irr_mngt = IrrigationManagement(irrigation_method=3, Schedule=irr)

    model = AquaCropModel(sim_start_time=f'{1979}/10/01',
                        sim_end_time=f'{1985}/05/30',
                        weather_df=weather_data,
                        soil=sandy_loam,
                        crop=wheat,
                        initial_water_content=InitWC,
                        irrigation_management=irr_mngt)

    model._initialize()

    def __init__(self):
        # Actions we can take: irrigate, do not irrigate
        self.action_space = Discrete(2)
        # Observation space
        self.observation_space = np.array(self.model._init_cond.biomass, self.model._init_cond.irr_cum)
        # Set state
        self.state = self.model._init_cond.biomass
        # Set season length
        self.season = 365
        
    def step(self, action):
        # Apply action
        #self.model.run_single_step(0,1) # The idea is to pass the action to the run_single_step method, this will define the depth of the irrigation of the next day
        self.state = self.model._init_cond.biomass
        self.season -= 1
        # Calculate reward

        
        # Check if season is done
        
        # Apply depth noise
        #self.state += random.randint(-1,1)
        # Set placeholder for info
        info = {}
        
        # Return step information
        return 
    
    def reset(self):
        self.model._initialize()
        return self.model