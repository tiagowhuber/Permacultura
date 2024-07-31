from aquacrop import AquaCropModel, Soil, Crop, InitialWaterContent, FieldMngt, GroundWater, IrrigationManagement
from aquacrop.utils import prepare_weather, get_filepath
from aquacrop.entities.paramStruct import ParamStruct

import pandas as pd
import matplotlib.pyplot as plt
#import seaborn as sns
import time
import datetime
import numpy as np

filepath=get_filepath('brussels_climate.txt')

weather_data = prepare_weather(filepath)
weather_data

sandy_loam = Soil(soil_type='SandyLoam')

wheat = Crop('Wheat', planting_date='10/01')

InitWC = InitialWaterContent(value=['WP'])


schedule = pd.read_csv('testsirrigateeveryday.csv') # read the csv file into a pandas DataFrame
schedule.columns = ['Date', 'Depth'] # name columns

irr_mngt = IrrigationManagement(irrigation_method=0, Schedule=schedule)

year_start = np.random.randint(1976, 2000)
print(f"Year start: {year_start}")
model = AquaCropModel(sim_start_time=f'{year_start}/10/01',
                    sim_end_time=f'{year_start+7}/05/30',
                    weather_df=weather_data,
                    soil=sandy_loam,
                    crop=wheat,
                    initial_water_content=InitWC,
                    irrigation_management=irr_mngt)


