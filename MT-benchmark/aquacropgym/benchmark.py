from aquacrop import AquaCropModel, Soil, Crop, InitialWaterContent, FieldMngt, GroundWater, IrrigationManagement
from aquacrop.utils import prepare_weather, get_filepath
from aquacrop.entities.paramStruct import ParamStruct

import pandas as pd
import matplotlib.pyplot as plt
import time
import datetime
import numpy as np

filepath=(get_filepath('champion_climate_no_rain.txt'))

weather_data = prepare_weather(filepath)
weather_data

sandy_loam = Soil(soil_type='SiltClayLoam')

Maize = Crop('MaizeChampionGDD', planting_date='05/01')

InitWC = InitialWaterContent(value=['WP'])

#max_irr_season = 400
#irr_mngt = IrrigationManagement(irrigation_method=2,IrrInterval=5)
irr_mngt = IrrigationManagement(irrigation_method=1, SMT=[50,50,50,50]*4)
#irr_mngt = IrrigationManagement(irrigation_method=5, depth=0.65)
#irr_mngt = IrrigationManagement(irrigation_method=1, SMT=[40,40,40,40]*4, MaxIrrSeason=max_irr_season)


model = AquaCropModel(sim_start_time=f'{1985}/05/01',
                    sim_end_time=f'{2015}/12/31',
                    weather_df=weather_data,
                    soil=sandy_loam,
                    crop=Maize,
                    initial_water_content=InitWC,
                    irrigation_management=irr_mngt)

model.run_model(till_termination=True)

final_stats = model._outputs.final_stats
print(final_stats)
final_stats.to_csv('fixedoutput.csv', columns=['Yield (tonne/ha)', 'Seasonal irrigation (mm)'], index=False)

# CROP_PRICE = 180
# IRRIGATION_COST = 1
# FIXED_COST = 1728

# end_reward = (CROP_PRICE * final_stats['Yield (tonne/ha)'].mean()
#               - IRRIGATION_COST * final_stats['Seasonal irrigation (mm)'].mean()
#               - FIXED_COST)

# print(f"End Reward: {end_reward}")
