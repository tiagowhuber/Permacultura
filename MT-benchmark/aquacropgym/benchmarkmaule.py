
from aquacrop import AquaCropModel, Soil, Crop, InitialWaterContent, FieldMngt, GroundWater, IrrigationManagement
from aquacrop.utils import prepare_weather, get_filepath
from aquacrop.entities.paramStruct import ParamStruct

import pandas as pd
import matplotlib.pyplot as plt
import time
import datetime
import numpy as np
from utils import calc_eto_faopm

#gendf = calc_eto_faopm('data/MauleWG30Y.dat', 1995, -35.4, 1072, True, ["simyear", "jday", "minTemp", "maxTemp", "precip", "rad"])
#weather_data = gendf

#weather_data.to_csv('weather_data.csv', index=False)

filepath=(get_filepath('maule_climate_modified.txt'))
#filepath=(get_filepath('maule_climate_no_precipitation.txt'))
weather_data = prepare_weather(filepath)
#weather_data = pd.read_csv(filepath)
weather_data['Date'] = pd.to_datetime(weather_data['Date'])

sandy_loam = Soil(soil_type='Loam')

wheat = Crop('Tomato', planting_date='01/01')

InitWC = InitialWaterContent(value=['WP'])

#max_irr_season = 290
#irr_mngt = IrrigationManagement(irrigation_method=2,IrrInterval=1)
irr_mngt = IrrigationManagement(irrigation_method=1, SMT=[50,50,50,50]*4)
#irr_mngt = IrrigationManagement(irrigation_method=5, depth=3.19)

model = AquaCropModel(sim_start_time=f'{1996}/05/01',
                    sim_end_time=f'{2020}/12/31',
                    weather_df=weather_data,
                    soil=sandy_loam,
                    crop=wheat,
                    initial_water_content=InitWC,
                    irrigation_management=irr_mngt)

model.run_model(till_termination=True)

final_stats = model._outputs.final_stats
print(final_stats)
final_stats.to_csv('fixedoutput.csv', columns=['Yield (tonne/ha)', 'Seasonal irrigation (mm)'], index=False)