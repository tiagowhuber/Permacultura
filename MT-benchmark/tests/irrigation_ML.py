from aquacrop import AquaCropModel, Soil, Crop, InitialWaterContent, IrrigationManagement
from aquacrop.utils import prepare_weather, get_filepath

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

path = get_filepath('tunis_climate.txt')
wdf = prepare_weather(path)
wdf

sim_start = '1979/09/01'
sim_end = '1984/05/30'

soil= Soil('Loam')

cabernet_sauvignon = Crop(
    c_name='Tomato',  # Crop name
    planting_date='09/01',  # Planting date
    CropType=3,  # Fruit/grain
    PlantMethod=1,  # Sown
    CalendarType=1,  # Calendar days
    #SwitchGDD=1,  # Convert calendar to GDD mode
    HarvestDate='04/30',  # Harvest date, considering typical growing season in Chile
    Emergence=100,  # Adjust based on typical emergence period
    MaxRooting=120,  # Adjust based on rooting depth requirements
    Senescence=200,  # Adjust based on typical senescence period
    Maturity=250,  # Adjust based on typical maturity period
    HIstart=150,  # Adjust based on yield formation start
    Flowering=-999,  # Not applicable for grapevines
    YldForm=100,  # Adjust based on yield formation duration
    #GDDmethod='ASCE',  # Adjust based on preferred GDD calculation method
    Tbase=10,  # Adjust based on grapevine temperature requirements
    Tupp=35,  # Adjust based on grapevine temperature requirements
    PolHeatStress=1,  # Grapevines are sensitive to heat stress
    Tmax_up=35,  # Adjust based on maximum temperature sensitivity
    Tmax_lo=40,  # Adjust based on maximum temperature sensitivity
    PolColdStress=1,  # Grapevines are sensitive to cold stress
    Tmin_up=5,  # Adjust based on minimum temperature sensitivity
    Tmin_lo=0,  # Adjust based on minimum temperature sensitivity
    TrColdStress=1,  # Grapevines are sensitive to cold stress affecting transpiration
    #GDD_up=10,  # Adjust based on grapevine GDD requirements
    #GDD_lo=5,  # Adjust based on grapevine GDD requirements
    Zmin=0.5,  # Adjust based on minimum rooting depth
    Zmax=2,  # Adjust based on maximum rooting depth
    fshape_r=0.75,  # Adjust based on root expansion characteristics
    SxTopQ=0.2,  # Adjust based on root water extraction characteristics
    SxBotQ=0.1,  # Adjust based on root water extraction characteristics
    SeedSize=10,  # Adjust based on grapevine planting density
    PlantPop=2500,  # Adjust based on grapevine planting density per hectare
    CCx=0.9,  # Adjust based on canopy cover requirements
    CDC=0.003,  # Adjust based on canopy decline coefficient
    CGC=0.005,  # Adjust based on canopy growth coefficient
    Kcb=1.1,  # Adjust based on grapevine crop coefficient
    fage=0.05,  # Adjust based on aging effects
    WP=1.2,  # Adjust based on water productivity
    WPy=110,  # Adjust based on water productivity during yield formation
    fsink=1.1,  # Adjust based on CO2 concentration effects
    HI0=0.4,  # Adjust based on reference harvest index
    dHI_pre=10,  # Adjust based on water stress effects before flowering
    a_HI=0.5,  # Adjust based on vegetative growth impact on harvest index
    b_HI=0.2,  # Adjust based on stomatal closure impact on harvest index
    dHI0=0.2,  # Adjust based on maximum allowable increase of harvest index
    Determinant=0,  # Indeterminate growth for grapevines
    exc=0.1,  # Adjust based on excess of potential fruits
    p_up1=0.7,  # Adjust based on upper soil water depletion threshold for canopy expansion
    p_up2=0.6,  # Adjust based on upper soil water depletion threshold for canopy stomatal control
    p_up3=0.5,  # Adjust based on upper soil water depletion threshold for canopy senescence
    p_up4=0.7,  # Adjust based on upper soil water depletion threshold for canopy pollination
    p_lo1=0.3,  # Adjust based on lower soil water depletion threshold for canopy expansion
    p_lo2=0.2,  # Adjust based on lower soil water depletion threshold for canopy stomatal control
    p_lo3=0.2,  # Adjust based on lower soil water depletion threshold for canopy senescence
    p_lo4=0.3,  # Adjust based on lower soil water depletion threshold for canopy pollination
    fshape_w1=0.75,  # Adjust based on water stress effects on canopy expansion
    fshape_w2=0.75,  # Adjust based on water stress effects on canopy stomatal control
    fshape_w3=0.75,  # Adjust based on water stress effects on canopy senescence
    fshape_w4=0.75  # Adjust based on water stress effects on pollination
)

wheat = Crop('Wheat', planting_date='09/01')
initWC = InitialWaterContent(value=['WP'])

schedule = pd.read_csv('irrigation_data.csv') # read the csv file into a pandas DataFrame
schedule.columns = ['Date', 'Depth'] # name columns

irr_mngt = IrrigationManagement(irrigation_method=3, Schedule=schedule)


model = AquaCropModel(sim_start,
                    sim_end,
                    wdf,
                    soil,
                    wheat,
                    initial_water_content=initWC,
                    irrigation_management=irr_mngt) # create model

model.solution_single_time_step(till_termination=True)  



print(model._outputs.final_stats.head())
