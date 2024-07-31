from aquacrop import AquaCropModel, Soil, Crop, InitialWaterContent, IrrigationManagement
from aquacrop.utils import prepare_weather, get_filepath

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin

path = get_filepath('champion_climate.txt')
wdf = prepare_weather(path)

def run_model(smts,max_irr_season,year1,year2):
    """
    funciton to run model and return results for given set of soil moisture targets
    """

    maize = Crop('Maize',planting_date='05/01') # define crop
    loam = Soil('SiltClayLoam') # define soil
    init_wc = InitialWaterContent(value=['WP'])
    #init_wc = InitialWaterContent(wc_type='Pct',value=[70]) # define initial soil water conditions

    irrmngt = IrrigationManagement(irrigation_method=1,SMT=smts,MaxIrrSeason=100000) # define irrigation management

    # create and run model
    model = AquaCropModel(f'{year1}/05/01',f'{year2}/10/31',wdf,loam,maize,
                          irrigation_management=irrmngt,initial_water_content=init_wc)

    model.run_model(till_termination=True)
    return model.get_simulation_results()

def evaluate(smts,max_irr_season,test=False):
    """
    funciton to run model and calculate reward (yield) for given set of soil moisture targets
    """
    # run model
    out = run_model(smts,max_irr_season,year1=1995,year2=1997)
    # get yields and total irrigation
    yld = out['Yield (tonne/ha)'].mean()
    tirr = out['Seasonal irrigation (mm)'].mean()
    print(f"Yield: {yld}, Total Irrigation: {tirr}")

    reward=yld

    # return either the negative reward (for the optimization)
    # or the yield and total irrigation (for analysis)
    if test:
        return yld,tirr,reward
    else:
        return -reward
    
#print(evaluate([70]*4,300))

def get_starting_point(num_smts,max_irr_season,num_searches):
    """
    find good starting threshold(s) for optimization
    """

    # get random SMT's
    x0list = np.random.rand(num_searches,num_smts)*100
    rlist=[]
    # evaluate random SMT's
    for xtest in x0list:
        r = evaluate(xtest,max_irr_season,)
        rlist.append(r)

    # save best SMT
    x0=x0list[np.argmin(rlist)]
    
    return x0

#print(get_starting_point(4,300,10))

def optimize(num_smts,max_irr_season,num_searches=10):
    """ 
    optimize thresholds to be profit maximising
    """
    # get starting optimization strategy
    x0=get_starting_point(num_smts,max_irr_season,num_searches)
    # run optimization
    res = fmin(evaluate, x0,disp=0,args=(max_irr_season,))
    # reshape array
    smts= res.squeeze()
    # evaluate optimal strategy
    return smts

#smts=optimize(4,300)

#print(evaluate(smts,300,True))

from tqdm.autonotebook import tqdm # progress bar

opt_smts=[]
yld_list=[]
tirr_list=[]
for max_irr in tqdm(range(700,1100,75)):
    

    # find optimal thresholds and save to list
    smts=optimize(4,max_irr)
    opt_smts.append(smts)

    # save the optimal yield and total irrigation
    yld,tirr,_=evaluate(smts,max_irr,True)
    yld_list.append(yld)
    tirr_list.append(tirr)

# create plot
fig, ax = plt.subplots(1, 1, figsize=(13, 8))

# plot results
ax.scatter(tirr_list, yld_list)
ax.plot(tirr_list, yld_list)

# labels
ax.set_xlabel('Total Irrigation (ha-mm)', fontsize=18)
ax.set_ylabel('Yield (tonne/ha)', fontsize=18)

# Update x-axis limit to reflect the range of max_irr values
ax.set_xlim([-20, 200])

# Update y-axis limits
ax.set_ylim([2, 15.5])

# annotate with optimal thresholds
bbox = dict(boxstyle="round", fc="1")
offset = [15, 15, 15, 15, 15, -125, -100, -5, 10, 10]
yoffset = [0, -5, -10, -15, -15, 0, 10, 15, -20, 10]
for i, smt in enumerate(opt_smts):
    smt = smt.clip(0, 100)
    ax.annotate('(%.0f, %.0f, %.0f, %.0f)' % (smt[0], smt[1], smt[2], smt[3]),
                (tirr_list[i], yld_list[i]), xytext=(offset[i], yoffset[i]), textcoords='offset points',
                bbox=bbox, fontsize=12)

plt.show()
