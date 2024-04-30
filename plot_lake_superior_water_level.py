"""


# Title: Lake Superior water level plot
# Author: Collin Roland
# Date Created: 20240430
# Summary: Plots monthly mean Lake Superior water level
# Date Last Modified: 20240430
# To do:  Nothing
"""
# %% Import packages
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib import rc, font_manager
import matplotlib.dates as mdates
from matplotlib.pyplot import savefig
import numpy as np
import os
from pathlib import Path
import pandas as pd

# %% Process lake level data 

LakeLvlPath = Path(r'D:\CJR\water_level\GLHYD_data_metric.csv')
LakeLvl = pd.read_csv(LakeLvlPath, skiprows=12)
LakeLvl = LakeLvl[['month','year','Superior']]
LakeLvl['day'] = 15
LakeLvl['month'] = [datetime.strptime(x, '%b').month for x in LakeLvl['month']]
LakeLvl['datetime'] = pd.to_datetime(LakeLvl[['year','month','day']])


LakeLvl2 = LakeLvl[(LakeLvl['year']>1990)]

# %% Plot lake laevel data

lidar_2009_start = pd.to_datetime('2009-03-01', format = '%Y-%m-%d')
lidar_2009_end = pd.to_datetime('2009-03-30', format = '%Y-%m-%d')
lidar_2019_start = pd.to_datetime('2019-08-01', format = '%Y-%m-%d')
lidar_2019_end = pd.to_datetime('2019-09-30', format = '%Y-%m-%d')

fig1, ax1 = plt.subplots(1,1,figsize=(10,6))
plt.plot(LakeLvl2['datetime'],LakeLvl2['Superior'],color='darkblue', label = 'Monthly mean')
ax1.grid(visible=True,which='major',color='#CCCCCC',linestyle='--')
ax1.grid(visible=True,which='major',color='#CCCCCC',linestyle=':')
plt.hlines(183.4, xmin=LakeLvl2['datetime'].iloc[0], xmax = LakeLvl2['datetime'].iloc[-1], linestyle='--', color='black', label = 'Long-term average')
ax1.set_ylabel('Water\nelevation\n(meters\nIGLD85)', color='black', fontsize = 12, rotation = 0,
               labelpad = 35)
ax1.set_xlabel('Year',color='black',fontsize=12)
# ax1.set_title('Lake Superior mean monthly water elevation',color='black')
ax1.set_xlim(LakeLvl2['datetime'].iloc[0],LakeLvl2['datetime'].iloc[-1])
plt.yticks(fontsize=10,color='black',weight='normal')
plt.xticks(fontsize=10,color='black',weight='normal')
ax1.tick_params(color='black', labelcolor='black')
ax1.axvspan(lidar_2009_start, lidar_2009_end, alpha = 0.7, color = 'darkgray',
            zorder = 1, label = 'LiDAR data acquisition')
ax1.axvspan(lidar_2019_start, lidar_2019_end, alpha = 0.7, color = 'darkgray',
            zorder = 1)
for spine in ax1.spines.values():
    spine.set_edgecolor('black')
ax1.legend()
ax1.set_facecolor("snow")
ax1.grid(visible = True, which = 'both', color = 'dimgray', linewidth = 0.5, linestyle = ":")
ax1.xaxis.set_major_locator(mdates.YearLocator(base = 2, month = 1, day = 1))
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
os.chdir(r'D:\CJR\PhDFigures\lake_superior_jglr')
fig1.savefig('figure_1.png', dpi = 150, bbox_inches = 'tight', pad_inches = 0.1, 
             transparent = False)
