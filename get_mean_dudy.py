#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 18 08:29:15 2025

@author: potato
"""


import yt
import pandas as pd
import numpy as np
import os
from glob import glob
import matplotlib.pyplot as plt


data_dir = "/Users/potato/Downloads/wall_vel_amr3"

outdir = '/Users/potato/Downloads/nrel_result_test/AMR3_turb'

csv_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
csv_files = sorted(csv_files)
dudy = []
time_list = []

num_file = len(csv_files)
time_list = []
for i in range(len(csv_files)):
    
    df = pd.read_csv(data_dir+'/'+csv_files[i])
    temp_dy = df["z_m"].iloc[0]
    mean_du = df["x_velocity"].mean()
    time_list.append(df['time_s'])
    dudy.append(mean_du/temp_dy)
    
    
  
fname = data_dir.split('/')[-1]
plt.plot(time_list,dudy,)
plt.xlabel('time (s)')
plt.ylabel('dudy (1/s)')
plt.title(f'mean dudy for {fname}')
plt.grid()
plt.savefig(f'{outdir}/mean_dudy_for_{fname}.png')
