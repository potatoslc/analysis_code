#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 12 08:15:16 2025

@author: potato
"""
import os
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd
d1 = '/Users/potato/Downloads/20250707_chpc_p2/'
header = 'plt_'

folder_direction =''

def read_file_all(fname):
    df = pd.read_csv(fname,header=0)
    sz1,sz2 = df.shape
    speed_list = []
    for i in range(sz1):
        xv,yv,zv = df['x_velocity'][i],df['y_velocity'][i],df['z_velocity'][i]
        speed_list.append(round(math.sqrt(xv**2 + yv**2 + zv**2),4))
        
    time = df['Time'][0]
    return time,speed_list


def read_file_get_dudy(fname,dc='x'):
    df = pd.read_csv(fname,header=0)
    sz1,sz2 = df.shape
    speed_list = []
    for i in range(sz1):
        xv,yv,zv = df['x_velocity'][i],df['y_velocity'][i],df['z_velocity'][i]
        speed_list.append(round(math.sqrt(xv**2 + yv**2 + zv**2),4))
        
    time = df['Time'][0]
    dudy_list =[]
    """
    if (dc not in {'x','y','z'}):        
        return -1
    """
    locs = sorted(set(df['Points:2']))[:2]
    dy = abs(locs[1]-locs[0])
    df['vel'] = speed_list
    df_n = df[df['Points:2'].isin(locs)]
    df_n = df_n.reset_index(drop=True)
    df_n = df_n[['Points:0', 'Points:1', 'Points:2','vel']]
    df_n = df_n.sort_values(by=['Points:1','Points:2'],ascending=[True, True])
    num_dict = {}
    for i in range(df_n.shape[0]):
        temp_str = str(df_n.iloc[i,0]) + str(df_n.iloc[i,1]) 
        if temp_str not in num_dict.keys():
            num_dict[temp_str] = [[df_n.iloc[i,2],df_n.iloc[i,3] ] ]
        else:
            num_dict[temp_str].append([df_n.iloc[i,2],df_n.iloc[i,3] ])
    
        
    for k,v in num_dict.items():
        if len(v)!=2:
            continue
        du = v[0][1] - v[1][1] if v[0][0]> v[1][0] else v[1][1] - v[0][1]
        dudy = round(du/dy,5)
        
        dudy_list.append(dudy)  
    return time,dudy_list


fig, ax = plt.subplots()

def init():
    ax.clear()
    ax.set_title("Histogram with Mean")
    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency")

def update(frame,dataframes):
    ax.clear()
    df = dataframes[frame]
    values = df  # 替换成你实际的列名
    mean_val = values.mean()

    # 绘制直方图
    ax.hist(values, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    ax.axvline(mean_val, color='red', linestyle='dashed', linewidth=2)
    
    ax.set_title(f"Frame {frame+1} | Mean = {mean_val:.2f}")
    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency")
    
def get_data(d1):
    all_data = []
    timeline = []
    mean_v = []
    sort_file = []
    #file_count = len([f for f in os.listdir(d1) if os.path.isfile(os.path.join(d1, f))])
    l1 = [f for f in os.listdir(d1) if os.path.isfile(os.path.join(d1, f))]
    file_count = len(l1)
    for i in range(file_count):
        seq_num = l1[i].split('_')[1].split('.')[0]
        sort_file.append([int(seq_num),l1[i]])
        
    sorted_f = sorted(sort_file, key=lambda x: x[0])
        
        
    for i in range(file_count):
        #file_name = d1+'/'+header+str(i)+'.csv'
        ttime,vlist = read_file_all(d1+'/'+sorted_f[i][1])
        timeline.append(ttime)
        all_data.append(vlist)
        mean_v.append(np.mean(vlist))
        
    return timeline,mean_v

def get_all_dudy_wall(d1):
    all_data = []
    timeline = []
    mean_v = []
    sort_file = []
    #file_count = len([f for f in os.listdir(d1) if os.path.isfile(os.path.join(d1, f))])
    l1 = [f for f in os.listdir(d1) if os.path.isfile(os.path.join(d1, f))]
    file_count = len(l1)
    for i in range(file_count):
        seq_num = l1[i].split('_')[1].split('.')[0]
        sort_file.append([int(seq_num),l1[i]])
        
    sorted_f = sorted(sort_file, key=lambda x: x[0])
        
        
    for i in range(file_count):
        #file_name = d1+'/'+header+str(i)+'.csv'
        ttime,vlist = read_file_get_dudy(d1+'/'+sorted_f[i][1])
        timeline.append(ttime)
        all_data.append(vlist)
        mean_v.append(np.median(vlist))
        
    return timeline,mean_v

t1,v1 = get_all_dudy_wall(d1)
df_out = pd.DataFrame({'timestep':t1,'dudy':v1})
#t1,v1 = get_data(d1)
#t2,v2 = get_data('/Users/potato/Downloads/20250616_p40_v8_c625-4_dp13d8_wbcn_long/')
#t3,v3 = get_data('/Users/potato/Downloads/20250611_p40_v8_c312-4/')


#ani = FuncAnimation(fig, update, frames=len(all_data), init_func=init, interval=300)

#plt.tight_layout()
#plt.show()
"""
plt.plot(t1,v1, label ='cellsize:7.81e-5 m, channel:0.04 m, dP=13.8' )
plt.plot(t2,v2, label ='cellsize:6.25e-4 m, channel:0.04 m' )
plt.plot(t3,v3, label ='cellsize:3.12e-4 m, channel:0.04 m' )
plt.xlabel("Time")
plt.ylabel("Mean Value")
plt.title("Time vs Mean velocity")
plt.legend()
plt.grid(True)
plt.show()
"""

