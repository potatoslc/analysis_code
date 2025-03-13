#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 14:37:07 2025
the distance from flame tip to inlet
@author: potato



"""

import pandas as pd
import os
import matplotlib.pyplot as plt
file_direct = "./speed_profile"
file_header = "speed_profile"
file_num = 0
for  _,_,files in os.walk(file_direct):
    file_num+=len(files)
def read_all(file_direct,file_header,file_num):
    d_T_profile = []
    failed_list = []
    for i in range(file_num): 
        full_path = file_direct+"/"+file_header+"_"+str(i)+".csv" 
        try:
            all_data = pd.read_csv(full_path)
            time_stamp = all_data["Time"][0]
            min_value = all_data["Points:0"].min() #get min value of the x-direction
            #min_rows = all_data.loc[min_loc,"Points:1"] might have multiple points at x=min_value
            filter_data = all_data[(all_data["Points:0"]==min_value)]
            #distance_value = filter_data ["Points:1"].min() currently dont need this
            speed_value = filter_data["x_velocity"].mean()
            d_T_profile.append([time_stamp,min_value,speed_value])
            
            
        except:
            failed_list.append(full_path)
            continue
    d_T_profile_df = pd.DataFrame(d_T_profile,columns=["Time","Distance(m)","x_velocity"])
    
    return d_T_profile_df

        
        
def plot_generation(df,option="both"):
    headers = df.columns[1:]
    header_lower = [x.lower() for x in headers]
    df_plt_x = df.iloc[:,0]
    df_plt_y = df.iloc[:,1:]
    fig, ax1 = plt.subplots()  
    ax1.set_xlabel("time (s)")
    for i in range(len(headers)):
        if option in header_lower[i]:
            df_plt_y = df[headers[i]]
            ax1.set_ylabel(headers[i])
            ax1.plot(df_plt_x,df_plt_y)
            break
    if option=="both":
        df_plt_y_1 = df_plt_y.iloc[:,0]
        df_plt_y_2 = df_plt_y.iloc[:,1]
        ax1.set_ylabel(headers[0])
        ax1.plot(df_plt_x,df_plt_y_1,color="black")
        ax2 = ax1.twinx()
        ax2.set_ylabel(headers[1])
        ax2.plot(df_plt_x,df_plt_y_2,color="red")
        fig.tight_layout()
        plt.show()
        
     
    
    return fig
all_profile= read_all(file_direct, file_header, file_num)

 