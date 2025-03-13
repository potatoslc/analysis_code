#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 14:37:07 2025

@author: potato



"""

import pandas as pd
import os
file_direct = "./test2"
file_header = "test2"
file_num = 0
for  _,_,files in os.walk(file_direct):
    file_num+=len(files)
def read_all(file_direct,file_header,file_num):
    distance_profile = []
    failed_list = []
    for i in range(file_num): 
        full_path = file_direct+"/"+file_header+"_"+str(i)+".csv" 
        try:
            all_data = pd.read_csv(full_path)
            time_stamp = all_data["Time"][0]
            min_loc = all_data["Points:0"].idxmin()
            min_value = all_data["Points:0"].min()
            #min_rows = all_data.loc[min_loc,"Points:1"] might have multiple points at x=min_value
            filter_data = all_data[(all_data["Points:0"]==min_value) & (all_data["Points:1"]>0)]
            distance_value = filter_data ["Points:1"].min()
            distance_profile.append([time_stamp,distance_value])
            
        except:
            failed_list.append(full_path)
            continue
    distance_profile_df = pd.DataFrame(distance_profile,columns=["Time","Distance(m)"])
    
    return distance_profile

        
        

