#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 17 16:15:11 2025

@author: potato
"""

import yt
import pandas as pd
import numpy as np
import os
from glob import glob

# ------------------------
# USER SETTINGS
# ------------------------
data_dir = "/Users/potato/Downloads/nrel_result_test"      # ← change to your directory
T_front = 320.0                     # temperature threshold [K]
header ='p45'
outdir = '/Users/potato/Downloads/nrel_result_test'
# ------------------------
# LOOP OVER ALL plt* DIRECTORIES


T_front = 320
flame_position =[]
time_list = []

# ------------------------
for plt_path in sorted(glob(os.path.join(data_dir, "plt*"))):
    if not os.path.isdir(plt_path):
        continue

    print(f"\n📂 Processing {plt_path} ...")
    ds = yt.load(plt_path)
    ad = ds.all_data()
    fname = plt_path.split('/')[-1]

    # --- raw coords ---
    T     = ad[("boxlib", "temp")].ndarray_view()


    x_raw = ad["x"].to("m").ndarray_view()
    y_raw = ad["y"].to("m").ndarray_view()
    z_raw = ad["z"].to("m").ndarray_view()
    time_val = float(ds.current_time.to_value())        
    # --- scale all coords by 100 ---
    x = x_raw * 100.0
    y = y_raw * 100.0
    z = z_raw * 100.0

    mask = T > T_front
    indices = np.where(mask)[0]
    idx_local = np.argmin(x_raw[mask])   # index inside compressed x_raw[mask]
    front_idx = indices[idx_local]       # global index
    
    # --- flame point in raw meters ---
    fx_raw = float(x_raw[front_idx])
    fy_raw = float(y_raw[front_idx])
    fz_raw = float(z_raw[front_idx])
    print("🔥 Flame point (raw meters):")
    print(f"  x = {fx_raw:.9e} m")
    print(f"  y = {fy_raw:.9e} m")
    print(f"  z = {fz_raw:.9e} m")
    scale = 100
    if fy_raw >= (0.0175-0.00125)/100:
        fy_raw = (0.0175-0.00125)/100
    if fy_raw <= 0.00125/100:
        fy_raw = 0.00125/100
    
    if fz_raw >= (0.0175-0.00125)/100:
        fz_raw = (0.0175-0.00125)/100
    if fz_raw <= 0.00125/100:
        fz_raw = 0.00125/100
        
        
    x_max = fx_raw + 0.0025/scale
    x_min = fx_raw
    y_max = fy_raw + 0.00125/scale
    y_min = fy_raw - 0.00125/scale
    z_max = fz_raw + 0.00125/scale
    z_min = fz_raw - 0.00125/scale
    
    sel = (x_raw >= x_min) & (x_raw <= x_max) & (z_raw >= z_min)  & (z_raw <= z_max)  & (y_raw >= y_min)  & ( y_raw <= y_max)  
    data = {
        "time_s": np.full(np.count_nonzero(sel), time_val),
        "T_k":T[sel],
    }
    species_fields = [
        f[1] for f in ds.field_list
        if f[0] == "boxlib" and f[1].startswith("Y(")
    ]
      

    for f in species_fields:
        key = ("boxlib", f)
        if key in ds.field_list:
            data[f] = ad[key].ndarray_view()[sel]
        else:
            print(f"  ⚠️ Field '{f}' not found in {plt_path}")      
    df = pd.DataFrame(data)
    df.to_csv(outdir+'/' +fname+'test1.csv')
        
        
        
        