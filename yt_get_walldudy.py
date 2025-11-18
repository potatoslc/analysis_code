#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 17 17:54:41 2025

@author: potato
"""

import yt
import pandas as pd
import numpy as np
import os
from glob import glob

# --- USER SETTINGS ---
data_dir = "/Users/potato/Downloads/nrel_result_test"      # ← change to your directory
T_front = 320.0                     # temperature threshold [K]
header ='p45'
outdir = '/Users/potato/Downloads/nrel_result_test'

x_min, x_max = 0.05, 0.08        # x window (after ×100 scaling)
fields = [ "x_velocity", "y_velocity", "z_velocity"]

T_front = 320
flame_position =[]
time_list = []
# --- LOOP OVER ALL plt* DIRECTORIES ---
for plt_path in sorted(glob(os.path.join(data_dir, "plt*"))):
    if not os.path.isdir(plt_path):
        continue

    print(f"\n📂 Processing {plt_path} ...")
    ds = yt.load(plt_path)
    ad = ds.all_data()
    curr_plt = plt_path.split('/')[-1]

    # --- get raw coordinates ---
    x_raw = ad["x"].to("m").ndarray_view()
    y_raw = ad["y"].to("m").ndarray_view()
    z_raw = ad["z"].to("m").ndarray_view()
    
    time_val = float(ds.current_time.to_value())        
    # --- scale all coords by 100 ---
    x = x_raw * 100.0
    y = y_raw * 100.0
    z = z_raw * 100.0


    # --- select the lowest z-plane (exact match) ---
    z_lowest = np.min(z_raw)
    sel = (z_raw == z_lowest) & (x >= x_min) & (x <= x_max)



    # --- collect data ---
    data = {
        "time_s": np.full(np.count_nonzero(sel), time_val),
        "x_m": x[sel],
        "y_m": y[sel],
        "z_m": z[sel],
        
    }
    
    for f in fields:
        key = ("boxlib", f)
        if key in ds.field_list:
            data[f] = ad[key].ndarray_view()[sel]
        else:
            print(f"  ⚠️ Field '{f}' not found in {plt_path}")

    # --- export ---
    df = pd.DataFrame(data)
        
    out_csv = outdir + '/'+header +'_'+curr_plt+'.csv'
    #print(out_csv)
    #break
    #out_csv = os.path.join(data_dir, os.path.basename(plt_path) + ".csv")
    df.to_csv(out_csv, index=False)

    print(f"✅ Saved {len(df)} lowest-z cells → {out_csv}")



print("\n🎉 Done! All lowest-z slices exported.")
