import yt
import numpy as np
import pandas as pd
import os
import subprocess
import time

                                                                                                                                                                                                                                             
e1= '/home/u0890475/Documents/PeleAnalysis/Src/isosurface3d.gnu.ex'                                                                                                                                                                               
e2='/home/u0890475/Documents/PeleAnalysis/Src/PythonScripts/mef2vtk.py'                                                                                                                                                                           
"""
e1 = '/scratch/potatoslc/peleanalysis/PeleAnalysis/Src/isosurface3d.gnu.x86-spr.ex'
e2 = '/scratch/potatoslc/peleanalysis/PeleAnalysis/Src/PythonScripts/mef2vtk.py'
"""  
def read_get_flamex(base_dir,filename,  Tcut = 303, x_cut=0.07,outdir=''):
    import yt
    import numpy as np
    d1 = base_dir+'/'+filename
    ds = yt.load(d1)

    ad = ds.all_data()

    YH2 = ad[("boxlib","Y(H2)")].v
    YH2_u = np.round(np.percentile(YH2,99),5)
    for i in range(1,10,1):
        c = i*0.1
        y_iso = (1.0 - c) * YH2_u

        cmd = [
            e1,
            f"infile={d1}",
            'isoCompName=Y(H2)',
            f"isoVal={y_iso}",
            'comps=23',
            'writeSurf=1', 
            
            
        ]
        subprocess.run(cmd,check=True)
    try:
        bd_split = base_dir.split('/')
        prefix = bd_split[-1]
    except:
        prefix=''
    mef_files = [f for f in os.listdir(base_dir) if f.endswith(".mef")]
    for i in range(len(mef_files)):

        temp_cmd2 = ['python',e2,base_dir+'/'+mef_files[i]]
        print(temp_cmd2)
        subprocess.run(temp_cmd2)
        
    os.makedirs(base_dir+'/'+prefix+'_plt_isosurface_vtk', exist_ok=True)
    vtk_folder = os.path.join(base_dir,prefix+ "_plt_isosurface_vtk")
    subprocess.run(f"mv {base_dir}/*.vtk {base_dir}/{prefix}_plt_isosurface_vtk/", shell=True)
    if not os.path.exists(vtk_folder):
        os.mkdir(vtk_folder)
    for filename in os.listdir(base_dir):
        if filename.endswith(".vtk"):
            src = os.path.join(base_dir, filename)
            dst = os.path.join(vtk_folder, filename)
            os.rename(src, dst)

import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--base_dir",
        required=True,
        help="directory containing plt file"
    )

    parser.add_argument(
        "--filename",
        required=True,
        help="plt folder name, for example plt000500"
    )

    args = parser.parse_args()

    read_get_flamex(
        base_dir=args.base_dir,
        filename=args.filename
    )