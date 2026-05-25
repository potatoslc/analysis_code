import yt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import bisect
from scipy.spatial import cKDTree
import time
import math


def read_get_flamex(d1,  Tcut = 310, x_cut=0.07, ):
    import yt
    import numpy as np
    import pandas as pd
    #load ds
    ds = yt.load(d1)
    ds.force_periodicity()
    # fields we will load per grid
    grad_keys = [
        ("boxlib", "Y(H2)_gradient_x"),
        ("boxlib", "Y(H2)_gradient_y"),
        ("boxlib", "Y(H2)_gradient_z"),
    ]
    Tkey = ("boxlib", "temp")
    
    # Collect chunks here
    chunks = []
    for g in ds.index.grids:
   
        if float(g.LeftEdge[0]) >= 0.07:
            continue
    
        # Pull minimal fields first (to build mask cheaply)
        x = g[("index", "x")].ndarray_view().ravel()
        T = g[Tkey].ndarray_view().ravel()
    
        mask = (T >= Tcut  ) & (x <= x_cut )
        if not np.any(mask):
            continue
    
        # Now pull only what you need (still per-grid, so small)
        y  = g[("index", "y")].ndarray_view().ravel()
        z  = g[("index", "z")].ndarray_view().ravel()
    
        data = {
            "x": x[mask],
            "y": y[mask],
            "z": z[mask],
            "T": T[mask],
        }
        chunks.append(pd.DataFrame(data))
    
    df_350 = pd.concat(chunks, ignore_index=True)
    print(df_350['x'].min(),df_350['x'].max())
    x_cold_region = df_350['x'].min()
    minidx = df_350['x'].idxmin()
    min_ptx = df_350['x'].iloc[minidx]
    min_pty = df_350['y'].iloc[minidx]
    min_ptz = df_350['z'].iloc[minidx]

    return [min_ptx,min_pty,min_ptz]





# get flame region cut df
def get_flame_cut(d1,Tcut:float=305,x_cold_region:float=0  , x_cutright =0.075   , grid_index=0, prodic_bdry =0 ):
    import yt
    import numpy as np
    import pandas as pd
    #load ds
    ds = yt.load(d1)
    if prodic_bdry ==1:
        ds.force_periodicity()
    max_YH2, x, y, z = ds.all_data().quantities.max_location(("boxlib","Y(H2)"))
    # add the 
    YH2_field = ("boxlib","Y(H2)")
    #max_YH2 = ad[YH2_field].max()
    
    def _c(field, data):
        c = 1.0 - data[YH2_field] / max_YH2
        return np.clip(c, 0.0, 1.0)

    ds.add_field(
        ("boxlib","c"),
        function=_c,
        sampling_type="cell",
        units="dimensionless",
        force_override=True
    )
    ds.add_gradient_fields(("boxlib", "Y(H2)"))
    Tkey = ("boxlib", "temp")
    mass_fields = [f for f in ds.field_list if "Y(" in f[1]]
    # fields we will load per grid
    grad_keys = [
        ("boxlib", "Y(H2)_gradient_x"),
        ("boxlib", "Y(H2)_gradient_y"),
        ("boxlib", "Y(H2)_gradient_z"),
    ]
  

    # Collect chunks here
    chunks = []
    
    
    for g in ds.index.grids:
        # Quick reject: if this grid is entirely to the right of x=0.07, skip it
    
        # Pull minimal fields first (to build mask cheaply)
        x = g[("index", "x")].ndarray_view().ravel()
        T = g[Tkey].ndarray_view().ravel()
    
    
        """
        Just cut the region at the boundary of isothermal wall x=0.07
        If need to find points of flame at isothermal wall, make sure set the value larger than 0.07 (e.g. x< 0.075)
        """
        mask = (x>=x_cold_region) & (x < x_cutright)
        if not np.any(mask):
            continue
    
        # Now pull only what you need (still per-grid, so small)
        y  = g[("index", "y")].ndarray_view().ravel()
        z  = g[("index", "z")].ndarray_view().ravel()
        dx = g[("index", "dx")].ndarray_view().ravel()
        c  = g[("boxlib","c")].to_ndarray().ravel()
        u  = g[('boxlib', 'x_velocity')].to_ndarray().ravel()
        v  = g[('boxlib', 'y_velocity')].to_ndarray().ravel()
        w  = g[('boxlib', 'z_velocity')].to_ndarray().ravel()
    
        gh2x = g[grad_keys[0]].ndarray_view().ravel()
        gh2y = g[grad_keys[1]].ndarray_view().ravel()
        gh2z = g[grad_keys[2]].ndarray_view().ravel()
        gh2mag = np.sqrt(gh2x*gh2x + gh2y*gh2y + gh2z*gh2z)
        

        heat_release = g[('boxlib', 'HeatRelease')].ndarray_view().ravel()
        data = {
            "x": x[mask],
            "y": y[mask],
            "z": z[mask],
            "T": T[mask],
            "c": c[mask],
            "gridsize": dx[mask],
            "gh2x": gh2x[mask],
            "gh2y": gh2y[mask],
            "gh2z": gh2z[mask],
            "gh2mag": gh2mag[mask],
            "Q":heat_release[mask],
            "u":u[mask],
            "v":v[mask],
            "w":w[mask],
        }
            # mass fractions (only masked cells)
        for f in mass_fields:
            data[f[1]] = g[f].ndarray_view().ravel()[mask]
    
        chunks.append(pd.DataFrame(data))




    if len(chunks) == 0:
        raise RuntimeError("No cells found for (T>350) & (x<0.07).")

    df_cut = pd.concat(chunks, ignore_index=True)
    df_cut['global_index'] =df_cut.index
    
    min_idx = df_cut[df_cut['T']>Tcut]['x'].idxmin()
    #print(min_idx)
    #return df_cut, [df_cut['x'].iloc[min_idx], df_cut['y'].iloc[min_idx], df_cut['z'].iloc[min_idx] ]

    min_gridsize = df_cut['gridsize'].min()
    
    if (grid_index==1):
        x_grid = np.array(df_cut['x'])/(min_gridsize/1)
        y_grid = np.array(df_cut['y'])/(min_gridsize/1)
        z_grid = np.array(df_cut['z'])/(min_gridsize/1)
        df_cut['x_grid'] = x_grid
        df_cut['y_grid'] = y_grid 
        df_cut['z_grid'] = z_grid 
        df_cut['x_grid'] = df_cut['x_grid'].astype(int)
        df_cut['y_grid'] = df_cut['y_grid'].astype(int)
        df_cut['z_grid'] = df_cut['z_grid'].astype(int)

    all_gridsize = df_cut['gridsize']
    min_gridsize = min(np.array(all_gridsize))
    all_gridsize = round(np.log2(all_gridsize/min_gridsize))
    all_gridsize = all_gridsize.astype(int)
    baseline_amr = max(all_gridsize)
    amr_level = baseline_amr - all_gridsize
    df_cut['amr'] = amr_level

    
    return df_cut, min_idx


def add_periodic_side(df, periodic_coord , edge_width=4):
    df_coord = df.columns
    
    if periodic_coord not in df_coord:
        return df,-2

    # i just assume the edge width is always smaller than df width
    y_unique = sorted(df[periodic_coord].unique() )
    max_y = max(y_unique)
    low_y = y_unique[0:edge_width]
    high_y = y_unique[-edge_width:]
    df_low_side = df[df[periodic_coord].isin(low_y) ]
    df_high_side = df[df[periodic_coord].isin(high_y) ]

    # reset the index
    high_side_list =   [  int(x)  for x in  (max_y - np.array(df_high_side[periodic_coord] ))  ]
    df_high_side[periodic_coord] = high_side_list

    low_side_list =  [  int(x)  for x in  (max_y + np.array(df_low_side[periodic_coord] ))  ]
    df_low_side[periodic_coord] = low_side_list

    df_comb = pd.concat([df,df_low_side,df_high_side],axis=0)
    df_comb=df_comb.reset_index(ignore=True)
    return df_comb



def find_pathline(df, x0,y0,z0,cutoff_value,max_length,curr_dx):
    pathline = []
    pathline_dict =[]
    find_next_point_cubic(df,x0,y0,z0,cutoff_value*0.5,int(max_length*0.66),curr_dx,pathline,pathline_dict)
    find_prvs_point_cubic(df,x0,y0,z0,cutoff_value,int(max_length),curr_dx,pathline,pathline_dict)
    return pathline,pathline_dict


def find_pathline_back(df, x0,y0,z0,cutoff_value,max_length,curr_dx):
    pathline = []
    pathline_dict =[]
    find_next_point_cubic(df,x0,y0,z0,cutoff_value/2,int(max_length/2),curr_dx,pathline,pathline_dict)
    #find_prvs_point_cubic(df,x0,y0,z0,cutoff_value,int(max_length/2),curr_dx,pathline,pathline_dict)
    return pathline,pathline_dict



def find_next_point_cubic(df,x0,y0,z0,cutoff_value,max_length,curr_dx,pathline,pathline_dict):
    # if the line is long enough, then return
    if len(pathline)>=2:
        pt0 = pathline[0]
        ptlast = pathline[-1]
        distance = ((pt0[0] - ptlast[0] )**2  + (pt0[1] - ptlast[1]  )**2 + (pt0[2] - ptlast[2]  )**2 )**0.5
        if distance >= cutoff_value:
            return 
    if len(pathline_dict) >= max_length:
            return 

    temp_weight_dict = get_weight_w_point(df,x0,y0,z0,curr_dx)
    if len(temp_weight_dict)==0:
        return
    #print(temp_weight_dict)

    # make sure the indexes of pathline and pathline dict are aligned
    pathline.append([x0,y0,z0])
    pathline_dict.append(temp_weight_dict)


    # that's the c_i/ |c| part 
    x_dirc, y_dirc, z_dirc = 0,0,0
    next_dx  =  0
    for k,v in temp_weight_dict.items():
        x_dirc += (df['gh2x'].iloc[k] * v)
        y_dirc += (df['gh2y'].iloc[k] * v)
        z_dirc += (df['gh2z'].iloc[k] * v)
        #next_dx = max(next_dx, df['gridsize'].iloc[k])
    next_dx = curr_dx
    norm_dist = (x_dirc**2+ y_dirc**2+z_dirc**2)**0.5
    norm_dist_sqrd = norm_dist**2

    curr_factor = 1/norm_dist if norm_dist<1 else 1
    
    #x_next = x0  - x_dirc/norm_dist * 0.1/curr_factor
    #y_next = y0  - y_dirc/norm_dist * 0.1/curr_factor
    #z_next = z0  - z_dirc/norm_dist * 0.1/curr_factor
    
    x_next = x0  - x_dirc/norm_dist * 0.25 * next_dx
    y_next = y0  - y_dirc/norm_dist * 0.25 * next_dx
    z_next = z0  - z_dirc/norm_dist * 0.25 * next_dx
    if z_next <=0:
        return
    #print(f'The next point is x: {x_next}, y: {y_next} , z: {z_next}')

    find_next_point_cubic(df,x_next,y_next,z_next,cutoff_value,max_length,next_dx,pathline,pathline_dict)
        
    return 





def find_prvs_point_cubic(df,x0,y0,z0,cutoff_value,max_length,curr_dx,pathline,pathline_dict):
    # if the line is long enough, then return
    if len(pathline)>=2:
        pt0 = pathline[0]
        ptlast = pathline[-1]
        distance = ((pt0[0] - ptlast[0] )**2  + (pt0[1] - ptlast[1]  )**2 + (pt0[2] - ptlast[2]  )**2 )**0.5
        if distance >= cutoff_value:
            return 
    if len(pathline_dict) >= max_length:
            return 

    temp_weight_dict = get_weight_w_point(df,x0,y0,z0,curr_dx)
    if len(temp_weight_dict)==0:
        return
    #print(temp_weight_dict)

    # make sure the indexes of pathline and pathline dict are aligned
    pathline.append([x0,y0,z0])
    pathline_dict.append(temp_weight_dict)


    # that's the c_i/ |c| part 
    x_dirc, y_dirc, z_dirc = 0,0,0
    next_dx  =  0
    for k,v in temp_weight_dict.items():
        x_dirc += (df['gh2x'].iloc[k] * v)
        y_dirc += (df['gh2y'].iloc[k] * v)
        z_dirc += (df['gh2z'].iloc[k] * v)
        #next_dx = max(next_dx, df['gridsize'].iloc[k])
    next_dx = curr_dx
    norm_dist = (x_dirc**2+ y_dirc**2+z_dirc**2)**0.5
    norm_dist_sqrd = norm_dist**2
    curr_factor = 1/norm_dist if norm_dist<1 else 1
    x_next = x0  + x_dirc/norm_dist * 0.25 * next_dx
    y_next = y0  + y_dirc/norm_dist * 0.25 * next_dx
    z_next = z0  + z_dirc/norm_dist * 0.25 * next_dx
    if z_next <=0:
        return
    #print(f'The next point is x: {x_next}, y: {y_next} , z: {z_next}')

    find_prvs_point_cubic(df,x_next,y_next,z_next,cutoff_value,max_length,next_dx,pathline,pathline_dict)
        
    return 


def find_pts(df,x0,y0,z0,curr_dx):
    pts_good = []
    coord_dict = {}
    
    
    xbl, xbr = x0 - 4*curr_dx, x0 + 4*curr_dx
    ybl, ybr = y0 - 4*curr_dx, y0 + 4*curr_dx
    zbl, zbr = z0 - 4*curr_dx, z0 + 4*curr_dx
    #print(f' trying to find points for {x0},{y0},{z0}')
    # the 01 indiation of cubic
    coord_idx = [[-1,-1,-1], [-1,-1,1], [-1,1,-1],[-1,1,1],
                [1,-1,-1], [1,-1,1], [1,1,-1],[1,1,1], ]


    for ec in coord_idx:
        x_edge = xbr if ec[0] ==1 else xbl
        y_edge = ybr if ec[1] ==1 else ybl
        z_edge = zbr if ec[2] ==1 else zbl

        subxl, subxr = min(x_edge,x0), max(x_edge,x0)
        subyl, subyr = min(y_edge,y0), max(y_edge,y0)
        subzl, subzr = min(z_edge,z0), max(z_edge,z0)


        if subzl <0:
            subzl += curr_dx/2
            subzr += curr_dx/2

        if (subxl >0.07 ) or (subxr >0.07):
            subxl += 3*curr_dx
            subxr += 3*curr_dx

        
        temp_cut = df[ (df['x'] > subxl) &  (df['x'] <= subxr) & (df['y'] > subyl) &  (df['y'] <= subyr) & (df['z'] > subzl) &  (df['z'] <= subzr)]


        
        df_cut = temp_cut.copy()
        #

        
        if (df_cut.shape[0]==0):
            #print(f'no point can be found for x:{subxl } to {subxr }, y: {subyl } to {subyr }, z:{subzl } to {subzr }')
            continue
         
        points = df_cut[['x','y','z']].to_numpy()
        tree = cKDTree(points)
        
        #print(f'subxl: {subxl}, subxr: {subxr}')
        #print(f'subyl: {subyl}, subyr: {subyr}')
        #print(f'subzl: {subzl}, subzr: {subzr}')
        #print(df_cut['x'].min(), df_cut['x'].max())
        query_point = np.array([x0, y0, z0])
        # index is the index of the original df
        # and this index need to be stored and used
        #dist, idx = tree.query(query_point)
        #print(f"for {ec} is nearest index is {idx} with distance {dist} x:{df_out['x'].loc[idx]}, y:{df_out['y'].loc[idx]}, z:{df_out['z'].loc[idx]}.")
        dist, i_local = tree.query(query_point)
        if(subxl>subxr):
            print('something wrong on x !')
            break

        if(subyl>subyr):
            print('something wrong on y !')
            break

        if(subzl>subzr):
            print('something wrong on z!')
            break
        
        idx_global = df_cut['global_index'].iloc[int(i_local)]
        
        #print(f'local index is {i_local} and global index is {idx_global}.')
        #print(f"xyz: {df[['x','y','z']].iloc[idx_global]}")
        
        pts_good.append(idx_global)
        coord_dict[tuple(ec)] = idx_global
        


    return pts_good, coord_dict


def get_flame_cut_curv(d1,Tcut:float=305,x_cold_region:float=0,max_resol=34.1e-6, prdic_bdry=0):
    import yt
    import numpy as np
    import pandas as pd
    #load ds
    ds = yt.load(d1)
    if prdic_bdry==1:
        ds.force_periodicity()
    # add the 
    ds.add_gradient_fields(("boxlib", "Y(H2)"))

    eps = 1e-12
    #yh2_ub = 0.01304
    YH2_field = ("boxlib","Y(H2)")
    ad= ds.all_data()
    max_YH2 = ad[YH2_field].max()
    
    def _c(field, data):
        c = 1.0 - data[YH2_field] / max_YH2
        return np.clip(c, 0.0, 1.0)
    
    ds.add_field(("boxlib","c"), function=_c, sampling_type="cell",
                 units="dimensionless", force_override=True)
    
    ds.add_gradient_fields(("boxlib","c"))

    half_cell = max_resol/2
    def _nx(field, data):
        gx = data[("boxlib","c_gradient_x")]
        g  = data[("boxlib","c_gradient_magnitude")]
        eps_u = data.ds.quan(1e-30, str(g.units))
        return gx / (g + eps_u)
    
    def _ny(field, data):
        gy = data[("boxlib","c_gradient_y")]
        g  = data[("boxlib","c_gradient_magnitude")]
        eps_u = data.ds.quan(1e-30, str(g.units))
        return gy / (g + eps_u)
    
    def _nz(field, data):
        gz = data[("boxlib","c_gradient_z")]
        g  = data[("boxlib","c_gradient_magnitude")]
        eps_u = data.ds.quan(1e-30, str(g.units))
        return gz / (g + eps_u)

    ds.add_field(("boxlib","nx"), function=_nx, sampling_type="cell",
                 units="dimensionless", force_override=True)
    ds.add_field(("boxlib","ny"), function=_ny, sampling_type="cell",
                 units="dimensionless", force_override=True)
    ds.add_field(("boxlib","nz"), function=_nz, sampling_type="cell",
                 units="dimensionless", force_override=True)

    ds.add_gradient_fields(("boxlib","nx"))
    ds.add_gradient_fields(("boxlib","ny"))
    ds.add_gradient_fields(("boxlib","nz"))
    
    def _kappa(field, data):
        return (data[("boxlib","nx_gradient_x")] +
                data[("boxlib","ny_gradient_y")] +
                data[("boxlib","nz_gradient_z")])
    
    ds.add_field(("boxlib","curvature"), function=_kappa, sampling_type="cell",
                 units="1/cm", force_override=True)   # unit label optional

    
    Tkey = ("boxlib", "temp")
    mass_fields = [f for f in ds.field_list if "Y(" in f[1]]
    # fields we will load per grid
    grad_keys = [
        ("boxlib", "Y(H2)_gradient_x"),
        ("boxlib", "Y(H2)_gradient_y"),
        ("boxlib", "Y(H2)_gradient_z"),
    ]
  

    # Collect chunks here
    chunks = []
    
    
    for g in ds.index.grids:
        # Quick reject: if this grid is entirely to the right of x=0.07, skip it
    
        # Pull minimal fields first (to build mask cheaply)
        x = g[("index", "x")].ndarray_view().ravel()
        T = g[Tkey].ndarray_view().ravel()
        dx = g[("index", "dx")].ndarray_view().ravel()
        c  = g[("boxlib","c")].to_ndarray().ravel()
    
        """
        Just cut the region at the boundary of isothermal wall x=0.07
        If need to find points of flame at isothermal wall, make sure set the value larger than 0.07 (e.g. x< 0.075)
        """
        mask = (x>=x_cold_region) & (x < 0.075) & (dx< max_resol*1.05)
        if not np.any(mask):
            continue
    
        # Now pull only what you need (still per-grid, so small)
        y  = g[("index", "y")].ndarray_view().ravel()
        z  = g[("index", "z")].ndarray_view().ravel()
        
        
        curv = g[('boxlib', 'curvature')].ndarray_view().ravel()
        
        gh2x = g[grad_keys[0]].ndarray_view().ravel()
        gh2y = g[grad_keys[1]].ndarray_view().ravel()
        gh2z = g[grad_keys[2]].ndarray_view().ravel()
        gh2mag = np.sqrt(gh2x*gh2x + gh2y*gh2y + gh2z*gh2z)
        

        heat_release = g[('boxlib', 'HeatRelease')].ndarray_view().ravel()
        data = {
            "x": x[mask],
            "y": y[mask],
            "z": z[mask],
            "T": T[mask],
            "gridsize": dx[mask],
            "gh2x": gh2x[mask],
            "gh2y": gh2y[mask],
            "gh2z": gh2z[mask],
            "gh2mag": gh2mag[mask],
            "Q":heat_release[mask],
            "k":curv[mask],
            "c":c[mask],
        }
            # mass fractions (only masked cells)
        for f in mass_fields:
            data[f[1]] = g[f].ndarray_view().ravel()[mask]
    
        chunks.append(pd.DataFrame(data))

    


    if len(chunks) == 0:
        raise RuntimeError("No cells found for (T>350) & (x<0.07).")

    df_cut = pd.concat(chunks, ignore_index=True)

    all_gridsize = df_cut['gridsize']
    min_gridsize = min(np.array(all_gridsize))
    all_gridsize = round(np.log2(all_gridsize/min_gridsize))
    all_gridsize = all_gridsize.astype(int)
    baseline_amr = max(all_gridsize)
    amr_level = baseline_amr - all_gridsize
    df_cut['amr'] = amr_level

    

    df_cut['x_grid'] = round((df_cut['x'] - min_gridsize/2 )/(min_gridsize ))
    df_cut['y_grid'] = round((df_cut['y'] - min_gridsize/2 )/(min_gridsize ))
    df_cut['z_grid'] = round((df_cut['z'] - min_gridsize/2 )/(min_gridsize ))
    
    
    #print(min_idx)
    #return df_cut, [df_cut['x'].iloc[min_idx], df_cut['y'].iloc[min_idx], df_cut['z'].iloc[min_idx] ]
    df_cut['global_index'] =df_cut.index
    min_idx = df_cut[df_cut['T']>Tcut]['x'].idxmin()
    return df_cut, min_idx



def get_weight_w_point(df0,x0,y0,z0,curr_dx):

    df =df0.copy()
    temp_ls, _ = find_pts(df,x0,y0,z0,curr_dx)
    if len(temp_ls)==0:
        return {}
    
    temp_dist = []
    subdf = df[['x', 'y', 'z']].loc[temp_ls]
    #print(f'the length of subdf is {subdf.shape[0]}')
    minv = 1e-12
    currmin = 100
    for i in range(subdf.shape[0]):
        tx = subdf['x'].iloc[i]
        ty = subdf['y'].iloc[i]
        tz = subdf['z'].iloc[i]
        minxyz_pre = [(tx - x0  )**2, (ty - y0  )**2 , (tz- z0  )**2 ] 
        
        minxyz = [ x for x in minxyz_pre if x>0]
        minxyz.append(currmin)
        currmin = min(minxyz)
        #print(f'for current {tx}, {ty}, {tz}, the diff is {minxyz}')

    minv = min(1e-4, currmin/1000)

    

    #print(minv)
    
    for i in range(subdf.shape[0]):
        x_diff = subdf['x'].iloc[i] - x0 if (subdf['x'].iloc[i] - x0)!=0 else minv
        y_diff = subdf['y'].iloc[i] - y0 if (subdf['y'].iloc[i] - y0)!=0 else minv
        z_diff = subdf['z'].iloc[i] - z0 if (subdf['z'].iloc[i] - z0)!=0 else minv

        
        temp_dist.append( 1/( (x_diff)**2 +   (y_diff)**2 +( z_diff)**2) )

    sum_weight = sum(temp_dist)
    weight_dict = {}
    check_ttl=0

    # find way to 
    for i in range(subdf.shape[0]):
        if subdf.index[i] not in weight_dict:
            weight_dict[subdf.index[i]] = temp_dist[i] /  sum_weight
        else:
            weight_dict[subdf.index[i]] += temp_dist[i] /  sum_weight

    #print(check_ttl)
    return weight_dict


def get_T_Mass_from_list(df,pathline,pathline_dict):
    index_col = ['x', 'y', 'z']
    header_col = [ec for ec in df.columns if 'Y('  in ec ]
    header_col.append('T')
    header_col.append('c')
    header_col.append('k')
    header_col.extend( ['gh2x','gh2y', 'gh2z','Q'])
    full_col = index_col+ header_col
    #final_df = pd.DataFrame(columns=full_col)
    # this list combined 2 items, [index, value]
    # then sort by index. 
    final_list = []
    for i in range(len(pathline_dict)):
        temp_dict = pathline_dict[i]
        temp_target_coord = pathline[i]
        
        temp_intp_list = []
        temp_intp_list.extend(pathline[i])
        temp_intp_weight = []
        #print(temp_dict)
        w=pd.Series(temp_dict, name="weight")
        temo_subdf = df[header_col].loc[w.index].copy()
        temo_subdf["weight"] = w
        #print(temo_subdf)
        #num_cols = df_selected.select_dtypes(include="number").columns
        df_weighted = temo_subdf[header_col].multiply(temo_subdf["weight"], axis=0)
        
        for i in range(len(header_col)):
            #print(df_weighted[header_col[i]])
            temp_intp_list.append(sum(df_weighted[header_col[i]]))
        final_list.append(temp_intp_list)
    
        
    return pd.DataFrame(final_list,columns = full_col)

def flame_cline_df(df1,df_coord,slabel="",savedir = "./"):
    sf_oh=[]
    sf_h =[]
    sf_o2=[]
    sf_o=[]
    sf_ho2=[]
    sf_h2o2=[]
    sf_h2o=[]
    sf_h2=[]
    sf_cline= []
    sf_t = []
    sf_pl=[]
    sf_coord=[]
    fln=[]
    curr_ct = 0
    rows = []
    for i in range(df_coord.shape[0]):
        temp_xs  = df_coord['x'].iloc[i]
        temp_ys  = df_coord['y'].iloc[i]
        temp_zs  = df_coord['z'].iloc[i]
        #print(temp_xs,temp_ys,temp_zs)                                                                                                                                                                                                      
    
    
        tpl2,tpd2 = find_pathline(df1,  temp_xs  ,  temp_ys ,  temp_zs  ,0.0045,200,34.1e-6  )
        print(f'find the path for {temp_xs}  ,  {temp_ys} , { temp_zs}. ')
        #print(tpl2)
        if len(tpl2) ==0:
            continue
        temp_df2 = get_T_Mass_from_list(df1,tpl2,tpd2)
        #print(tpl2)                                                                                                                                                                                                                     
        #print('found line dataframe!')                                                                                                                                                                                                  
        temp_cline = 1- temp_df2['Y(H2)'] /0.01304
        temp_oh = temp_df2['Y(OH)']
        temp_o2 = temp_df2['Y(O2)']
        temp_o = temp_df2['Y(O)']
        temp_ho2 = temp_df2['Y(HO2)']
        temp_h2o2 = temp_df2['Y(H2O2)']
        temp_h2o = temp_df2['Y(H2O)']
        temp_t2 = temp_df2['T']
        temp_h = temp_df2['Y(H)']
        temp_k = temp_df2['k']
        for j in range(len(tpl2)):
             rows.append({
                 "point_id": i,
                 "step": j,
                 "x": tpl2[i][0],
                 "y": tpl2[i][1],
                 "z": tpl2[i][2],
    
                 "c": temp_cline[j],
                 "Y(OH)": temp_df2['Y(OH)'].iloc[j],
                 "Y(O2)": temp_df2['Y(O2)'].iloc[j],
                 "Y(O)": temp_df2['Y(O)'].iloc[j],
                 "Y(HO2)": temp_df2['Y(HO2)'].iloc[j],
                 "Y(H2O2)": temp_df2['Y(H2O2)'].iloc[j],
                 "Y(H2O)": temp_df2['Y(H2O)'].iloc[j],
                 "Y(H)": temp_df2['Y(H)'].iloc[j],
                 "T": temp_df2['T'].iloc[j],
             })
        #print('Section 1 is ok')                                                                                                                                                                                                        
        sf_oh.append(temp_oh)
        sf_o2.append(temp_o2)
        sf_o.append(temp_o)
        sf_ho2.append(temp_ho2)
        sf_h2o2.append(temp_h2o2)
        #print('section 2a is ok')                                                                                                                                                                                                       
        sf_h2o.append(temp_h2o)
        sf_cline.append(temp_cline)
        #print('section 2b is ok')                                                                                                                                                                                                       
        sf_t.append(temp_t2)
        sf_coord.append([temp_xs,temp_ys,temp_zs])
        #print('section 2c is ok')                                                                                                                                                                                                       
        sf_h.append(temp_h)
        sf_pl.append(tpl2)
        
        print(f'there are {len(temp_h)} points for this c line of #{i} point of {temp_xs}, {temp_ys},{temp_zs}.')
    
        curr_ct+=1
    
                                                                                                                                                                                                                                         
                                                                                                                                                                                                                           
    
    
    df_all = pd.DataFrame(rows)
    df_all.to_csv(savedir+'/'+slabel+'.csv',index=False)
        



    import matplotlib.pyplot as plt
    plt.scatter(df_all['c'],df_all['Y(H)'],c='b',s=2)
    plt.title(f'Progress Variable vs Y(H) of {slabel}.')
    plt.savefig(f'{savedir}/Progress Variable vs Y(H) of {slabel}.png',dpi=600)
    plt.show()
    #print(f'the size of current line is {len(sf_h[i])}.')                                                                                                                                                                               
    
    plt.scatter(sf_cline[i],sf_oh[i],c='b',s=2)
    plt.title(f'Progress Variable vs Y(OH) of {slabel}.')
    plt.savefig(f'{savedir}/Progress Variable vs Y(OH) of {slabel}.png',dpi=600)
    plt.show()
    
    plt.scatter(sf_cline[i],sf_o[i],c='b',s=2)
    plt.title(f'Progress Variable vs Y(O) of {slabel}.')
    plt.savefig(f'{savedir}/Progress Variable vs Y(O) of {slabel}.png',dpi=600)
    plt.show()
    
    plt.scatter(sf_cline[i],sf_o2[i],c='b',s=2)
    plt.title(f'Progress Variable vs Y(O2) of {slabel}.')
    plt.savefig(f'{savedir}/Progress Variable vs Y(O2) of {slabel}.png',dpi=600)
    plt.show()
    
    """                                                                                                                                                                                                                                  
    plt.scatter(sf_cline[i],sf_h2[i],c='b',s=2)                                                                                                                                                                                          
    plt.title(f'Progress Variable vs Y(H2) of {slabel}.')                                                                                                                                                                                
    plt.savefig(f'{savedir}/Progress Variable vs Y(H2) of {slabel}.png',dpi=600)                                                                                                                                                         
    plt.show()                                                                                                                                                                                                                           
    """
    
    plt.scatter(sf_cline[i],sf_h2o2[i],c='b',s=2)
    plt.title(f'Progress Variable vs Y(H2O2) of {slabel}.')
    plt.savefig(f'{savedir}/Progress Variable vs Y(H2O2) of {slabel}.png',dpi=600)
    plt.show()
    
    plt.scatter(sf_cline[i],sf_h2o[i],c='b',s=2)
    plt.title(f'Progress Variable vs Y(H2O) of {slabel}.')
    plt.savefig(f'{savedir}/Progress Variable vs Y(H2O) of {slabel}.png',dpi=600)
    plt.show()
    
    plt.scatter(sf_cline[i],sf_ho2[i],c='b',s=2)
    plt.title(f'Progress Variable vs Y(HO2) of {slabel}.')
    plt.savefig(f'{savedir}/Progress Variable vs Y(HO2) of {slabel}.png',dpi=600)
    plt.show()

    return



def add_xlower_ghostcell(df,x_cold_hot, T_cold_hot:list , ghost_direct_name = ' ',x_cut:list=[],):
    # either grid limit or physical domain limit is fine
    # x_cold_hot is defined as which x_grid is cold and which is hot,
    # smaller or equal than  x_cold_hot: cold, larger than x_cold_hot:hot
    # T_cold_hot[0] indicate the cold temperature, T_cold_hot[1] indicate the hot temperature
    #print(df[ghost_direct_name].min(),df[ghost_direct_name].max() )
    df_cut = df[ (df[ghost_direct_name]<= x_cut[1]) &   (df[ghost_direct_name ]>= x_cut[0])]
    #print(df_cut.shape)
    df_cut_low = df_cut[df_cut[ghost_direct_name] == 0]
    df_cut_low_copy = df_cut_low.copy()
    for i in range(df_cut_low.shape[0]):

        # apply Neumann zero gradient conditon, dH2/dy =0
        if df_cut_low_copy[x_grid_name].iloc[i] <= x_cold_hot:
            df_cut_low_copy.loc[i,'T'] = 2*T_cold_hot[0] - df_cut_low.loc[i,'T']
        else:
            df_cut_low_copy.loc[i,'T'] = 2*T_cold_hot[1] - df_cut_low.loc[i,'T']

    df_cut_out  = pd.concat([df_cut,  df_cut_low_copy ])

    return df_cut_out


def get_padded_box_arrays(
    d1,
    x_cut_region=(0.066, 0.072),
    y_cut_region=None,
    z_cut_region=None,
    yh2_ub=0.01304,
    ny_pad=4,
    add_zlower_ghost=True,
    zghost_mode="edge",
):
    import yt
    import numpy as np

    ds = yt.load(d1)

    # --- fields ---
    T_field   = ("boxlib", "temp")
    HRR_field = ("boxlib", "HeatRelease")
    Ux_field  = ("boxlib", "x_velocity")
    Uy_field  = ("boxlib", "y_velocity")
    Uz_field  = ("boxlib", "z_velocity")
    YH2_field = ("boxlib", "Y(H2)")
    pressure_field =('boxlib', 'RhoRT')
    density_field = ('boxlib', 'density')
    mass_fields = [f for f in ds.field_list if "Y(" in f[1]]

    # --- grid info ---
    level = ds.index.max_level
    dims_full = ds.domain_dimensions * (2 ** level)

    dx = float(ds.domain_width[0] / dims_full[0])
    dy = float(ds.domain_width[1] / dims_full[1])
    dz = float(ds.domain_width[2] / dims_full[2])

    dom_x0 = float(ds.domain_left_edge[0])
    dom_y0 = float(ds.domain_left_edge[1])
    dom_z0 = float(ds.domain_left_edge[2])

    dom_x1 = float(ds.domain_right_edge[0])
    dom_y1 = float(ds.domain_right_edge[1])
    dom_z1 = float(ds.domain_right_edge[2])

    # =========================
    # X CUT（必须给）
    # =========================
    x_left, x_right = x_cut_region
    x_left  = max(x_left,  dom_x0)
    x_right = min(x_right, dom_x1)

    i0 = int(np.floor((x_left  - dom_x0) / dx))
    i1 = int(np.ceil((x_right - dom_x0) / dx))

    # =========================
    # Y CUT（None → full）
    # =========================
    if y_cut_region is None:
        j0, j1 = 0, int(dims_full[1]) - 1
        y_is_full = True
    else:
        y_left, y_right = y_cut_region
        y_left  = max(y_left,  dom_y0)
        y_right = min(y_right, dom_y1)

        j0 = int(np.floor((y_left  - dom_y0) / dy))
        j1 = int(np.ceil((y_right - dom_y0) / dy))
        y_is_full = False

    # =========================
    # Z CUT（None → full）
    # =========================
    if z_cut_region is None:
        k0, k1 = 0, int(dims_full[2]) - 1
    else:
        z_left, z_right = z_cut_region
        z_left  = max(z_left,  dom_z0)
        z_right = min(z_right, dom_z1)

        k0 = int(np.floor((z_left  - dom_z0) / dz))
        k1 = int(np.ceil((z_right - dom_z0) / dz))

    # --- local dims ---
    nx_local = i1 - i0 + 1
    ny_local = j1 - j0 + 1
    nz_local = k1 - k0 + 1

    # --- left edge ---
    left_edge = ds.domain_left_edge.copy()
    left_edge[0] = dom_x0 + i0 * dx
    left_edge[1] = dom_y0 + j0 * dy
    left_edge[2] = dom_z0 + k0 * dz

    # --- covering grid ---
    cg = ds.covering_grid(
        level=level,
        left_edge=left_edge,
        dims=np.array([nx_local, ny_local, nz_local], dtype=int),
    )

    # --- extract fields ---
    x = cg[("index", "x")].v
    y = cg[("index", "y")].v
    z = cg[("index", "z")].v

    T   = cg[T_field].v
    Q   = cg[HRR_field].v
    Ux  = cg[Ux_field].v
    Uy  = cg[Uy_field].v
    Uz  = cg[Uz_field].v
    YH2 = cg[YH2_field].v
    P_rho = cg[pressure_field].v
    rho = cg[density_field].v
    mass_data = {f[1]: cg[f].v for f in mass_fields}

    # --- progress variable ---
    c = 1.0 - YH2 / yh2_ub

    # =========================
    # Y padding
    # =========================
    pad_cfg_y = ((0, 0), (ny_pad, ny_pad), (0, 0))
    pad_mode_y = "wrap" if y_is_full else "edge"

    x_pad  = np.pad(x,  pad_cfg_y, mode=pad_mode_y)
    y_pad  = np.pad(y,  pad_cfg_y, mode=pad_mode_y)
    z_pad  = np.pad(z,  pad_cfg_y, mode=pad_mode_y)

    T_pad  = np.pad(T,  pad_cfg_y, mode=pad_mode_y)
    Q_pad  = np.pad(Q,  pad_cfg_y, mode=pad_mode_y)
    Ux_pad = np.pad(Ux, pad_cfg_y, mode=pad_mode_y)
    Uy_pad = np.pad(Uy, pad_cfg_y, mode=pad_mode_y)
    Uz_pad = np.pad(Uz, pad_cfg_y, mode=pad_mode_y)
    c_pad  = np.pad(c,  pad_cfg_y, mode=pad_mode_y)
    rho_pad = np.pad(rho,  pad_cfg_y, mode=pad_mode_y)
    P_rho_pad=np.pad(P_rho,  pad_cfg_y, mode=pad_mode_y)
    mass_pad = {k: np.pad(v, pad_cfg_y, mode=pad_mode_y) for k, v in mass_data.items()}

    # --- periodic coordinate fix（只有 full 才需要）---
    if y_is_full and ny_pad > 0:
        Ly = dom_y1 - dom_y0
        y_pad[:, :ny_pad, :]  -= Ly
        y_pad[:, -ny_pad:, :] += Ly

    # =========================
    # Z ghost
    # =========================
    if add_zlower_ghost:
        pad_cfg_z = ((0, 0), (0, 0), (1, 0))

        x_pad  = np.pad(x_pad,  pad_cfg_z, mode=zghost_mode)
        y_pad  = np.pad(y_pad,  pad_cfg_z, mode=zghost_mode)
        z_pad  = np.pad(z_pad,  pad_cfg_z, mode=zghost_mode)

        T_pad  = np.pad(T_pad,  pad_cfg_z, mode=zghost_mode)
        Q_pad  = np.pad(Q_pad,  pad_cfg_z, mode=zghost_mode)
        Ux_pad = np.pad(Ux_pad, pad_cfg_z, mode=zghost_mode)
        Uy_pad = np.pad(Uy_pad, pad_cfg_z, mode=zghost_mode)
        Uz_pad = np.pad(Uz_pad, pad_cfg_z, mode=zghost_mode)
        c_pad  = np.pad(c_pad,  pad_cfg_z, mode=zghost_mode)
        P_rho_pad=np.pad(P_rho_pad,  pad_cfg_z, mode=zghost_mode)
        rho_pad = np.pad(rho_pad,  pad_cfg_z, mode= zghost_mode )
        mass_pad = {k: np.pad(v, pad_cfg_z, mode=zghost_mode) for k, v in mass_pad.items()}

        z_pad[:, :, 0] = z_pad[:, :, 1] - dz

    # =========================
    # bbox
    # =========================
    bbox = np.array([
        [float(x_pad.min() - dx/2), float(x_pad.max() + dx/2)],
        [float(y_pad.min() - dy/2), float(y_pad.max() + dy/2)],
        [float(z_pad.min() - dz/2), float(z_pad.max() + dz/2)],
    ])

    return {
        "bbox": bbox,
        "dx": dx,
        "dy": dy,
        "dz": dz,
        "ny_pad": ny_pad,
        "has_zlower_ghost": add_zlower_ghost,
        "x": x_pad,
        "y": y_pad,
        "z": z_pad,
        "T": T_pad,
        "Q": Q_pad,
        "ux": Ux_pad,
        "uy": Uy_pad,
        "uz": Uz_pad,
        "c": c_pad,
        "mass": mass_pad,
        "P":P_rho_pad,
        "rho":rho_pad,
    }

def compute_curvature_from_padded_arrays(
    box,
    Tcut=None,
    keep_only_original_core=True,
):
    import yt
    import numpy as np
    import pandas as pd

    x = box["x"]
    y = box["y"]
    z = box["z"]
    T = box["T"]
    Q = box["Q"]
    ux = box["ux"]
    uy = box["uy"]
    uz = box["uz"]
    c = box["c"]
    mass = box["mass"]

    bbox = box["bbox"]
    ny_pad = box["ny_pad"]
    has_zlower_ghost = box["has_zlower_ghost"]
    dx = box["dx"]
    P = box["P"]

    data_dict = {
        "c": c,
        "temp": T,
        "HeatRelease": Q,
        "x_velocity": ux,
        "y_velocity": uy,
        "z_velocity": uz,
    }
    for k, v in mass.items():
        data_dict[k] = v

    ds_u = yt.load_uniform_grid(
        data=data_dict,
        domain_dimensions=c.shape,
        bbox=bbox,
        length_unit="m",
        nprocs=1,
        unit_system="mks",
    )

    ds_u.add_gradient_fields(("stream", "c"))
    ds_u.add_gradient_fields(("stream", "Y(H2)"))

    def _nx(field, data):
        gx = data[("stream", "c_gradient_x")]
        g  = data[("stream", "c_gradient_magnitude")]
        eps = data.ds.quan(1e-30, str(g.units))
        return gx / (g + eps)

    def _ny(field, data):
        gy = data[("stream", "c_gradient_y")]
        g  = data[("stream", "c_gradient_magnitude")]
        eps = data.ds.quan(1e-30, str(g.units))
        return gy / (g + eps)

    def _nz(field, data):
        gz = data[("stream", "c_gradient_z")]
        g  = data[("stream", "c_gradient_magnitude")]
        eps = data.ds.quan(1e-30, str(g.units))
        return gz / (g + eps)

    ds_u.add_field(("stream", "nx"), function=_nx, sampling_type="cell",
                   units="dimensionless", force_override=True)
    ds_u.add_field(("stream", "ny"), function=_ny, sampling_type="cell",
                   units="dimensionless", force_override=True)
    ds_u.add_field(("stream", "nz"), function=_nz, sampling_type="cell",
                   units="dimensionless", force_override=True)

    ds_u.add_gradient_fields(("stream", "nx"))
    ds_u.add_gradient_fields(("stream", "ny"))
    ds_u.add_gradient_fields(("stream", "nz"))


    

    

    def _kappa(field, data):
        return (
            data[("stream", "nx_gradient_x")]
            + data[("stream", "ny_gradient_y")]
            + data[("stream", "nz_gradient_z")]
        )

    ds_u.add_field(("stream", "curvature"), function=_kappa,
                   sampling_type="cell", units="1/m", force_override=True)

    ad = ds_u.all_data()

    shape = c.shape

    out = {
        "x": x,
        "y": y,
        "z": z,
        "T": T,
        "Q": Q,
        "ux": ux,
        "uy": uy,
        "uz": uz,
        "umag": np.sqrt(ux**2 + uy**2 + uz**2),
        "c": c,
        "k": ad[("stream", "curvature")].v.reshape(shape),
        "cgx": ad[("stream", "c_gradient_x")].v.reshape(shape),
        "cgy": ad[("stream", "c_gradient_y")].v.reshape(shape),
        "cgz": ad[("stream", "c_gradient_z")].v.reshape(shape),
        "cgmag": ad[("stream", "c_gradient_magnitude")].v.reshape(shape),
        "gh2x": ad[("stream", "Y(H2)_gradient_x")].v.reshape(shape),
        "gh2y": ad[("stream", "Y(H2)_gradient_y")].v.reshape(shape),
        "gh2z": ad[("stream", "Y(H2)_gradient_z")].v.reshape(shape),
    }
    out["gh2mag"] = np.sqrt(out["gh2x"]**2 + out["gh2y"]**2 + out["gh2z"]**2)

    for k, v in mass.items():
        out[k] = v

    # remove padding before flattening
    y_slice = slice(ny_pad, -ny_pad if ny_pad > 0 else None)
    z_slice = slice(1, None) if has_zlower_ghost else slice(None)

    if keep_only_original_core:
        for k in list(out.keys()):
            out[k] = out[k][:, y_slice, z_slice]

    # flatten only once, at the end
    df = pd.DataFrame({k: v.ravel() for k, v in out.items()})

    if Tcut is not None:
        df = df[df["T"] > Tcut].copy()

    min_grid = dx
    half_cell = min_grid / 2.0
    df["x_grid"] = np.round((df["x"] - half_cell) / min_grid).astype(int)
    df["y_grid"] = np.round((df["y"] - half_cell) / min_grid).astype(int)
    df["z_grid"] = np.round((df["z"] - half_cell) / min_grid).astype(int)
    df["global_index"] = df.index

    return df, ds_u

def get_curvature_m3(d1,save_to=None,periodic_add:int=4,x_cut_region:list=[0.05,0.075],yh2_ub=0.01304,Tcut=305,output_dir='./', ):
    import os
    box = get_padded_box_arrays(
        d1,
        x_cut_region=(x_cut_region[0], x_cut_region[1]),
        z_cut_region=(0,1.1e-3),
        yh2_ub=yh2_ub,
        ny_pad=periodic_add,
        add_zlower_ghost=True,
    )
    df_curv = compute_curvature_from_padded_arrays_lite(
        box,
        Tcut=Tcut,
        keep_only_original_core=True,
    )
    filename = d1.split('/')[-1]
    
    if save_to:
        os.makedirs(save_to,exist_ok=True)
        total_out_direct = save_to+'/'+filename+'_curvature.csv'
        df_curv.to_csv(total_out_direct,index=False)
    

    return df_curv
def compute_divergence_from_padded_arrays(
    box,
    Tcut=None,
    keep_only_original_core=True,
):
    import yt
    import numpy as np
    import pandas as pd

    # --- unpack ---
    x = box["x"]
    y = box["y"]
    z = box["z"]
    T = box["T"]
    Q = box["Q"]
    ux = box["ux"]
    uy = box["uy"]
    uz = box["uz"]
    c = box["c"]
    mass = box["mass"]

    bbox = box["bbox"]
    ny_pad = box["ny_pad"]

    shape=ux.shape
    
    has_zlower_ghost = box["has_zlower_ghost"]
    dx = box["dx"]

    # --- build uniform grid (same as curvature) ---
    data_dict = {
        "x_velocity": (ux, "m/s"),
        "y_velocity": (uy, "m/s"),
        "z_velocity": (uz, "m/s"),
        "temp": (T, "K"),
    }
    for k, v in mass.items():
        data_dict[k] = v

    ds_u = yt.load_uniform_grid(
        data=data_dict,
        domain_dimensions=ux.shape,
        bbox=bbox,
        length_unit="m",
        nprocs=1,
        unit_system="mks",
    )

    # --- gradients ---
    ds_u.add_gradient_fields(("stream", "x_velocity"))
    ds_u.add_gradient_fields(("stream", "y_velocity"))
    ds_u.add_gradient_fields(("stream", "z_velocity"))

    # --- divergence ---
    def _div_u(field, data):
        return (
            data[("stream", "x_velocity_gradient_x")]
            + data[("stream", "y_velocity_gradient_y")]
            + data[("stream", "z_velocity_gradient_z")]
        )

    ds_u.add_field(
        ("stream", "div_u"),
        function=_div_u,
        sampling_type="cell",
        units="1/s",
        force_override=True,
    )

    ad = ds_u.all_data()

    dudx = ad[  ("stream", "x_velocity_gradient_x") ].v.reshape(shape)

    dvdy = ad[ ("stream", "y_velocity_gradient_y") ].v.reshape(shape)
    
    dwdz = ad[ ("stream", "z_velocity_gradient_z") ].v.reshape(shape)

    dvdx = ad[ ("stream", "y_velocity_gradient_x") ].v.reshape(shape)
    
    dzdx = ad[ ("stream", "z_velocity_gradient_x") ].v.reshape(shape)
    
    # --- assemble output (same style as curvature) ---
    out = {
        "x": x,
        "y": y,
        "z": z,
        "T": T,

        "ux": ux,
        "uy": uy,
        "uz": uz,
        #"umag": np.sqrt(ux**2 + uy**2 + uz**2),
        "dudx":dudx,
        "dvdy":dvdy,
        "dwdz":dwdz,
        
        "dvdx":dvdx,
        "dzdx":dzdx,
        #"div_u": ad[("stream", "div_u")].v.reshape(shape),
    }



    # --- flatten ---
    df = pd.DataFrame({k: v.ravel() for k, v in out.items()})
    """
    if Tcut is not None:
        df = df[df["T"] > Tcut].copy()
    """
    # --- grid index（照抄你的）---
    min_grid = dx
    half_cell = min_grid / 2.0
    df["x_grid"] = np.round((df["x"] - half_cell) / min_grid).astype(int)
    df["y_grid"] = np.round((df["y"] - half_cell) / min_grid).astype(int)
    df["z_grid"] = np.round((df["z"] - half_cell) / min_grid).astype(int)
    #df["global_index"] = df.index


    df=df.drop(columns=["x","y","z"],errors="ignore")
    return df, ds_u


def compute_curvature_from_padded_arrays_lite_old(
    box,
    Tcut=None,
    keep_only_original_core=True,
):

    import yt
    import numpy as np
    import pandas as pd

    # -----------------------------
    # unpack
    # -----------------------------

    x = box["x"]
    y = box["y"]
    z = box["z"]
    
    T = box["T"]
    Q = box["Q"]
    #yh2=box['Y(H2)']
    c = box["c"]
    P = box["P"]
    bbox = box["bbox"]

    ny_pad = box["ny_pad"]
    yh2 = box['mass']["Y(H2)"]
    yh = box['mass']["Y(H)"]
    yo = box['mass']["Y(O)"]
    yoh = box['mass']["Y(OH)"]
    yh2o = box['mass']["Y(H2O)"]
    yo2 = box['mass']["Y(O2)"]
    yho2 = box['mass']["Y(HO2)"]
    yh2o2 = box['mass']["Y(H2O2)"]
    yn2 = box['mass']["Y(N2)"]
    
    has_zlower_ghost = box["has_zlower_ghost"]

    dx = box["dx"]

    # -----------------------------
    # build uniform grid
    # -----------------------------

    data_dict = {
        "c": c,
    }

    ds_u = yt.load_uniform_grid(
        data=data_dict,
        domain_dimensions=c.shape,
        bbox=bbox,
        length_unit="m",
        nprocs=1,
        unit_system="mks",
    )

    # -----------------------------
    # gradients
    # -----------------------------

    ds_u.add_gradient_fields(("stream", "c"))

    def _nx(field, data):

        gx = data[("stream", "c_gradient_x")]

        g = data[("stream", "c_gradient_magnitude")]

        eps = data.ds.quan(1e-30, str(g.units))

        return gx / (g + eps)

    def _ny(field, data):

        gy = data[("stream", "c_gradient_y")]

        g = data[("stream", "c_gradient_magnitude")]

        eps = data.ds.quan(1e-30, str(g.units))

        return gy / (g + eps)

    def _nz(field, data):

        gz = data[("stream", "c_gradient_z")]

        g = data[("stream", "c_gradient_magnitude")]

        eps = data.ds.quan(1e-30, str(g.units))

        return gz / (g + eps)

    ds_u.add_field(
        ("stream", "nx"),
        function=_nx,
        sampling_type="cell",
        units="dimensionless",
        force_override=True,
    )

    ds_u.add_field(
        ("stream", "ny"),
        function=_ny,
        sampling_type="cell",
        units="dimensionless",
        force_override=True,
    )

    ds_u.add_field(
        ("stream", "nz"),
        function=_nz,
        sampling_type="cell",
        units="dimensionless",
        force_override=True,
    )

    # -----------------------------
    # second gradients
    # -----------------------------

    ds_u.add_gradient_fields(("stream", "nx"))
    ds_u.add_gradient_fields(("stream", "ny"))
    ds_u.add_gradient_fields(("stream", "nz"))

    # -----------------------------
    # curvature
    # -----------------------------

    def _kappa(field, data):

        return (
            data[("stream", "nx_gradient_x")]
            + data[("stream", "ny_gradient_y")]
            + data[("stream", "nz_gradient_z")]
        )

    ds_u.add_field(
        ("stream", "curvature"),
        function=_kappa,
        sampling_type="cell",
        units="1/m",
        force_override=True,
    )

    ad = ds_u.all_data()

    shape = c.shape

    # -----------------------------
    # ONLY KEEP NECESSARY DATA
    # -----------------------------

    out = {

        "x": x,

        "y": y,

        "z": z,

        "T": T,
        "P": P,
        "Q": Q,
        "Y(H2)": yh2,
        "k": ad[
            ("stream", "curvature")
        ].v.reshape(shape),
    }

    # -----------------------------
    # remove padding
    # -----------------------------

    y_slice = slice(
        ny_pad,
        -ny_pad if ny_pad > 0 else None
    )

    z_slice = (
        slice(1, None)
        if has_zlower_ghost
        else slice(None)
    )

    if keep_only_original_core:

        for k in list(out.keys()):

            out[k] = out[k][
                :,
                y_slice,
                z_slice
            ]

    # -----------------------------
    # flatten
    # -----------------------------

    df = pd.DataFrame({

        k: v.ravel()

        for k, v in out.items()

    })

    # -----------------------------
    # temperature filter
    # -----------------------------

    if Tcut is not None:

        df = df[
            df["T"] > Tcut
        ].copy()

    # -----------------------------
    # grid index
    # -----------------------------

    min_grid = dx

    half_cell = min_grid / 2.0

    df["x_grid"] = np.round(
        (df["x"] - half_cell) / min_grid
    ).astype(int)

    df["y_grid"] = np.round(
        (df["y"] - half_cell) / min_grid
    ).astype(int)

    df["z_grid"] = np.round(
        (df["z"] - half_cell) / min_grid
    ).astype(int)

    # -----------------------------
    # remove xyz
    # optional
    # -----------------------------

    df = df[
        [
            "x_grid",
            "y_grid",
            "z_grid",
            "T",
            "Q",
            "P",
            "Y(H2)",
            "k",
        ]
    ]

    return df



def compute_curvature_from_padded_arrays_lite(
    box,
    Tcut=None,
    keep_only_original_core=True,
):

    import yt
    import numpy as np
    import pandas as pd

    # ============================================================
    # unpack
    # ============================================================

    x = box["x"]
    y = box["y"]
    z = box["z"]

    T = box["T"]
    Q = box["Q"]

    c = box["c"]

    P = box["P"]

    rho = box["rho"]
    mass = box["mass"]

    bbox = box["bbox"]

    ny_pad = box["ny_pad"]

    has_zlower_ghost = box["has_zlower_ghost"]

    dx = box["dx"]

    # ============================================================
    # species mass fraction
    # ============================================================

    yh2   = box["mass"]["Y(H2)"]
    yh    = box["mass"]["Y(H)"]
    yo    = box["mass"]["Y(O)"]
    yoh   = box["mass"]["Y(OH)"]
    yh2o  = box["mass"]["Y(H2O)"]
    yo2   = box["mass"]["Y(O2)"]
    yho2  = box["mass"]["Y(HO2)"]
    yh2o2 = box["mass"]["Y(H2O2)"]
    yn2   = box["mass"]["Y(N2)"]

    # ============================================================
    # molecular weights
    # ============================================================

    W_H2   = 2.01588
    W_H    = 1.00794
    W_O    = 15.999
    W_OH   = 17.007
    W_H2O  = 18.01528
    W_O2   = 31.998
    W_HO2  = 33.006
    W_H2O2 = 34.0147
    W_N2   = 28.0134

    # ============================================================
    # cp calculation
    # ============================================================

    R = 8314.462618

    def nasa_cp_mass(T_in, a, W):

        cp_R = (
            a[0]
            + a[1]*T_in
            + a[2]*T_in**2
            + a[3]*T_in**3
            + a[4]*T_in**4
        )

        return R * cp_R / W

    a_h2 = [3.29812431, 8.24944174e-04, -8.14301529e-07, -9.47543433e-11, 4.13487224e-13]

    a_h = [2.50000000,0,0,0,0]

    a_o = [2.94642878,-1.63816649e-03,2.42103170e-06,-1.60284319e-09,3.89069636e-13]

    a_oh = [3.99201543,-2.40131752e-03,4.61793841e-06,-3.88113333e-09,1.36411470e-12]

    a_h2o = [4.19864056,-2.03643410e-03,6.52040211e-06,-5.48797062e-09,1.77197817e-12]

    a_o2 = [3.78245636,-2.99673416e-03,9.84730201e-06,-9.68129509e-09,3.24372837e-12]

    a_ho2 = [4.30179801,-4.74912051e-03,2.11582891e-05,-2.42763894e-08,9.29225124e-12]

    a_h2o2 = [4.27611269,-5.42822417e-04,1.67335701e-05,-2.15770813e-08,8.62454363e-12]

    a_n2 = [3.53100528,-1.23660987e-04,-5.02999433e-07,2.43530612e-09,-1.40881235e-12]

    cp_h2 = nasa_cp_mass(T, a_h2, W_H2)

    cp_h = nasa_cp_mass(T, a_h, W_H)

    cp_o = nasa_cp_mass(T, a_o, W_O)

    cp_oh = nasa_cp_mass(T, a_oh, W_OH)

    cp_h2o = nasa_cp_mass(T, a_h2o, W_H2O)

    cp_o2 = nasa_cp_mass(T, a_o2, W_O2)

    cp_ho2 = nasa_cp_mass(T, a_ho2, W_HO2)

    cp_h2o2 = nasa_cp_mass(T, a_h2o2, W_H2O2)

    cp_n2 = nasa_cp_mass(T, a_n2, W_N2)

    cp_mix = (

        yh2 * cp_h2

        + yh * cp_h

        + yo * cp_o

        + yoh * cp_oh

        + yh2o * cp_h2o

        + yo2 * cp_o2

        + yho2 * cp_ho2

        + yh2o2 * cp_h2o2

        + yn2 * cp_n2
    )

    # ============================================================
    # thermal conductivity interpolation
    # ============================================================

    T_table = np.array([
        300,400,500,600,700,800,900,
        1000,1100,1200,1300,1400,
        1500,1600,1700,1800,1900,2000
    ],dtype=float)

    lambda_table = {

        "H2": np.array([
            0.186822,0.228484,0.264547,0.298359,
            0.331276,0.363881,0.396427,0.429013,
            0.461661,0.494361,0.527083,0.559794,
            0.592455,0.625032,0.657492,0.689804,
            0.721941,0.753880
        ]),

        "H": np.array([
            0.293975,0.371383,0.439693,0.501773,
            0.559264,0.613202,0.664287,0.713019,
            0.759767,0.804812,0.848376,0.890633,
            0.931730,0.971783,1.010892,1.049142,
            1.086605,1.123343
        ]),

        "O": np.array([
            0.050761,0.061072,0.070506,0.079265,
            0.087488,0.095277,0.102707,0.109835,
            0.116706,0.123355,0.129812,0.136099,
            0.142236,0.148241,0.154126,0.159905,
            0.165587,0.171182
        ]),

        "OH": np.array([
            0.059315,0.071948,0.083444,0.094560,
            0.105563,0.116551,0.127546,0.138548,
            0.149540,0.160505,0.171424,0.182281,
            0.193059,0.203746,0.214330,0.224800,
            0.235149,0.245370
        ]),

        "H2O": np.array([
            0.026240,0.037015,0.048280,0.060463,
            0.073536,0.087356,0.101766,0.116620,
            0.131794,0.147183,0.162699,0.178270,
            0.193837,0.209349,0.224764,0.240049,
            0.255174,0.270115
        ]),

        "O2": np.array([
            0.026532,0.033939,0.041219,0.048224,
            0.054933,0.061362,0.067540,0.073496,
            0.079258,0.084851,0.090296,0.095612,
            0.100816,0.105921,0.110940,0.115884,
            0.120763,0.125584
        ]),

        "HO2": np.array([
            0.029391,0.039455,0.049221,0.058749,
            0.068069,0.077203,0.086166,0.094970,
            0.103625,0.112141,0.120524,0.128782,
            0.136920,0.144944,0.152858,0.160668,
            0.168378,0.175991
        ]),

        "H2O2": np.array([
            0.037047,0.050833,0.064185,0.077221,
            0.089982,0.102484,0.114734,0.126738,
            0.138499,0.150023,0.161314,0.172377,
            0.183217,0.193838,0.204247,0.214448,
            0.224445,0.234245
        ]),

        "N2": np.array([
            0.026463,0.032735,0.038990,0.045164,
            0.051224,0.057156,0.062956,0.068621,
            0.074153,0.079556,0.084832,0.089985,
            0.095020,0.099941,0.104751,0.109455,
            0.114057,0.118559
        ]),
    }

    def interp_lambda(T_in, sp):

        T_use = np.clip(T_in,300.0,2000.0)

        return np.interp(
            T_use,
            T_table,
            lambda_table[sp]
        )

    # ============================================================
    # lambda mixture
    # ============================================================

    denom = (

        yh2 / W_H2

        + yh / W_H

        + yo / W_O

        + yoh / W_OH

        + yh2o / W_H2O

        + yo2 / W_O2

        + yho2 / W_HO2

        + yh2o2 / W_H2O2

        + yn2 / W_N2
    )

    denom = np.maximum(denom,1e-300)

    xh2 = (yh2 / W_H2) / denom

    xh = (yh / W_H) / denom

    xo = (yo / W_O) / denom

    xoh = (yoh / W_OH) / denom

    xh2o = (yh2o / W_H2O) / denom

    xo2 = (yo2 / W_O2) / denom

    xho2 = (yho2 / W_HO2) / denom

    xh2o2 = (yh2o2 / W_H2O2) / denom

    xn2 = (yn2 / W_N2) / denom

    lambda_mix = (

        xh2 * interp_lambda(T,"H2")

        + xh * interp_lambda(T,"H")

        + xo * interp_lambda(T,"O")

        + xoh * interp_lambda(T,"OH")

        + xh2o * interp_lambda(T,"H2O")

        + xo2 * interp_lambda(T,"O2")

        + xho2 * interp_lambda(T,"HO2")

        + xh2o2 * interp_lambda(T,"H2O2")

        + xn2 * interp_lambda(T,"N2")
    )

    # ============================================================
    # rho alpha
    # ============================================================

    rho_alpha = lambda_mix / cp_mix

    omega_T = Q / cp_mix

    # ============================================================
    # build yt grid
    # ============================================================

    data_dict = {

        "c": (c,""),

        "temp": (T,"K"),

        "rho_alpha": (rho_alpha,"kg/(m*s)"),

        "omega_T": (omega_T,"kg*K/(m**3*s)"),

        "density": (rho,"kg/m**3"),
    }

    ds_u = yt.load_uniform_grid(
        data=data_dict,
        domain_dimensions=c.shape,
        bbox=bbox,
        length_unit="m",
        nprocs=1,
        unit_system="mks",
    )

    # ============================================================
    # curvature
    # ============================================================

    ds_u.add_gradient_fields(("stream", "c"))

    def _nx(field, data):

        gx = data[("stream", "c_gradient_x")].to("1/m")

        g = data[("stream", "c_gradient_magnitude")].to("1/m")

        eps = data.ds.quan(1e-8,"1/m" )
        out = gx / (g + eps)
        return out.to("")

    def _ny(field, data):

        gy = data[("stream", "c_gradient_y")].to("1/m")

        g = data[("stream", "c_gradient_magnitude")].to("1/m")

        eps = data.ds.quan(1e-8,"1/m" )

        out = gy / (g + eps)
        return out.to("")


    def _nz(field, data):

        gz = data[("stream", "c_gradient_z")].to("1/m")

        g = data[("stream", "c_gradient_magnitude")].to("1/m")

        eps = data.ds.quan(1e-8,"1/m" )

        out = gz / (g + eps)
        return out.to("")


    ds_u.add_field(
        ("stream", "nx"),
        function=_nx,
        sampling_type="cell",
        units="",
        force_override=True,
    )

    ds_u.add_field(
        ("stream", "ny"),
        function=_ny,
        sampling_type="cell",
        units="",
        force_override=True,
    )

    ds_u.add_field(
        ("stream", "nz"),
        function=_nz,
        sampling_type="cell",
        units="",
        force_override=True,
    )

    ds_u.add_gradient_fields(("stream", "nx"))

    ds_u.add_gradient_fields(("stream", "ny"))

    ds_u.add_gradient_fields(("stream", "nz"))

    def _kappa(field, data):

        return (
            data[("stream", "nx_gradient_x")]
            + data[("stream", "ny_gradient_y")]
            + data[("stream", "nz_gradient_z")]
        ).to("1/m")

    ds_u.add_field(
        ("stream", "curvature"),
        function=_kappa,
        sampling_type="cell",
        units="1/m",
        force_override=True,
    )

    # ============================================================
    # Sd calculation
    # ============================================================

    ds_u.add_gradient_fields(("stream","temp"))

    def _flux_x(field, data):

        return (
            data[("stream","rho_alpha")]
            *
            data[("stream","temp_gradient_x")]
        )

    def _flux_y(field, data):

        return (
            data[("stream","rho_alpha")]
            *
            data[("stream","temp_gradient_y")]
        )

    def _flux_z(field, data):

        return (
            data[("stream","rho_alpha")]
            *
            data[("stream","temp_gradient_z")]
        )

    ds_u.add_field(
        ("stream","flux_x"),
        function=_flux_x,
        sampling_type="cell",
        units="kg*K/(m**2*s)",
        force_override=True,
    )

    ds_u.add_field(
        ("stream","flux_y"),
        function=_flux_y,
        sampling_type="cell",
        units="kg*K/(m**2*s)",
        force_override=True,
    )

    ds_u.add_field(
        ("stream","flux_z"),
        function=_flux_z,
        sampling_type="cell",
        units="kg*K/(m**2*s)",
        force_override=True,
    )

    ds_u.add_gradient_fields(("stream","flux_x"))

    ds_u.add_gradient_fields(("stream","flux_y"))

    ds_u.add_gradient_fields(("stream","flux_z"))

    def _Sd(field, data):

        div_flux = (

            data[("stream","flux_x_gradient_x")]

            + data[("stream","flux_y_gradient_y")]

            + data[("stream","flux_z_gradient_z")]
        )

        gradTmag = data[
            ("stream","temp_gradient_magnitude")
        ]

        eps = data.ds.quan(
            1e-10,
            str(gradTmag.units)
        )

        return (

            div_flux

            + data[("stream","omega_T")]

        ) / (

            data[("stream","density")]

            * (gradTmag + eps)
        )

    ds_u.add_field(
        ("stream","Sd"),
        function=_Sd,
        sampling_type="cell",
        units="m/s",
        force_override=True,
    )

    # ============================================================
    # output
    # ============================================================

    ad = ds_u.all_data()

    shape = c.shape

    out = {

        "x": x,

        "y": y,

        "z": z,

        "T": T,

        "Q": Q,

        "P": P,

        "rho": rho,

        "cp": cp_mix,

        "lambda": lambda_mix,

        "k": ad[
            ("stream","curvature")
        ].v.reshape(shape),

        "Sd": ad[
            ("stream","Sd")
        ].v.reshape(shape),
        "gradT":ad[("stream", "temp_gradient_magnitude")].v.reshape(shape),
    }
    for k, v in mass.items():
        out[k] = v

    # ============================================================
    # remove padding
    # ============================================================

    y_slice = slice(
        ny_pad,
        -ny_pad if ny_pad > 0 else None
    )

    z_slice = (
        slice(1,None)
        if has_zlower_ghost
        else slice(None)
    )

    if keep_only_original_core:

        for k in list(out.keys()):

            out[k] = out[k][
                :,
                y_slice,
                z_slice
            ]

    # ============================================================
    # flatten
    # ============================================================

    df = pd.DataFrame({

        k: v.ravel()

        for k,v in out.items()
    })

    # ============================================================
    # temperature cut
    # ============================================================

    if Tcut is not None:

        df = df[
            df["T"] > Tcut
        ].copy()

    # ============================================================
    # grid index
    # ============================================================

    min_grid = dx

    half_cell = min_grid / 2.0

    df["x_grid"] = np.round(
        (df["x"] - half_cell) / min_grid
    ).astype(int)

    df["y_grid"] = np.round(
        (df["y"] - half_cell) / min_grid
    ).astype(int)

    df["z_grid"] = np.round(
        (df["z"] - half_cell) / min_grid
    ).astype(int)

    # ============================================================
    # final columns
    # ============================================================

    df = df[
        [
            "x_grid",
            "y_grid",
            "z_grid",
            "T",
            "Q",
            "P",
            "rho",
            "cp",
            "Y(H2)",
            "lambda",
            "k",
            "Sd",
        ]
    ]

    return df





def get_strain_rate(d1,periodic_add:int=4,x_cut_region:list=[0.05,0.075],y_cut_region=None,z_cut_region=[0,5.44e-4],yh2_ub=0.01304,Tcut=305, save_to=None):
    import os
    filename = d1.split('/')[-1]
    box1 = get_padded_box_arrays(
        d1,
        x_cut_region=x_cut_region,
        y_cut_region=y_cut_region,
        z_cut_region=z_cut_region,
        yh2_ub=0.01304,
        ny_pad=4,
        add_zlower_ghost=True,
    )
    df_t2,_ = compute_divergence_from_padded_arrays(box1)
    
    if save_to:
        os.makedirs(save_to,exist_ok=True)
        total_out_direct = save_to+'/'+filename+'_strain_rate.csv'
        df_t2.to_csv(total_out_direct,index=False)

    return df_t2


def get_combined_incsv(d1,yh2_u,save_dir,Tcut):
    flame_position  = read_get_flamex(d1,Tcut =305, )
    _ = get_curvature_m3(d1, periodic_add=4,x_cut_region=[flame_position[0],0.0705],yh2_ub=yh2_u,Tcut=Tcut,save_to=save_dir)
    _ = get_strain_rate(d1, periodic_add=4,x_cut_region=[flame_position[0]-0.005,0.07], z_cut_region=[0,5.47e-4], yh2_ub=yh2_u,Tcut=Tcut,save_to=save_dir)
