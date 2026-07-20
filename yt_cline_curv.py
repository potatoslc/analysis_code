import yt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import bisect
from scipy.spatial import cKDTree
import time
import math
import argparse
import yt_flame_allfunc_v2_2 as ytf 


def compute_curvature_from_padded_arrays_lite(
        box,
        Tcut=None,
        keep_only_original_core=True,
        get_curvature=False,
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
    #IRH2 = box["IRH2"]
    yh2_ub = np.percentile(yh2,99)
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

    lambda_coeffs={
        "H2":[-0.850373,0.505140,-0.110480,0.010639,-0.000378],
        "O2":[0.083824,-0.050128,0.011207,-0.001097,0.000040],
        "H2O":[-0.401999,0.250482,-0.058071,0.005928,-0.000223],
        "H":[-0.191615,0.095538,-0.016523,0.001297,-0.000037],
        "O":[0.026830,-0.015741,0.003722,-0.000381,0.000015],
        "OH":[-0.245023,0.150100,-0.033853,0.003364,-0.000123],
        "HO2":[-0.018484,0.011374,-0.002644,0.000285,-0.000011],
        "H2O2":[-0.074315,0.046267,-0.010888,0.001157,-0.000045],
        "N2":[0.002613,0.001593,-0.000984,0.000165,-0.000008],
        }

    def species_lambda(T_in,coeff):
        logT=np.log(T_in)
        poly=(
            coeff[0]
            +coeff[1]*logT
            +coeff[2]*logT**2
            +coeff[3]*logT**3
            +coeff[4]*logT**4
        )

        return np.sqrt(T_in)*poly
    denom_x=(
        yh2/W_H2
        +yh/W_H
        +yo/W_O
        +yoh/W_OH
        +yh2o/W_H2O
        +yo2/W_O2
        +yho2/W_HO2
        +yh2o2/W_H2O2
        +yn2/W_N2
        )

    denom_x=np.maximum(denom_x,1e-10)
    xh2=(yh2/W_H2)/denom_x
    xh=(yh/W_H)/denom_x
    xo=(yo/W_O)/denom_x
    xoh=(yoh/W_OH)/denom_x
    xh2o=(yh2o/W_H2O)/denom_x
    xo2=(yo2/W_O2)/denom_x
    xho2=(yho2/W_HO2)/denom_x
    xh2o2=(yh2o2/W_H2O2)/denom_x
    xn2=(yn2/W_N2)/denom_x

    lambda_h2=species_lambda(T,lambda_coeffs["H2"])
    lambda_o2=species_lambda(T,lambda_coeffs["O2"])
    lambda_h2o=species_lambda(T,lambda_coeffs["H2O"])
    lambda_h=species_lambda(T,lambda_coeffs["H"])
    lambda_o=species_lambda(T,lambda_coeffs["O"])
    lambda_oh=species_lambda(T,lambda_coeffs["OH"])
    lambda_ho2=species_lambda(T,lambda_coeffs["HO2"])
    lambda_h2o2=species_lambda(T,lambda_coeffs["H2O2"])
    lambda_n2=species_lambda(T,lambda_coeffs["N2"])

    term1=(
        xh2*lambda_h2
        +xh*lambda_h
        +xo*lambda_o
        +xoh*lambda_oh
        +xh2o*lambda_h2o
        +xo2*lambda_o2
        +xho2*lambda_ho2
        +xh2o2*lambda_h2o2
        +xn2*lambda_n2
        )

    denom_lambda=(
        xh2/lambda_h2
        +xh/lambda_h
        +xo/lambda_o
        +xoh/lambda_oh
        +xh2o/lambda_h2o
        +xo2/lambda_o2
        +xho2/lambda_ho2
        +xh2o2/lambda_h2o2
        +xn2/lambda_n2
        )

    denom_lambda=np.maximum(denom_lambda,1e-300)
    term2=1.0/denom_lambda
    lambda_mix=0.5*(term1+term2)
    # ============================================================
    # rho alpha
    # ============================================================

    rho_alpha = lambda_mix / cp_mix
    omega_T = Q / cp_mix

    # ============================================================
    # build yt grid
    # ============================================================


    #ds_u.add_gradient_fields(("stream", "Y(H2)"))

    
    def binary_D(T,P,b0,b1,b2,b3,b4):
        logT = np.log(T)
        poly = b0 + b1*logT + b2*logT**2 + b3*logT**3 + b4*logT**4
        return  T**1.5 * poly/P

    

    
    
    
    D_H2_O2 = binary_D(T,P,-0.008079,0.004641,-0.000876,0.000077,-2.473269e-06)
    D_H2_H2O = binary_D(T,P,-0.009701,0.004014,-0.000468,0.000019,9.241024e-08)
    D_H2_H = binary_D(T,P,-0.024947,0.013946,-0.002601,0.000225,-7.131544e-06)
    D_H2_O = binary_D(T,P,-0.009170,0.005424,-0.001032,0.000092,-2.985782e-06)
    D_H2_OH = binary_D(T,P,-0.009139,0.005406,-0.001029,0.000092,-2.975863e-06)
    D_H2_HO2 = binary_D(T,P,-0.008071,0.004637,-0.000875,0.000077,-2.471030e-06)
    D_H2_H2O2 = binary_D(T,P,-0.008064,0.004633,-0.000874,0.000077,-2.468921e-06)
    D_H2_N2 = binary_D(T,P,-0.007323,0.004247,-0.000803,0.000071,-2.284269e-06)

    epsD = 1e-20

    den_D = xo2/D_H2_O2 + xh2o/D_H2_H2O + xh/D_H2_H + xo/D_H2_O + xoh/D_H2_OH + xho2/D_H2_HO2 + xh2o2/D_H2_H2O2 + xn2/D_H2_N2

    den_D = np.maximum(den_D, epsD)

    D_H2_mix = (1.0 - xh2) / den_D

    rhoD_H2 = rho * D_H2_mix

    data_dict = {
        "c": (c,""),
        "Y(H2)": (yh2,""),
        "temp": (T,"K"),
        "rho_alpha": (rho_alpha,"kg/(m*s)"),
        "omega_T": (omega_T,"kg*K/(m**3*s)"),
        "density": (rho,"kg/m**3"),
        #"IRH2": (IRH2,"kg/(m**3*s)"),
        "rhoD_H2": (rhoD_H2,"kg/(m*s)"),
        "D_H2_mix":(D_H2_mix,"m**2/s"),
        
    }
    
    
    ds_u = yt.load_uniform_grid(
        data=data_dict,
        domain_dimensionscompute_curv=c.shape,
        bbox=bbox,
        length_unit="m",
        nprocs=1,
        unit_system="mks",
    )

    ds_u.add_gradient_fields(("stream","Y(H2)"))

    def _YH2_flux_x(field,data):
        return data[("stream","rhoD_H2")] * data[("stream","Y(H2)_gradient_x")]

    def _YH2_flux_y(field,data):
        return data[("stream","rhoD_H2")] * data[("stream","Y(H2)_gradient_y")]

    def _YH2_flux_z(field,data):
        return data[("stream","rhoD_H2")] * data[("stream","Y(H2)_gradient_z")]


    ds_u.add_field(("stream","YH2_flux_x"), function=_YH2_flux_x, sampling_type="cell", units="kg/(m**2*s)", force_override=True)

    ds_u.add_field(("stream","YH2_flux_y"), function=_YH2_flux_y, sampling_type="cell", units="kg/(m**2*s)", force_override=True)

    ds_u.add_field(("stream","YH2_flux_z"), function=_YH2_flux_z, sampling_type="cell", units="kg/(m**2*s)", force_override=True)

    ds_u.add_gradient_fields(("stream","YH2_flux_x"))
    ds_u.add_gradient_fields(("stream","YH2_flux_y"))
    ds_u.add_gradient_fields(("stream","YH2_flux_z"))
    
    def _div_rhoD_grad_YH2(field,data):
        return data[("stream","YH2_flux_x_gradient_x")] + data[("stream","YH2_flux_y_gradient_y")] + data[("stream","YH2_flux_z_gradient_z")]

    ds_u.add_field(("stream","div_rhoD_grad_YH2"), function=_div_rhoD_grad_YH2, sampling_type="cell", units="kg/(m**3*s)", force_override=True)
    def _Sr_H2(field,data):

        gradY = data[("stream","Y(H2)_gradient_magnitude")]
        eps = data.ds.quan(1e-8,str(gradY.units))

        return -1 / (data[("stream","density")] * (gradY + eps))

    def _Sn_H2(field,data):

        gradY = data[("stream","Y(H2)_gradient_magnitude")]
        eps = data.ds.quan(1e-8,str(gradY.units))

        return -data[("stream","div_rhoD_grad_YH2")] / (data[("stream","density")] * (gradY + eps))

    ds_u.add_field(("stream","Sn_H2"), function=_Sn_H2, sampling_type="cell", units="m/s", force_override=True)

    def _Sd_YH2(field,data):
        return data[("stream","Sr_H2")] + data[("stream","Sn_H2")]

    #ds_u.add_field(("stream","Sd_YH2"), function=_Sd_YH2, sampling_type="cell", units="m/s", force_override=True)
    # ============================================================
    # curvature
    # ============================================================

    ds_u.add_gradient_fields(("stream", "c"))
    def _nx(field, data):
        gx = data[("stream", "c_gradient_x")].to("1/m")
        g = data[("stream", "c_gradient_magnitude")].to("1/m")
        eps = data.ds.quan(1e-6,"1/m" )
        out = -1*gx/(g+eps)
        return out.to("")
    
    def _ny(field, data):
        gy = data[("stream", "c_gradient_y")].to("1/m")
        g = data[("stream", "c_gradient_magnitude")].to("1/m")
        eps = data.ds.quan(1e-6,"1/m" )
        out = -1*gy /(g+eps)
        return out.to("")


    def _nz(field, data):
        gz = data[("stream", "c_gradient_z")].to("1/m")
        g = data[("stream", "c_gradient_magnitude")].to("1/m")
        eps = data.ds.quan(1e-6,"1/m" )
        out = -1*gz /(g+eps)
        return out.to("")



    def _nmag(field,data):
        return data[("stream", "c_gradient_magnitude")].to("1/m")
    
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
    ds_u.add_field(
        ("stream", "nmag"),
        function=_nmag,
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

    # calculate strain rate:
    ds_u.add_gradient_fields(("stream", "x_velocity"))
    ds_u.add_gradient_fields(("stream", "y_velocity"))
    ds_u.add_gradient_fields(("stream", "z_velocity"))
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
    def _flx(field, data):
        
        return data[("stream","flux_x_gradient_x")] + data[("stream","flux_y_gradient_y")] + data[("stream","flux_z_gradient_z")]
 
    ds_u.add_field(
	("stream","flx"),
        function=_flx,
	sampling_type="cell",
	units="K*kg/(m**3*s)",
	force_override=True,
    )

    
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

        return  -1*(
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
    #data_dict["rhoD_H2"] = (rhoD_H2, "kg/(m*s)")

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
        "rho": rho,
        "cp": cp_mix,
        "lambda": lambda_mix,
        "k": ad[
            ("stream","curvature")
        ].v.reshape(shape),
        "Sd": ad[
            ("stream","Sd")
        ].v.reshape(shape),
        "gradTmag":ad[("stream", "temp_gradient_magnitude")].v.reshape(shape),
        "flx":ad[("stream", "flx")].v.reshape(shape),
        "divnx":ad[("stream", "nx_gradient_x")].v.reshape(shape),
        "divny":ad[("stream", "ny_gradient_y")].v.reshape(shape),
        "divnz":ad[("stream", "nz_gradient_z")].v.reshape(shape),
        "cgx": ad[("stream", "c_gradient_x")].v.reshape(shape),
        "cgy": ad[("stream", "c_gradient_y")].v.reshape(shape),
        "cgz": ad[("stream", "c_gradient_z")].v.reshape(shape),
        "nz":ad[("stream", "nz")].v.reshape(shape),
        "ny":ad[("stream", "ny")].v.reshape(shape),
        "nx":ad[("stream", "nx")].v.reshape(shape),
        "D_H2_mix":ad[("stream", "D_H2_mix")].v.reshape(shape),
        "div_u":ad[("stream", "div_u")].v.reshape(shape),
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
            "cgx",
            "cgy",
            "cgz"
            #"P",
            #"rho",
            #"cp",
            "Y(H2)",
            "Y(H)",
            "Y(O2)",
            "Y(H2O)",
            "Y(O)",
            "Y(H2O2)",
            "Y(OH)",
            "y(HO2)",
            #"lambda",
            "k",
            "Sd",
            "div_u",
        ]
    ]
    # 找到每个y_grid对应的front位置
    front_df = (
        df[df['T'] > Tcut]
        .groupby(['y_grid','z_grid'])['x_grid']
        .min()
        .rename('x_front')
    )
    df2 = df.join(front_df,on=['y_grid','z_grid'])
    df2 = df2[df2['x_grid'] >= df2['x_front']-20  ].copy()
    df2.drop(columns='x_front',inplace=True)
    
    
    return df2


def process_plt(d1):
    from scipy.signal import find_peaks

    flame_position  = ytf.read_get_flamex(d1,Tcut =300, )
    box = ytf.get_padded_box_arrays(
        d1,
        x_cut_region=(flame_position-0.001, 0.0695),
        z_cut_region=(0,0.55e-3),
        yh2_ub=yh2_ub,
        ny_pad=periodic_add,
        add_zlower_ghost=True,
    )
    df_comb = compute_curvature_from_padded_arrays_lite(box,Tcut=300, keep_only_original_core=True)
    filename = d1.split('/')[-1]
    fullname = filename+"_k_divu.csv"
    df_comb.to_csv(fullname,index=False)
    df_comb['oindex']=df_comb.index
    df_front_list = (df_comb.groupby(['y_grid'],as_index=False)['x_grid'].min())
    mins,_=find_peaks(df_front_list['y_grid'])
    print(mins)
    original_index = df_front_list['oindex'].iloc[mins]
    peak_points = df_comb.iloc[original_index]

    #from here, do the progress variable gradient line


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir",required=True,help="Input pltfile")
    args = parser.parse_args()
    process_plt(args.input_dir)


if __name__== "__main__":
    main()






