"u0890475 reversion, v1"

import cantera as ct
import numpy as np
import h5py
import yaml
with open('mechanism.yaml', 'r') as file:
    mechanism_info = yaml.safe_load(file)
    
species_dict ={}
species_dict_seq={}
try:
    species_info = mechanism_info["species"]
    species_ttl_number = len(species_info)
    print("The species in the mechanism in sequence is: ")
    for i in range(len(species_info)):
        species_dict_seq[i]=species_info[i]["name"]
        species_dict[species_info[i]["name"]]=i
        print(str(i)+": "+species_info[i]["name"])
except:
    pass

def ctFlame(P,T,phi,Vel):
	loglevel = 1
	g = ct.Solution('mechanism.yaml')
	mdotR = Vel*g.density
	ct.transport_model = 'Multi'
	air = "O2:0.209,N2:0.791" # mol fraction of air with only N2,O2
	g.set_equivalence_ratio(phi=phi, fuel="H2:1", oxidizer=air)
	#g.X = {'H2':1, 'O2':.5/phi, 'N2':(.5*3.76)/phi}
	g.TP = T,P
	gridd = np.linspace(0,.005,1024)
	flame = ct.FreeFlame(gas = g,grid = gridd)
    #https://cantera.org/2.0/doxygen/html/classCantera_1_1FreeFlame.html
	flame.transport_model = 'multicomponent'
	flame.inlet.mdot = mdotR
	flame.set_refine_criteria(ratio=4, slope=0.05, curve=0.1, prune=0.01)
	flame.solve(loglevel=loglevel, refine_grid = False, auto = False)
	return flame


phi = 0.375
flame = ctFlame(101325,298,phi,8)
Y1 = flame.Y
T = flame.T

mass_deficit_init = 1-(Y1[0][0] +Y1[1][0] + Y1[8][0]) # might change if use different mechanism
initprof_2 = [  x[0] if x[0]>0 else 0 for x in Y1]
initprof_2[int(species_dict["N2"])]+= (mass_deficit_init)
init_array = np.array(initprof_2)
outprof_2=[x[-2] for x in Y1]
mass_deficit_outlet =1- sum(outprof_2)
outprof_2[int(species_dict["N2"])]+= (mass_deficit_outlet)
out_array= np.array(outprof_2)
T_ab = T[-2]

print("Inlet total mass deficit is " + str(mass_deficit_init) )
print("Outlet total mass deficit is " + str(mass_deficit_outlet))
    
title='Inlet_Profile_phi' + str(round(phi*100,-1))
hwrite = h5py.File(title + '.h5', 'w')
hwrite.create_dataset("inlet",data=init_array)
hwrite.create_dataset("outlet",data=out_array)
hwrite.create_dataset("T_ab",data=T_ab)
hwrite.close()


"""
rewrite the hard coding part

# [H2, O2, H2O, H, O, OH, HO2, H2O2, N2]
# the species list from mechaniusm.yaml file for LiDryer folder
YH2 = Y1[0]
YO2 = Y1[1]
YH2O = Y1[2]
YH = Y1[3]
YO = Y1[4]
YOH = Y1[5]
YHO2 = Y1[6]
YH2O2 = Y1[7]
YN2 = Y1[8]

initprof = np.zeros([9,1])
initprof[0] = YH2[0]
initprof[1] = YO2[0]
initprof[2] = YH2O[0]
initprof[3] = YH[0]
initprof[4] = YO[0]
initprof[5] = YOH[0]
initprof[6] = YHO2[0]
initprof[7] = YH2O2[0]
initprof[8] = YN2[0]


mass_deficit =1-(initprof[8] + initprof[0] + initprof[1])
initprof[8] = initprof[8] + mass_deficit # add to nitrogen

initprof[0] = YH2[0]
initprof[1] = YO2[0]
initprof[2] = 0
initprof[3] = 0
initprof[4] = 0
initprof[5] = 0
initprof[6] = 0
initprof[7] = 0
initprof[8] = initprof[8]


## outlet profiles
outprof = np.zeros([9,1])
outprof[0] = YH2[-2]
outprof[1] = YO2[-2]
outprof[2] = YH2O[-2]
outprof[3] = YH[-2]
outprof[4] = YO[-2]
outprof[5] = YOH[-2]
outprof[6] = YHO2[-2]
outprof[7] = YH2O2[-2]
outprof[8] = YN2[-2]


mass_deficit_outlet = 1-sum(outprof)
outprof[8] = outprof[8] + mass_deficit_outlet # add to nitrogen

T_ab = T[-2]

title = 'Inlet_Profile_phi' + str(round(phi*100,-1))
hf = h5py.File(title + '2.h5', 'w')
hf.create_dataset('inlet', data=initprof)
hf.create_dataset('outlet', data=outprof)
hf.create_dataset('T_ab',data = T_ab)
hf.close()

print("phi = " + str(phi))
print("inlet")
print(str(initprof))
print("outlet")
print(str(outprof))
print("T_ab")
print(str(T_ab))


# how to read the h5 files because I'm stupid
f2 = h5py.File(title + '2.h5','r') # open in reading
Inlets = f['inlet']
Outlets = f['outlet']
#print(Inlets[::])


print("Inlet total mass deficit" + str(1 - sum(initprof)))
print("Outlet total mass deficit" + str(1 - sum(outprof)))


f.close()
"""