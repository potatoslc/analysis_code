### Critically Strained Boundary Layer Flashback Code
### Authors: Alex Novoselov, Dominik Ebi
### Purpose: Uses CANTERA to automatically compute one dimensional premixed counterflow flames at their extinction limit, outputting properties needed to compute boundary layer flashback limits. Specifically, the model requires solution of the equation << (s_l,ext)^2 / (alpha * g) = 1 >> where s_l,ext is the extinction limit flame speed of the premixed counterflow flame, alpha is the thermal diffusivity (taken at the location of maximum temperature gradient), and g is the axial wall velocity gradient.
### Usage: This is an in-development research code. Things may not always work without some tinkering. You must have python3 and CANTERA installed for the code to work. It can be run from the command line as "python3 blf.py". Extinction limits are then computed at conditions specified in the user variable "condfile" (see Files section below), and flame properties are recorded in a file called "extinction" within the subdirectory specified by the user variable "outputdir" (see Files section below). The chemical mechanism used is specified by the user defined variable "mechfile" (see Files section below). Two mechanisms are readily provided in CANTERA .cti format for pure hydrogen (H2_Li.cti) and methane/hydrogen (SDmech.cti). Users should not modify the "Numerics" section unless they are familiar with what the different values do (e.g., have read the CANTERA documentation). The section labeled "Stretch properties" defines the initial inlet velocity ("u0"), the width of the counterflow simulation ("width"), and the acceptable error on the extinction limit as a fraction of the extinction limit ("u_del"). For example, u_del = 0.01 means the difference between the highest velocity where the flame burns, and the lowest velocity where extinction occurs is within 1 percent of the velocity where extinction occurs.


##### Input Parameters #####

### Stretch Properties
u0 = 0.125            # m/s
u_del = 0.01            # percent error off burning vel from extinction
width = 0.005           # half width of domain

manualgrid = True    # Use a refined initial grid if crashing
numgrid = 250          # Number of points for a manual grid

### Numerics
loglevel=0
ratio=3
slope=0.1
curve=0.2
prune=0.03

### Files
#mechfile = 'SDmech.cti'
# may need to change that 
mechfile = 'H2_Li.yaml'
#mechfile = 'H2_Li.cti'
outputdir = 'output/'
condfile = 'conditions'
condusecel = 0         # Set to 1 if the conditions file uses celcius, 0 for kelvin 
useprevsol = False     # Slower when true, but better convergence (i.e. no floating point exceptions)

############################

import cantera as ct
import numpy as np
import sys
from numpy import savetxt
import os

### Functions ###

# Differentiation function for data that has variable grid spacing Used here to compute normal strain-rate
def derivative(x, y):
    dydx = np.zeros(y.shape, y.dtype.type)

    dx = np.diff(x)
    dy = np.diff(y)
    dydx[0:-1] = dy/dx

    dydx[-1] = (y[-1] - y[-2])/(x[-1] - x[-2])

    return dydx

# Compute all the flame properties needed for the model
def computeFlameProperties(oppFlame):

    # Evaluate flame front location
    dTdx = derivative(oppFlame.grid, oppFlame.T)
    FlameFrontLoc=dTdx.argmax()
    if (FlameFrontLoc == 0):
        # error check incase flame anchored to boundary
        FlameFrontLoc = 1
    
    # Compute the derivative of axial velocity to obtain normal strain rate
    try:
        strainRates = derivative(oppFlame.grid, oppFlame.u) #u is not found, maybe Uo?
    except:
        strainRates = derivative(oppFlame.grid, oppFlame.Uo)

    # Obtain the location of the max. strain rate upstream of the pre-heat zone.
    # This is the characteristic strain rate
    # Only search between inlet and flame front location to prevent evaluation in burnt gas
    maxStrLocation = abs(strainRates[:FlameFrontLoc]).argmax()
    
    if (maxStrLocation == 0):
        # error check incase flame anchored to boundary
        maxStrLocation = 1

    minVelocityPoint = oppFlame.Uo[:maxStrLocation].argmin()
    if (minVelocityPoint == 0):
        # error check incase flame anchored to boundary
        minVelocityPoint = 1

    # Characteristic Strain Rate = K
    strainRatePoint = abs(strainRates[:minVelocityPoint]).argmax()
    K = abs(strainRates[strainRatePoint])
    
    # Properties computed at location of maximum temperature gradient
    Sd_05 = oppFlame.Uo[FlameFrontLoc]
    thermdiff_05 = (oppFlame.thermal_conductivity[FlameFrontLoc]/oppFlame.cp_mass[FlameFrontLoc]/oppFlame.density[FlameFrontLoc])
    
    return K, Sd_05, thermdiff_05

def solveOpposedFlame(oppFlame, massFlux, loglevel, ratio, slope, curve, prune):
    """
    Execute this function to run the Oppposed Flow Simulation This function
    takes a CounterFlowTwinPremixedFlame object as the first argument
    """

    oppFlame.reactants.mdot = massFlux
    oppFlame.set_refine_criteria(ratio=ratio, slope=slope, curve=curve, prune=prune)
    oppFlame.solve(loglevel, auto=True)

### Begin Program ###

if (loglevel == 0):
    ct.suppress_thermo_warnings()

# Load conditions to solve for extinction
cond = np.genfromtxt(condfile, delimiter=',')

# write the extinction file header
if (os.path.exists(outputdir) == False):
    os.system("mkdir " + outputdir)
if (not os.path.exists(outputdir+'extinction')):
    file = open(outputdir+'extinction', 'a')
    header = 'P [Pa]				T [K]			        phi				xH2				K [1/s]				Sd_05 [m/s]			thermdiff_05 [m2/s]\n'
    file.write(header)
    file.close()

max_burn_vel = 0
min_ext_vel = 0

for i in range(len(cond)-1):
    # Loop through all conditions saved in textfile

    P = cond[i+1][0]*101325
    T = cond[i+1][1]+273.15*condusecel
    xH2 = cond[i+1][2]
    phi = cond[i+1][3]
    
    print('P = ' + str(P) + ', T = ' + str(T) + ', xH2 = ' + str(xH2) + ', phi = ' + str(phi))

    axial_velocity = u0
    Kprev = 0
    Sd_05prev = 0
    thermdiff_05prev = 0
    ext_sol = False

    if (max_burn_vel >= u0):
        axial_velocity = max_burn_vel/1.75
        #why /1.75
    # Reset u0 if the hydrogen content increases across entries in the conditions file. Can change to look at e.g. T instead
    if (cond[i+1][2] > cond[i][2]):
        axial_velocity = u0

    while True:
        # Iterate through velocities to find extinction condition
        # Doubles u0 continously until finding an extinct solution
        # Then, takes the average between highest burning velocity and lowest extinction velocity computed until within error
        
        print("Axial Velocity = {0:1f}".format(axial_velocity))
        gas = ct.Solution(mechfile)

        # Check whether methane is in the system and set equivalence ratio accordingly
        gas.set_equivalence_ratio(phi, {'H2':1.0}, {'O2':1.0, 'N2':3.76})
        for j in range(len(gas.species())):
            if (str(gas.species(j)) == '<Species CH4>'):
                gas.set_equivalence_ratio(phi, {'CH4':1-xH2, 'H2':xH2}, {'O2':1.0, 'N2':3.76})
            
        gas.TP = T, P
        massFlux = gas.density * axial_velocity # kg/m2/s
        
        if manualgrid == True:
            grid = np.linspace(0, width, numgrid)
            oppFlame = ct.CounterflowTwinPremixedFlame(gas, grid=grid)
        else:
            oppFlame = ct.CounterflowTwinPremixedFlame(gas, width=width)
        # 'Multi' for Multi-component formulation, 'Mix' for mixture-averaged formulations
        #oppFlame.transport_model = 'multicomponent'
        oppFlame.transport_model = 'Multi'

        if (useprevsol == True):
            # Load saved solution if it exists
            if (os.path.exists(outputdir + "tempsave.xml")):
                oppFlame.restore(outputdir+"tempsave.xml","1",loglevel=loglevel)

        # Run the solver (and check for errors e.g. unsolvable)
        try:
            solveOpposedFlame(oppFlame, massFlux, loglevel, ratio, slope, curve, prune)
            Tmax = np.max(oppFlame.T)
        except:
            # Assume it means extinct
            Tmax = T

        # Branching depending on whether solution is extinct or not
        if (Tmax > T*1.1):
            # Flame is burning

            if (useprevsol == True):
                # Save the solution to use as startprofile
                if (os.path.exists(outputdir+"tempsave.xml")):
                    os.system("rm " + outputdir + "tempsave.xml")
                os.system("touch " + outputdir + "tempsave.xml")
                oppFlame.save(outputdir + "tempsave.xml", "1", "", loglevel=loglevel)
            
            # Compute the strain rate, just before the flame. This is not necessarily
            # the maximum We use the max. strain rate just upstream of the pre-heat zone
            # as this is the strain rate that computations comprare against, like when
            # plotting Su vs. K, as well as the flame speed and thermal diffusivity
            K, Sd_05, thermdiff_05 = computeFlameProperties(oppFlame)

            Kprev = K
            Sd_05prev = Sd_05
            thermdiff_05prev = thermdiff_05

            # Save flame structure
            filename = outputdir + 'PMCF' + '_T_' + str(int(round(T))) + '_P_' + str(int(round(P))) + '_K_' + str(int(round(K)))
            #oppFlame.write_csv(filename, species='Y')
            oppFlame.save(filename+".csv",basis="mole", overwrite=True)
                
            max_burn_vel = axial_velocity

            if ext_sol == False:
            # Double velocity if no extinct flame has been computer yet
                axial_velocity = axial_velocity*2.0

            else:
            # Average velocity between highest velocity burning and lowest velocity extinct solutions
                axial_velocity = (max_burn_vel + min_ext_vel)/2.0

                if ((min_ext_vel - max_burn_vel <= u_del*max_burn_vel)):
                    # Save extinction parameters and end loop for this condition when within error
                    data=np.zeros((1,7))
                    data[0,0]=P
                    data[0,1]=T
                    data[0,2]=phi
                    data[0,3]=xH2
                    data[0,4]=Kprev
                    data[0,5]=Sd_05prev
                    data[0,6]=thermdiff_05prev
                
                    file = open(outputdir+'extinction', 'ab')
                    savetxt(file, data, delimiter='\t')
                    file.close()

                    if (useprevsol == True):
                        os.system("rm " + outputdir + "tempsave.xml")
                    
                    break
                
        else:
            # Flame is extinct
            
            ext_sol = True

            min_ext_vel = axial_velocity
            axial_velocity = (max_burn_vel + min_ext_vel)/2.0

            if ((min_ext_vel - max_burn_vel <= u_del*max_burn_vel)):
                # Save extinction parameters and end loop for this condition when within error
                data=np.zeros((1,7))
                data[0,0]=P
                data[0,1]=T
                data[0,2]=phi
                data[0,3]=xH2
                data[0,4]=Kprev
                data[0,5]=Sd_05prev
                data[0,6]=thermdiff_05prev

                file = open(outputdir+'extinction', 'ab')
                savetxt(file, data, delimiter='\t')
                file.close()

                if (useprevsol == True):
                    os.system("rm " + outputdir + "tempsave.xml")
                        
                break
