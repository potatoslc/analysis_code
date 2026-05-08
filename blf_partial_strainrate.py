### Critically Strained Boundary Layer Flashback Code
### Modified: save flame profiles at selected fractions of extinction characteristic strain rate

import cantera as ct
import numpy as np
import os
from numpy import savetxt

##### Input Parameters #####

### Stretch Properties
u0 = 2.0          # m/s
u_del = 0.01          # velocity tolerance near extinction
width = 0.005         # half width of domain

manualgrid = True
numgrid = 250

### Numerics
loglevel = 0
ratio = 3
slope = 0.1
curve = 0.2
prune = 0.03

### Files
mechfile = 'H2_Li.yaml'
outputdir = 'output/'
condfile = 'conditions'
condusecel = 0
useprevsol = False

### Target strain ratios relative to extinction characteristic strain
### Example: 0.10 means 10% of K_ext
target_strain_ratios = [1]

### Target search settings
target_K_tol = 0.03
target_max_iter = 50

############################


def derivative(x, y):
    dydx = np.zeros(y.shape, y.dtype.type)
    dx = np.diff(x)
    dy = np.diff(y)

    dydx[0:-1] = dy / dx
    dydx[-1] = (y[-1] - y[-2]) / (x[-1] - x[-2])

    return dydx


def get_velocity(oppFlame):
    if hasattr(oppFlame, "velocity"):
        return oppFlame.velocity
    elif hasattr(oppFlame, "u"):
        return oppFlame.u
    else:
        return oppFlame.Uo


def computeFlameProperties(oppFlame):
    dTdx = derivative(oppFlame.grid, oppFlame.T)
    FlameFrontLoc = dTdx.argmax()

    if FlameFrontLoc == 0:
        FlameFrontLoc = 1

    U = get_velocity(oppFlame)
    strainRates = derivative(oppFlame.grid, U)

    maxStrLocation = abs(strainRates[:FlameFrontLoc]).argmax()

    if maxStrLocation == 0:
        maxStrLocation = 1

    minVelocityPoint = U[:maxStrLocation].argmin()

    if minVelocityPoint == 0:
        minVelocityPoint = 1

    strainRatePoint = abs(strainRates[:minVelocityPoint]).argmax()
    K = abs(strainRates[strainRatePoint])

    Sd_05 = U[FlameFrontLoc]

    thermdiff_05 = (
        oppFlame.thermal_conductivity[FlameFrontLoc]
        / oppFlame.cp_mass[FlameFrontLoc]
        / oppFlame.density[FlameFrontLoc]
    )

    return K, Sd_05, thermdiff_05


def make_gas(P, T, xH2, phi):
    gas = ct.Solution(mechfile)

    gas.set_equivalence_ratio(
        phi,
        {'H2': 1.0},
        {'O2': 1.0, 'N2': 3.76}
    )

    for j in range(len(gas.species())):
        if str(gas.species(j)) == '<Species CH4>':
            gas.set_equivalence_ratio(
                phi,
                {'CH4': 1 - xH2, 'H2': xH2},
                {'O2': 1.0, 'N2': 3.76}
            )

    gas.TP = T, P

    return gas


def make_flame(gas):
    if manualgrid:
        grid = np.linspace(0, width, numgrid)
        oppFlame = ct.CounterflowTwinPremixedFlame(gas, grid=grid)
    else:
        oppFlame = ct.CounterflowTwinPremixedFlame(gas, width=width)

    oppFlame.transport_model = 'Multi'

    return oppFlame


def solveOpposedFlame(oppFlame, massFlux, loglevel, ratio, slope, curve, prune):
    oppFlame.reactants.mdot = massFlux
    oppFlame.set_refine_criteria(ratio=ratio, slope=slope, curve=curve, prune=prune)
    oppFlame.solve(loglevel, auto=True)


def solve_flame_at_velocity(P, T, xH2, phi, axial_velocity):
    gas = make_gas(P, T, xH2, phi)
    massFlux = gas.density * axial_velocity
    oppFlame = make_flame(gas)

    try:
        solveOpposedFlame(oppFlame, massFlux, loglevel, ratio, slope, curve, prune)
        Tmax = np.max(oppFlame.T)
    except:
        Tmax = T

    return oppFlame, Tmax


def save_extinction_data(P, T, phi, xH2, Kext, Sd_05ext, thermdiff_05ext):
    data = np.zeros((1, 7))

    data[0, 0] = P
    data[0, 1] = T
    data[0, 2] = phi
    data[0, 3] = xH2
    data[0, 4] = Kext
    data[0, 5] = Sd_05ext
    data[0, 6] = thermdiff_05ext

    file = open(outputdir + 'extinction', 'ab')
    savetxt(file, data, delimiter='\t')
    file.close()





def save_target_strain_flames(P, T, xH2, phi, Kext, max_burn_vel):
    for target_strain_ratio in target_strain_ratios:
        target_K = target_strain_ratio * Kext

        print('\n=================================')
        print('Searching target strain flame')
        print('Kext =', Kext)
        print('Target ratio =', target_strain_ratio)
        print('Target K =', target_K)
        print('=================================\n')

        target_velocity_low = 1.0e-6
        target_velocity_high = max_burn_vel

        best_oppFlame = None
        best_K = None
        best_velocity = None
        best_error = 1.0e99

        for it in range(target_max_iter):
            target_velocity = 0.5 * (target_velocity_low + target_velocity_high)

            print('Target search iter =', it)
            print('Target velocity =', target_velocity)

            oppFlame2, Tmax2 = solve_flame_at_velocity(P, T, xH2, phi, target_velocity)

            if Tmax2 <= T * 1.1:
                target_velocity_high = target_velocity
                continue

            K2, Sd2, therm2 = computeFlameProperties(oppFlame2)
            rel_err = abs(K2 - target_K) / target_K

            print('Current K =', K2)
            print('Relative Error =', rel_err)

            if rel_err < best_error:
                best_error = rel_err
                best_oppFlame = oppFlame2
                best_K = K2
                best_velocity = target_velocity

            if rel_err < target_K_tol:
                break

            if K2 > target_K:
                target_velocity_high = target_velocity
            else:
                target_velocity_low = target_velocity

        if best_oppFlame is None:
            print('WARNING: target strain flame was not found.')
            continue

        filename_target = (
            outputdir
            + 'target_strain_'
            + str(int(target_strain_ratio * 100))
            + 'pct'
            + '_T_'
            + str(int(round(T)))
            + '_P_'
            + str(int(round(P)))
            + '_phi_'
            + str(round(phi, 3))
            + '_xH2_'
            + str(round(xH2, 3))
        )

        best_oppFlame.save(filename_target + '.csv', basis='mass', overwrite=True)



        print('\nFOUND/SAVED TARGET STRAIN FLAME')
        print('Saved:', filename_target + '.csv')
        print('Target K =', target_K)
        print('Actual K =', best_K)
        print('Best velocity =', best_velocity)
        print('Best relative error =', best_error)
        print('=================================\n')


### Begin Program ###

if loglevel == 0:
    ct.suppress_thermo_warnings()

cond = np.genfromtxt(condfile, delimiter=',')

if not os.path.exists(outputdir):
    os.makedirs(outputdir)

if not os.path.exists(outputdir + 'extinction'):
    file = open(outputdir + 'extinction', 'a')
    header = 'P [Pa]\tT [K]\tphi\txH2\tK [1/s]\tSd_05 [m/s]\tthermdiff_05 [m2/s]\n'
    file.write(header)
    file.close()

max_burn_vel = 0
min_ext_vel = 0

for i in range(len(cond) - 1):
    P = cond[i + 1][0] * 101325
    T = cond[i + 1][1] + 273.15 * condusecel
    xH2 = cond[i + 1][2]
    phi = cond[i + 1][3]

    print('P = ' + str(P) + ', T = ' + str(T) + ', xH2 = ' + str(xH2) + ', phi = ' + str(phi))

    axial_velocity = u0
    Kprev = 0
    Sd_05prev = 0
    thermdiff_05prev = 0
    ext_sol = False

    if max_burn_vel >= u0:
        axial_velocity = max_burn_vel / 1.75

    if cond[i + 1][2] > cond[i][2]:
        axial_velocity = u0

    while True:
        print("Axial Velocity = {0:1f}".format(axial_velocity))

        gas = make_gas(P, T, xH2, phi)
        massFlux = gas.density * axial_velocity
        oppFlame = make_flame(gas)

        if useprevsol:
            if os.path.exists(outputdir + "tempsave.xml"):
                oppFlame.restore(outputdir + "tempsave.xml", "1", loglevel=loglevel)

        try:
            solveOpposedFlame(oppFlame, massFlux, loglevel, ratio, slope, curve, prune)
            Tmax = np.max(oppFlame.T)
        except:
            Tmax = T

        if Tmax > T * 1.1:
            if useprevsol:
                if os.path.exists(outputdir + "tempsave.xml"):
                    os.system("rm " + outputdir + "tempsave.xml")
                os.system("touch " + outputdir + "tempsave.xml")
                oppFlame.save(outputdir + "tempsave.xml", "1", "", loglevel=loglevel)

            K, Sd_05, thermdiff_05 = computeFlameProperties(oppFlame)

            Kprev = K
            Sd_05prev = Sd_05
            thermdiff_05prev = thermdiff_05



            max_burn_vel = axial_velocity

            if ext_sol == False:
                axial_velocity = axial_velocity * 2.0
            else:
                axial_velocity = 0.5 * (max_burn_vel + min_ext_vel)

                if min_ext_vel - max_burn_vel <= u_del * max_burn_vel:
                    Kext = Kprev

                    save_extinction_data(
                        P=P,
                        T=T,
                        phi=phi,
                        xH2=xH2,
                        Kext=Kprev,
                        Sd_05ext=Sd_05prev,
                        thermdiff_05ext=thermdiff_05prev
                    )

                    save_target_strain_flames(
                        P=P,
                        T=T,
                        xH2=xH2,
                        phi=phi,
                        Kext=Kext,
                        max_burn_vel=max_burn_vel
                    )

                    if useprevsol:
                        os.system("rm " + outputdir + "tempsave.xml")

                    break

        else:
            ext_sol = True
            min_ext_vel = axial_velocity
            axial_velocity = 0.5 * (max_burn_vel + min_ext_vel)

            if min_ext_vel - max_burn_vel <= u_del * max_burn_vel:
                Kext = Kprev

                save_extinction_data(
                    P=P,
                    T=T,
                    phi=phi,
                    xH2=xH2,
                    Kext=Kprev,
                    Sd_05ext=Sd_05prev,
                    thermdiff_05ext=thermdiff_05prev
                )

                save_target_strain_flames(
                    P=P,
                    T=T,
                    xH2=xH2,
                    phi=phi,
                    Kext=Kext,
                    max_burn_vel=max_burn_vel
                )

                if useprevsol:
                    os.system("rm " + outputdir + "tempsave.xml")

                break