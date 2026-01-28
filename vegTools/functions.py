import numpy as np
import pandas as pd
import fluidfoam
from fluidfoam import OpenFoamSimu
import os
import math


"""
Get average profiles
"""

def get_profiles(list_var, z, V):
    """
    Compute volume-weighted mean profiles along the z-axis.

    Inputs:
    - list_var: list of arrays to compute the mean (e.g., velocity components)
    - z: array of z-coordinates of each cell with readmesh of fluidfoam
    - V: array of cell volumes with writeCellVolumes of openfoam

    Outputs:
    - unique_z: unique z values defining slices
    - mean_values: list of mean values for each variable per slice
    """
    df = pd.DataFrame({'z': np.round(z, 6), 'V': V})
    for i, var in enumerate(list_var):
        df[f'var_{i}'] = var * V

    df_sum = df.groupby('z', sort=True).sum().reset_index()
    unique_z = df_sum['z'].values
    V_tot_per_z = df_sum['V'].values  
    
    mean_values = []
    for i in range(len(list_var)) : 
        mean_values.append(df_sum[f'var_{i}'].values / V_tot_per_z )
    
    return mean_values

def get_viscous_stress(z ,gradU,V, rho = None , nu = None) : 
    """
    Compute volume-weighted mean profiles of viscous stress.

    Inputs:
    - z: array of z-coordinates of each cell with readmesh of fluidfoam
    - grad_U : velocity gradient of averged over time velocity (postProcess -func 'grad(UMean)')
    - V: array of cell volumes with writeCellVolumes of openfoam

    Outputs:
    - unique_z: unique z values defining slices
    - viscous stress in [Pa]
    """

    if (rho is None) & (nu is None) : 
        rho = 1000 #[kg/m3]
        nu = 1e-6 #[m2/s]

    #DataFrame Creation 
    df = pd.DataFrame({'z': np.round(z, 6),'grad_U' : gradU * V, 'V': V})
    
    df_sum = df.groupby('z', sort=True).sum().reset_index()
    unique_z = df_sum['z'].values
    V_tot_per_z = df_sum['V'].values

    gradU_mean = df_sum['grad_U'] / V_tot_per_z

    viscous_stress = np.array(rho * nu * gradU_mean)
    return unique_z , viscous_stress

def get_primeprime(var, z , mean_per_slice):
    """
    This function computes the spatial fluctuations (var') 
    of a variable without sorting along the z-axis.
    It also preserves the original shape of the OpenFoam fields. 
    The function get_profiles can then still be used.

    Inputs:
    - var: 1D array containing the variable from which to compute the primeprime values.
    - z: 1D array of z-coordinates of each cell with readmesh of fluidfoam
    - mean_per_slice: 1D array containing the average value for each slice (computed using get_profiles function).

    Output:
    - var_primeprime: 1D array with the spatial variation of the variable var (var'').
    Same shape as list_var
    """
    # Find unique_z which is sorted thanks to the function np.unique
    unique_z = np.unique(np.round(z,6))

    
    ### For each cell in the mesh, find its position in the mesh closer
    ### from the values in unique_z array
    # ------------------------------------------------#
    #Version which takes too much time for large meshes 
    # indices = np.zeros(len(z)) 
    # for cell in range(len(z)): 
    #     idz = np.argmin(np.abs(unique_z - z[cell])) 
    #     indices[cell] = idz
    # ------------------------------------------------#


    # ------------------------------------------------#
    #Faster version to find indices 
    #indices contains array from 0 to len(unique_z)
    indices = np.searchsorted(unique_z, z, side='right') - 1
    indices = np.clip(indices, 0, len(unique_z) - 1) 
    # ------------------------------------------------#

    # Subtraction of the mean corresponding to the slice
    var_primeprime = var - mean_per_slice[indices]

    return var_primeprime

def get_mean_profiles_x(list_var, x, V, nb_inter):
    """
    Compute volume-weighted mean profiles along the x-axis using fixed intervals.

    Inputs:
    - list_var: list of arrays to average (e.g., pressure, TKE)
    - x: array of x-coordinates (from mesh)
    - V: array of cell volumes
    - nb_inter: number of intervals along the x-axis

    Outputs:
    - x_centers: center of each interval
    - mean_values: list of mean profiles (1D arrays, one per input variable)
    """
    x_min, x_max = np.min(x), np.max(x)
    bins = np.linspace(x_min, x_max, nb_inter + 1)
    x_centers = (1/2) * (bins[:-1] + bins[1:])
    df = pd.DataFrame({'x': x, 'V': V})
    for i, var in enumerate(list_var):
        df[f'var_{i}'] = var * V

    df['interval'] = pd.cut(df['x'], bins=bins, labels=False)

    df_sum = df.groupby('interval').sum().reset_index()

    V_tot_per_x = df_sum['V'].values

    mean_values = []
    for i in range(len(list_var)):
        mean_i = df_sum[f'var_{i}'].values / V_tot_per_x
        mean_values.append(mean_i)

    return x_centers, mean_values

def get_mean_profiles_x_structured(list_var, x, V):
    """
    Compute volume-weighted mean profiles along the x-axis.

    Inputs:
    - list_var: list of arrays to compute the mean (e.g., velocity components)
    - x: array of z-coordinates of each cell with readmesh of fluidfoam
    - V: array of cell volumes with writeCellVolumes of openfoam

    Outputs:
    - unique_x: unique x values defining slices
    - mean_values: list of mean values for each variable per slice
    """
    df = pd.DataFrame({'x': np.round(x, 6), 'V': V})
    for i, var in enumerate(list_var):
        df[f'var_{i}'] = var * V

    df_sum = df.groupby('x', sort=True).sum().reset_index()
    unique_x = df_sum['x'].values
    V_tot_per_x = df_sum['V'].values  
    
    mean_values = []
    for i in range(len(list_var)) : 
        mean_values.append(df_sum[f'var_{i}'].values / V_tot_per_x )
    
    return unique_x, mean_values

def get_fdz(simu,phi, mode = 'total', D = 0.01,num_cylinder = 2, rho = 1000) : 
    """
    Given a class simu, this function computes profiles of 
    drag force against cylinder(s) 
    Inputs : 
        - simu  = (OpenFoamSimu class)
        - phi = density of vegetation
        - (optional) mode : 
                - total = Fdp + Fdv
                - viscous = Fdv
                - form = Fdp
        - (optional) D = stem diameter
        - (optional) num_cylinder = number of cylinders in the mesh
        - (optional) rho = density of fluid
    Outputs : 
        - fdz = profile of drag force
    """
    df_forces_total = []  #Forces for each cylinder
    for k in range(num_cylinder): 
        # Read mesh at cylinder
        zc = getattr(simu,f'z_cylinder_{k+1}')

        #Read forces on each cylinder
        forceForm_name = f'forceFormCylbar_cylinder_{k+1}0'  # Force on x axis
        forceVis_name = f'forceVisCylbar_cylinder_{k+1}0'  # Force on x axis
        
        if mode == "total" : 
            forcex_value = getattr(simu, forceForm_name) + getattr(simu, forceVis_name)  # force [N]
        elif mode == "viscous" : 
            print(f"Viscous Drag only cylinder{k+1}")
            forcex_value = getattr(simu, forceVis_name)  # force [N]
        elif mode == "form" : 
            print(f"Form Drag only cylinder{k+1}")
            forcex_value = getattr(simu, forceForm_name)  # force [N]
        
        # Compute magnitude of force 
        df_force = pd.DataFrame({'zc': np.round(zc, 6), 'Fdx':  forcex_value})
        df_forces_total.append(df_force) 


    # Concantenate all cylinders dataFrames
    df_force_total = pd.concat(df_forces_total)

    # Sum forces on each cylinders
    df_force_slice = df_force_total.groupby('zc').sum().reset_index()

    simu.dz_slice = get_dz_slice(simu.z, simu.V)
    #frontal_area = simu.dz_slice * D * num_cylinder  #Frontal area for all cylinders, assuming that mesh is structured on z axis
    #Compute Drag coefficient
    simu.Ubar_mean = get_profiles([simu.Ubar0],simu.z , simu.V)[0]
    Cd = df_force_slice['Fdx'] / (num_cylinder * 0.5 * rho * simu.Ubar_mean**2 * D * simu.dz_slice)
    fd_z = np.array((  df_force_slice['Fdx'] * 2 * phi   ) / (rho * np.pi * simu.dz_slice * D**2))  #[ m²/s²]

    return fd_z

def get_Cd(simu,phi, mode = 'total', D = 0.01,num_cylinder = 2, rho = 1000) : 
    """
    Given a class simu, this function computes profiles of 
    drag coefficient against cylinder(s) 
    Inputs : 
        - simu  = (OpenFoamSimu class)
        - phi = density of vegetation
        - (optional) mode : 
                - total = Fdp + Fdv
                - viscous = Fdv
                - form = Fdp
        - (optional) D = stem diameter
        - (optional) num_cylinder = number of cylinders in the mesh
        - (optional) rho = density of fluid
    Outputs : 
        - Cd = profile of drag coefficient
    """
    df_forces_total = []  #Forces for each cylinder
    for k in range(num_cylinder): 
        # Read mesh at cylinder
        zc = getattr(simu,f'z_cylinder_{k+1}')

        #Read forces on each cylinder
        forceForm_name = f'forceFormCylbar_cylinder_{k+1}0'  # Force on x axis
        forceVis_name = f'forceVisCylbar_cylinder_{k+1}0'  # Force on x axis
        
        if mode == "total" : 
            forcex_value = getattr(simu, forceForm_name) + getattr(simu, forceVis_name)  # force [N]
        elif mode == "viscous" : 
            print(f"Viscous Drag only cylinder{k+1}")
            forcex_value = getattr(simu, forceVis_name)  # force [N]
        elif mode == "form" : 
            print(f"Form Drag only cylinder{k+1}")
            forcex_value = getattr(simu, forceForm_name)  # force [N]
        
        # Compute magnitude of force 
        df_force = pd.DataFrame({'zc': np.round(zc, 6), 'Fdx':  forcex_value})
        df_forces_total.append(df_force) 


    # Concantenate all cylinders dataFrames
    df_force_total = pd.concat(df_forces_total)

    # Sum forces on each cylinders
    df_force_slice = df_force_total.groupby('zc').sum().reset_index()

    simu.dz_slice = get_dz_slice(simu.z, simu.V)
    #Compute Drag coefficient
    simu.Ubar_mean = get_profiles([simu.Ubar0],simu.z , simu.V)[0]
    Cd = df_force_slice['Fdx'] / (num_cylinder * 0.5 * rho * simu.Ubar_mean**2 * D * simu.dz_slice)

    return np.array(Cd)

"""
Get geometric characteristic on the mesh 
"""
def get_dz_slice(z, V):
    """
    Compute the mean dz for each slice z.

    Inputs:
    - z: array of z-coordinates
    - V: array of cell volumes

    Outputs:
    - dz_slice: array of dz values per slice
    """

    df = pd.DataFrame({'z': np.round(z, 6), 'V': V})
    df_sum = df.groupby('z', sort=True).sum().reset_index()
    
    unique_z = df_sum['z'].values
    V_tot_per_z = df_sum['V'].values  

    # Compute the bottom cell layer thickness
    z_min = np.min(np.unique(z))
    cells_bottom = np.where(z == z_min)
    V_bottom = V[cells_bottom]
    dz_bottom = 2 * z_min 
    dS_bottom = V_bottom / dz_bottom
    S0 = np.sum(dS_bottom)

    # Compute dz per slice
    dz_slice = np.divide(V_tot_per_z,S0)
    return dz_slice

def get_nc_alongx(ratio, dr , phi , D = None) : 
    if D is None : 
        D = 0.01 #m
    
    #Domain size
    W = np.sqrt(np.pi/(2*phi))*D 
    w = W/2

    #dx mesh size in x axis (size of outter cell)
    dx = ratio * dr * 4 * w /(np.pi * D )

    #Number of cell in cylinder region 
    nc = np.round(w/dx)

    #Return twice nc to get number of cells in the whole domain
    return int(2*nc)


"""
Get informations on patch bottom 
"""
def get_u_star(Tau,z,V,dz,mode,rho = None , nu = None) : 
    """
    Compute u* for differents methods of calculation on the bottom patch 

    Inputs:
    - Tau : bottom stress (e.g WallShearStress from OF , 
       or viscous stress computed in post Process in python )
    - z: array of z-coordinates
    - V: array of cell volumes
    - dz : Fisrt dz from bottom 
    - mode : 
        - viscous : to compute viscous stress only 
        - total1 : to compute fisrt <Tau> then <u*>
        - total2 : to compute directly <sqrt(Tau/rho)> == <u*>

    Outputs:
    - u_star : friction velocity [m/s]
    """
    if (rho is None) & (nu is None) : 
        rho = 1000 #[kg/m3]
        nu = 1e-6 #[m2/s]
        #Get stress at the bottom taking only viscous stress contribution
    
    if mode not in ["viscous", "total1", "total2"]:
        raise ValueError(f'mode must be : viscous , total1 or total2')
    
    if mode == 'viscous' : 
        Taub = Tau
        u_star = np.sqrt(np.abs(Taub) / rho )

    if mode == 'total1' :       
        #Get cells at bottom 
        z_min = np.min(np.unique(z))
        cells_bottom = np.where(z == z_min)[0]
        V_bottom = V[cells_bottom]
        dz_bottom = 2 * z_min 
        dS_bottom = V_bottom / dz_bottom
        S0 = np.sum(dS_bottom)


        df = pd.DataFrame({'Tau' : Tau * dS_bottom})
        df_sum = df.sum()

        Taub = df_sum['Tau'] / S0
        u_star = np.sqrt(np.abs(Taub) / rho )


    if mode == 'total2' :       
        #Get cells at bottom 
        z_min = np.min(np.unique(z))
        cells_bottom = np.where(z == z_min)[0]
        V_bottom = V[cells_bottom]
        dz_bottom = 2 * z_min 
        dS_bottom = V_bottom / dz_bottom
        S0 = np.sum(dS_bottom)

        df = pd.DataFrame({'u_star' : np.sqrt(np.abs(Tau)/rho) * dS_bottom})
        df_sum = df.sum()

        u_star = df_sum['u_star'] / S0

    return u_star

def average_bottom(list_var ,z , V , dz) : 
    """
    Compute volume-weighted mean value of a scalar field on the bottom patch.

    Inputs:
    - list_var: list of arrays to compute the mean (e.g., velocity components)
    - V: array of cell volumes with writeCellVolumes of openfoam
    - dz : Fisrt dz from bottom 

    Outputs:
    - mean_values: list of mean values for each variable on the bottom patch
    """

    #Get cells at bottom 
    z_min = np.min(np.unique(z))
    cells_bottom = np.where(z == z_min)
    V_bottom = V[cells_bottom]
    dz_bottom = 2 * z_min 
    dS_bottom = V_bottom / dz_bottom
    S0 = np.sum(dS_bottom)

    #DataFrame creation
    df = pd.DataFrame({'dS' : dS_bottom})
    for i, var in enumerate(list_var):
        df[f'var_{i}'] = var[cells_bottom] * df['dS'].values

    df_sum = df.sum()
    S0 = df_sum['dS']  
    
    mean_values = []
    for i in range(len(list_var)) : 
        mean_values.append(df_sum[f'var_{i}'] / S0 )
    
    return mean_values


"""
Extract informations from Description.ods
"""

def read_description(PATH_description,cases) : 
    df_description = pd.read_excel(PATH_description +  "/Description.ods")
    
    #Get run name from cases 
    run_name = os.path.basename(cases)
    mask = df_description["name"] == run_name
    df_selected = df_description[mask]

    # Si aucune ligne trouvée
    if df_selected.empty:
        raise ValueError(f"Aucune entrée trouvée pour '{run_name}' dans Description.ods")

    # On récupère la ligne sous forme de dict clé = nom de la colonne
    info = df_selected.iloc[0].to_dict()

    # Remplacer NaN par chaîne vide pour les valeurs non string
    for k, v in info.items():
        if isinstance(v, float) and math.isnan(v):
            info[k] = ""

     # Optionnel : retire les champs vides
    info = {k:v for k,v in info.items() if v != ""}

    # Si title n'existe pas dans les colonnes, on le crée vide
    if "title" not in info:
        info["title"] = ""
    
    return info

"""
Import functions from veg_relations
"""

def dragcoef_etminan(phiv, Rep):
    # return the drag coefficient according to Etminan et al. (2017)
    coef = (1-phiv)/(1-(2*phiv/np.pi)**(0.5))
    Rec = Rep*coef
    Cdc = 1+10*Rec**(-2/3)
    Cdp = coef**2*Cdc

    return Cdp, Cdc

def dragcoef_tanino(phiv):
    # return the drag coefficient according to Tanino and Nepf (2008)
    Cd = 2*(0.46+3.8*phiv)
    return Cd

def dragcoef_tinoco(phiv, Rep):
    # return the drag coefficient according to Tanino and Nepf (2008)
    Cd = 2*(0.58+6.5*phiv)
    return Cd

def TKETanino(phiv, Rep, dv, nu=1e-6, drag_law='Etminan'):
    '''
    Compute the theoretical TKE in vegetation array following Tanino and Nepf (2008)
    '''

    #Compute empirical coefficients which values depend on phiv
    try: #phiv is an array with more than 1 value
        delta = 1.21*np.ones(len(phiv))
        lt = dv*np.ones(len(phiv))
        I = np.where(phiv > np.pi/(4*2.79**2))
        delta[I] = 0.77

        sn = ((np.pi/(4*phiv))**(0.5)-1)*dv
        lt[I] = sn[I]
    except TypeError: #phiv is a float
        if phiv <= np.pi/(4*2.79**2):
            delta = 1.21
            lt = dv
        else:
            delta = 0.77
            lt = ((np.pi/(4*phiv))**(0.5)-1)*dv

    if drag_law == 'Etminan':
        #Compute drag coefficient following Etminan et al. (2017)
        Cdp, Cdc = dragcoef_etminan(phiv, Rep)
        Cd = Cdp
    elif drag_law == 'Tanino':
        Cd = dragcoef_tanino(phiv)
    elif drag_law == 'Tinoco':
        Cd = dragcoef_tinoco(phiv)
    #Theoretical turbulent kinetic energy
    kth = delta*(lt/dv*phiv/((1-phiv)*np.pi/2)*Cd)**(2/3)*(Rep*nu/dv)**2

    return kth

def ustar_condefrias(phiv, Rep, dv, nu=1e-6, drag_law='Etminan', C=9.5, Cf=0.0025):
    '''
    Conde-Frias et al. (2023) law for bed friction velocity ustar
    '''

    kth = TKETanino(phiv, Rep, dv, nu, drag_law)

    ustar = np.maximum(C*(kth/Rep)**(0.5), Cf**(0.5)*Rep*nu/dv)
    return ustar

def ustar_etminan(phiv, Rep, dv, nu=1e-6, drag_law='Etminan', C=5.15):
    '''
    Etminan et al. (2018) law for bed friction velocity ustar
    '''

    #Theoretical turbulent kinetic energy according to Tanino and Nepf (2008) 
    kth = TKETanino(phiv, Rep, dv, nu, drag_law)
    #Compute drag coefficient
    if drag_law == 'Etminan':
        #Compute drag coefficient following Etminan et al. (2017)
        Cdp, Cdc = dragcoef_etminan(phiv, Rep)
    elif drag_law == 'Tanino':
        Cd = dragcoef_tanino(phiv)
        Cdc = Cd
    #Constriction Velocity
    coef = (1-phiv)/(1-(2*phiv/np.pi)**(0.5))
    Rec = Rep*coef
    Uc = Rec*nu/dv
    Up = Rep*nu/dv
    #frontal area
    a = phiv/(np.pi/4*dv)
    #Height of the boundary layer
    Hv = C*np.sqrt(nu*kth/(Cdc*a*Uc**3))
    ReHv = Up*Hv/nu
    #friction velocity
    ustar = np.sqrt(2/ReHv)*Up
    return ustar

def ustar_yang(phiv, Rep, dv, nu=1e-6, Cf=0.0025):
    '''
    Yand and Nepf (2015) law for bed friction velocity ustar
    '''

    #Bulk Velocity
    Up = Rep*nu/dv
    Ub = Up/(1-phiv)
    #friction velocity
    ustar = np.maximum(2*np.sqrt(nu*Up/dv), Cf**(0.5)*Up)
    return ustar

def get_gradP_veg(phiv, Rep, dv, nu=1e-6, rhof=1000, drag_law='Etminan'):

    if drag_law == 'Etminan':
        #Compute drag coefficient following Etminan et al. (2017)
        Cdp, Cdc = dragcoef_etminan(phiv, Rep)
        Cd = Cdp
    elif drag_law == 'Tanino':
        Cd = dragcoef_tanino(phiv)
    elif drag_law == 'Tinoco':
        Cd = dragcoef_tinoco(phiv)

    #Constriction Velocity
    coef = (1-phiv)/(1-(2*phiv/np.pi)**(0.5))
    Rec = Rep*coef
    Uc = Rec*nu/dv
    Up = Rep*nu/dv
    
    Kv = rhof*2/np.pi*Cd/((1-phiv)*dv)*Up
    gradP = phiv*Kv*Up

    return gradP
