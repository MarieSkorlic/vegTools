import numpy as np
import pandas as pd
import fluidfoam
from fluidfoam import OpenFoamSimu
import os
import math


"""
Get average profiles
"""
def get_mean_profiles(list_var, z, V, Cx, Cy):
    """
    Compute volume-weighted mean profiles along the z-axis.

    Inputs:
    - list_var: list of arrays to compute the mean (e.g., velocity components)
    - z: array of z-coordinates of each cell with readmesh of fluidfoam
    - V: array of cell volumes with writeCellVolumes of openfoam
    - Cx, Cy: coordinates with writeCellCentres of openfoam 

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
    
    return unique_z, mean_values

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

def get_disp_stress(z ,UU,Ui,Uj,V,Cx,Cy, rho = None) : 
    """
    Compute volume-weighted mean profiles of dispersive stress.

    Inputs:
    - z: array of z-coordinates of each cell with readmesh of fluidfoam
    - Ui,Uj : velocity used in the calculation, should be averaged in time 
    - V: array of cell volumes with writeCellVolumes of openfoam

    Outputs:
    - unique_z: unique z values defining slices
    - dispersive stress in [Pa]
    """

    if rho is None : 
        rho = 1000 #[kg/m3]
    
    unique_z , mean_values = get_mean_profiles([UU , Ui , Uj],z, V, Cx, Cy)
    disp_stress =  - rho * ( mean_values[0] - mean_values[1] * mean_values[2])
    return unique_z , disp_stress

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

def get_reynolds_stress(z ,R ,V, rho = None ) : 
    """
    Compute volume-weighted mean profiles of Reynolds stress.
    Reynolds Stress from RANS profiles computed with turbulent viscosity

    Inputs:
    - z: array of z-coordinates of each cell with readmesh of fluidfoam
    - R : Reynolds Stress given by openfoam : should be averaged in time 
    - V: array of cell volumes with writeCellVolumes of openfoam

    Outputs:
    - unique_z: unique z values defining slices
    - Reynolds stress in [Pa]
    """

    if (rho is None) : 
        rho = 1000 #[kg/m3]

    #DataFrame Creation 
    df = pd.DataFrame({'z': np.round(z, 6),'R' : R*V, 'V': V})
    
    df_sum = df.groupby('z', sort=True).sum().reset_index()
    unique_z = df_sum['z'].values
    V_tot_per_z = df_sum['V'].values

    R_mean = df_sum['R'] / V_tot_per_z

    reynolds_stress =   rho * np.array(R_mean)
    return unique_z , reynolds_stress

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

    #Extract data from the selected line
    name_run = df_selected["name"].values[0]
    phi = df_selected["phi"].values[0]
    Rep = df_selected["Rep"].values[0]
    ratio = df_selected["ratio"].values[0]
    dr = df_selected["dr"].values[0]
    marker = df_selected["marker"].values[0]
    color = df_selected["color"].values[0]
    title = df_selected["title"].values[0]
    if type(title)!= str : 
        if (math.isnan(title))  : 
            title = ' '

    return title , phi , Rep , ratio , dr , marker , color




