"""Class to load post processed data from a simulation
=====================================================================

.. autoclass:: OpenFoamSimu

.. automethod:: OpenFoamSimu.keys

.. automethod:: OpenFoamSimu.readopenfoam

"""

import os, sys
import subprocess
import numpy as np
from fluidfoam import readmesh, readfield, OpenFoamFile, readscalar , readvector , readsymmtensor
from vegTools.functions import get_profiles, get_dz_slice,get_fdz, read_description,get_primeprime
from mathtools import derivate_wall
import netCDF4 as nc
from pathlib import Path

class Error(Exception):
    pass

class DirectorySimuError(Error):
    def __init__(self, simu):
        super(DirectorySimuError,self).__init__(
                "No directory found for simulation named {}".format(simu))

class simuPostPro(object):
    """
    Class to load all data saved at timeStep of an openFoam simulation

    Args:
        path: str, reference path where simulations are stored.\n
            You may want to provide path if all your simulations are located
            inside path and subfolders of path. You can do it by modifying
            in the __init__ path='/path/to/the/simulations/'\n
        simu: str, name of the simu that has to be loaded.\n
            If simu=None, it will lists all existing simulation names in path
            and ask you to choose.\n
        timeStep: str, timeStep to load. If None, load the last time step\n
        structured: bool, true if the mesh is structured\n
        dataToLoad: list of str, list containing the name of the varaibles 
            to read and load. If None, read and load all saved variables.
    """

    


    def __init__(self,  path=None, boundary = None, simu=None, timeStep=None, structured=False,
                dataToLoad=None,deletefile = False, precision=15, order='F'):
        
        if boundary is None : 
            boundary = ['cylinder_1' , 'cylinder_2','bottom']


        if path == None and simu == None:
            # If nothing if given, consider the current directory as the 
            # simulation to load
            self.directory = os.getcwd()+'/'
            self.simu = os.getcwd().split('/')[-1]
            path = './'

        elif simu == None:
            # If only path is provided, consider all subfolders as possible
            # simulations to load
            self.directory = self._choose_simulation(path)
            self.simu = self.directory.split("/")[-2]
            path = './'

        else:
            # If path and simu are provided, consider the given directory
            # as the simulation to load
            self.simu = simu
            if path.endswith('/') is False:
                path += '/'
            self.directory = path + simu

            if self.directory.endswith('/') is False: 
                self.directory += '/'
        
        if deletefile : 
            os.system(f'rm -r {self.directory}fieldsPostPro.nc')
        
        
        """
        If file containing postProcessed fields not in self.directory,
        read simulation, compute postProcess fields and write in 
        netCDF : fieldsPostPro.nc
        """
        filenamePostPro = 'fieldsPostPro.nc'
        filenameRawData = 'fields.nc'
        if not os.path.exists(self.directory + '/' + filenamePostPro) :
            # If raw data has not been extracted through 
            if not os.path.exists(self.directory + '/' + filenameRawData) :
                #Read Mesh 
                self.readmesh(structured=structured, precision=precision,boundary = boundary,
                    order=order)
                #Read fields
                self.readopenfoam(timeStep=timeStep,boundary = boundary, structured=structured, 
                        dataToLoad=dataToLoad, precision=precision,
                        order=order)
                #Read phi and Rep from .ods file 
                self.compute_simulations_data()
                #Compute profiles
                self.compute_profiles()
                #self.compute_boundary_data(boundary)
                self.compute_profiles_stresses()
                self.compute_profiles_TKE()
                self.compute_profiles_DKE()
                #Compute macroscopic quantities
                self.compute_macroscopic_quantities()
                #Write and readNetCDF for postProcessed Data
                self.writeNetCDFPostPro(boundary = boundary)
                self.readNetCDF(filename=filenamePostPro)
            else : 
                self.readNetCDF(filename = filenameRawData)
                #Read phi and Rep from .ods file 
                self.compute_simulations_data()
                #Compute profiles
                self.compute_profiles()
                #self.compute_boundary_data(boundary)
                self.compute_profiles_stresses()
                self.compute_profiles_TKE()
                self.compute_profiles_DKE()
                #Compute macroscopic quantities
                self.compute_macroscopic_quantities()
                #Write and readNetCDF for postProcessed Data
                self.writeNetCDFPostPro(boundary = boundary)
                self.readNetCDF(filename=filenamePostPro)
        else : 
            self.readNetCDF(filename=filenamePostPro)

            

    def compute_simulations_data(self) :
        PATH_description = str(Path(self.directory).parent)
        case = os.path.basename(self.simu)
        #Read csv Description.ods to extract 'phi'
        data = read_description(PATH_description,case)
        self.phif,self.phi, self.Rep   = 1 - data['phi'], data['phi'] , data['Rep']

    def compute_profiles(self) :
        self.profiles = {}
        OF_python_names = {
                    'Ubar0': 'Ubar',
                    'Ubar1': 'Vbar',
                    'Ubar2': 'Wbar',
                    }
        
        for var in self.variables :
            field = getattr(self,var)
            name = OF_python_names.get(var,var)
            # volFields
            if len(field) == len(self.x) : 
                profile = get_profiles([field],self.z,self.V)[0]
                self.profiles[f'{name}_mean'] = profile
        
        #Compute TKE_profile (obligé car pas dans simu.variables)
        Uprime2 , Vprime2 , Wprime2 = get_profiles([self.UUbar - self.Ubar0 * self.Ubar0 , 
                                                                   self.VVbar - self.Ubar1 * self.Ubar1 , 
                                                                   self.WWbar - self.Ubar2 * self.Ubar2],
                                                                   self.z , self.V)
        #Resolved TKE
        k_resol = (1/2) * (Uprime2 + Vprime2 + Wprime2)
        self.profiles['k_resol'] = k_resol
        
        #SGS TKE
        k_SGS = get_profiles([self.turbulenceProperties_kbar] , self.z , self.V)[0]
        self.profiles['k_SGS'] = k_SGS

        #Total TKE
        k_tot = k_resol + k_SGS
        self.profiles['k_tot'] = k_tot

        # DKE
        DKE = (0.5) * (
                         get_profiles([self.Ubar0 * self.Ubar0],self.z,self.V)[0] - self.profiles['Ubar_mean'] * self.profiles['Ubar_mean']
                       + get_profiles([self.Ubar1 * self.Ubar1],self.z,self.V)[0] - self.profiles['Vbar_mean'] * self.profiles['Vbar_mean']
                       + get_profiles([self.Ubar2 * self.Ubar2],self.z,self.V)[0] - self.profiles['Wbar_mean'] * self.profiles['Wbar_mean'])
        self.profiles['DKE'] = DKE


        # Drag force
        fdz_form = get_fdz(self,self.phi,'form')
        self.profiles['fdz_form'] = fdz_form
        fdz_viscous = get_fdz(self,self.phi,'viscous')
        self.profiles['fdz_viscous'] = fdz_viscous
        fdz = get_fdz(self,self.phi,'total')
        self.profiles['fdz'] = fdz


    def compute_profiles_stresses(self , rho = 1000 , nu = 1e-6) :
        # ---- Viscous Stress [Pa]
        gradUbar_mean = get_profiles([self.grad_Ubar6],self.z,self.V)[0]
        viscous_stress = rho * gradUbar_mean * nu
        self.profiles['viscous_stress'] = viscous_stress

        # ---- Reynolds Stress [Pa]
        reynolds_stress_SGS = rho * get_profiles([- self.turbulenceProperties_Rbar2] , self.z , self.V)[0]
        self.profiles['reynolds_stress_SGS'] = reynolds_stress_SGS
        #Resolved reynolds stresses = - <bar{u'w'}> = - ( <bar{uw}> - <bar{u}> * <bar{w}>  )
        reynolds_stress_resol = - rho * get_profiles([self.UWbar - self.Ubar0 * self.Ubar2],self.z , self.V)[0]
        self.profiles['reynolds_stress_resol'] = reynolds_stress_resol
        #Total Reynolds Stress 
        reynolds_stress = + reynolds_stress_resol + reynolds_stress_SGS
        self.profiles['reynolds_stress'] = reynolds_stress
        # ---- Dispersive stress [Pa]  (- <bar{u}''bar{w}''> = -(<bar{u}bar{w}> - <bar{u}><bar{w}>) )
        Ubar_Wbar_mean, Ubar_mean, Wbar_mean = get_profiles([self.Ubar0 * self.Ubar2 , self.Ubar0 , self.Ubar2],self.z , self.V)
        disp_stress = - rho * (Ubar_Wbar_mean - Ubar_mean * Wbar_mean)
        self.profiles['disp_stress'] = disp_stress

    def compute_profiles_TKE(self , rho = 1000 , nu = 1e-6) :
        unique_z = np.unique(np.round(self.z , 6))

        # ---- Production term 
        # - Shear production
        Rxz_resol = get_profiles([self.UWbar - self.Ubar0 * self.Ubar2],self.z , self.V)[0]
        Rxz_SGS =  get_profiles([self.turbulenceProperties_Rbar2] , self.z , self.V)[0]
        Rxz_tot = Rxz_resol + Rxz_SGS

        gradUbar_mean = get_profiles([self.grad_Ubar6],self.z , self.V)[0]
        Ps = - self.phif * (Rxz_tot * gradUbar_mean)
        self.profiles['Ps'] = Ps

        # - Wake production
        Rij = [ (self.UUbar - self.Ubar0 * self.Ubar0) + self.turbulenceProperties_Rbar0 , (self.UVbar - self.Ubar0 * self.Ubar1) + self.turbulenceProperties_Rbar1 , (self.UWbar - self.Ubar0 * self.Ubar2) + self.turbulenceProperties_Rbar2 ,
                    (self.UVbar - self.Ubar0 * self.Ubar1) + self.turbulenceProperties_Rbar1 , (self.VVbar - self.Ubar1 * self.Ubar1) + self.turbulenceProperties_Rbar3  , (self.VWbar - self.Ubar1 * self.Ubar2) + self.turbulenceProperties_Rbar4,
                    (self.UWbar - self.Ubar0 * self.Ubar2) + self.turbulenceProperties_Rbar2 , (self.VWbar - self.Ubar1 * self.Ubar2) + self.turbulenceProperties_Rbar4, (self.WWbar - self.Ubar2 * self.Ubar2) + self.turbulenceProperties_Rbar5 ]

        Rijgrad = np.multiply(Rij ,
                [
                self.grad_Ubar0 , self.grad_Ubar3 , self.grad_Ubar6,
                self.grad_Ubar1 , self.grad_Ubar4 , self.grad_Ubar7,
                self.grad_Ubar2 , self.grad_Ubar5 , self.grad_Ubar8
                ])

        Rijgrad = np.sum(Rijgrad , axis = 0)
        Rijgrad = get_profiles([Rijgrad] , self.z , self.V)[0]

        Pw = -self.phif * ( Rijgrad - 
                            get_profiles([self.UWbar - self.Ubar0 * self.Ubar2],self.z, self.V)[0] 
                            * get_profiles([self.grad_Ubar6],self.z,self.V)[0]  
                            ) 
        self.profiles['Pw'] = Pw
        

        # ---- Dissipation 
        epsilonP1_bar_mean = nu * get_profiles([self.epsilonP1bar] , self.z , self.V)[0]
        epsilonP2_bar_mean = - nu * (get_profiles([self.grad_Ubar6],self.z , self.V)[0] * 
                                        get_profiles([self.grad_Ubar6] , self.z , self.V)[0])

        
        # SGS dissipation < \overline{nut * Sij * Sij} > 
        epsSGS = get_profiles([self.epsilonSGSbar], self.z , self.V)[0]
        self.profiles['epsSGS'] = epsSGS 
        # Resolved dissipation
        epsilon_resol = self.phif * (epsilonP1_bar_mean + epsilonP2_bar_mean)
        self.profiles['epsilon_resol'] = epsilon_resol
        # Total dissipation 
        epsilon_tot = self.phif * (epsilonP1_bar_mean + epsilonP2_bar_mean + epsSGS)
        self.profiles['epsilon_tot'] = epsilon_tot



        # ---- Transport term
        #Transport term with fulcutating pressure  
        Tp = - self.phif *  derivate_wall( 
                                                unique_z ,
                                                get_profiles([self.pWbar - self.pbar * self.Ubar2], 
                                                self.z , self.V)[0]
                                                )
        self.profiles['Tp'] = Tp


        # # Transport term with fluctuating velocity
        ## -- ## 
        T1 = 2 * (
            self.Ubar0**2
            + self.Ubar1**2
            + self.Ubar2**2
        ) * self.Ubar2


        T2 = (
            self.UUWbar  # u u w
            + self.VVWbar  # v v w
            + self.WWWbar  # w w w
            )


        T3 = - 2 * (
            self.UWbar * self.Ubar0
            + self.VWbar * self.Ubar1
            + self.WWbar * self.Ubar2
            )


        T4 = - (
            self.UUbar
            + self.VVbar
            + self.WWbar
            ) * self.Ubar2


        ## -- ##
        UiUiW_prime = T1 + T2 + T3 + T4
        UiUiW_prime = get_profiles([UiUiW_prime], self.z, self.V)[0]

        Tv = - self.phif *  derivate_wall(
            unique_z,
            0.5 * (UiUiW_prime)
            )
        self.profiles['Tv'] = Tv

        # TKE transport by molecular diffusion
        k_SGS = get_profiles([self.turbulenceProperties_kbar] , self.z , self.V)[0]
        Uprime2 , Vprime2 , Wprime2 = get_profiles([self.UUbar - self.Ubar0 * self.Ubar0 , self.VVbar - self.Ubar1 * self.Ubar1 , self.WWbar - self.Ubar2 * self.Ubar2],self.z , self.V)
        k_resol = (1/2) * (Uprime2 + Vprime2 + Wprime2)
        k_tot = k_resol + k_SGS

        Tnu = self.phif * nu * derivate_wall(unique_z,derivate_wall(unique_z,k_tot))
        self.profiles['Tnu'] = Tnu 


        #TKE transport by spatial flucutation of velocity
        Wbar_mean = get_profiles([self.Ubar2] , self.z , self.V)[0]
        Wbar_primeprime = get_primeprime(self.Ubar2 , self.z, Wbar_mean)
        UprimeUprime = self.UUbar - self.Ubar0 * self.Ubar0 + self.turbulenceProperties_Rbar0
        VprimeVprime = self.VVbar - self.Ubar1 * self.Ubar1 + self.turbulenceProperties_Rbar3
        WprimeWprime = self.WWbar - self.Ubar2 * self.Ubar2 + self.turbulenceProperties_Rbar5

        Td = (0.5) * get_profiles([
                                          UprimeUprime * Wbar_primeprime
                                        + VprimeVprime * Wbar_primeprime
                                        + WprimeWprime * Wbar_primeprime],
                                        self.z , self.V)[0]
        
    
        Td = -self.phif * derivate_wall(unique_z,Td)
        self.profiles['Td'] = Td

        # Total transport term 
        T = (Tp + Tv + Tnu + Td)
        self.profiles['T'] = T 

        totalTKE = (Ps + Pw + T - epsilon_tot)
        self.profiles['totalTKE'] = totalTKE


    def compute_profiles_DKE(self , rho = 1000 , nu = 1e-6) : 
        unique_z = np.unique(np.round(self.z , 6))
        # ---- Production term
        # - Shear production 
        Dxz = (get_profiles([self.Ubar0 * self.Ubar2],self.z , self.V)[0] 
                    -(get_profiles([self.Ubar0],self.z , self.V)[0]* get_profiles([self.Ubar0 * self.Ubar2],self.z , self.V)[0])) 
        gradUbar_mean = get_profiles([self.grad_Ubar6],self.z , self.V)[0]
        Pspp = -self.phif * Dxz * gradUbar_mean
        self.profiles['Pspp'] = Pspp

        # - Production from form drag
        Ubar_mean = get_profiles([self.Ubar0],self.z,self.V)[0]
        Pppp = self.profiles['fdz_form'] * Ubar_mean
        self.profiles['Pppp'] = Pppp


        ## -- Dissipation term 
        # - P1
        sqrgrad = [self.grad_Ubar0**2 , self.grad_Ubar3**2 , self.grad_Ubar6**2,
                self.grad_Ubar1**2 , self.grad_Ubar4**2 , self.grad_Ubar7**2,
                self.grad_Ubar2**2 , self.grad_Ubar5**2 , self.grad_Ubar8**2]
        sqrgrad = np.sum(sqrgrad, axis = 0) #Sum all components

        epspp1 = self.phif * nu * get_profiles([sqrgrad],self.z , self.V)[0]

        #P2
        epspp2 = -self.phif * nu * (
                    get_profiles([self.grad_Ubar6],self.z, self.V)[0])**2

        #Total espilon 
        epsilonpp = epspp1 + epspp2
        self.profiles['epsilonpp'] = epsilonpp


        ## -- Transport term 
        # -Tv''
        Ubar_mean , Vbar_mean , Wbar_mean = get_profiles([self.Ubar0,self.Ubar1,self.Ubar2],self.z , self.V)
        Uprimeprime = get_primeprime(self.Ubar0 , self.z , Ubar_mean)
        Vprimeprime = get_primeprime(self.Ubar1 , self.z , Vbar_mean)
        Wprimeprime = get_primeprime(self.Ubar2 , self.z , Wbar_mean)


        Rxz = ((self.UWbar - self.Ubar0 * self.Ubar2) + self.turbulenceProperties_Rbar2)
        Ryz = ((self.VWbar - self.Ubar1 * self.Ubar2) + self.turbulenceProperties_Rbar4)
        Rzz = ((self.WWbar - self.Ubar2 * self.Ubar2) + self.turbulenceProperties_Rbar5)

        Tvpp = (Rxz * Uprimeprime 
                    + Ryz * Vprimeprime 
                    + Rzz * Wprimeprime)

        Tvpp = - self.phif * derivate_wall(unique_z,
                        get_profiles([Tvpp],self.z , self.V)[0])
        self.profiles['Tvpp'] = Tvpp

        # - Td''
        Tdpp = (Uprimeprime * Uprimeprime * Wprimeprime
                + Vprimeprime * Vprimeprime * Wprimeprime
                + Wprimeprime * Wprimeprime * Wprimeprime
                    )           
        Tdpp = -(0.5) * self.phif * derivate_wall(
                        unique_z , 
                        get_profiles([Tdpp],self.z , self.V)[0])
        self.profiles['Tdpp'] = Tdpp

        # - Tp''
        pbar_mean = get_profiles([self.pbar],self.z , self.V)[0]
        pprimeprime = get_primeprime(self.pbar,self.z, pbar_mean)
        Tppp = -self.phif * derivate_wall(unique_z,
                                    get_profiles(
                                    [pprimeprime * Wprimeprime],
                                    self.z , self.V)[0])
        self.profiles['Tppp'] = Tppp

        # -Tnu''
        Tnupp = ( Uprimeprime * self.grad_Ubar6
                + Vprimeprime * self.grad_Ubar7
                + Wprimeprime * self.grad_Ubar8
                )
        Tnupp = self.phif * nu * derivate_wall(
                                unique_z,
                                get_profiles([Tnupp],self.z, self.V)[0])
        self.profiles['Tnupp'] = Tnupp

        #Total transport
        Tpp = Tvpp + Tdpp + Tppp + Tnupp
        self.profiles['Tpp'] = Tpp

        #totalDKE
        totalDKE = (Pspp + Pppp + Tpp - epsilonpp - self.profiles['Pw'])
        self.profiles['totalDKE'] = totalDKE


    def compute_macroscopic_quantities(self) : 
        #TKE
        TKE_per_cell = (1./2.) * ((
                                    (self.U0 - self.Ubar0)**2 + 
                                    (self.U1 - self.Ubar1)**2 + 
                                    (self.U2 - self.Ubar2)**2 ) 
                                    + self.turbulenceProperties_kbar)
        self.k_brack = np.sum(TKE_per_cell * self.V) / np.sum(self.V)


        self.q25TKE = np.quantile(np.sqrt(TKE_per_cell), 0.25, 
                            method = 'inverted_cdf',
                            weights = self.V)

        self.q75TKE = np.quantile(np.sqrt(TKE_per_cell), 0.75, 
                        method = 'inverted_cdf',
                        weights = self.V)


        #u*
        dz_bottom = get_dz_slice(self.z, self.V)[0]  #dz of the first cell 
        z_min = np.min(np.unique(self.z))
        dz_bottom = 2 * z_min 
        mask_cells_bottom = np.where(self.z == z_min)
        V_bottom = self.V[mask_cells_bottom]
        dS_bottom = V_bottom / dz_bottom  #Surface of each cell at the bottom 
        S0 = np.sum(dS_bottom) #Surace of the bottom patch 
        #u*(simulation)
        self.ustar_simu = np.sum(np.sqrt(np.abs(self.wallShearStress_bottom0)) * dS_bottom)/S0

        # Compute quantiles u* plot 
        self.q25u_star = np.quantile(np.sqrt(np.abs(self.wallShearStress_bottom0)), 0.25, 
                                method = 'inverted_cdf',
                                weights = dS_bottom)

        self.q75u_star = np.quantile(np.sqrt(np.abs(self.wallShearStress_bottom0)), 0.75, 
                                method = 'inverted_cdf',
                                weights = dS_bottom)


    def writeNetCDFPostPro(self,boundary):
        filename = "fieldsPostPro.nc"
        filepath = os.path.join(self.directory, filename)
        with nc.Dataset(filepath, 'w', format='NETCDF4') as ds:
            # --- Dimensions --- #
            ds.createDimension('unique_z', len(np.unique(np.round(self.z,6))))
            for bound in boundary : 
                ds.createDimension(f'dim_{bound}', len(getattr(self,f'x_{bound}')))


            
            # # --- Mesh on boundaries --- # 
            # for bound in boundary : 
            #     namex = f'x_{bound}'
            #     namey = f'y_{bound}'
            #     namez = f'z_{bound}'
            #     for coord in [namex , namey , namez] :
            #             var = ds.createVariable(coord, 'f8',(f'dim_{bound}'))
            #             var[:] = getattr(self,coord)
            
            # # --- Data on boundaries --- # 
            # print(f'###### {ds.variables.keys()}')
            # for bound in boundary : 
            #     for var in self.variables : 
            #         if (f'{bound}' in var 
            #             and len(getattr(self,var)) == len(getattr(self,f'x_{bound}'))
            #             and var not in [f'x_{bound}',f'y_{bound}',f'z_{bound}'] ): 
            #             var_bound = ds.createVariable(var,'f8',(f'dim_{bound}',))
            #             var_bound[:] = getattr(self,var)

            

            # --- Mesh-coordinates --- #
            var = ds.createVariable('unique_z', "f8", ("unique_z",))
            var[:] = np.unique(np.round(self.z,6))

            var = ds.createVariable('dz_slice', "f8", ("unique_z",))
            var[:] = get_dz_slice(self.z,self.V)


            # --- Profiles --- # 
            for name, data in self.profiles.items():
                nc_var = ds.createVariable(name, 'f8', ('unique_z',))
                nc_var[:] = data
            
            

            # --- Macroscopic quantities --- # 
            for scalar_name in ['ustar_simu', 
                                'k_brack',
                                'q25TKE','q75TKE',
                                'q75u_star', 'q25u_star',
                                'phif', 'Rep','phi'
                                ]:
                if hasattr(self, scalar_name):
                    var = ds.createVariable(scalar_name, 'f8')
                    var.assignValue(getattr(self, scalar_name))


        print("Done writing NetCDF for PostProcessed Data")



    def readNetCDF(self,filename): 
        ncfile = nc.Dataset(self.directory + filename , "r" , format="NETCDF4")

        # Récupère tous les noms de variables et les change
        self.variables = [v.replace('.', '_').replace('Mean','bar').replace(':','_') 
                        for v in ncfile.variables.keys()]

    

        # Lecture des données dans les attributs
        for python_var, orig_var in zip(self.variables, ncfile.variables.keys()):
            data = np.asarray(ncfile.variables[orig_var][:])
            setattr(self, python_var, data)

        ncfile.close()
 
    def readmesh(self,boundary = None ,timeStep=None, structured=False, precision=10, order='F'):
        
        if timeStep is None:
            dir_list = os.listdir(self.directory)
            time_list = []

            for directory in dir_list:
                try:
                    float(directory)
                    time_list.append(directory)
                except:
                    pass
            time_list.sort(key=float)
            timeStep = time_list[-1]

        elif type(timeStep) is int:
            #timeStep should be in a str format
            timeStep = str(timeStep)

        self.timeStep = timeStep

        # Check if cell center position is written in the output directory
        try:
            field = OpenFoamFile(path=self.directory, time_name=self.timeStep,
                                 name='C', structured=False, precision=precision,
                                 order=order)
            values = field.values
            shape = (3, values.size // 3)
            values = np.reshape(values, shape, order=order)
            if structured and not field.uniform:
                try:
                    values[0:3, :] = values[0:3, self.ind]
                    shape = (3,) + tuple(self.shape)
                    values = np.reshape(values, shape, order=order)
                except:
                    print("Variable {} could not be loaded".format(var))
                    self.variables.remove(var)
            X, Y, Z = values[0], values[1], values[2]

            # # --- Boundaries coordinates --- #
            for bound in boundary :
                field = OpenFoamFile(path=self.directory, 
                                time_name=self.timeStep,
                                name='C', 
                                boundary = bound,
                                structured=False, 
                                precision=precision,
                                order=order)
                values = field.values
                shape = (3, values.size // 3)
                values = np.reshape(values, shape, order=order)
        
                xc, yc, zc = values[0],values[1],values[2]
                setattr(self,f'x_{bound}',xc)
                setattr(self,f'y_{bound}',yc)
                setattr(self,f'z_{bound}',zc)

        except FileNotFoundError:
            X, Y, Z = readmesh(self.directory, boundary = None, structured=structured,
                            precision=precision, order=order)
        self.x = X
        self.y = Y
        self.z = Z
        if structured:
            nx = np.unique(X).size
            ny = np.unique(Y).size
            nz = np.unique(Z).size
            self.ind = np.array(range(nx*ny*nz))
            self.shape = (nx, ny, nz)

        # # --- Boundaries coordinates --- #
        for bound in boundary :
            xc,yc,zc = readmesh(self.directory, boundary = bound, structured=structured,
                            precision=precision, order=order)
            
            setattr(self,f'x_{bound}',xc)
            setattr(self,f'y_{bound}',yc)
            setattr(self,f'z_{bound}',zc)

    def readopenfoam(self, boundary, timeStep=None, structured=False, dataToLoad=None,
                     precision=10, order='F'):
        """
        Reading SedFoam results
        Load the last time step saved of the simulation

        Args:
            timeStep : str or int, timeStep to load. If None, load the last time step\n
            structured : bool, true if the mesh is structured
        """

        if timeStep is None:
            dir_list = os.listdir(self.directory)
            time_list = []

            for directory in dir_list:
                try:
                    float(directory)
                    time_list.append(directory)
                except:
                    pass
            time_list.sort(key=float)
            timeStep = time_list[-1]

        elif type(timeStep) is int:
            #timeStep should be in a str format
            timeStep = str(timeStep)

        self.timeStep = timeStep

        #List all variables saved at the required time step removing potential
        #directory that cannot be loaded
        if dataToLoad is None:
            self.variables = []
            basepath = self.directory+self.timeStep+'/'
            for fname in os.listdir(basepath):
                path = os.path.join(basepath, fname)
                if os.path.isdir(path):
                    # skip directories
                    continue
                else:
                    self.variables.append(fname)
                    
            #Remove C, Cx, Cy and Cz if present
            var_to_remove = ['C', 'Cx', 'Cy', 'Cz']
            for var in var_to_remove:
                if var in self.variables:
                    self.variables.remove(var)
        else:
            self.variables = dataToLoad
        

        for var in self.variables:
            #Check if file is in path 
            file_path = os.path.join(self.directory, self.timeStep, var)
            if not os.path.exists(file_path):
                continue           

            field = OpenFoamFile(
                path=self.directory,
                time_name=self.timeStep,
                name=var,
                structured=False,
                precision=precision,
                order=order
            )
            values = field.values

            # ---- volume scalar ---- #
            if field.type_data == "scalar":
                if structured and not field.uniform:
                    try:
                        values = np.reshape(values[self.ind], self.shape, order=order)
                    except:
                        print(f"Variable {var} could not be loaded")
                        self.variables.remove(var)
                        continue
                self.add_variable(var, values)

                # ---- surface scalar ---- #
                if np.shape( getattr(self, var) )[-1] == 1 : 
                    for bound in boundary:
                        try:
                            s_values = readscalar(
                                path=self.directory,
                                time_name=self.timeStep,
                                name=var,   
                                boundary=bound
                            )
                            self.add_variable(f"{var}_{bound}", s_values)
                        except FileNotFoundError:
                            continue

            # ---- vector fields ---- #
            elif field.type_data == "vector":
                # volume vector
                shape = (3, values.size // 3)
                values = np.reshape(values, shape, order=order)
                if structured and not field.uniform:
                    try:
                        values[0:3, :] = values[0:3, self.ind]
                        shape = (3,) + tuple(self.shape)
                        values = np.reshape(values, shape, order=order)
                    except:
                        print(f"Variable {var} could not be loaded")
                        self.variables.remove(var)
                        continue
                
                # Add field
                self.add_variable(var, values)

                # Add all components
                for i in range(values.shape[0]):
                    self.add_variable(f"{var}{i}", values[i])
                

                # surface vector
                if np.shape( getattr(self, var) )[-1] == 1 :
                    for bound in boundary:
                        try:
                            s_values = readvector(
                                path=self.directory,
                                time_name=self.timeStep,
                                name=var,        
                                boundary=bound
                            )
                            
                            #Components
                            for i in range(s_values.shape[0]):
                                self.add_variable(f"{var}_{bound}{i}", s_values[i])
                        except FileNotFoundError:
                            continue

            # ---- symmtensor fields ---- #
            elif field.type_data == "symmtensor":
                shape = (6, values.size // 6)
                values = np.reshape(values, shape, order=order)
                if structured and not field.uniform:
                    try:
                        values[0:6, :] = values[0:6, self.ind]
                        shape = (6,) + tuple(self.shape)
                        values = np.reshape(values, shape, order=order)
                    except:
                        print("Variable {} could not be loaded".format(var))
                        self.variables.remove(var)
                        continue
                        
                # Add field
                self.add_variable(var, values)

                # Add all components
                for i in range(values.shape[0]):
                    self.add_variable(f"{var}{i}", values[i])

            # ---- tensor fields ---- #
            elif field.type_data == "tensor":
                shape = (9, values.size // 9)
                values = np.reshape(values, shape, order=order)
                if structured and not field.uniform:
                    try:
                        values[0:9, :] = values[0:9, self.ind]
                        shape = (9,) + tuple(self.shape)
                        values = np.reshape(values, shape, order=order)
                    except:
                        print("Variable {} could not be loaded".format(var))
                        self.variables.remove(var)
                        continue
                        
                # Add field
                self.add_variable(var, values)

                # Add all components
                for i in range(values.shape[0]):
                    self.add_variable(f"{var}{i}", values[i])

            #Set attributes
            self.__setattr__(var.replace('.', '_').replace('Mean','bar').replace(':','_'), values)

            #Finally clean names of variables, by removing useless names : 
            self.variables = [v for v in self.variables if ':' not in v and 'Mean' not in v]

    def keys(self):
        """
        Print the name of all variables loaded from simulation results
        """
        print("Loaded available variables are :")
        print(self.variables)

    def _choose_simulation(self, path):
        """
        Make a list of all directories located in path containing a simulation.
        Ask the user which simulation to load

        Args:
            path : str, reference path where simulations are stored.
        """
        directories = []
        subDirectories = [x[0] for x in os.walk(path)]

        for f in subDirectories:
            #A directory is detected to be a simulation if it contains a 0_org/ folder
            if f + "/constant" in subDirectories:
                directories.append(f)

        # If no directories found
        if len(directories) == 0:
            raise DirectorySimuError(path)

        for i in range(len(directories)):
            print("{} : {}".format(i, directories[i]))
        chosenSimulation = -1
        while type(chosenSimulation) is not int or (
                chosenSimulation < 0 or chosenSimulation > len(directories) - 1):
            chosenSimulation = int( input(
                "Please, choose one simulation ! (integer between {} and {})".format(
                    0, len(directories) - 1))
            )
        directory = directories[chosenSimulation]

        return directory + "/"

    def _find_directory(self, path, simu):
        """
        Look for the directory of simu in all the sub directories of path. If several
        directories are found, the program asks which directory is the good one.

        Args:
            path : str, reference path where simulations are stored.
            simu : str, name of the simu that has to be loaded. If None, it will
                lists all existing simulation names in path and ask you to choose
        """
        directories = []
        subDirectories = [x[0] for x in os.walk(path)]

        for f in subDirectories:
            if f.endswith(simu):
                directories.append(f)

        # If no directories found
        if len(directories) == 0:
            raise DirectorySimuError(simu)

        # If several directories found, ask for the one wanted
        elif len(directories) > 1:
            print("The following simulations has been found :")
            for i in range(len(directories)):
                print("{} : {}".format(i, directories[i]))
            chosenSimulation = -1
            while type(chosenSimulation) is not int or (
                    chosenSimulation < 0 or chosenSimulation > len(directories) - 1):
                chosenSimulation = int(input(
                    "Please, choose one simulation ! (integer between {} and {})".format(
                        0, len(directories) - 1)
                    )
                )
            directory = directories[chosenSimulation]

        else:
            directory = directories[0]

        return directory + "/"

if __name__ == "__main__":

    simu = "box"
    timeStep = "4"

    for d in dirs:
        rep = os.path.join(os.path.dirname(__file__), "../output_samples")

        mySimu = OpenFoamSimu(path=rep, simu=simu, timeStep=timeStep, structured=True)

        mySimu.keys()

        mySimu.U

