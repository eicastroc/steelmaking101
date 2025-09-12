# -*- coding: utf-8 -*-
"""
Created on Mon Oct 32 13:51:53 2024

@author: ecastro

"""


import numpy as np


#%%#####################################################################
def logK13(T:float) -> float:
    """
    log Equilibrium constant for sulfur-oxygen 
    dissolution reaction (Reac. 1.3)

    Parameters
    ----------
    T : Float
        Temperature, in [K].
        
    Returns
    -------
        Float
        logarithm base 10 of equilibrium constant for Reac. 1.3
    """
    return -935.0/T + 1.375


#%%#####################################################################
def logLs(Cs: float, T: float, fs: float, ho:float) -> float:
    """ 
    log of Sulfur partition coefficient Ls
        Ls = (%S)/[%S]

    Parameters
    ----------
    Cs: Float
        Sulfide capacity of the slag, in [wt%]
    T : Float
        Temperature, in [K].
    fs: Float
        Henrian activity coefficient, for sulfur
    ho: Float
        Henrian activity, for oxygen [wt%]
    
    Returns
    -------
        Float
        logarithm base 10 of sulfur partition coefficient
    """
    return np.log10(Cs) + logK13(T) + np.log10(fs) - np.log10(ho)


#%%#####################################################################
def desulfRatio(Msl: float, Ls: float) -> float:
    """ 
    Ratio of equilibrium and initial sulfur contents in the metal
    as given by a sulfur mass balance.
    
    Parameters
    ----------
    Msl: Float
         kg of molten slag per ton of metal, [kg/ton]
    Ls : Float
         Sulfur partition coefficient, Ls=(%S)/[%S]
    Returns
    -------
        Float
        Ratio of equilibrium and initial sulfur contents in the metal
    """
    return 1 / (1 + (Msl/1000) * Ls)


#%%#####################################################################
def desulfPct(Msl: float, Ls: float) -> float:
    """ 
    Percentage desulfurization at thermodynamic equilibrium
    
    Parameters
    ----------
    Msl: Float
         kg of molten slag per ton of metal, [kg/ton]
    Ls : Float
         Sulfur partition coefficient, Ls=(%S)/[%S]
    Returns
    -------
        Float
        Percentage of desulfurization at thermodynamic equilibrium cond.
    """
    return (1 - (1 / (1 + (Msl/1000) * Ls))) * 100


#%%#####################################################################
def mSlagEq(Ls: float, wS0: float, wS: float) -> float:
    """ 
    Percentage desulfurization at thermodynamic equilibrium
    
    Parameters
    ----------
    Ls : Float
         Sulfur partition coefficient, Ls=(%S)/[%S]
    wS0 : Float
         Initial sulfur content in the metal, wt%
    wS : Float
         Equilibrium sulfur content in the metal, wt%
    Returns
    -------
        Float
        kg of molten slag per ton of metal, [kg/ton]
    """
    return (1000/Ls) * (wS0/wS - 1)


#%%#####################################################################
def desulfRate(t:float, wS:float, c:float) -> float:
    """
    Desulfurization rate differential equation
    taking into account uptake of sulfur by the slag
    
    Parameters
    ----------
    t : Float
        time, in [s]
    wS: Float
        sulfur content, in [wt%]
    c:  tupple with Floats
        ks: desulfurization rate constant, in [1/s]
        ws0: initial sulfur content in metal, in [wt%]
        Msl: kg of slag per ton of metal, in [kg/ton]
        Ls: sulfur partition coeficient, Ls=(%S)/[%S]
    Returns
    -------
    dwSdt : float
        Desulfurization rate, in [[%S]/s]
    """
    ks, wS0, Msl, Ls = c
    Y = Msl/1000 * Ls
    dwSdt = -ks * (wS * (1 + 1/Y) - wS0/Y)
    return dwSdt


#%%#####################################################################
def opticalBasicity(slagComp: dict, T: float) -> float:
    """
    calculation of optical basicity

    Parameters
    ----------
    slagComp : dict
        Dictionary with slag composition, a valid entry is:
            
            slagComp = {'Al2O3': 9,'CaO': 56, 'MgO': 8, 'SiO2': 25, 'FeO': 1, 'MnO': 1}
            
    T : Float
        Temperature, in [K].
        
    Returns
    -------
        Float
        optical basicity of slag, ideally a number between 0 and 1.
    """
    
    ### Constants for calculation
    molarMass = {'Al2O3': 101.9613, 'CaO': 56.0794, 'MgO': 40.3014, 'SiO2': 60.0843, 'FeO': 71.8464, 'MnO': 70.9374}
    anions = {'Al2O3': 3, 'CaO': 1, 'MgO': 1, 'SiO2': 2, 'FeO': 1, 'MnO': 1}

    ### Basicities of molecules (could be modified...)
    lambTh = {'Al2O3': 0.66, 'CaO': 1.0, 'MgO': 0.92, 'SiO2': 0.47, 'FeO': 0.94, 'MnO': 0.95} ## (A.Ghosh, Sec. Steelmaking, App.2.4)
    
    #### Mass pct of components in slag
    slagW = np.asarray([slagComp['Al2O3'], slagComp['CaO'], slagComp['MgO'], slagComp['SiO2'], slagComp['FeO'], slagComp['MnO']])
    
    #### slag molar masses
    slagM = np.asarray([molarMass['Al2O3'], molarMass['CaO'], molarMass['MgO'], molarMass['SiO2'], molarMass['FeO'], molarMass['MnO']])
    
    #### slag number of oxygen anions
    slagA = np.asarray([anions['Al2O3'], anions['CaO'], anions['MgO'], anions['SiO2'], anions['FeO'], anions['MnO']])

    #### slag basicity of components
    slagLamb = np.asarray([lambTh['Al2O3'], lambTh['CaO'], lambTh['MgO'], lambTh['SiO2'], lambTh['FeO'], lambTh['MnO']])
    
    ### Molar fraction of components in slag
    slagX = (slagW/slagM) / np.sum(slagW/slagM)
 
    return np.sum(slagX * slagA * slagLamb) / np.sum(slagX * slagA)


#%%#####################################################################
def logCs(slagComp:dict, T:float)-> float:
    """
    calculation of sulfide capacity, according to:
    
    Zhang G H, Chou K C, Pal U
    Estimation of Sulfide Capacities of Multicomponent Slags using Optical Basicity
    ISIJ International 53, 761-767 (2013)

    Makes use of the function opticalBasicity()
    
    Parameters
    ----------
    slagComp : dict
        Dictionary with slag composition, a valid entry is:
            
            slagComp = {'Al2O3': 9,'CaO': 56, 'MgO': 8, 'SiO2': 25, 'FeO': 1, 'MnO': 1}
            
    T : Float
        Temperature, in [K].
        
    Returns
    -------
        Float
        logarithm base 10 of sulfide capacity, Cs
    """    
    
    lamb = opticalBasicity(slagComp, T)
    return -6.08 + 4.49/lamb + (15893 - 15864/lamb)/T


#%%#####################################################################
def kSCalc(eps:float) -> float:
    """
    calculation of desulfurization constant, according to:
    
    Making, Shaping, Treating of Steel, 11th ed. (1998)
    Steelmaking and Refining Volume
    R.J. Fruehan, ed.
    p. 671 (Eqs. 11.2.3 to 11.2.5)
    
    Parameters
    ----------
    eps : Float
        specific stirring power, in [W.tonne-1]
        
    Returns
    -------
        Float
        Desulfurization rate constante, in [s-1]
    """    
    return 1/60 * ((eps<60)*0.013*(eps)**(0.25) + (eps>=60)*8e-6*(eps)**(2.1))


#%%#####################################################################
def stirPowerCalc(V:float, T:float, M:float, H:float, P0:float) -> float:
    """
    calculation of stirring power, according to:
    
    Making, Shaping, Treating of Steel, 11th ed. (1998)
    Steelmaking and Refining Volume
    R.J. Fruehan, ed.
    p. 670 (Eq. 11.2.1)
    
    Parameters
    ----------
    V : Float
        gas flow rate, in [Nm3.min-1]
    
    T : Float
        temperature, in [K]

    M : Float
        Mass of steel bath, in [tonne]

    H : Float
        Depth of gas injection, in [m]

    P0 : Float
        Gas pressure at the bath surface, in [atm]
        
    Returns
    -------
        Float
        specific stirring power, in [W.tonne-1]
    """    
    return 14.23 * (V*T/M) * np.log10((1+H)/(1.48*P0))