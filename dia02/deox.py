# -*- coding: utf-8 -*-
"""
Created on Mon Oct 32 13:51:53 2024

@author: ecastro

All the thermodynamic data is taken from the following publication

Zhang, L., Ren, Y., Duan, H., Yang, W., & Sun, L. (2015). 
Stability diagram of Mg-Al-O system inclusions in molten steel. 
Metallurgical and Materials Transactions B, 46(4), 1809-1825.
"""

import numpy as np
import matplotlib.pyplot as plt

#%%## Deoxidation equilibrium constants ################################


def logK_Al(T:float=1873)->float:
    """
    Deoxidation Equilibrium constant of Aluminum
    
    (Al2O3) = 2[Al] + 3[O]
    K = a_Al**2 * a_O**3 / a_Al2O3
    logK = 11.62 - 45300/T

    Parameters
    ----------
    T : Float
        Temperature in K

    Returns
    -------
    Float
        Base-10 Logarithm of equilibrium constant for the reaction: 
            (Al2O3) = 2[Al] + 3[O]
    """
    return -45300/T + 11.62 


def logK_Mg(T:float=1873)->float:
    """
    Deoxidation Equilibrium constant of Magnesium
    
    (MgO) = [Mg] + [O]
    K = a_Mg * a_O / a_MgO
    logK = -4.28 - 4700/T

    Parameters
    ----------
    T : Float
        Temperature in K

    Returns
    -------
    Float
        Base-10 Logarithm of equilibrium constant for the reaction: 
            (MgO) = [Mg] + [O]
    """
    return -4700/T -4.28 


def logK_MgAl(T:float=1873)->float:
    """
    Deoxidation Equilibrium constant of Spinel
    
    (MgO.Al2O3) = 2[Al] + [Mg] + 4[O]
    (MgO.Al2O3) = (MgO) + (Al2O3)
    K = a_Al**2 a_Mg * a_O**4 / a_MgOAl2O3
    logK = 6.376 - 51083.2/T

    Parameters
    ----------
    T : Float
        Temperature in K

    Returns
    -------
    Float
        Base-10 Logarithm of equilibrium constant for the reaction: 
            (MgO.Al2O3) = 2[Al] + [Mg] + 4[O]
    """
    return 6.376 - 51083.2/T 


#%%## First order interaction coefficients e_i^j #######################
"""    First order interaction coefficients e_i^j    """

### i = Al
def e_Al_Al(T):
    return 80.5/T
def e_Al_Mg(T):
    return -0.13
def e_Al_O(T):
    return 3.21 - 9720/T

### i = Mg
def e_Mg_Al(T):
    return -0.12
def e_Mg_Mg(T):
    return 0
def e_Mg_O(T):
    return 958 -2.592e6/T

### i = O
def e_O_Al(T):
    return 1.90 - 5750/T
def e_O_O(T):
    return 0.76 - 1750/T
def e_O_Mg(T):
    return 630 - 1.705e6/T


#%%## Second order interaction coefficients r_i^j, r_i^{j,k} ###########
"""    Second order interaction coefficients r_i^j, r_i^{j,k}    """

### i = Al
def r_Al_Al(T:float=1873)->float:
    return 0 # -0.0011 + 0.17/T # Sighworth & Elliot
def r_Al_Mg(T:float=1873)->float:
    return 0
def r_Al_O(T:float=1873)->float:
    return -107 + 2.75e5/T
def r_Al_AlMg(T:float=1873)->float:
    return 0
def r_Al_AlO(T:float=1873)->float:
    return -0.021 - 13.78/T
def r_Al_MgO(T:float=1873)->float:
    return -260

### i = Mg
def r_Mg_Al(T:float=1873)->float:
    return 0
def r_Mg_Mg(T:float=1873)->float:
    return 0
def r_Mg_O(T:float=1873)->float:
    return -1.904e6 + 4.22e9/T
def r_Mg_AlMg(T:float=1873)->float:
    return 0
def r_Mg_AlO(T:float=1873)->float:
    return -230
def r_Mg_MgO(T:float=1873)->float:
    return 2.143e5 - 5.156e8/T

### i = O  -- Oxygen
def r_O_Al(T:float=1873)->float:
    return 0.0033 - 25/T
def r_O_Mg(T:float=1873)->float:
    return 7.05e4 -1.696e8/T
def r_O_O(T:float=1873)->float:
    return 0 
def r_O_AlMg(T:float=1873)->float:
    return -150
def r_O_AlO(T:float=1873)->float:
    return 127.3 +3.273e5/T
def r_O_MgO(T:float=1873)->float:
    ## In the original article Fig. 5 could not be reproduced.
    ## When this term is neglected, Fig. 5 is a perfect match.
    return -2.513e6 + 5.573e9/T


#%%### Wrappers for interaction parameters (Wagner)#####################

def e(key:dict, T:float=1873, debug:bool=True)->float:
    """
    Wagner formalism interaction coefficients:
    This function returns the first-order interaction coefficients e_i^j,
    i.e. interaction coefficient for solute i in the presence of solute j

    Parameters
    ----------
    key : STRING
        A valid i,j component pair, e.g. "Al,O", "Si,O", etc.
    T : FLOAT, optional
        Temperature of the melt. The default is 1873.
    debug: Boolean
        If true, error rise if a pair not present, otherwise a warning and the
        interaction parameter is set to zero.
    Returns
    -------
    e_dict[key](T) : FLOAT
        The first-order interaction coefficient e_i^j
    """   
    e_dict = {
          ### Al
          'Al,Al': e_Al_Al, 'Al,Mg': e_Al_Mg, 'Al,O' : e_Al_O,
          ### Mg
          'Mg,Al': e_Mg_Al, 'Mg,Mg': e_Mg_Mg, 'Mg,O' : e_Mg_O, 
          ### O
          'O,Al' : e_O_Al,  'O,Mg' : e_O_Mg,  'O,O'  : e_O_O,
          }
    if key in e_dict.keys():
        return e_dict[key](T)
    elif debug == True:
        raise ValueError(f"Interaction parameter not available for the given pair: {key}")
    else:
        print(f"Interaction parameter not available for the given pair: {key} (set to zero)")
        return 0
    


def r(key:dict, T:float=1873, debug:bool=True)->float:
    """
    Wagner formalism interaction coefficients:
    This function returns the second-order interaction coefficients 
    r_i^j or r_i^{j,k}
    i.e. interaction coefficient for solute i in the presence of either
    solute j, or solutes j,k.

    Parameters
    ----------
    key : STRING
        A valid i,j or i,jk component pair, e.g. "Al,O", "Al,AlO", etc.
    T : FLOAT, optional
        Temperature of the melt. The default is 1873.
    debug: Boolean
        If true, error rise if a pair not present, otherwise a warning and the
        interaction parameter is set to zero.
    Returns
    -------
    e_dict[key](T) : FLOAT
        The second-order interaction coefficient r_i^j or r_i^{j,k}
    """   
    r_dict = {
          ###################### r_i^j
          ### Al
          'Al,Al': r_Al_Al, 'Al,Mg': r_Al_Mg, 'Al,O' : r_Al_O,
          ### Mg
          'Mg,Al': r_Mg_Al, 'Mg,Mg': r_Mg_Mg, 'Mg,O' : r_Mg_O, 
          ### O
          'O,Al' : r_O_Al,  'O,Mg' : r_O_Mg,  'O,O'  : r_O_O,    
          
          ###################### r_i^{j,k}
          ### Al
          'Al,AlMg': r_Al_AlMg, 'Al,AlO' : r_Al_AlO, 'Al,MgO' : r_Al_MgO,
          ### Mg
          'Mg,AlMg': r_Mg_AlMg, 'Mg,AlO' : r_Mg_AlO, 'Mg,MgO' : r_Mg_MgO,
          ### O
          'O,AlO'  : r_O_AlO,   'O,AlMg' : r_O_AlMg, 'O,MgO'  : r_O_MgO,
          }
    if key in r_dict.keys():
        return r_dict[key](T)
    elif debug == True:
        raise ValueError(f"Interaction parameter not available for the given pair: {key}")
    else:
        print(f"Interaction parameter not available for the given pair: {key} (set to zero)")
        return 0
    

#%%## Root-finding functions for single deox equilibrium ###############

def optFun_AlO(pctO:float, *params:list)->float:
    """
    Thermodynamic equilibrium Al-O
    Activity coefficients calculated with none, only first order, or
    first and second order interaction coefficients
    
    logK = 2*log(fAl) + 2*log[%Al] + 3*log(fO) + 3*log[%O] - log(a_Al2O3)

    log(fAl) = e_Al_Al*[%Al] + e_e_Al_O*[%O]                 \
             + r_Al_Al*[%Al]**2 + r_Al_O*[%O]**2 + r_Al_O*[%O]**2  \
             + r_Al_AlO*[%Al]*[%O]
             
    log(fO) = e_O_Al*[%Al] + e_O_O*[%O]                    \
            + r_O_Al*[%O]**2 + r_O_AlO[%Al]*[%O]
    
    Parameters
    ----------
    pctO : Float
        weight percent oxygen.
    *params : List or tupples
        pctAl: weight percent aluminum (float)    
        T: Temperature in K (float)
        aAl2O3: thermodynamic activity of Al2O3 (float)
        order: 0 (none), 1(e_i^j), 2(e_i^j, r_i^j, r_i^{j,k}) (int)
        
    Returns
    -------
    eps : float
        LHS residual used to numerically find the root of the equation.
    """
    pctAl, T, aAl2O3, order = params # unpacking of parameters

    log_K = logK_Al(T)
    log_aAl2O3 = np.log10(aAl2O3)
        
    if order == 0:
        log_fAl = 0
        log_fO  = 0
    elif order == 1:
        log_fAl = e('Al,Al',T)*pctAl + e('Al,O',T)*pctO
        log_fO  = e('O,Al',T)*pctAl + e('O,O',T)*pctO
    elif order == 2:
        log_fAl = e('Al,Al',T)*pctAl + e('Al,O',T)*pctO \
                + r('Al,O',T)*pctO**2 + r('Al,AlO',T)*pctAl*pctO
        log_fO  = e('O,Al',T)*pctAl + e('O,O',T)*pctO \
                + r('O,Al',T)*pctAl**2 + r('O,AlO',T)*pctAl*pctO
    else:
        valid = [0, 1, 2]
        raise ValueError(f"order: {order} is not valid, choose either {valid }")
    eps = 2*log_fAl + 2*np.log10(pctAl) + 3*log_fO + 3*np.log10(pctO) \
        - log_aAl2O3 - log_K
    return eps


def optFun_MgO(pctO:float, *params:list)->float:
    """
    Thermodynamic equilibrium Mg-O
    Activity coefficients calculated with none, only first order, or
    first and second order interaction coefficients
    
    logK = log(fMg) + log[%Mg] + log(fO) + log[%O] - log(a_MgO)

    log(fMg) = e_Mg_Mg*[%Mg] + e_Mg_O*[%O]           \
             + r_Mg_O*[%O]**2 + r_Mg_MgO*[%Mg]*[%O]
    
    log(fO) = e_O_Mg*[%Mg] + e_O_O*[%O]              \
            + r_O_Mg*[%Mg]**2 + r_O_MgO*[%Mg]*[%O]

    logK = (e_Mg_Mg + e_O_Mg)*[%Mg] + (e_Mg_O + e_O_O)*[%O]    \
         + log[%Mg] + log[%O] - log(a_MgO)                     \
         + r_Mg_O*[%O]**2 + r_O_Mg*[%Mg]**2                    \
         + (r_Mg_MgO + r_O_MgO)*[%Mg][%O]

    Parameters
    ----------
    pctO : Float
        weight percent oxygen.
    *params : List or tupple
        pctMg: weight percent Magnesium
        T: Temperature in K.
        aMgO: thermodynamic activity of MgO
        order: 0 (none), 1(e_i^j), 2(e_i^j, r_i^j, r_i^{j,k}) (int)
        
    Returns
    -------
    eps : float
        LHS residual used to numerically find the root of the equation.
    """
    pctMg, T, aMgO, order = params # unpacking of parameters

    log_K = logK_Mg(T)
    log_aMgO = np.log10(aMgO)
        
    if order == 0:
        log_fMg = 0
        log_fO  = 0
    elif order == 1:
        log_fMg = e('Mg,Mg',T)*pctMg + e('Mg,O',T)*pctO
        log_fO  = e('O,Mg',T)*pctMg  + e('O,O',T)*pctO
    elif order == 2:
        log_fMg = e('Mg,Mg',T)*pctMg + e('Mg,O',T)*pctO           \
            + r('Mg,O',T)*pctO**2 + r('Mg,MgO')*pctMg*pctO
        log_fO = e('O,Mg',T)*pctMg + e('O,O',T)*pctO              \
           + r('O,Mg',T)*pctMg**2 + r('O,MgO')*pctMg*pctO
    else:
        valid = [0, 1, 2]
        raise ValueError(f"order: {order} is not valid, choose either {valid }")
    eps = log_fMg + np.log10(pctMg) + log_fO + np.log10(pctO) \
        - log_aMgO - log_K
    return eps


#%%## Root-finding functions for multi deox equilibrium ###############


def optFun_alumina(pctO:float, *params:list)->float:
    """
    Thermodynamic equilibrium Al-O
    Activity coefficients calculated with none, only first order, or
    first and second order interaction coefficients
    
    logK = 2*log(fAl) + 2*log[%Al] + 3*log(fO) + 3*log[%O] - log(a_Al2O3)

    log(fAl) = e_Al_Al*[%Al] + e_Al_Mg*[%Mg] + e_Al_O*[%O]         \
             + r_Al_Al*[%Al]**2 + r_Al_Mg*[%Mg]**2 + r_Al_O*[%O]**2 \
             + r_Al_AlMg*[%Al]*[%Mg] + r_Al_AlO*[%Al]*[%O] + r_Al_MgO*[%Mg][%O]
             
    log(fO) = e_O_Al*[%Al] + e_O_Mg*[%Mg] + e_O_O*[%O]  \
            + r_O_Al*[%O]**2 + r_O_Mg*[%Mg]**2 + r_O_O*[%O]**2 \
            + r_O_AlMg*[%Al]*[%Mg] + r_O_AlO[%Al]*[%O] + r_O_MgO*[%Mg][%O]

    Parameters
    ----------
    pctO : Float
        weight percent oxygen.
    *params : List or tuple
        pctAl: weight percent Aluminum
        pctMg: weight percent Magnesium
        T: Temperature in K.
        aMgO: thermodynamic activity of MgO
        order: 0 (none), 1(e_i^j), 2(e_i^j, r_i^j, r_i^{j,k}) (int)
        
    Returns
    -------
    eps : float
        LHS residual used to numerically find the root of the equation.
    """
    pctAl, pctMg, T, aAl2O3, order = params # unpacking of parameters

    log_K = logK_Al(T)
    log_aAl2O3 = np.log10(aAl2O3)
        
    if order == 0:
        log_fAl = 0
        log_fO  = 0
    elif order == 1:
        log_fAl = e('Al,Al',T)*pctAl + e('Al,Mg',T)*pctMg + e('Al,O',T)*pctO
        log_fO  = e('O,Al',T)*pctAl  + e('O,Mg',T)*pctMg  + e('O,O',T)*pctO
    elif order == 2:
        log_fAl = e('Al,Al',T)*pctAl + e('Al,Mg',T)*pctMg + e('Al,O',T)*pctO \
                + r('Al,O',T)*pctO**2                                        \
                + r('Al,AlO',T)*pctAl*pctO + r('Al,MgO')*pctMg*pctO
        log_fO  = e('O,Al',T)*pctAl  + e('O,Mg',T)*pctMg  + e('O,O',T)*pctO \
                + r('O,Al',T)*pctAl**2 + r('O,Mg',T)*pctMg**2               \
                + r('O,AlMg',T)*pctAl*pctMg + r('O,AlO',T)*pctAl*pctO       \
                + r('O,MgO')*pctMg*pctO
    else:
        valid = [0, 1, 2]
        raise ValueError(f"order: {order} is not valid, choose either {valid }")
    eps = 2*log_fAl + 2*np.log10(pctAl) + 3*log_fO + 3*np.log10(pctO) \
        - log_aAl2O3 - log_K
    return eps


def optFun_periclase(pctO, *params):
    """
    Thermodynamic equilibrium Mg-O
    Activity coefficients calculated with none, only first order, or
    first and second order interaction coefficients
    
    logK = log(fMg) + log[%Mg] + log(fO) + log[%O] - log(a_MgO)

    log(fMg) = e_Mg_Al*[%Al] + e_Mg_Mg*[%Mg] + e_Mg_O*[%O] \
             + r_Mg_Al*[%Al]**2 + r_Mg_Mg*[%Mg]**2 + r_Mg_O*[%O]**2 \
             + r_Mg_AlMg*[%Al]*[%Mg] + r_Mg_AlO*[%Al]*[%O] + r_Mg_MgO*[%Mg]*[%O]
    
    log(fO) = e_O_Al*[%Al] + e_O_Mg*[%Mg] + e_O_O*[%O] \
             + r_O_Al*[%Al]**2 + r_O_Mg*[%Mg]**2 + r_O_O*[%O]**2 \
             + r_O_AlMg*[%Al]*[%Mg] + r_O_AlO*[%Al]*[%O] + r_O_MgO*[%Mg]*[%O]

    Parameters
    ----------
    pctO : Float
        weight percent oxygen.
    *params : List or tupple
        pctAl: weight percent Aluminum
        pctMg: weight percent Magnesium
        T: Temperature in K.
        aMgO: thermodynamic activity of MgO
        order: 0 (none), 1(e_i^j), 2(e_i^j, r_i^j, r_i^{j,k}) (int)
        
    Returns
    -------
    eps : float
        LHS residual used to numerically find the root of the equation.
    """
    pctAl, pctMg, T, aMgO, order = params # unpacking of parameters

    log_K = logK_Mg(T)
    log_aMgO = np.log10(aMgO)
        
    if order == 0:
        log_fMg = 0
        log_fO  = 0
    elif order == 1:
        log_fMg = e('Mg,Al',T)*pctAl + e('Mg,Mg',T)*pctMg + e('Mg,O',T)*pctO
        log_fO  = e('O,Al',T)*pctAl  + e('O,Mg',T)*pctMg  + e('O,O',T)*pctO
    elif order == 2:
        log_fMg = e('Mg,Al',T)*pctAl + e('Mg,Mg',T)*pctMg + e('Mg,O',T)*pctO \
                + r('Mg,O',T)*pctO**2                                        \
                + r('Mg,AlO',T)*pctAl*pctO + r('Mg,MgO')*pctMg*pctO
        log_fO  = e('O,Al',T)*pctAl  + e('O,Mg',T)*pctMg  + e('O,O',T)*pctO \
                + r('O,Al',T)*pctAl**2 + r('O,Mg',T)*pctMg**2               \
                + r('O,AlMg',T)*pctAl*pctMg + r('O,AlO',T)*pctAl*pctO       \
                + r('O,MgO')*pctMg*pctO
    else:
        valid = [0, 1, 2]
        raise ValueError(f"order: {order} is not valid, choose either {valid }")
    eps = log_fMg + np.log10(pctMg) + log_fO + np.log10(pctO) \
        - log_aMgO - log_K
    return eps


def optFun_spinel(pctO:float, *params:list)->float:
    """
    Thermodynamic equilibrium Mg-O
    Activity coefficients calculated with none, only first order, or
    first and second order interaction coefficients
    
    logK = 2log(fAl) + 2log[%Al] + log(fMg) + log[%Mg] + 4log(fO) + 4log[%O] \
         - log(a_MgOAl2O3)

    Parameters
    ----------
    pctO : Float
        weight percent oxygen.
    *params : List or tupple
        pctAl: weight percent Aluminum
        pctMg: weight percent Magnesium
        T: Temperature in K.
        aMgO: thermodynamic activity of MgO
        order: 0 (none), 1(e_i^j), 2(e_i^j, r_i^j, r_i^{j,k}) (int)
        
    Returns
    -------
    eps : float
        LHS residual used to numerically find the root of the equation.
    """
    pctAl, pctMg, T, aMgOAl2O3, order = params # unpacking of parameters

    log_K = logK_MgAl(T)
    log_aMgOAl2O3 = np.log10(aMgOAl2O3)
        
    if order == 0:
        log_fAl = 0
        log_fO  = 0
    elif order == 1:
        log_fAl = e('Al,Al',T)*pctAl + e('Al,Mg',T)*pctMg + e('Al,O',T)*pctO
        log_fMg = e('Mg,Al',T)*pctAl + e('Mg,Mg',T)*pctMg + e('Mg,O',T)*pctO
        log_fO  = e('O,Al',T)*pctAl  + e('O,Mg',T)*pctMg  + e('O,O',T)*pctO
    elif order == 2:
        log_fAl = e('Al,Al',T)*pctAl + e('Al,Mg',T)*pctMg + e('Al,O',T)*pctO \
                + r('Al,O',T)*pctO**2                                        \
                + r('Al,AlO',T)*pctAl*pctO + r('Al,MgO')*pctMg*pctO
                
        log_fMg = e('Mg,Al',T)*pctAl + e('Mg,Mg',T)*pctMg + e('Mg,O',T)*pctO\
                + r('Mg,O')*pctO**2 \
                + r('Mg,AlO',T)*pctAl*pctO + + r('Mg,MgO')*pctMg*pctO
                
        log_fO  = e('O,Al',T)*pctAl  + e('O,Mg',T)*pctMg  + e('O,O',T)*pctO \
                + r('O,Al',T)*pctAl**2 + r('O,Mg',T)*pctMg**2               \
                + r('O,AlMg',T)*pctAl*pctMg + r('O,AlO',T)*pctAl*pctO       \
                + r('O,MgO')*pctMg*pctO
    else:
        valid = [0, 1, 2]
        raise ValueError(f"order: {order} is not valid, choose either {valid }")        
    eps = 2*log_fAl + 2*np.log10(pctAl) + log_fMg + np.log10(pctMg) \
        + 4*log_fO + 4*np.log10(pctO) - log_aMgOAl2O3 - log_K

    return eps


#%%
## Función auxiliar
def draw_log_scales(x0:float, x1:float, ax=None, axis=0, color='lightgray'):
    orders = np.logspace(x0, x1, int(x1-x0)+1)
    multiples = np.linspace(1, 9, 9)

    if ax == None:
        fig, ax = plt.subplots()

    for order in orders:
        for multiple in multiples:
            if axis==0:
                ax.axvline(np.log10(order*multiple), color=color, alpha=0.25)
            else:
                ax.axhline(np.log10(order*multiple), color=color, alpha=0.25)