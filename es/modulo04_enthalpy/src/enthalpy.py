
import numpy as np
from typing import Any, Callable, Union, Literal
from scipy.interpolate import interp1d


#%% Definitions of auxiliary stuff

from scipy.constants import sigma # Stefan-Boltzmann constant [W/(m2.K4)]

# Definition of types to improve code readability
DynamicParam = Union[float, Callable[[float, float], float]]
GeometryParam = Union[float, tuple[float, float], tuple[float, float, float]]

# Conversion of temperature from Celcius to Kelvin
def c2k(T):
    return T + 273.15

# Conversion of temperature from Kelvin to Celcius
def k2c(T):    
    return T - 273.15


#%% Newtonian cooling with the enthalpic method

def dHdt(
    t: float, 
    H: float, 
    thermophysical: tuple[float, Any], 
    geometry: GeometryParam, 
    convective: tuple[DynamicParam, DynamicParam], 
    radiative: tuple[DynamicParam, DynamicParam, DynamicParam]
) -> list[float]:
    """
    dh/dt derivative of specific enthalpy with respect to time, 
    for lumped capacitance model with convective and radiative heat fluxes.
    This function is tested and ready to use with a numerical integrator
    such as scipy.integrate.solve_ivp.

    Parameters
    ----------
    t : float
        time, [s]
    H : float
        specific enthalpy, [J/kg]
    thermophysical: tuple
        A tuple containing the following thermophysical properties:
            * rho : float
                density of the Newtonian solid, [kg/m3]
            * h2T : scipy.interpolate._interpolate.interp1d
                interpolator enthalpy [J/kg] --> Temperature [K]
    geometry : float or tuple
        Geometric representation of the solid. Can accept:
            * float : The direct surface-area-to-volume ratio (SA:V), [1/m].
            * tuple (A, V) : Total surface area [m2] and volume [m3].
            * tuple (A_conv, A_rad, V) : Convective surface area [m2], radiative 
              surface area [m2], and volume [m3].
    convective : tuple
        A tuple containing parameters for convective heat flux calculation
            * h_con : float or function
                Convective heat transfer coefficient, [W/(m2.K)]
                Can be a constant or a function of time and temperature: h_con(t, T).
            * T_con : float or function
                Convective far-field temperature, [K]
                Can be a constant or a function of time and temperature: T_con(t, T)
    radiative : tuple
        A tuple containing parameters for radiative heat flux calculation
            * eps_rad : float or function
                Gray body emissivity, [-]
                Can be a constant or a function of time and temperature: eps_rad(t, T)
            * F_rad : float or function
                view factor, [-]
                Can be a constant or a function of time and temperature: F_rad(t, T)
            * T_rad : float or function
                Radiative far-field temperature, [K]
                Can be a constant or a function of time and temperature: T_rad(t, T)

    Returns
    -------
    dh_dt : numpy.array
        The time derivative of specific enthalpy [J/(kg.s)], enclosed in a 
        single-element list for compatibility with vector integrators.
    """
    ######################################## pre-processing

    # unpack thermophysical parameters
    rho, h2T = thermophysical

    # dynamic processing of geometry
    if isinstance(geometry, (int, float)):
        A_V_conv = float(geometry)
        A_V_rad = float(geometry)
    elif isinstance(geometry, (tuple, list, np.ndarray)):
        if len(geometry) == 2:
            A_tot, V = geometry
            A_V_conv = A_tot / V
            A_V_rad = A_tot / V
        elif len(geometry) == 3:
            A_conv, A_rad, V = geometry
            A_V_conv = A_conv / V
            A_V_rad = A_rad / V
        else:
            raise ValueError("Si 'geometry' es una tupla/arreglo, debe tener 2 o 3 elementos.")
    else:
        raise TypeError("'geometry' debe ser un float o una tupla/lista de 2 o 3 elementos.")

    # unpack convective and radiative heat transfer parameters
    h_con, T_con = convective
    eps_rad, F_rad, T_rad = radiative 


    # find current temperature using the interpolator
    H_scalar = float(H[0]) if isinstance(H, (np.ndarray, list)) else float(H) # ensure H is a scalar
    T = float(h2T(H_scalar))


    ##################################### Evaluate convective heat flux [W/m2]
    # convective heat transfer coefficient
    hc = h_con(t, T) if callable(h_con) else h_con
    # far field temperature
    Tf = T_con(t, T) if callable(T_con) else T_con

    q_conv = -hc * (T - Tf)

    #################################### Evaluate radiative heat flux [W/m2]
    # thermal emissivity
    eps = eps_rad(t, T) if callable(eps_rad) else eps_rad
    # view factor
    F = F_rad(t, T) if callable(F_rad) else F_rad
    # far field temperature
    Tr = T_rad(t, T) if callable(T_rad) else T_rad
    
    q_rad = -sigma * eps * F * (T**4 - Tr**4)

    ############################ Evaluate volumetric heat transfer rate [W/m3]
    Q_tot = A_V_conv * q_conv + A_V_rad * q_rad

    ##### Return derivative of specific enthalpy w.r.t time, dh/dt, [J/(kg.s)]
    return np.asarray([Q_tot / rho])


# Creation of interpolators between enthalpy and temperature
def create_h_T_interpolators(
    h_T_table: Union[np.ndarray, Any],
    kind : Literal['linear', 'nearest', 'nearest-up', 'zero', 'slinear', 'quadratic', 'cubic', 'previous', 'next'] = 'linear'
) -> tuple[interp1d, interp1d]:
    """
    Create bi-directional 1D interpolators between specific enthalpy [J/kg] and Temperature [K].

    This utility generates two independent interpolators allowing seamless conversion 
    from enthalpy to temperature and vice versa, typically used to look up state 
    variables during thermodynamic or heat transfer simulations.

    Parameters
    ----------
    h_T_table : np.ndarray
        A NumPy structured array or record array containing the reference thermophysical 
        data. It must include at least the following named fields:
            * 'H' : array-like
                Specific enthalpy values, typically in [J/kg].
            * 'T' : array-like
                Temperature values, typically in [K].
    kind : str or int, optional
        Specifies the kind of interpolation as a string or as an integer 
        specifying the order of the spline interpolator to use. 
        Options include: 'linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic'.
        Default is 'linear'.

    Returns
    -------
    H2T : scipy.interpolate.interp1d
        Interpolator function to compute Temperature given Enthalpy: T = H2T(H).
    T2H : scipy.interpolate.interp1d
        Interpolator function to compute Enthalpy given Temperature: H = T2H(T).
    """
    # handle data used as input for the interpolators
    try:
        # Try structured array or Pandas DataFrame access
        H_data = h_T_table['H']
        T_data = h_T_table['T']
    except (ValueError, IndexError, TypeError, KeyError):
        # Fallback to standard 2D positional arrays
        # Convert input to numpy array safely
        try:
            arr = np.asarray(h_T_table)
        except Exception as e:
            raise TypeError("Input 'h_T_table' cannot be converted to a NumPy array.") from e
        # Verify dimensions and shapes
        if arr.ndim != 2 or arr.shape[1] < 2:
            raise ValueError(
                f"Expected a 2D array-like object with at least 2 columns (H and T), "
                f"but got shape {arr.shape}."
            )
        H_data = arr[:, 0]
        T_data = arr[:, 1]
    
    # create enthalpy [J/kg] --> temperature [K] interpolator
    H2T = interp1d(
        H_data,
        T_data,
        kind = kind,
        bounds_error = False,
        fill_value = 'extrapolate'
    )

    # create temperature [K] --> enthalpy [J/kg] interpolator
    T2H = interp1d(
        T_data,
        H_data,
        kind = kind,
        bounds_error = False,
        fill_value = 'extrapolate'
    )

    return H2T, T2H
#%% Newtonian cooling with Temperature method (classic implementation)