import numpy as np



def opticalBasicity(slagComp: dict, T: float, corr=False, lambcoeffs='Mills') -> float:
    """
    calculation of optical basicity, according to either:

        Duffy J A, Ingram M D  (1976)
        An Interpretation of Glass Chemistry in Terms of the Optical Basicity Concept  
        Journal of Non-crystalline Solids 21, pp. 373-410 (1976)


        Mills K C (1993)
        The Influence of Structure on the Physico-chemical Properties of Slags
        ISIJ International 33, pp. 148-155        

    Parameters
    ----------
    slagComp : dict
        Dictionary with slag composition, a valid entry is:
            
            slagComp = {'Al2O3': 9,'CaO': 56, 'MgO': 8, 'SiO2': 25, 'FeO': 1, 'MnO': 1}
            
    T : Float
        Temperature, in [K].

    corr : Boolean
        Flag for correction of optical basicity.
            True: calculation according to Mills K C (1993).
            False: calcualtion according to Duffy J A, Ingram M D  (1976).

    lambcoeffs : String
        This is for performing modifications to calculations of binary
        basicity, specifically the lambTh coefficients, according to 
        specific models.

        Valid entries are: 'Mills', 'Ghosh', 'Zhang'
        
    Returns
    -------
        Float
        optical basicity of slag, ideally a number between 0 and 1.
    """
    
    ### Constants for calculation
    molarMass = {'Al2O3': 101.9613, 'CaO': 56.0794, 'MgO': 40.3014, 'SiO2': 60.0843, 'FeO': 71.8464, 'MnO': 70.9374}
    anions = {'Al2O3': 3, 'CaO': 1, 'MgO': 1, 'SiO2': 2, 'FeO': 1, 'MnO': 1}

    ### Basicities of molecules (could be modified...)
    lambcoeffs = lambcoeffs.lower()
    lambTh = {'Al2O3': 0.60, 'CaO': 1.0, 'MgO': 0.78, 'SiO2': 0.48, 'FeO': 1.00, 'MnO': 1.00} ## (Ken Mills, Course 2011)
    if lambcoeffs == 'slag':    ## (Slag Atlas, 1995)
        lambTh = {'Al2O3': 0.605, 'CaO': 1.0, 'MgO': 0.78, 'SiO2': 0.48, 'FeO': 1.00, 'MnO': 1.00} 
    if lambcoeffs == "ghosh":    ## (A.Ghosh, Sec. Steelmaking, App.2.4, 2000)
        lambTh = {'Al2O3': 0.66, 'CaO': 1.0, 'MgO': 0.92, 'SiO2': 0.47, 'FeO': 0.94, 'MnO': 0.95} 
    if lambcoeffs == "zhang":    ## (Zhang (2013))
        lambTh = {'Al2O3': 0.61, 'CaO': 1.0, 'MgO': 0.78, 'SiO2': 0.48, 'FeO': 1.24, 'MnO': 1.43} 


    
    
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

    ### Corrected basicity
    if corr==True:
            ### Aliases to keep equations readable
        Al = int(0); Ca = int(1); Mg = int(2); Si = int(3); Fe = int(4); Mn = int(5)

        ### Molar fraction of Non-bridging oxygens
        Xnb = slagX[Ca] + slagX[Mg] + slagX[Fe] + slagX[Mn]
        f = (Xnb - slagX[Al]) / Xnb

        ### Modify molar fractions (for accounting for bridging/non-bridging)
        slagX[Ca] *= f
        slagX[Mg] *= f
        slagX[Fe] *= f
        slagX[Mn] *= f
 
    return np.sum(slagX * slagA * slagLamb) / np.sum(slagX * slagA)



def logCs_Gaye(slagComp, T):
    """
    calculation of modified sulfide capacity, according to:
    
        Faral M, Gaye H.  
        Metal slag equilibria.   
        In: Proc Second Intern Symposium Metall Slags and Fluxes, TMS-AIME. 1984. p. 159–79

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
        logarithm base 10 of modified sulfide capacity, Cs'
    """    
    Ca, Mg, Si, Al = slagComp['CaO'], slagComp['MgO'], slagComp['SiO2'], slagComp['Al2O3']

    Bnum = 5.623*Ca + 4.15*Mg - 1.115*Si + 1.457*Al
    Aden = Ca + 1.391*Mg+ 1.867*Si + 1.65*Al

    logCsp = Bnum/Aden + 2.82 - 13300/T
    logK = -935.0/T + 1.375

    return logCsp - logK


def logCs_Sommer(slagComp, T, corr=False):
    """
    calculation of sulfide capacity, according to:
    
        Sosinsky N J, Sommerville I D. 
        The composition and temperature dependence of the sulfide capacity of metallurgical slags. 
        Metall Trans B 17, 331–337 (1986)

    Makes use of the function opticalBasicity()
    
    Parameters
    ----------
    slagComp : dict
        Dictionary with slag composition, a valid entry is:
            
            slagComp = {'Al2O3': 9,'CaO': 56, 'MgO': 8, 'SiO2': 25, 'FeO': 1, 'MnO': 1}
            
    T : Float
        Temperature, in [K].

    lambCorr : Boolean
        Use of corrected optical basicity

    corr : Boolean
        Flag for correction of optical basicity.

    Returns
    -------
        Float
        logarithm base 10 of sulfide capacity, Cs
    """    
    

    lam = opticalBasicity(slagComp, T, corr)


    return (22690 - 54640*lam)/T  + 43.6*lam -25.2



def logCs_Young(slagComp, T, corr=False):
    """
    calculation of sulfide capacity, according to:
    
        Young R W.
        Use of the optical basicity concept for determining phosphorus and sulphur slag/metal partitions.
        European Report EUR 13176 EN (1991).

        NOTE: There is a typo in the original source.
               The coefficients are are taken from:
                https://doi.org/10.2355/isijinternational.ISIJINT-2021-514

    Makes use of the function opticalBasicity()
    
    Parameters
    ----------
    slagComp : dict
        Dictionary with slag composition, a valid entry is:
            
            slagComp = {'Al2O3': 9,'CaO': 56, 'MgO': 8, 'SiO2': 25, 'FeO': 1, 'MnO': 1}
            
    T : Float
        Temperature, in [K].

    corr : Boolean
        Flag for correction of optical basicity.

    Returns
    -------
        Float
        logarithm base 10 of sulfide capacity, Cs
    """    
    
    lam = opticalBasicity(slagComp, T, corr)

    Si = slagComp['SiO2']
    Al = slagComp['Al2O3']
    Fe = slagComp['FeO']
    Cs1 = -13.913 + 42.84*lam - 23.82*lam**2 - 11710/T - 0.02223*Si- 0.02275*Al
    Cs2 = -0.6261 + 0.4808*lam - 0.7917*lam**2 - 1697/T - (2587*lam)/T + 5.144e-4*Fe

    return (lam<0.8)*Cs1 + (lam>=0.8)*Cs2



def logCs_Tani(slagComp, T, corr=False):
    """
    calculation of sulfide capacity, according to:
    
    Taniguchi Y, Sano N, Seetharaman S
    Sulphide Capacities of CaO–Al2O3–SiO2–MgO–MnO Slags in the Temperature Range 1673–1773K
    ISIJ International 49, 156-163 (2009)

    Makes use of the function opticalBasicity()
    
    Parameters
    ----------
    slagComp : dict
        Dictionary with slag composition, a valid entry is:
            
            slagComp = {'Al2O3': 9,'CaO': 56, 'MgO': 8, 'SiO2': 25, 'FeO': 1, 'MnO': 1}
            
    T : Float
        Temperature, in [K].

    corr : Boolean
        Flag for correction of optical basicity.

    Returns
    -------
        Float
        logarithm base 10 of sulfide capacity, Cs
    """    
    
    lam = opticalBasicity(slagComp, T, corr)
    Mg = slagComp['MgO']
    Mn = slagComp['MnO']
    Si = slagComp['SiO2']
    Al = slagComp['Al2O3']

    return 7.350 + 94.89*np.log10(lam) - (10051+lam*(-338*Mg+287*Mn))/T + 0.2284*Si + 0.1379*Al - 0.0587*Mg + 0.0841*Mn



def logCs_Zhang(slagComp, T, corr=True):
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

    corr : Boolean
        Flag for correction of optical basicity.

    Returns
    -------
        Float
        logarithm base 10 of sulfide capacity, Cs
    """    
    
    lam = opticalBasicity(slagComp, T, corr, lambcoeffs = "Zhang")

    return -6.08 + 4.49/lam + (15893 - 15864/lam)/T



def logCs_Hao(slagComp, T, corr=True):
    """
    calculation of sulfide capacity, according to:
    
    Hao X & Wang X
    A New Sulfide Capacity Model for CaO-Al2O3-SiO2-MgO Slags Based on Corrected Optical Basicity
    Steel Research International 87, 359 - 363 (2016)

    Makes use of the function opticalBasicity()
    
    Parameters
    ----------
    slagComp : dict
        Dictionary with slag composition, a valid entry is:
            
            slagComp = {'Al2O3': 9,'CaO': 56, 'MgO': 8, 'SiO2': 25, 'FeO': 1, 'MnO': 1}
            
    T : Float
        Temperature, in [K].

    corr : Boolean
        Flag for correction of optical basicity.

    Returns
    -------
        Float
        logarithm base 10 of sulfide capacity, Cs
    """    
    
    lam = opticalBasicity(slagComp, T, corr)

    return (12410/lam - 27109)/T + 19.45 - 11.85/lam


def logCs_KTH(slagComp, T, corr=True):
    """
        Sulfide capacity calculation according to the KTH sulfide capacity
        model presented in references:
        
            Nzotta, MM., Sichen, D., & Seetharaman, S. (1998). 
            Sulphide capacities in some multi component slag systems. 
            ISIJ international, 38(11), 1170-1179.
            
            Nzotta, MM., Sichen, D. & Seethraman, S. (1999).
            A study of the sulfide capacities of iron-oxide containing slags
            Metallurgical and Materials Transactions B, 30(5), 909-920.

        
        Condo et al. proposed modifications, where binary parameters for
        Al2O3-MgO are proposed, and ternary parameters
        for Al2O3-CaO-MgO system were discarded: 
        
            Condo A F T, Qifeng S, Sichen D
            Sulfide Capacities in the Al2O3-CaO-MgO-SiO2 system.
            Steel Research International 89, 1800061 (2018).
                        
    Parameters
    ----------
    slagComp : dict
        Dictionary with slag composition, a valid entry is:
            
            slagComp = {'Al2O3': 9,'CaO': 56, 'MgO': 8, 'SiO2': 25, 'FeO': 1, 'MnO': 1}
            
    T : Float
        Temperature, in [K].

    corr : Boolean
        Flag for correction of model parameters.
            True: calculation according to Nzotta, MM. et al. (1999).
            False: calcualtion according to Condo A F T (2018)
        
    Returns
    -------
        Float
        logarithm base 10 of sulfide capacity, Cs.
    """    
    
    ### Constants for calculation
    R = 8.314 # Universal gas constant
    molarMass = {'Al2O3': 101.9613, 'CaO': 56.0794, 'MgO': 40.3014, 'SiO2': 60.0843, 'FeO': 71.8464, 'MnO': 70.9374}
    cations = {'Al2O3': 2, 'CaO': 1, 'MgO': 1, 'SiO2': 1, 'FeO': 1, 'MnO': 1}

    #### Mass pct of components in slag
    slagW = np.asarray([slagComp['Al2O3'], slagComp['CaO'], slagComp['MgO'], slagComp['SiO2'], slagComp['FeO'], slagComp['MnO']])
    
    #### slag molar masses
    slagM = np.asarray([molarMass['Al2O3'], molarMass['CaO'], molarMass['MgO'], molarMass['SiO2'], molarMass['FeO'], molarMass['MnO']])
    
    #### slag number of cations
    slagC = np.asarray([cations['Al2O3'], cations['CaO'], cations['MgO'], cations['SiO2'], cations['FeO'], cations['MnO']])
    
    ### Molar fraction of components in slag
    slagX = (slagW/slagM) / np.sum(slagW/slagM)
    
    ### Ionic fraction of  cations in slag
    slagY = (slagC*slagX) / np.sum(slagC*slagX)

    ### Aliases to keep equations readable
    Al = int(0); Ca = int(1); Mg = int(2); Si = int(3); Fe = int(4); Mn = int(5)
    
    ######################## Computation of Xi parameter
    ### pure components (perfect mixing reference state)
    xi_Al = 157705.28
    xi_Ca   = -33099.43
    xi_Mg   = 9573.07
    xi_Si  = 168872.59
    xi_Fe   = 0
    xi_Mn   = -36625.46
    Xi_pure = slagX[Al]*xi_Al + slagX[Ca]*xi_Ca + slagX[Mg]*xi_Mg + slagX[Si]*xi_Si + slagX[Fe]*xi_Fe + slagX[Mn]*xi_Mn

    ## binary interactions
    xi_Al_Ca  = slagY[Al]*slagY[Ca]*(98282.7968+55.07340941*T)
    if corr==False:
        xi_Al_Mg = 0.0
    else:
        xi_Al_Mg  = slagY[Al]*slagY[Mg]*(3055800-1481.1*T)
    xi_Al_Si  = slagY[Al]*slagY[Si]*(186850.468)
    xi_Al_Mn  = slagY[Al]*slagY[Mn]*(433987.403+249.121594*T*(slagY[Al]-slagY[Mn]))
    xi_Ca_Si   = slagY[Ca]*slagY[Si]*(97271.7695+72.8749746*T)
    xi_Ca_Fe   = slagY[Ca]*slagY[Fe]*(1.74180413e2+9.3184392e1*T-1.14946043e5*(slagY[Ca]-slagY[Fe]))
    xi_Mg_Si   = slagY[Mg]*slagY[Si]*(697403.222-224.084556*T)
    xi_Mn_Si   = slagY[Mn]*slagY[Si]*(-322911.47+212.02998*T+134860.658*(slagY[Mn]-slagY[Si]))
    xi_Fe_Si   = slagY[Fe]*slagY[Si]*(-3.85381423e5+2.09908747e2*T+1.6193563e5*(slagY[Fe]-slagY[Si]))
    xi_Fe_Mn   = slagY[Mn]*slagY[Fe]*(8.47784954e5-3.48193022e2*T)
    Xi_bin = xi_Al_Ca + xi_Al_Mg + xi_Al_Si + xi_Al_Mn + xi_Ca_Si + xi_Ca_Fe + xi_Mg_Si + xi_Mn_Si + xi_Fe_Si + xi_Fe_Mn

    ## ternary interactions
    if corr==False:
        xi_Al_Ca_Mg = slagY[Al]*slagY[Ca]*slagY[Mg]*(4165955.5-1066.5663*T-3040801.89*slagY[Al])
    else:
        xi_Al_Ca_Mg = 0.0
    xi_Al_Ca_Si = slagY[Al]*slagY[Ca]*slagY[Si]*(-2035792.64+686.044695*T)
    xi_Al_Mg_Mn = slagY[Al]*slagY[Mg]*slagY[Mn]*(-1561497.23+2722.78645*T-12274184.6*slagY[Al])
    xi_Al_Mg_Si = slagY[Al]*slagY[Mg]*slagY[Si]*(156192.588-290.498555*T+949447.247*slagY[Al])
    xi_Al_Mn_Si = slagY[Al]*slagY[Mn]*slagY[Si]*(1565848.46-662.494162*T-5322903.11*slagY[Al])
    xi_Ca_Mg_Si  = slagY[Ca]*slagY[Mg]*slagY[Si]*(-1526497.71+625.663842*T+1485255.98*slagY[Ca])
    xi_Ca_Mn_Si  = slagY[Ca]*slagY[Mn]*slagY[Si]*(-1179891.59+621.243714*T-1191111.79*slagY[Ca])
    xi_Mg_Mn_Si  = slagY[Mg]*slagY[Mn]*slagY[Si]*(9103609.27-4426.00708*T-2869664.62*slagY[Mg])
    xi_Fe_Mg_Si  = slagY[Fe]*slagY[Mg]*slagY[Si]*(1.55017390e6-1.12815899e3*T+1.43235112e6*slagY[Fe])
    xi_Al_Fe_Si = slagY[Al]*slagY[Fe]*slagY[Si]*(-2.62147542e6+1.46552872e3*T-2.7752503e6*slagY[Al])
    xi_Fe_Mn_Si  = slagY[Fe]*slagY[Mn]*slagY[Si]*(-8.24444429e5+1.64329498e2*T+7.22404274e5*slagY[Fe])
    xi_Ca_Fe_Si  = slagY[Ca]*slagY[Fe]*slagY[Si]*(-1.22135555e6+6.50216976e2*T-1.59675926e6*slagY[Ca])
    Xi_tern = xi_Al_Ca_Mg + xi_Al_Ca_Si + xi_Al_Mg_Mn + xi_Al_Mg_Si + xi_Al_Mn_Si + xi_Ca_Mg_Si + xi_Ca_Mn_Si + xi_Mg_Mn_Si + xi_Fe_Mg_Si + xi_Al_Fe_Si + xi_Fe_Mn_Si + xi_Ca_Fe_Si
             
    ######################## Calculation of sulfide capacity
    ## Free energy of ionic S,O reaction
    dG = 118535 - 58.8157*T
    ### Final computation of Xi parameter
    Xi = Xi_pure + Xi_bin + Xi_tern
    ### Computation of sulfide capacity
    Cs = np.exp(-dG/(R*T)) * np.exp(-Xi/(R*T))
  
    return np.log10(Cs)