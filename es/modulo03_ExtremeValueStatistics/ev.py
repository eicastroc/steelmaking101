import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gumbel_r


def eCDF(Y: np.array):
    """ 
    empirical cumulative distribution function for inclusion size,
    or any other microstructural feature that can be analyzed with
    Extreme Value methodology.

    Parameters
    ----------
    Y: array with measurements of a microstructural feature.
    
    Returns
    -------
    Ysorted: np.ndarray
        microstructural feature measurements, in ascending order.
    ecdf: numpy.ndarray
        empirical cumulative distribution function values.
    """
    Ysorted = np.sort(Y)  # sort values
    Nvalues = len(Ysorted)
    ecdf = np.linspace(1, Nvalues, Nvalues) / (Nvalues+1)    # estimate ecdf
    return Ysorted, ecdf


def fitEVmom(Y: np.array, verbose:bool = False):
    """
    Fit (Gumbel) Extreme Value distribution to measurements
    of a microstructural feature (e.g., inclusion size) using
    Moments Method (as presented in ASTM E2283 norm)

    Parameters
    ----------
    Y: np.array
        Measurements of the microstructural feature that will be studied 
                using Extreme Value Methodology. In the case of inclusions, 
                Y correponds to the longest inclusion length in an specimen.
    
    Returns
    -------
    lamb: Float
        location parameter of the fitted (Gumbel) Extreme Value distriution.
    delta: Float
        scale parameter of the fitted (Gumbel) Extreme Value distriution.
    """
    Ymean = np.mean(Y) # mean
    Ystd = np.std(Y)   # std. dev. pop.
    delta = Ystd * np.sqrt(6) / np.pi   # scale parameter
    lamb = Ymean - 0.5772 * delta       # location parameter
    if verbose==True:
        print('Gumbel distribution params. (*MoM est.)')
        print('lambda: {:.4f}'.format(lamb))
        print('delta : {:.4f}\n'.format(delta))
    return lamb, delta

def fitEVml(Y:np.array, verbose:bool = False):
    """
    Fit (Gumbel) Extreme Value distribution to measurements
    of a microstructural feature (e.g., inclusion size) using
    Maximum Likelihood methods (as implemented in scipy.stats)
    
    Parameters
    ----------
    Y: np.array
        Measurements of the microstructural feature that will be studied 
                using Extreme Value Methodology. In the case of inclusions, 
                Y correponds to the longest inclusion length in an specimen.
    
    Returns
    -------
    lamb: Float
        location parameter of the fitted (Gumbel) Extreme Value distriution.
    delta: Float
        scale parameter of the fitted (Gumbel) Extreme Value distriution.
    """    
    evDist = gumbel_r.fit(Y) #ML fitting
    delta = evDist[1]   # scale parameter
    lamb = evDist[0]    # location parameter
    if verbose==True:
        print('Gumbel distribution params. (*ML est.)')
        print('lambda: {:.4f}'.format(lamb))
        print('delta : {:.4f}\n'.format(delta))
    return lamb, delta


def calcSE(x:np.array, lamb:float, delta:float):
    """
    Estimate the Standard Error (SE) of size estimation for
    a population of microstructural features that has been fit
    to the (Gumbel) Extreme Value distribution.
    
    Parameters
    ----------
    x: np.array
        size of microstructural feature
    lamb: Float
        location parameter of the fitted (Gumbel) Extreme Value distribution.
    delta: Float
        scale parameter of the fitted (Gumbel) Extreme Value distribution.
    N: Int
        number of measurements
    
    Returns
    -------
    SE: Float
        standard error of size estimations
    """    
    y = (x - lamb) / delta
    n = len(y)
    se = delta * np.sqrt((1.109 + 0.514*y + 0.608*y**2)/n)
    return se


def calcRedVar(F:np.array):
    """ 
    reduced variable, y, used in the procedure of linearization 
    of the (Gumbel) Extreme Value CDF.
    
    Parameters
    ----------
    F: np.array
        Cumulative density function F(y)
    
    Returns
    -------
    y: float
        linealized (Gumbel) Extreme Value cumulative density function.
    """    
    y = -1.0 * np.log(-np.log(F))
    return y


def gumbelPlot(ax, x:np.array, method:str="ml", ci=True,
               marker='o', color='k', lab="emp."):


    # work on ECDF
    x_sorted, ecdf = eCDF(x) #sort data
    y = calcRedVar(ecdf) # reduced variable

    # fit distribution
    method = method.lower()
    if method=="ml":
        lamb, delta = fitEVml(x_sorted, verbose=True)
    elif method=="mom":
        lamb, delta = fitEVmom(x_sorted, verbose=True)
    else:
        raise NameError(f"{method} is not a valid method: [ml, mom]")
    
    # microestructural feature size estimations
    x_est = delta * y + lamb

    # confidence interval
    se = calcSE(x_est, lamb, delta)

    # Empirical distribution
    ax.plot(x_sorted, y, ls='', marker=marker, c=color, mfc="None", label=lab)

    # fitted distribution with 95%CI
    ax.plot(x_est, y, ls='-', color=color)
    if ci==True:
        xmin, xmax = x_est-2.0*se, x_est+2.0*se
        ax.plot(xmin, y, ls='--', color=color, alpha=0.5)
        ax.plot(xmax, y, ls='--', color=color, alpha=0.5)

    #ax.set(xlabel="x", ylabel="Red. Var.")
    #ax.legend()
    ax.grid(ls='--')


#%%
"""
The functions below are for a more convinient plotting of microstructural
features, based on the plot templates presented by Murakami:

    Metal Fatigue: Effects of Small Defects and Nonmetallic Inclusions (2nd ed)
    Yukitaka Murakami
    Elsevier (2019)
    ISBN: 978-0-12-813876-2
    DOI: https://doi.org/10.1016/C2016-0-05272-5
"""

def make_patch_spines_invisible(ax):
    """
    Source: 
    """
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for sp in ax.spines.values():
        sp.set_visible(False)


def makeInclusionsPlotTemplate(figsize=None, xmin=0, xmax=100, ymin=-2, ymax=12):
    """
    Make a Gumbel plot template for inclusions ratings 
    based on Appendixes: [A, B, C], of:

    Metal Fatigue: Effects of Small Defects and Nonmetallic Inclusions (2nd ed)
    Yukitaka Murakami
    Elsevier (2019)
    ISBN: 978-0-12-813876-2
    DOI: https://doi.org/10.1016/C2016-0-05272-5
    
    """

    fig, ax = plt.subplots(figsize=figsize)
    fig.subplots_adjust(right=0.75)

    # first vertical axis scale (Probabilities)
    ps = np.array([
        1e-2, 1e-1, 5e-1, 8e-1, 9e-1, 0.95, 0.98, 0.99, 
        0.995, 0.998, 0.999, 0.9995, 0.9998, 
        0.9999, 0.99995, 0.99998, 0.99999
    ])
    yticks = -1.0*np.log(-1.0*np.log(ps))
    ind = [index for index, value in enumerate(yticks) if ymin <= value <= ymax]
    ps = ps[ind]
    yticks = yticks[ind]
    ax.set(
        xlim=(xmin, xmax),
        ylim=(ymin, ymax),
        ylabel="F%",
        yticks = yticks,
        yticklabels = np.round(100*ps, 4)
    )

    # second vertical axis scale (Reduced variable)
    par1 = ax.twinx()
    par1.set(
        ylim=(ymin, ymax),
        ylabel="Reduced Variable",
        yticks=range(ymin, ymax+1)
    )
    par1.spines["left"].set_position(("axes", -0.20)) # red one
    make_patch_spines_invisible(par1)
    par1.spines["left"].set_visible(True)
    par1.yaxis.set_label_position('left')
    par1.yaxis.set_ticks_position('left')

    # third vertical axis scale (Return periods)
    tr = np.array([
        5, 10, 20, 50, 100, 200, 500, 1000, 
        2000, 5000, 10000, 20000, 50000, 100000
        ])
    pr = 1 - 1/tr
    yticks = -1.0*np.log(-1.0*np.log(pr))
    ind = [index for index, value in enumerate(yticks) if ymin <= value <= ymax]
    tr = tr[ind]
    yticks = yticks[ind]
    par2 = ax.twinx()
    par2.set(
        ylim=(ymin, ymax),
        ylabel='Return period',
        yticks=yticks,
        yticklabels=tr
    )

    ax.grid(ls="--")
    fig.tight_layout()

    return fig, ax


def plotInclusions(ax, x:np.array, method:str="ml", ci=True,
               marker='o', color='k', mfc="None", label="emp."):


    # work on ECDF
    x_sorted, ecdf = eCDF(x) #sort data
    y = calcRedVar(ecdf) # reduced variable

    # Empirical distribution
    ax.plot(x_sorted, y, ls='', marker=marker, c=color, 
            mfc=mfc, label=label)

    # fit distribution
    method = method.lower()
    if method=="ml":
        lamb, delta = fitEVml(x_sorted, verbose=True)
    elif method=="mom":
        lamb, delta = fitEVmom(x_sorted, verbose=True)
    else:
        raise NameError(f"{method} is not a valid method: [ml, mom]")
    
    # microestructural feature size estimations
    ymin, ymax = ax.get_ylim()
    y_est = np.linspace(ymin, ymax, 100)
    x_est = delta * y_est + lamb

    # fitted distribution with 95%CI
    ax.plot(x_est, y_est, ls='-', color=color)
    se = calcSE(x_est, lamb, delta)
    if ci==True:
        xmin, xmax = x_est-2.0*se, x_est+2.0*se
        ax.plot(xmin, y_est, ls='--', color=color, alpha=0.5)
        ax.plot(xmax, y_est, ls='--', color=color, alpha=0.5)

    
    ax.legend()