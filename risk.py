import pandas as pd
from scipy.stats import (jarque_bera, norm)
import numpy as np

class ReturnSeries:
    """
    Given a csv file path return a return series.

    path - a string file path
    name - the name of the output col
    col_key - the name of the input col
    is_percent - if the values are in percentage format
    """

    def __init__(self, path, name, col_key):
        self.__path = path
        self.__name = name
        self.__col_key = col_key
        df = pd.read_csv(path, header=0, index_col=0, parse_dates=True)
        df = df[col_key]
        df.index = df.index.to_period('M')
        df = df.pct_change()
        df = df.dropna()
        df = df.rename(name)
        self.returns = df

    def get_returns(self):
        return self.returns


def get_ffme_returns():
    ffme = pd.read_csv("data/Portfolios_Formed_on_ME_monthly_EW.csv",
                       header=0, index_col=0, parse_dates=True, na_values=-99.99)
    rets = ffme[['Lo 10', 'Hi 10']]
    rets.columns = ['SmallCap', 'LargeCap']
    rets /= 100
    rets.index = pd.to_datetime(rets.index, format="%Y%m").to_period('M')
    return rets


def get_hfi_returns():
    hfi = pd.read_csv("data/edhec-hedgefundindices.csv", header=0, index_col=0,
                      parse_dates=True)
    hfi /= 100
    hfi.index = hfi.index.to_period('M')
    return hfi


def drawdown(principal, return_series):
    """ Takes a principal and a time series of asset returns.
        returns a DataFrame with columns for
        the wealth index,
        the previous peaks, and
        the percentage drawdown
    """
    wealth_index = principal * (1 + return_series).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index - previous_peaks) / previous_peaks
    return pd.DataFrame({
        "Wealth": wealth_index,
        "Previous Peak": previous_peaks,
        "Drawdown": drawdowns
    })


def skewness(r):
    """
    Computes the skewness of a Series or DataFrame
    Return: Float or Series
    """
    return higher_moment(r, 3)


def kurtosis(r):
    return higher_moment(r, 4)


def higher_moment(r, k):
    return ((r - r.mean())**k).mean()/r.std(ddof=0)**k


def is_normal(r, level=0.01):
    _, p_value = jarque_bera(r)
    return p_value > level


def cornish_fisher_expansion(s, k, za):
    return za + (1/6)*(za**2 - 1)*s + (1/24)*(za**3 - 3*za)*(k-3)-(1/36)*(2*(za**3)-5*za)*(s**2)

def semideviation(r):
    return r[r<0].std(ddof=0)

def var_historic(r, level=5):
    """
    Returns the historic VaR of a DataFrame or Series.
    It will compute the the level-percentile of the data.
    """
    if isinstance(r, pd.DataFrame):
        return r.aggregate(var_historic, level=level)
    elif isinstance(r, pd.Series):
        return -np.percentile(r, level)
    else:
        raise TypeError("Expected r to be a Series or DataFrame")

def var_gaussian(r, level=5, modified=False):        
    z = norm.ppf(level/100)
    if modified:
        skew = skewness(r)
        kurt = kurtosis(r)
        z = cornish_fisher_expansion(skew, kurt, z)
    return -(r.mean() + z*r.std(ddof=0))

def cvar_historic(r, level=5):
    if isinstance(r, pd.DataFrame):
        return r.aggregate(cvar_historic, level=level)
    elif isinstance(r, pd.Series):
        is_beyond = r <= -var_historic(r, level)
        return -r[is_beyond].mean()
    else:
        raise TypeError("Expected r to be Series or DataFrame")