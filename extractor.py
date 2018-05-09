
import pandas as pd
import matplotlib.pyplot as plt
import math
from pandas.plotting import autocorrelation_plot
import numpy as np
import pylab as pl
from scipy import signal
import scipy.fftpack

import matplotlib.pylab as pylab
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (5, 5),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'18',
         'ytick.labelsize':'18'}
pylab.rcParams.update(params)


def autocorr(self, lag=1):

    """
    Lag-N autocorrelation

    Parameters
    ----------
    lag : int, default 1
        Number of lags to apply before performing autocorrelation.

    Returns
    -------
    autocorr : float
    """
    return self.corr(self.shift(lag))

def crosscorr(datax, datay, lag=0):

    """ Lag-N cross correlation.
    Parameters
    ----------
    lag : int, default 0
    datax, datay : pandas.Series objects of equal length

    Returns
    ----------
    crosscorr : float
    """
    return datax.corr(datay.shift(lag))


def bacteriaExtract(df, BacteriaId):

    dout = pd.DataFrame()

    for i in range (0, df.shape[0]):
        if df['BacteriaId'].iloc[i] == BacteriaId:
            #print df.iloc[i]
            dout = dout.append(df.iloc[i], ignore_index=True)

    return dout

def whatIsBacId(df):

    bacteriaId = df['BacteriaId'].iloc[0]

    return bacteriaId

def dc_blocker(x, r=0.9):

    """ Lag-N cross correlation.
       Parameters
       ----------
       x : pandas.Series objects

       Returns
       ----------
       y : pandas.Series objects
       """
    y = []
    y.append(0)
    for j in range(1, x.shape[0]):
        y.append(x.iloc[j] - x.iloc[j-1] + r*y[j-1])
    # for n in range(1, x.shape[0]):
    #     y[n] = x[n] - x[n - 1] + r * y[n - 1]
    return y

data = 'data/Ecoli-13C-TEC-Dual.MOV.csv'
frame_rate = 30

output = 'data/ecoli_perb/'
df = pd.read_csv(data)
bacteriaName = 'ecoli'

for i in range (0, 20):
    bacteriaId = i
    print("analysing bacteria %i..." % i)
    dX = []
    dY = []
    ddX = []
    ddY = []
    dis = []
    phase = []
    velocity = []
    # dT = 1 / float(frame_rate)
    dT = 0.033
    size = df.shape[0]
    chunk = 300
    arraySize = (size - size%chunk) / chunk

    dnew = bacteriaExtract(df, bacteriaId)

    dnew['oldX'] = dnew['X']
    dnew['oldY'] = dnew['Y']
    dnew['X'] = dc_blocker(dnew['X'])
    dnew['Y'] = dc_blocker(dnew['Y'])

    '''L2 Euclidean Distance Lag = 1'''

    for i in range(0, dnew.shape[0]):
        if i == 0:
            dX.append(0)
            dY.append(0)
        else:
            dX.append(dnew['X'].iloc[i] - dnew['X'].iloc[i - 1])
            dY.append(dnew['Y'].iloc[i] - dnew['Y'].iloc[i - 1])


    '''Displacement Vector'''

    dnew['dX'] = dX
    dnew['dY'] = dY

    chunker = np.array_split(dnew[:(dnew.shape[0] - dnew.shape[0]%chunk)], (dnew.shape[0] - dnew.shape[0]%chunk)/chunk)

    # print(chunker[1])

    for j in range(0, len(chunker)):
        fileOutput = '%s%s_%i_%i.csv' % (output, bacteriaName, bacteriaId, j)
        np.savetxt(fileOutput, chunker[j], fmt='%.3f', delimiter=',', header="BacteriaId,Frame,Time,X,Y,oldX,oldY,dX,dY")

    print("complete bacteria %i..." % bacteriaId)
