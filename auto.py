import pandas as pd
import matplotlib.pyplot as plt
import math
from pandas.plotting import autocorrelation_plot
import numpy as np
import pylab as pl
from scipy import signal
import scipy.fftpack

def acf(series):
    n = len(series)
    data = np.asarray(series)
    mean = np.mean(data)
    c0 = np.sum((data - mean) ** 2) / float(n)

    def r(h):
        acf_lag = ((data[:n - h] - mean) * (data[h:] - mean)).sum() / float(n) / c0
        return round(acf_lag, 3)
    x = np.arange(n) # Avoiding lag 0 calculation
    acf_coeffs = map(r, x)
    return acf_coeffs

OUTPUT_DIR = 'data/ecoli_perb_aut/'
INPUT_DIR ='data/ecoli_perb/'
bacteriaName = 'ecoli'
bacteriaNumber = 20
chunks = 10

for i in range(0, bacteriaNumber):
    print "start bacteria %i" % i
    for j in range(0, chunks):
        INPUT_PATH = '%s%s_%i_%i.csv' %(INPUT_DIR, bacteriaName, i, j)
        OUTPUT_PATH = '%s%s_%i_%i.csv' % (OUTPUT_DIR, bacteriaName, i, j)
        df = pd.read_csv(INPUT_PATH)
        df['autodX'] = acf(df['dX'])
        df['autodY'] = acf(df['dY'])
        np.savetxt(OUTPUT_PATH, df, fmt='%.3f', delimiter=',', header="BacteriaId,Frame,Time,X,Y,oldX,oldY,dX,dY,autoX,autoY")

