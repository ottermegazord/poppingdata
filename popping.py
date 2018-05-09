import pandas as pd
import matplotlib.pyplot as plt
import math
from pandas.plotting import autocorrelation_plot
import numpy as np
import pylab as pl
from scipy import signal
import scipy.fftpack

class Popping():

    def __init__(self, data, frameRate, bacteriaNumber):
        self.df = pd.read_csv(data)
        self.sampleFreq = 1 / float(frameRate)
        self.df = self.df.sort_values(by=['BacteriaId'])
        self.bacteriaNumber = bacteriaNumber

    def printer(self):
        # Outputs dataframe
        return self.df

    def whatisbacid(self):
        return self.df['BacteriaId'].iloc[0]

    def bacteriaExtract(self, BacteriaId):

        dout = pd.DataFrame()

        for i in range(0, self.df.shape[0]):
            if self.df['BacteriaId'].iloc[i] == BacteriaId:
                dout = dout.append(self.df.iloc[i], ignore_index=True)

        return dout


    def euclidianDistance(self, bacteriaNumber):
        dnew = self.bacteriaExtract(bacteriaNumber)
        dX = []
        dY = []
        for i in range(0, dnew.shape[0]):
            if i == 0:
                dX.append(0)
                dY.append(0)
            else:
                dX.append(dnew['X'].iloc[i] - dnew['X'].iloc[i - 1])
                dY.append(dnew['Y'].iloc[i] - dnew['Y'].iloc[i - 1])
        dnew['dX'] = dX
        dnew['dY'] = dY
        return dnew








