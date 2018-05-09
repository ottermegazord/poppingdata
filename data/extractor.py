
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

data = 'data/Ecoli-HangingDrop-20x-Dilution-1.MOV-20Trackers.csv'
frame_rate = 30

output = 'data/ecoli/output.csv'
df = pd.read_csv(data)
bacteriaId = 1
dX = []
dY = []
ddX = []
ddY = []
dis = []
phase = []
velocity = []
# dT = 1 / float(frame_rate)
dT = 0.033


dnew = bacteriaExtract(df, bacteriaId)

print(dnew.shape[0])
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

print dnew.to_csv(output, sep=',', index=False, header=True)



# dXnp = np.asarray(dnew['dX'])
# dYnp = np.asarray(dnew['dY'])
#
# print(dXnp)
#
# yf = scipy.fftpack.fft(dYnp)
# xf = np.linspace(0.0, 1.0/(2.0*dT), dnew.shape[0]/2)
#
# fig, ax = plt.subplots()
# ax.plot(xf, 2.0/dnew.shape[0] * np.abs(yf[:dnew.shape[0]//2]))
# plt.show()
#
#
# ps = np.abs(np.fft.fft(dXnp))**2
# freqs = np.fft.fftfreq(dXnp.size, dT)
# idx = np.argsort(freqs)
#
# dYnp = np.asarray(dY)
# psY = np.abs(np.fft.fft(dYnp))**2
# freqsY = np.fft.fftfreq(dYnp.size, dT)
# idY = np.argsort(freqsY)
#
#
# plt.figure()
# plt.suptitle('FFT Cartesian', fontsize=13)
# plt.plot(freqsY[idY], psY[idY], label='y')
# plt.plot(freqs[idx], ps[idx], label='x')
# plt.legend()
# plt.show()
#
# # print(dX - dnew['dX'])
#
# # dnew['newX'] = dc_blocker(dnew['X'])
# # dnew['newY'] = dc_blocker(dnew['Y'])
#
# # print(dnew['dX'] - dX)
#
# # print(dc_blocker(dnew['dX']))
#
#
# for i in range(0, dnew.shape[0]):
#     x = float((dnew['dX'][i]))
#     y = float((dnew['dY'][i]))
#     dis.append(math.sqrt(x**2 + y**2))
#     phase.append(math.atan2(y,x)/math.pi*180 + 180)
#
#
# dnew['dis'] = dis
# dnew['phase'] = phase
#
# # xVelocity = []
# # yVelocity = []
# # velocity = []
# # angVel = []
#
# # for i in range(0, dnew.shape[0]):
# #     if i == 0:
# #         velocity.append(0)
# #         # xVelocity.append(0)
# #         # yVelocity.append(0)
# #         angVel.append(0)
# #     else:
# #         velocity.append((dnew['dis'].iloc[i] - dnew['dis'].iloc[i - 1]) / dT)
# #         # xVelocity.append((dnew['dX'].iloc[i] - dnew['dX'].iloc[i - 1]) / dT)
# #         # yVelocity.append((dnew['dY'].iloc[i] - dnew['dY'].iloc[i - 1]) / dT)
# #         angVel.append((dnew['phase'].iloc[i] - dnew['phase'].iloc[i - 1]) / dT)
#
# dnew['velocity'] = dnew['dis'] / dT
# dnew['phaseRad'] = (dnew['phase'] * math.pi) / float(180)
# # dnew['xVelocity'] = xVelocity
# # dnew['yVelocity'] = yVelocity
#
# # dnew['angVel'] = angVel
#
#
#
# dVelnp = np.asarray(dnew['velocity'])
# dPhasenp = np.asarray(dnew['phaseRad'])
#
# ps = np.abs(np.fft.fft(dVelnp))**2
# freqs = np.fft.fftfreq(dVelnp.size, dT)
# idVel = np.argsort(freqs)
#
# psPhase = np.abs(np.fft.fft(dPhasenp))**2
# freqs = np.fft.fftfreq(dPhasenp.size, dT)
# idPhase = np.argsort(freqs)
#
#
# plt.figure()
# plt.suptitle('FFT Polar', fontsize=13)
# plt.plot(freqsY[idPhase], psY[idPhase], label='phase')
# #plt.plot(freqs[idVel], ps[idVel], label='velocity')
# plt.legend()
# plt.savefig('graphs/fftpolar.png')
# plt.show()
#
# #
# xcov = [crosscorr(dnew['dY'], dnew['dY'], lag=i) for i in range(dnew.shape[0])]
#
#
# print("Average autocorrelation of X: %.9f" % dnew['dX'].autocorr(lag=1))
# print("Average autocorrelation of Y: %.9f" % dnew['dY'].autocorr(lag=1))
# print("Average autocorrelation of Displacement: %.9f" % dnew['dis'].autocorr(lag=1))
# print("Average autocorrelation of Phase: %.9f" % dnew['phase'].autocorr(lag=1))
# # print("Average crossocorrelation between Displacement and Phase: %.9f" % crosscorr(dnew['dis'],dnew['phase'],lag=0))
#
#
# plt.figure()
# plt.suptitle('Bacteria Tracking', fontsize=20)
# plt.plot(dnew['oldX'], dnew['oldY'], label='With Drift')
# plt.plot(dnew['X'], dnew['Y'], label='Without Drift')
# plt.legend()
# plt.savefig('graphs/XY.png')
# plt.show()
#
#
# plt.figure()
# plt.suptitle('Displacement against time', fontsize=13)
# plt.plot(dnew['Time'], dnew['dis'])
# plt.savefig('graphs/dis.png')
# plt.show()
#
# plt.figure()
# plt.suptitle('Phase(Degrees) against time', fontsize=13)
# plt.plot(dnew['Time'], dnew['phase'])
# plt.savefig('graphs/phase.png')
# plt.show()
#
# # plt.figure()
# # plt.suptitle('xVelocity', fontsize=13)
# # plt.plot(dnew['Time'], dnew['xVelocity'])
# # plt.savefig('graphs/xVelocity.png')
# # plt.show()
# #
# # plt.figure()
# # plt.suptitle('yVelocity', fontsize=13)
# # plt.plot(dnew['Time'], dnew['yVelocity'])
# # plt.savefig('graphs/yVelocity.png')
# # plt.show()
#
#
# plt.figure()
# plt.suptitle('Absolute velocity against time', fontsize=13)
# plt.plot(dnew['Time'], dnew['velocity'])
# plt.savefig('graphs/velocity.png')
# plt.show()
#
# # plt.figure()
# # plt.suptitle('Angular Velocity', fontsize=13)
# # plt.plot(dnew['Time'], dnew['angVel'])
# # plt.savefig('graphs/angDis.png')
# # plt.show()
#
#
# plt.figure()
# plt.suptitle('Autocorrelation: X and Y without drift', fontsize=13)
# #autocorrelation_plot(dnew['dis'])
# autocorrelation_plot(dnew['dX'], label='X direction')
# autocorrelation_plot(dnew['dY'], label='Y direction')
# plt.legend()
# plt.savefig('graphs/auto_dis.png')
# plt.show()
#
# plt.figure()
# plt.suptitle('Autocorrelation: Absolute Velocity', fontsize=13)
# autocorrelation_plot(dnew['velocity'])
# plt.savefig('graphs/auto_vel.png')
# plt.show()
#
# plt.figure()
# plt.suptitle('Autocorrelation: Phase', fontsize=13)
# autocorrelation_plot(dnew['phase'])
# plt.savefig('graphs/auto_phase.png')
# plt.show()
#
# plt.figure()
# plt.suptitle('Cross Correlation between Absolute Velocity and Phase', fontsize=13)
# plt.plot(xcov)
# plt.savefig('graphs/xcov.png')
# plt.show()
#
#
# # plt.figure()
# # plt.suptitle('Autocorrelation: Angular Velocity', fontsize=13)
# # autocorrelation_plot(df['angVel'])
# # plt.savefig('graphs/auto_angVel.png')
# # plt.show()
# #
# # plt.figure()
# # plt.suptitle('Autocorrelation: Radial Velocity', fontsize=13)
# # autocorrelation_plot(df['velocity'])
# # plt.savefig('graphs/auto_velocity.png')
# # plt.show()
# #
# # plt.figure()
# # plt.suptitle('Cross Correlation between Angular Velocity and Velocity', fontsize=13)
# # plt.plot(xcov)
# # plt.savefig('graphs/xcov2.png')
# # plt.show()
