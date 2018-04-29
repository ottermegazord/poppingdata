
import pandas as pd
import matplotlib.pyplot as plt
import math
from pandas.plotting import autocorrelation_plot

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

data = 'data/ecoli1.csv'
frame_rate = 30


df = pd.read_csv(data)
dX = []
dY = []
ddX = []
ddY = []
dis = []
phase = []
velocity = []
dT = 1 / float(frame_rate)


'''L2 Euclidean Distance Lag = 1'''

for i in range(0, df.shape[0]):
    if i == 0:
        dX.append(0)
        dY.append(0)
    else:
        dX.append(df['X'].iloc[i] - df['X'].iloc[i - 1])
        dY.append(df['Y'].iloc[i] - df['Y'].iloc[i - 1])


'''Displacement Vector'''

df['dX'] = dX
df['dY'] = dY


for i in range(0, df.shape[0]):
    x = float((df['dX'][i]))
    y = float((df['dY'][i]))
    dis.append(math.sqrt(x**2 + y**2))
    phase.append(math.atan2(y,x)/math.pi*180 + 180)


df['dis'] = dis
df['phase'] = phase

# xVelocity = []
# yVelocity = []
# velocity = []
# angVel = []

# for i in range(0, df.shape[0]):
#     if i == 0:
#         velocity.append(0)
#         # xVelocity.append(0)
#         # yVelocity.append(0)
#         angVel.append(0)
#     else:
#         velocity.append((df['dis'].iloc[i] - df['dis'].iloc[i - 1]) / dT)
#         # xVelocity.append((df['dX'].iloc[i] - df['dX'].iloc[i - 1]) / dT)
#         # yVelocity.append((df['dY'].iloc[i] - df['dY'].iloc[i - 1]) / dT)
#         angVel.append((df['phase'].iloc[i] - df['phase'].iloc[i - 1]) / dT)

df['velocity'] = df['dis'] / dT
df['phaseRad'] = (df['phase'] * math.pi) / float(180)
# df['xVelocity'] = xVelocity
# df['yVelocity'] = yVelocity

# df['angVel'] = angVel


xcov = [crosscorr(df['dis'], df['phase'], lag=i) for i in range(df.shape[0])]



print("Average autocorrelation of Displacement: %.9f" % df['dis'].autocorr(lag=1))
print("Average autocorrelation of Phase: %.9f" % df['phase'].autocorr(lag=1))
print("Average crossocorrelation between Displacement and Phase: %.9f" % crosscorr(df['dis'],df['phase'],lag=0))




plt.figure()
plt.suptitle('Displacement against time', fontsize=13)
plt.plot(df['Time'], df['dis'])
plt.savefig('graphs/dis.png')
plt.show()

plt.figure()
plt.suptitle('Phase(Degrees) against time', fontsize=13)
plt.plot(df['Time'], df['phase'])
plt.savefig('graphs/phase.png')
plt.show()

# plt.figure()
# plt.suptitle('xVelocity', fontsize=13)
# plt.plot(df['Time'], df['xVelocity'])
# plt.savefig('graphs/xVelocity.png')
# plt.show()
#
# plt.figure()
# plt.suptitle('yVelocity', fontsize=13)
# plt.plot(df['Time'], df['yVelocity'])
# plt.savefig('graphs/yVelocity.png')
# plt.show()


plt.figure()
plt.suptitle('Absolute velocity against time', fontsize=13)
plt.plot(df['Time'], df['velocity'])
plt.savefig('graphs/velocity.png')
plt.show()

# plt.figure()
# plt.suptitle('Angular Velocity', fontsize=13)
# plt.plot(df['Time'], df['angVel'])
# plt.savefig('graphs/angDis.png')
# plt.show()


plt.figure()
plt.suptitle('Autocorrelation: Displacement', fontsize=13)
autocorrelation_plot(df['dis'])
plt.savefig('graphs/auto_dis.png')
plt.show()

plt.figure()
plt.suptitle('Autocorrelation: Absolute Velocity', fontsize=13)
autocorrelation_plot(df['velocity'])
plt.savefig('graphs/auto_vel.png')
plt.show()

plt.figure()
plt.suptitle('Autocorrelation: Phase', fontsize=13)
autocorrelation_plot(df['phase'])
plt.savefig('graphs/auto_phase.png')
plt.show()

plt.figure()
plt.suptitle('Cross Correlation between Absolute Velocity and Phase', fontsize=13)
plt.plot(xcov)
plt.savefig('graphs/xcov.png')
plt.show()


# plt.figure()
# plt.suptitle('Autocorrelation: Angular Velocity', fontsize=13)
# autocorrelation_plot(df['angVel'])
# plt.savefig('graphs/auto_angVel.png')
# plt.show()
#
# plt.figure()
# plt.suptitle('Autocorrelation: Radial Velocity', fontsize=13)
# autocorrelation_plot(df['velocity'])
# plt.savefig('graphs/auto_velocity.png')
# plt.show()
#
# plt.figure()
# plt.suptitle('Cross Correlation between Angular Velocity and Velocity', fontsize=13)
# plt.plot(xcov)
# plt.savefig('graphs/xcov2.png')
# plt.show()
