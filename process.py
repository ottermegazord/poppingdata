import popping as pop


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

data1 = 'data/Ecoli-HangingDrop-20x-Dilution-1.MOV-20Trackers.csv'
data2 = 'data/Rhodosprillum-HangingDrop-3ul.MOV-20Trackers.csv'
fileOutput = 'data/output.csv'
frameRate = 0.033
bacteriaNumber = 2

popping1 = pop.Popping(data1, frameRate, bacteriaNumber)
popping2 = pop.Popping(data2, frameRate, bacteriaNumber)

print popping1.euclidianDistance(1)


