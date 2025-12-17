from   matplotlib.ticker import FormatStrFormatter
from   matplotlib import pyplot as plt
from   matplotlib import rc
from   matplotlib.lines import Line2D
import matplotlib           as mpl
import pylab                as pl
import matplotlib.colors    as col
from   mpl_toolkits.mplot3d import Axes3D

import matplotlib.patches as mpatches
import matplotlib.lines   as mlines



rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
mpl.rcParams['lines.linewidth'] = 2.
mpl.rcParams['font.size'] = 14.
mpl.rcParams['axes.labelsize'] = 16.
mpl.rcParams['axes.titlesize'] = 18.
mpl.rcParams['lines.markersize'] = 6.
mpl.rcParams['lines.markeredgewidth'] = 2.

mpl.rcParams['lines.linewidth'] = 3.5
mpl.rcParams['font.size'] = 18.
mpl.rcParams['axes.labelsize'] = 20.
mpl.rcParams['axes.titlesize'] = 22.
mpl.rcParams['lines.markersize'] = 10.
mpl.rcParams['lines.markeredgewidth'] = 2.5


# mpl.rcParams['xtick.labelsize'] = 14.
# mpl.rcParams['ytick.labelsize'] = 14.
# mpl.rcParams['legend.fontsize'] = 14.

xfig = 8
yfig = 0.65*xfig
leg_fs = 12.


lstyles = [':','--','-','_.']
colors = ['#99CC00','#45bfb6','#f43e1a','#a93f95','#0143ad']
cols11 = ['#AB2014','#EC9328','#ECB928','#B8DC34','#46B628','#28B698','#2896B6','#2873B6',\
          '#0D500A','#0A747C','#2833B6','#180A7C','#7C28B6','#5F1257','#370C32']
marks  = ['+','x','2','o','s','^','s','*']
