"""Global defaults and shared type aliases for the codebase: numeric/tensor-network
type definitions, default compression cutoffs and bond dimensions, default boundary
conditions and finite-difference settings, plus command-line input-flag parsing."""
from typing import Union, Optional, Sequence, Any, Iterable, Type, Callable, Literal
from setup_.enums import *
import sys, getopt, os
import importlib.util
import pdb

import numpy as np
import scipy.sparse

import matplotlib

def _configure_matplotlib_backend():
    # Respect user choice if they explicitly set a backend.
    if os.environ.get("MPLBACKEND"):
        return
    # If tkinter isn't available (common for minimal Python installs), avoid
    # defaulting to Tk-based interactive backends.
    if importlib.util.find_spec("tkinter") is None:
        matplotlib.use("Agg")

_configure_matplotlib_backend()
import matplotlib.pyplot as plt


import quimb.tensor as qtn
from setup_.quimb_TN1D import MatrixProductTensor

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from axis import Axis


Numeric = Union[int, float, complex, np.number]

MPSType = Optional[qtn.MatrixProductState]
MPOType = Optional[qtn.MatrixProductOperator]
MPTType = Optional[MatrixProductTensor]
TN1Type = Optional[qtn.TensorNetwork1D]
TNType = Union[Optional[qtn.TensorNetwork], TN1Type, MPOType, MPSType]
Iter = list or tuple # Union[list, tuple]

CUTOFF = 1.0e-28  # 1.0e-30
CUTOFF_MODE = 'rsum2'
MAXBOND = None
MINBOND = 4

DEFAULT_BC = BCType.SYMMETRIC   # ZEROGRADIENT
DEFAULT_ORDER = 1
DEFAULT_FDTYPE = FDType.CENTER

DEFAULT_SL_ORDER = 3

DEEP_GRID_CHECK = False     ## True for debugging


#### parameters that can be set by command line flags
flags = {}
def _set_args(argv):
    try:
        opts, args = getopt.getopt(argv,
                                   "hP:D:c:k:A:L:o:l:s:d:b:",     # help, MP, DMAX, cutoff, k, A, L, order, layout, sqrt, dir
                                   ["use_mp=",      # use multiprocessing versions of some functions
                                    "max_bond=",    # maximum TT rank
                                    "cutoff=",      # TT cutoff
                                    "norm_cutoff=", # TT norm cutoff
                                    "k=",           # perturbation wavevector
                                    "A=",           # perturbation strength
                                    "branch=",      # branch of wave being targeted
                                    "Lx=",          # QTT length in real space
                                    "Lve=",         # QTT length in velocity space for electrons
                                    "Lvi=",         # QTT length in velocity space for ions
                                    "layout=",      # layout string for QTT
                                    "order=",       # order of spatial derivatives
                                    "te_order=",    # time integration scheme for Vlasov/Boltzmann
                                    "te_order_EM=", # time integration scheme for Maxwell's eqn
                                    "dt=",          # time step (do_adapt_dt=False)
                                    "cfl=",         # cfl time step (do_adapt_dt=True)
                                    "T=",           # simulation run time
                                    "order=",       # order when computing spatial derivatives
                                    "sqrt",         # flag signaling to take sqrt of f
                                    "do_tt",        # flag signaling to do TT (not quantized) calc
                                    "do_full",      # flag signaling to do full calc (parallel_group TT)
                                    "no_save",      # save data
                                    "clear_checkpoints",    # flag signaling to delete checkpoints and restart from T=0
                                    "load_from_T="          # load data (specify T if not the current value of T)
                                    "restart_from_nt=",     # restart (specify nt if not the default (last available))
                                    "save_every_nt=",       # save_every_nt
                                    "fdir_iden=",           # fdir identifying string/number
                                    "no_evolve_ion",        # evolve_ion=False
                                    "dissipation=", # amount of dissipation (eta)
                                    "upwind",       # do upwind (if not specified by te_order)
                                    ] )
    except getopt.GetoptError:
        return

    print('in func')
    print('opts', opts, args)
    for opt, arg in opts:
        print('opt', opt, 'arg', arg)
        if opt == '-h':
            print('help')
            exit()
        elif opt in ('-P', '--use_mp'):
            flags['USE_MP'] = arg
        elif opt in ('-D', '--max_bond'):
            flags['DMAX'] = None if arg=='None' else int(arg)
        elif opt in ('-c', '--cutoff'):
            if arg in ['max', 'max2', 'max3', 'max4']:
                flags['cutoff'] = arg
            elif arg != 'None':
                flags['cutoff'] = float(arg)
        elif opt in ('--norm_cutoff',):
            if arg != 'None':
                flags['norm_cutoff'] = float(arg)
        elif opt in ('-k', '--k'):
            flags['k'] = float(arg)
        elif opt in ('-A', '--A'):
            flags['perturbation'] = float(arg)
        elif opt in ('-b', '--branch',):
            ## if multiple branches (for waves) are possible, specify which wave to look at
            ## e.g., for X wave, low freq or high freq
            ## e.g., for Whistler wave, left or right polarized
            flags['branch'] = int(arg)
        elif opt in ('-L',):
            flags['Lx'] = int(arg)
            flags['Lve'] = int(arg)
            flags['Lvi'] = int(arg)
        elif opt in ('--Lx',):
            flags['Lx'] = int(arg)
        elif opt in ('--Lve',):
            flags['Lve'] = int(arg)
        elif opt in ('--Lvi',):
            flags['Lvi'] = int(arg)
        elif opt in ('-o', '--order'):
            flags['order'] = int(arg)
        elif opt in ('--te_order',):
            flags['te_order'] = int(arg)
        elif opt in ('--te_order_EM',):
            flags['te_order_EM'] = int(arg)
        elif opt in ('--cfl',):
            flags['adapt_dt'] = True
            flags['cfl'] = float(arg)
        elif opt in ('--dt',):
            flags['adapt_dt'] = False
            flags['dt'] = float(arg)
        elif opt in ('--dissipation',):
            if arg in ['max']:
                flags['dissipation'] = arg
            else:
                flags['dissipation'] = float(arg)
        elif opt in ('-s', '--sqrt'):
            # assert (isinstance(arg, bool), f'is_sqrt argument should be bool, not {type(arg)}')
            flags['is_sqrt'] = True
        elif opt in ('--do_tt',):
            ## TT but not quantized
            flags['do_tt'] = True
        elif opt in ('--do_full',):
            ## no TT decomposition
            flags['do_full'] = True
        elif opt in ('--upwind',):
            flags['do_upwind'] = True
        elif opt in ('--no_evolve_ion',):
            flags['evolve_ion'] = False
        elif opt in ('--no_save',):
            flags['save_data'] = False
        elif opt in ('--load_from_T',):
            flags['load_data'] = int(arg)
        elif opt in ('--clear_checkpoints',):
            flags['restart'] = False
            flags['load'] = False
        elif opt in ('--restart_from_nt',):
            flags['restart'] = int(arg)
        elif opt in ('-d','--fdir_iden',):
            ## argument to distinguish folders when saving data
            flags['dnum'] = arg
        elif opt in ('--save_every_nt',):
            flags['save_every_nt'] = int(arg)
        elif opt in ('--layout', '-l'):
            ## layout string describing the TN geometry
            ## e.g., COMBBB, SFFB, PGFF, PFFF
            x_map = arg[-2]
            v_map = arg[-1]
            layout = arg[:-2]
            if layout in ['COMB', 'C']:
                layout = LayoutType.COMB
            elif layout in ['COMB_PF', 'CPF']:
                layout = LayoutType.COMB_PF
            elif layout in ['COMB_PG', 'CPG']:
                layout = LayoutType.COMB_PG
            elif layout in ['SEQ', 'SF', 'S']:
                layout = LayoutType.SEQUENTIAL
            elif layout in ['PAR', 'PF', 'P']:
                layout = LayoutType.PARALLEL
            elif layout in ['PG']:
                layout = LayoutType.PARALLEL_GROUP
            else:
                raise ValueError(f'not a valid layout string {layout}')
            flags['layout'] = layout
            flags['x_map_key'] = x_map
            flags['v_map_key'] = v_map
    return

init_args = sys.argv
print('init args', init_args)
if len(init_args) > 1:
    _set_args(init_args[1:])

USE_MP = flags.get('USE_MP', False)

