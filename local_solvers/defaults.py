from enum import Enum, IntEnum


class SweepDirection(IntEnum):
    ## int denotes where the MPS needs to be canonicalized to
    LEFT = -1
    RIGHT = 1

class SolveMethod(Enum):
    CGD = 'CGD'  # local CGD using quimb tensors
    CGDx = 'CGDx'  # local CGD using numpy
    LSQ = 'LSQ'  # least squares regression


DEFAULT_MAX_ITER = 100 # 20  # 10
DEFAULT_MAX_TOT_ITER = 50
DEFAULT_CONV_TOL = 1.0e-6
DEFAULT_SOLVE = SolveMethod.CGD
DEFAULT_MAX_WRONG_ITER = 10