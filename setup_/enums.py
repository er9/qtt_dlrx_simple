from enum import Enum, IntEnum

class DataType(Enum):
    MPS = 'MPS'
    MPX = 'MPX'
    MPO = 'MPO'
    TN3 = 'TN3'
    Num = 'Num'

class LayoutType(Enum):
    PARALLEL_GROUP = 'PG'  # 'parallelG'
    PARALLEL = 'PF'   # 'parallelF'
    SEQUENTIAL = 'SF' # 'sequentialF'
    COMB = 'COMB'
    COMB_PF = 'CPF' # X: PF, V: PF as comb
    COMB_PG = 'CPG'

class AxisMapType(Enum):
    BINARY = '0'
    FLIPPED = '1'
    SIGNED = '2'

class CompressType(Enum):
    SVD = 'svd'
    DMRG = 'dmrg'
    MG = 'mg'

class BCType(IntEnum):
    ANTIPERIODIC = -1
    PERIODIC = 1
    OPEN = 0
    ZEROGRADIENT = 3
    ZEROVALUE = -3
    ABSORBING = -20
    REFLECTING = -21
    SYMMETRIC = 2
    ANTISYMMETRIC = -2
    DIRICHLET = -4
    NEUMANN = 4

class FDType(Enum):
    CENTER = 'center'
    FORWARD = 'forward'
    BACKWARD = 'backward'
    FVM = 'fvm'

class BasisType(Enum):
    SPATIAL = 'RealSpace'
    FOURIER = 'Fourier'
    HERMITE = 'Hermite'

class CoordinateSystemType(Enum):
    CARTESIAN = 'cartesian'
    CYLINDRICAL = 'cylindrical'
    SPHERICAL = 'spherical'

class CoordinateType(IntEnum):    # is recognized as an int
    X = 0
    Y = 1
    Z = 2
    R = 0
    THETA = 1
    PHI = 2
    SCALAR = -1

class CollisionType(Enum):
    LB = 'LB' # 'Lenard-Bernstein'
    H6 = 'H6' # 'hyper6'
    H2 = 'H2'
    H4 = 'H4'

class LocalSolverType(Enum):
    TDDMRG = 'tdDMRG'
    TDCross = 'TDCross'
    TDVP = 'TDVP'
    MIXED = 'MIXED'
    DMRG = 'DMRG'
    LINSOLVE = 'GMRES'
    Cross = 'Cross'

def parse_BCType(bc: BCType or str) -> BCType:
    if isinstance(bc, BCType):
        return bc
    else:
        if bc in ['zg', 'ZG', 'zero gradient']:
            return BCType.ZEROGRADIENT
        if bc in ['zero value', 'zero', 'z', 'Z']:
            return BCType.ZEROVALUE
        if bc in ['open','o','O']:
            return BCType.OPEN
        if bc in ['p','periodic','P']:
            return BCType.PERIODIC
        if bc in ['ap','antiperiodic','AP']:
            return BCType.ANTIPERIODIC
        # if bc in ['r','reflecting','R']:
        #     return BCType.REFLECTING
        if bc in ['s','symmetric','S']:
            return BCType.SYMMETRIC
        if bc in ['as','antisymmetric','AS']:
            return BCType.ANTISYMMETRIC
        if bc in ['a','absorbing','A']:
            return BCType.ABSORBING

def parse_FDType(fd_type: FDType or str) -> FDType:
    if isinstance(fd_type, FDType):
        return fd_type
    else:
        if fd_type in ['center', 'C', 'c']:
            return FDType.CENTER
        if fd_type in ['forward','F','f']:
            return FDType.FORWARD
        if fd_type in ['backward','B','b']:
            return FDType.BACKWARD


# def get_axis_map(map_type: AxisMapType) -> 'AxisMap':
#     if map_type is AxisMapType.BINARY:
#         return BinaryMap()
#     elif map_type is AxisMapType.FLIPPED:
#         return FlipBinaryMap()
#     elif map_type is AxisMapType.SIGNED:
#         return MirrorMap()

