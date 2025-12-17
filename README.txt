Quickstart:
    - different branches for different tests:
        - EM2D:  2D Maxwell's simulation of wavepacket in cavity
            - tests/test_EM2.py
        - advec:  advection test problem in Fourier space
            - tests/test_vlasovEM_test_nox-k.py
        - burgers: upwind Burger's test case
            - tests/test_burgers_1D.py
    - general operations (addition, element-wise multiplication, function evaluations, etc)
        - see test_xfunc_v3_mixed.py


Running the code:  see any of the tests files for reference on how to run the code
    define Coordinates for the system
        Coordinate(name, CoordinateType)  import from coord/coord_sys.py
    optional: define CoordinateSystem (eg. CartesianCoordinateSpace)
    define Axis objects -- a frozen object so essentially immutable
        Axis(L,q,coordinate,**kwargs)
        specifies grid discretization and basis type (SPATIAL for now)
    define Grid
        probably a Grid1D object
        Grid(id, axes, layout_type)
            axes: tuple of Axis objects. order matters
            layout_type: LayoutType.SEQUENTIAL, PARALLEL, PARALLEL_GROUP
                defined in setup/enums.py
    define configurations (import from setup/configs.py)
        MaterialConfiguration:  define plasma parameters here
        CompressionConfiguration:  define GridTN compression levels here
        DerivativeConfiguration:  specify boundary conditions
                                  finite difference type (FORWARD, BACKWARD, CENTER)
                                  finite difference order
    define GridTN1D objects (matrix product states):
        GridTN1D(grid, data [qtn.MatrixProductState],
                 ax_deriv_configs: {dict['Axis', 'DerivativeConfiguration'])
        can also obtain from Grid object:
            grid_obj.map_state_to_mps(data, ax_deriv_configs=..., **kwargs)
            grid_obj.make_mps_ndim(dict['Axis': 'qtn.MatrixProductState'])
                (but as of now would need to update ax_deriv_configs separately)
    define ScalarField, Field objects (in field.py)
        ScalarField(name, grid, data: 'GridTN'=None, **kwargs)
        Field(name, grid, data: dict['Coordinate','GridTN'] = None, **kwargs)
            note: components of a vector field are specified by Coordinates
                  Coordinates are commonly associated with each Axis
                  in other words, all fields will have components indexed by Coordinates
                  specified by the Axis objects in the Grid. Fields can have components
                  not included in the Grid.  (e.g. Ez on an (x,y) grid)
            note: all GridTNs making up a field must all live on the same grid.
            note: we use a CompressionConfiguration to specify compression parameters
                  ('max_bond'=None,'cutoff'=1.0e-20,'cutoff_mode'='rsum2')
                  for different 'levels' of compression. (Levels mean that in we can
                  choose different compression parameters for different parts of the
                  algorithm. Right now can probably specify level=1 and leave the rest
                  as defaults.)
    define the PDE system. see details in each class




Structure:  (not all files may be relevant)
setup_/
    enums.py:   contains Enums or types for various classes

    configs.py:  contains Configuration classes
        MaterialConfiguration:  define plasma parameters here
        CompressionConfiguration:  define GridTN compression levels here
        DerivativeConfiguration:  specify boundary conditions
                                  finite difference type (FORWARD, BACKWARD, CENTER)
                                  finite difference order

    defaults.py:  typing, default global values, reading input flags
    helper.py:  common functions for initializing tset files.
    paths.py:  define where to save files
    quimb_TN1D.py:  generalization of quimb class for 1-D tensor network with more than 2 physical indices per tensor core

axis_map/
    Define different quantization mappings. default is map_binary.
    binary:  coarse to fine
    flipbinary:  fine to coarse
    mirror:  coarse to fine, but mirror the coarsest cell (see Ripoll's QTT paper)
    flipmirror:  mirror but with the tensor cores reversed.

basis/
    Different basis functions. default is spatial
    Hermite (symmetric, asymmetric)
    k (Fourier)
    Spatial (real-space)
    finite difference coefficients for spatial grid

coord/
    Different coordinate systems. only Cartesian is fully implemented

layout/
    Different QTT layouts for multi-dimensional systems.
    Default is sequential
    ParallelF:  interleaved ordering with tensors corresponding to different dimensions remaining factorized
    ParallelG:  interleaved ordering, but tensors corresponding to different dimensions are contracted with each other

tests/
    various test files.

axis.py:  Axis object defining grid points or spectral modes
field.py:  Field object (vector field) and ScalarField object (scalar field)
grid.py:  Grid object. collection of Axis objects to define n-dimensional grid.
    grid1D.py:  1D QTT Grid object
    grids_composite.py:    parent class for grid_comb.py
        grid_comb.py:  QTT Grid object for comb (tree-like) QTT layout
gridTN.py:  Parent class:  Tensor network object with a Grid object
    gridTN_1D.py:  1D QTT or MPS / MPO with Grid1D object
    gridTN_composite.py:  parent class for gridTN_1Dcomb
        gridTN_1Dcomb.py:  QTN with comb layout with GridsComb object

helper_block_dmrg.py:  helper class for solving blocked linear equations using DMRG (not fully tested)
helper_block_tddmrg_3.py:  helper class for solving TD-DMRG / alternating projection DLR for blocked equations
helper_block_tdvp_3.py: helper class for solving TDVP / projector splitting DLR for blocked equations
helper_dlr.py:  standard DLR (outdated)
helper_dmrg.py:  helper class for solving (not-blocked) linear equations using DMRG (not fully tested?)
helper_dmrg_2.py:  Same as above but with some algorithmic modifications / potential fixes? seemingly equivalent.
helper_quimb.py:  helper functions for quimb objects
helper_sl.py: helper functions for SL time integration (not relevant for these tests)
helper_tdvp.py:  helper functions for TDVP / projector splitting DLR for non-blocked equations
helper_tdvp_v2.py:  updated version of above
helper_TE.py:  time evolution subroutines

pde_system.py: PDE object. used to call time stepping
    pde_boltzmann.py:  PDE object for Boltzmann's equations
    pde_burgers.py:  PDE object for Burger's equations
    pde_EM.py: PDE object for Maxwell's equations
    pde_vlasov.py: PDE object for Vlasov equations
        pde_vlasovEM.py:  Vlasov-Maxwell's equations
        pde_vlasovES.py:  Vlasov-Poisson equations


