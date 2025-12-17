Quickstart:
    - see tests/test_vlasov-damp-final.py for reference on how to run the code
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




Structure:
setup/
    enums.py:   contains Enums or types for various classes

    configs.py:  contains Configuration classes
        MaterialConfiguration:  define plasma parameters here
        CompressionConfiguration:  define GridTN compression levels here
        DerivativeConfiguration:  specify boundary conditions
                                  finite difference type (FORWARD, BACKWARD, CENTER)
                                  finite difference order



