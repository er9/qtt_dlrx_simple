import numpy as np

from setup_.configs import *
from coord.coord_sys import Coordinate
from gridTN import GridTN

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from axis import Axis
    from grid import Grid
    from coord.coord_sys import CoordinateSystem

    Field_Component_Type = Union[ 'GridTN', 'TNType', 'Numeric', 'np.ndarray']

""" class definition of Field object
    can be scalar field (only 1 component) or vector field (up to ndim components)
    define a coordinate space for spatial axes here
    te_compress_opts:  compression options for compressing at each time step
"""


# def set_io_type(func):
#     """ decorator for differential operators gradient, laplacian, etc.
#         if comps=None or 'all':  return Field object with components set by results from func
#         if comps is a list:      return result for all components from func as a list
#         if comps is an int:      return result for single component from func
#
#         if output of func is a dict, convert to a field object
#     """
#     def wrapper(self,*args,comps=None,**kwargs):
#
#         if isinstance(comps,int):
#             comps = [comps]
#             return_type = 'single'
#         elif comps is None or comps=='all':
#             comps = list(self.components.keys())
#             return_type = 'field'
#         elif isinstance(comps,list) or isinstance(comps,tuple):
#             return_type = 'list'
#         else:
#             print("comps needs to be an int, list or tuple, or None or 'all'")
#             raise(TypeError)
#
#         out = func(self,*args,comps=comps,**kwargs)   ## out is originally defined to be a list
#
#         if out is None:                    return out
#
#         elif isinstance(out, qtn.MatrixProductState):
#             if   return_type == 'single':  return out
#             elif return_type == 'list':    return [out]
#             else:                          return self.create_like({0: out})
#
#         elif isinstance(out[0], dict):  # has multiple components, (ie. grad); convert to Field objects
#             new_fields = []
#             for i in range(len(comps)):
#                 new_fields += [self.create_like(out[i])]
#
#             if return_type == 'single':    return new_fields[0]
#             else:                          return new_fields
#
#         elif isinstance(out, dict):
#             if   return_type == 'single' or return_type == 'list':  return out
#             else:
#                 new_fields = {}
#                 for i in range(len(comps)):    new_fields[comps[i]] = out[i]
#                 return self.create_like(new_fields)
#
#         elif isinstance(out,list):
#             if   return_type == 'single':  return out[0]
#             elif return_type == 'list':    return out
#             elif return_type == 'field':
#                 new_field = self.create_like({comps[i]: out[i] for i in range(len(comps))})
#                 return new_field
#
#     return wrapper


class Field:

    def __init__(self, name: str, grid: 'Grid', data: dict['Coordinate', 'Field_Component_Type'],
                 compress_config: Optional['CompressionConfiguration'] = None,
                 # deriv_configs: Optional[dict]=None,
                 is_sqrt=False):  # , init_split_opts: dict = None):  # , compress_opts: Optional[dict]=None):
        """
        name (str):  name of the field
        ncomp (int): maximum number of components (eg. 1 if scalar field, ndim if vector field)
        grid:  Grid object on which components live
        data:  dictionary containing MPSs indexed by integer denoting which component of the field
               the MPS represents (eg. 0 if MPS is representing a scalar field)
               keys:  ax.axID if a vector field OR None if a scalar field
        boundary_conditions: nested dict.
                first level: dict of boundary conditions for each field component.
                             define for all possible components
                second level: dictionary of boundary conditions along each axis
                              periodic, antiperiodic, open (default), reflecting, zero gradient
                              (elements are tuples to specify left end/right end of axis)
        coords: list of coordinate systems (predefined)
                coordinate systems take dict
                coords_ax:  dictionary in which the coordinate position (BCType or int) keys the desired axis

        """
        self._name = name
        self.grid: 'Grid' = grid
        self.compress_config = compress_config if compress_config is not None else CompressionConfiguration()
        self._components: dict['Coordinate', 'Field_Component_Type'] = {}
        self.initialize_field(data)  # , init_split_opts)    # indexed by axIDs or None

        # self.coord_sys_all = set([ax._coord_sys for ax in self.grid.axes])

        ## probably should move to gridTN, if keeping?
        self.is_sqrt = is_sqrt

        # self.deriv_params = {compID: {ax.axID: {} for ax in self.grid.axes} for compID in self.componentIDs}
        # self._set_deriv_params(deriv_params)

        # self._compress_opts_all = {}
        # self.set_compress_opts(compress_opts)

    # @property
    # def grid(self):
    #     return self._components[next(iter(self._components))].grid

    def get_name(self):
        return self._name

    def set_name(self, new_name):
        # old_name = self._name
        self._name = new_name
        # for i, comps in self.components.items():
        #     if comps is None or comps.data is None:
        #         continue
        #     if isinstance(comps.data, qtn.MatrixProductState):
        #         comps.data.reindex_all(f'{self._name},comp{i}'+',i({})',inplace=True)
        #     elif isinstance(comps.data, qtn.MatrixProductOperator):
        #         comps.data.reindex_lower_sites(f'{self._name},comp{i}' + ',i({})', inplace=True)
        #         comps.data.reindex_upper_sites(f'{self._name},comp{i}' + ',o({})', inplace=True)
        #     comps.data.drop_tags((old_name,))
        #     comps.data.add_tag(self._name)

    name = property(fget=get_name, fset=set_name)

    def _get_component_names(self):
        return list(self._components.keys())

    componentIDs = property(fget=_get_component_names)

    def _get_ncomp(self):
        return len(self._components)

    ncomp = property(fget=_get_ncomp)

    # ## get and set compress_opts
    # def get_compress_opts(self,level=1):
    #     try:
    #         return self._compress_opts_all[level]
    #     except KeyError:
    #         self.set_compress_opts({},level)
    #         return self._compress_opts_all[level]
    #
    # def set_compress_opts(self,compress_opts,level=1):
    #     try:
    #         self._compress_opts_all[level].update(compress_opts)
    #     except KeyError:
    #         self._compress_opts_all[level] = compress_opts

    @property
    def components(self) -> dict['Coordinate', 'GridTN']:
        return self._components

    def _set_component(self, compID, new_comp: 'Field_Component_Type'):
        if new_comp is not None:
            assert (isinstance(compID, Coordinate)), \
                'componentID must be Coordinate object'
            assert (isinstance(new_comp, (GridTN, )) or isinstance(new_comp, (int, np.number, float, complex))), \
                'component must be a GridTN object or number'
            # assert (new_comp.grid == self.grid), \
            #     'component must live on Grid specified by self.grid'
            if not isinstance(new_comp, (int, np.number, float, complex)):
                if new_comp.grid != self.grid:
                    if set(new_comp.grid.axes).issubset(self.grid.axes):
                        new_comp = self.grid.pad_gtn_to_grid(new_comp)
                    else:
                        raise ValueError('component must live within Grid specified by self.grid')

        self._components[compID] = new_comp

    def _get_component(self, compID) -> 'GridTN':
        return self._components[compID]

    def __getitem__(self, compID) -> 'GridTN':
        """ return ith component of the field. for convenience
        """
        return self._get_component(compID)

    def __setitem__(self, compID, new_comp: 'GridTN'):
        """ set ith component of the field. for convenience
        """
        self._set_component(compID, new_comp)

    def save_data(self, fstr):
        for compID, gtn in self.components.items():
            if gtn is not None:
                gtn.save_data(fstr + str(compID))
            # print('saved data', fstr+str(comp))

    def reload_data(self, fstr, compIDs: list['Coordinate'] = None, comp_ax_deriv_configs=None, safe_pass=True):
        grid = self.grid
        return grid.load_field_data(fstr, self, compIDs=compIDs, comp_ax_deriv_configs=comp_ax_deriv_configs,
                                    safe_pass=safe_pass)

    @classmethod
    def load_data(cls, grid: 'Grid', fstr, field_name, compIDs: list['Coordinate'],
                  comp_ax_deriv_configs=None, compress_config=None):
        field_obj = cls(field_name, grid, data={compID: None for compID in compIDs}, compress_config=compress_config)
        return grid.load_field_data(fstr, field_obj, compIDs=compIDs, comp_ax_deriv_configs=comp_ax_deriv_configs)

    def replace_compIDs(self, compID_map: dict['Coordinate', 'Coordinate']):
        """ compID_map:  dict of old to new compIDs
        """
        new_components = {}
        for c_old, c_new in compID_map.items():
            try:
                new_components[c_new] = self._components[c_old]
            except KeyError:
                pass  # assume the component is None
                # new_components[c_new] = self.grid.make_empty_gridTN()
        self._components = new_components

    def create_like(self, new_components=None) -> 'Field':
        """ creates new field with the same parameters as self
        """
        if new_components is None:
            new_components = {c: None for c in self.componentIDs}
        else:
            for k, comp in new_components.items():
                if isinstance(comp, qtn.TensorNetwork):
                    new_components[k] = self.grid.make_gridTN(comp, ax_deriv_configs=self[k].ax_deriv_configs)

        new_field = self.__class__(self.name, self.grid, new_components, compress_config=self.compress_config,
                                   is_sqrt=self.is_sqrt)  # , init_split_opts=split_opts)
        # new_field._compress_opts_all = self._compress_opts_all
        return new_field

    def create_like_vector(self, new_components: Optional[dict['Coordinate', 'GridTN']] = None) -> 'Field':
        """ creates new field with the same parameters as self
        """
        if new_components is None:
            new_components = {c: None for c in self.componentIDs}
        else:
            for k, comp in new_components.items():
                if isinstance(comp, qtn.TensorNetwork):
                    new_components[k] = self.grid.make_gridTN(comp, ax_deriv_configs=self[k].ax_deriv_configs)

        new_field = Field(self.name, self.grid, new_components, compress_config=self.compress_config,
                          is_sqrt=self.is_sqrt)  # , init_split_opts=split_opts)
        return new_field

    def create_like_scalar(self, new_component: Optional['GridTN'] = None,
                           new_ax_deriv_configs: Optional[dict['Axis', 'DerivativeConfiguration']] = None) \
            -> 'ScalarField':
        """ creates new scalar field with the same parameters as self
        """
        if isinstance(new_component, qtn.TensorNetwork):
            if new_ax_deriv_configs is None:
                new_ax_deriv_configs = self._components[next(iter(self._components))].ax_deriv_configs
            new_component = self.grid.make_gridTN(new_component, new_ax_deriv_configs)

        new_field = ScalarField(self.name, self.grid, new_component, compress_config=self.compress_config,
                                is_sqrt=self.is_sqrt)  # , init_split_opts=split_opts)

        return new_field

    def copy(self, deep=True) -> 'Field':
        """ makes a copy of itself
        """
        new_comps = {}
        for k, val in self._components.items():
            try:
                new_comps[k] = val.copy(deep=deep)
            except(TypeError, AttributeError):
                new_comps[k] = val
        new_field = self.create_like(new_comps)
        return new_field

    def clear(self):
        self._components = {}

    def max_bond(self, compID: 'Coordinate') -> Optional[int]:
        """ get max_bond of mps for all components
        """
        comp = self._components.get(compID, None)
        if comp is None:
            return 0
        return comp.max_bond()

    def max_bonds(self, compIDs: Optional[list['Coordinate']] = None) -> Union[dict['Coordinate', Optional[int]], list]:
        """ get max_bond of mps for all components
        """
        if compIDs is None:
            compIDs = self.componentIDs
            max_bonds = {}
            for compID in compIDs:
                max_bonds[compID] = self.max_bond(compID)
        else:  # return list
            max_bonds = []
            for compID in compIDs:
                max_bonds += [self.max_bond(compID)]

        return max_bonds

    def initialize_field(self, field_vecs_dict: dict['Coordinate', 'Field_Component_Type']):
        """ initialize the field, writing the vectors as MPS
            keys of field_vecs_dict must be Coordinate
            missing keys are ok: correspond to zero field
        """
        # print('field vec dict', field_vecs_dict)
        for compID, data in field_vecs_dict.items():
            if isinstance(data, (GridTN, )) or data is None:
                self._set_component(compID, data)
            elif isinstance(data, qtn.TensorNetwork):
                comp = self.grid.make_gridTN(data)
                self._set_component(compID, comp)
            elif isinstance(data, np.ndarray):
                comp = self.grid.map_state_to_mps(data, split_opts=self.compress_config[1])
                self._set_component(compID, comp)
            elif isinstance(data, (list, tuple)) or data is None:
                self._set_component(compID, data)
            elif isinstance(data, (float, complex)) or data is None:
                self._set_component(compID, data)
            else:
                raise TypeError('Field components need to be GridTN or np.ndarray')

    def get_field_data(self, compIDs: Optional[Sequence['Coordinate']] = None, ax_select: dict['Axis', int] = None) \
            -> dict['Coordinate', np.ndarray]:
        """ return TN field as real space ndarray
        """
        if compIDs is None:
            compIDs = self.componentIDs

        out = {}
        for i in compIDs:
            out[i] = self.get_comp_data(i, ax_select=ax_select)
        return out

    def get_comp_data(self, compID: 'Coordinate', ax_select: dict['Axis', int] = None, ax_sum: Sequence['Axis'] = None) \
            -> Optional['np.ndarray']:
        """ return TN field as real space ndarray
        """
        comp = self._components.get(compID, None)
        if comp is None:
            return None  # 0.
        if ax_sum is not None and len(ax_sum) > 0:
            comp = comp.integrate(integ_axes=ax_sum, is_sqrt=self.is_sqrt)
        return comp.get_data(ax_select=ax_select)

    def update_comp_deriv_params(self, compIDs=None, axes=None, left_bc: Optional[BCType or str] = None,
                                 right_bc: Optional[BCType or str] = None, order: Optional[int] = None,
                                 fd_type: Optional[FDType or str] = None, offset: Optional[int] = None):
        """ update deriv_params of field components (GTNs)
            in place operation
        """

        if compIDs is None:
            compIDs = self.componentIDs

        for compID in compIDs:
            try:
                if axes is None:   axes = self[compID].grid.axes
            except KeyError:
                print(f'Warning: comp {compID} not in field')

            for ax in axes:
                try:
                    self[compID].update_deriv_params(ax, left_bc, right_bc, order, fd_type, offset)
                except KeyError:
                    print(f'Warning: deriv_params for comp {compID}, ax {ax} not set')

    ## move to GridTN  ##
    # def mult_cc(self,compress=1):
    #     """ compute g*(x)g(x) for all components
    #     """
    #     new_comps = {}
    #     for k,comp in self.components.items():
    #         new_comps[k] = helper.elemental_multiply(comp.conj(),comp,compress=True,
    #                                                  **self.grid.get_compress_opts(compress))
    #     squared = self.create_like(new_components=new_comps)
    #     squared.is_sqrt = False
    #     return squared

    def pad_to_new_grid(self, new_grid: 'Grid', inplace=False):
        """ pad all field components such that they exist in (target_ndim)-dimensional space
            fields are constant along all dimensions not specified by self.axes
            need either new_grid or target_ndim
        """
        new_field = self if inplace else self.create_like()

        new_field.grid = new_grid
        for compID in new_field.componentIDs:
            comp = self.components.get(compID, None)
            if comp is None:    continue
            new_field[compID] = new_grid.pad_gtn_to_grid(comp)
            # if comp.data_type == DataType.MPS:
            #     new_field[compID] = new_grid.pad_mps_to_grid(comp.data, self.grid.axes)
            # elif comp.data_type == DataType.MPO:
            #     new_field[compID] = new_grid.pad_mpo_to_grid(comp.data, self.grid.axes)

        return new_field

    def distance(self, other, compIDs=None, normalize=False, comp_norms: dict = None):
        """ get distance between components wrt other components
        """
        if compIDs is None:
            compIDs = set.union(set(self.componentIDs), set(other.componentIDs))

        comp_distances = {}
        for compID in compIDs:
            try:
                comp1 = self[compID]
            except KeyError:  # self component = 0
                comp1 = None

            try:
                comp2 = other[compID]
            except KeyError:  # other component = 0
                comp2 = None

            if comp1 is None and comp2 is None:
                comp_distances = 0.0
            elif comp1 is None:
                if normalize:
                    comp_distances[compID] = np.inf
                else:
                    comp_distances[compID] = comp2.frobenius_norm()
            elif comp2 is None:
                if normalize:
                    comp_distances[compID] = 1.0
                else:
                    comp_distances[compID] = comp1.frobenius_norm() if comp_norms is None else comp_norms[compID]
            else:
                comp_distances[compID] = comp1.distance(comp2)
                if normalize:
                    norm_val = comp1.frobenius_norm() if comp_norms is None else comp_norms[compID]
                    comp_distances[compID] /= norm_val

        return comp_distances

    def integrate(self, compIDs=None, axes=None, new_grid=None, new_ax_deriv_configs=None, inplace=False):
        """ integrate over axes specified by axIDs. integrates over all if axIDs=None -> scalar
            otherwise, returns a Field object with reduced dimensionality
            inplace:  if new GridTN object is generated or not
        """
        # if new_grid is None and axes is not None:
        #     new_grid = self.grid.create_like([ax for ax in self.grid.axes if ax not in axes])

        new_field = self if inplace else self.copy()
        compIDs = self.componentIDs if compIDs is None else compIDs

        results = {}
        for compID in compIDs:
            component = new_field.components.get(compID, None)
            if component is None:
                continue

            # if axIDs is None:
            #     axes = None
            # else:
            #     axes = [component.grid.get_ax(axID) for axID in axIDs]
            if axes is None:
                axes = component.grid.axes

            # print('integrate is sqrt', self.is_sqrt)
            component = component.integrate(integ_axes=axes, is_sqrt=self.is_sqrt, new_grid=new_grid,
                                            new_ax_deriv_configs=new_ax_deriv_configs)
            results[compID] = component

            if isinstance(component, GridTN):
                if new_grid is None:   new_grid = component.grid
                if new_ax_deriv_configs is None:  new_ax_deriv_configs = component.ax_deriv_configs

        new_field._components.update(results)  ## update with a scalar though?
        new_field.grid = new_grid
        return new_field

    def meas_expec(self, obs_gtn_dict, compIDs=None, integ_axes=None, new_grid=None, new_ax_deriv_configs=None):
        """ integrate over axes specified by axIDs. integrates over all if axIDs=None -> scalar
            otherwise, returns a Field object with reduced dimensionality
            inplace:  if new GridTN object is generated or not
        """
        if new_grid is None and integ_axes is not None:
            out_axes = [ax for ax in self.grid.axes if ax not in integ_axes]
            if len(out_axes) > 0:
                new_grid = self.grid.create_like([ax for ax in self.grid.axes if ax not in integ_axes])
            else:
                new_grid = None

        compIDs = self.componentIDs if compIDs is None else compIDs

        results = {}
        for compID in compIDs:
            component = self.components.get(compID, None)
            if component is None:
                continue

            if isinstance(obs_gtn_dict, GridTN):
                obs_gtn = obs_gtn_dict
            else:
                try:
                    obs_gtn = obs_gtn_dict[compID]
                except KeyError:
                    continue

            # if axIDs is None:
            #     axes = None
            # else:
            #     axes = [component.grid.get_ax(axID) for axID in axIDs]
            if integ_axes is None:
                integ_axes = component.grid.axes

            # print('meas expec', component)
            # print('meas expec', obs_gtn)
            # print('is sqrt', self.is_sqrt)
            component = component.meas_expec(obs_gtn, integ_axes=integ_axes, is_sqrt=self.is_sqrt, new_grid=new_grid,
                                             new_ax_deriv_configs=new_ax_deriv_configs)
            results[compID] = component
            # print('results', results)

            if isinstance(component, GridTN):
                if new_ax_deriv_configs is None:  new_ax_deriv_configs = component.ax_deriv_configs

        new_field = Field(self.name, new_grid, results, compress_config=self.compress_config, is_sqrt=False)
        # print('new grid', new_grid)
        # print('new field', new_field.components)
        # new_field._components.update(results)  ## update with a scalar though?
        # new_field.grid = new_grid
        return new_field

    def scalar_add(self, scalar_vals, compIDs=None, inplace=False):
        """ add components in comps by scalar_vals
        """
        field = self if inplace else self.copy()

        if isinstance(compIDs, tuple):
            compIDs = [compIDs]
        elif compIDs is None:
            compIDs = field.componentIDs

        if np.isscalar(scalar_vals):
            scalar_vals = {i: scalar_vals for i in compIDs}

        for i in compIDs:
            comp = field.components.get(i, None)
            if comp is None or comp.data is None:
                continue
            else:
                comp.scalar_add(scalar_vals[i], inplace=True)

        return field

    def scalar_multiply(self, scalar, compIDs=None, inplace=False):
        """ multiply components in comps by scalar
        """
        field = self if inplace else self.copy()

        # if isinstance(compIDs, tuple):
        #     compIDs = [compIDs]
        if compIDs is None:
            compIDs = field.componentIDs

        for i in compIDs:
            comp = field.components.get(i, None)
            if comp is None:  continue
            if isinstance(comp, (GridTN, )):
                comp.scalar_multiply(scalar, inplace=True)
            else:
                comp *= scalar

        return field

    def elemental_multiply(self, mps_vec, compIDs=None, zipup=False, inplace=False, compress_level: int = 0):
        """ multiply components in comps by scalar
            if not inplace, only return multiplied components
        """
        field = self if inplace else self.copy()

        # if isinstance(compIDs, tuple):
        #     compIDs = [compIDs]
        if compIDs is None:
            compIDs = field.componentIDs

        for i in compIDs:
            comp = field.components.get(i, None)
            if comp is None or comp.data is None:   continue
            comp.elemental_multiply(mps_vec, inplace=True, compress=compress_level, zipup=zipup,
                                    compress_type=field.compress_config.compress_type,
                                    compress_opts=field.compress_config[compress_level],
                                    sub_compress_opts=field.compress_config.get_all_sub_compress_opts(compress_level))

        return field

    # def xmultiply(self, x_axes: Sequence['Axis'], compIDs=None, inplace=False, compress=False, compress_opts=None):
    #     """ perform x_i * f(x_i) elemental multiplication
    #     """
    #     field = self if inplace else self.copy()
    #
    #     if isinstance(compIDs, tuple): ##???? when does this caes arise?
    #         compIDs = [compIDs]
    #     elif compIDs is None:
    #         compIDs = field.componentIDs
    #
    #     for i in compIDs:
    #         field[i].xmultiply(x_axes, inplace=True, compress=compress, compress_opts=compress_opts)
    #     return field

    def xmultiply(self, x_axes: Sequence['Axis'], x_powers: Sequence[int] = None, compID=None, offsets=None,
                  scales=None, compress_level=0) -> 'Field':
        """ compute x_i {x_i} * f(x_i) elemental multiplication for all x_i in x_axes
            not an inplace operation; generates new field with components x_i
        """
        if compID is None:
            compID = next(iter(self.componentIDs))

        if x_powers is None:
            x_powers = [1] * len(x_axes)

        new_comps = {}
        for x_ax, x_power in zip(x_axes, x_powers):
            comp = self.components.get(compID, None)
            if comp is None or comp.data is None:  continue
            # out = comp.xmultiply([x_ax], x_power=x_power, offsets=offsets, scales=scales, inplace=False,
            #                      compress_type=self.compress_config.compress_type,
            #                      compress=compress_level, compress_opts=self.compress_config[compress_level])
            # print('XMULT')
            xmult_mpo = comp.grid.get_xmultiply_mpo([x_ax], x_power=x_power, offsets=offsets, scales=scales)
            out_comp = self.grid.add_gtns((xmult_mpo, comp), compress_type=self.compress_config.compress_type,
                                          compress=compress_level, compress_opts=self.compress_config[compress_level], )
            new_comps[x_ax.coordinate] = out_comp

        return self.create_like_vector(new_components=new_comps)

    # @profile
    def xdot(self, x_axes: Sequence['Axis'], compIDs=None, offsets: dict['Axis', 'Numeric'] = None,
             scales: dict['Axis', 'Numeric'] = None, compress_level: int = 0,
             inner_compress_level: int = 0) -> Optional['ScalarField']:
        """ new xdot:
            compute sum( x_i {x_i} * g(x_i) {x_i}) elemental multiplication for all x_i in x_axes
            not an inplace operation; generates new field with components x_i
        """
        if compIDs is None:
            compIDs = [x_ax.coordinate for x_ax in x_axes]

        # out_comp = None
        targets = []
        for i in range(len(x_axes)):
            x_ax = x_axes[i]
            comp = self.components.get(compIDs[i], None)
            if comp is None:
                continue

            # op = comp.grid.get_xmultiply_mpo([x_ax], offsets=offsets, scales=scales)
            # targets += [(op, comp)]
            out = comp.xmultiply([x_ax], offsets=offsets, scales=scales, inplace=False, compress=inner_compress_level,
                                 compress_type=self.compress_config.compress_type,
                                 compress_opts=self.compress_config[inner_compress_level])
            targets += [out]

            # if out_comp is None:
            #     out_comp = out
            # else:
            #     out_comp.add(out, inplace=True, compress=False)
            #                  # compress=inner_compress_level,
            #                  # compress_opts=self.compress_config[inner_compress_level])

        # if out_comp is not None and compress_level:
        #     out_comp.compress(compress_opts=self.compress_config[compress_level])

        out_comp = self.grid.add_gtns(*targets, compress_type=self.compress_config.compress_type,
                                      compress=compress_level, sub_compress=inner_compress_level,
                                      compress_opts=self.compress_config[compress_level],
                                      sub_compress_opts=self.compress_config[inner_compress_level])

        return self.create_like_scalar(new_component=out_comp)

    # @profile
    # def xdot(self, x_axes: Sequence['Axis'], offsets: dict['Axis','Numeric'] = None,
    #          scales: dict['Axis','Numeric'] = None, compID=None, compress_level:int = 0,
    #          inner_compress_level:int = 0) -> Optional['ScalarField']:
    #     """ compute x_i {x_i} * f(x_i) elemental multiplication for all x_i in x_axes
    #         not an inplace operation; generates new field with components x_i
    #     """
    #     if compID is None:
    #         compID = next(iter(self.componentIDs))
    #
    #     comp = self.components.get(compID, None)
    #     if comp is None:        return None
    #
    #     # axIDs = [ax.axID for ax in x_axes]
    #
    #     # if isinstance(offsets,int) or isinstance(offsets,float):
    #     #     offsets_dict = {axID: offsets for axID in axIDs}
    #     #     offsets_ = offsets
    #     # elif isinstance(offsets,dict):
    #     #     offsets_dict = {}
    #     #     offsets_ = []
    #     #     for axID in axIDs:
    #     #         try:    offsets_dict[axID] = offsets[axID]
    #     #         except KeyError:  offsets_dict[axID] = 0.0
    #     #         offsets_ += [offsets_dict[axID]]
    #     #     offsets_ = tuple(offsets)
    #     # else:  # if offsets is None:
    #     #     offsets_dict = {axID: 0.0 for axID in axIDs}
    #     #     offsets_ = offsets
    #     #
    #     # if isinstance(scales, int) or isinstance(scales, float):
    #     #     scales_dict = {axID: scales for axID in axIDs}
    #     #     scales_ = scales
    #     # elif isinstance(scales, dict):
    #     #     scales_dict = {}
    #     #     scales_ = []
    #     #     for axID in axIDs:
    #     #         try:
    #     #             scales_dict[axID] = scales[axID]
    #     #         except KeyError:
    #     #             scales_dict[axID] = 0.0
    #     #         scales += [scales_dict[axID]]
    #     #     scales_ = tuple(scales_)
    #     # else:  # scales is None:
    #     #     scales_dict = {axID: 0.0 for axID in axIDs}
    #     #     scales_ = scales
    #
    #     out_comp = None
    #     for x_ax in x_axes:
    #         out = comp.xmultiply([x_ax], offsets=offsets, scales=scales, inplace=False, compress=inner_compress_level,
    #                              compress_opts=self.compress_config[inner_compress_level])
    #         if out_comp is None:
    #             out_comp = out
    #         else:
    #             out_comp.add(out, inplace=True, compress=False)
    #                          # compress=inner_compress_level,
    #                          # compress_opts=self.compress_config[inner_compress_level])
    #
    #     if compress_level:
    #         out_comp.compress(compress_opts=self.compress_config[compress_level])
    #
    #     return self.create_like_scalar(new_component=out_comp)

    def __mul__(self, const):
        return self.scalar_multiply(const)

    def add(self, other, inplace=False, compress_level=0):
        """ generate a new field whose components are the sum of self.components + other.components
            other can be Field object or Field.components (dict)
            not an in-place operation
        """
        new = self if inplace else self.copy()

        if other is None:  return new

        new_comps = {}

        all_comps = set.union(set(new.componentIDs), set(other.componentIDs))
        for i in all_comps:
            comp1 = new.components.get(i, None)
            comp2 = other.components.get(i, None)

            if comp1 is None and comp2 is None:
                continue
            elif comp1 is None:
                if comp2.grid != new.grid:
                    comp2 = new.grid.pad_gtn_to_grid(comp2)
                else:
                    comp2 = comp2.copy()
                new_comps[i] = comp2
            elif comp2 is None:
                new_comps[i] = comp1.copy()
            else:
                # new_comps[i] = comp1.add(comp2, inplace=True, compress=compress_level,
                #                          compress_opts=new.compress_config[compress_level],
                #                          sub_compress_opts=new.compress_config.get_all_sub_compress_opts(compress_level)
                #                          )

                ### need to define initial state closer to comp2; else if comp2's norm is too small
                ### then it settles on a local minimum and we get incorrect dynamics
                # out_comp = self.grid.add_gtns(comp1, comp2, compress_type=self.compress_config.compress_type,
                #                               compress=compress_level, compress_opts=self.compress_config[compress_level],
                #                               )

                if isinstance(comp1, (GridTN, )):
                    comp_ = comp1.add(comp2, compress=False)
                    if compress_level:
                        comp_ = comp_.compress(inplace=True, compress_type=self.compress_config.compress_type,
                                               compress_opts=self.compress_config[compress_level])
                else:
                    comp_ = comp2 + comp1
                # print('comp_', comp_.max_bond(), self.compress_config[compress_level].get('max_bond', None))
                # print('comp nrm', comp_.norm())
                # print('comp_', comp_.max_bond())

                # print('comp_out', out_comp.max_bond(), self.compress_config[compress_level].get('max_bond', None))
                new_comps[i] = comp_

            new[i] = new_comps[i]
        # new.components = new_comps
        return new

    def add_dmrg(self, *others: 'Field', inplace=False, compress_level=0, **dmrg_opts):
        """ generate a new field whose components are the sum of self.components + other.components
            other can be Field object or Field.components (dict)
            not an in-place operation
        """
        new = self if inplace else self.copy()

        if others is None:  return new

        new_comps = {}

        print('other', others)
        all_comps = set.union(set(new.componentIDs), *[set(other.componentIDs) for other in others])
        for i in all_comps:
            comps = []

            comp1 = new.components.get(i, None)
            if comp1 is not None:
                comps += [comp1]

            for other in others:
                comp2 = other.components.get(i, None)
                if comp2 is not None:
                    comps += [comp2.copy()]

            if len(comps) == 0:
                continue
            elif len(comps) == 1:
                comp = comps[0]
                comp = new.grid.pad_gtn_to_grid(comp) if comp.grid != new.grid else comp.copy()
                new_comps[i] = comp
            else:

                if isinstance(comp1, GridTN):
                    comp_ = comps[0].add_dmrg(*comps[1:], compress_opts=self.compress_config[compress_level],
                                              inplace=inplace, **dmrg_opts)
                else:
                    comp_ = np.sum(comps)
                # print('comp_', comp_.max_bond(), self.compress_config[compress_level].get('max_bond', None))
                # print('comp nrm', comp_.norm())
                # print('comp_', comp_.max_bond())

                # print('comp_out', out_comp.max_bond(), self.compress_config[compress_level].get('max_bond', None))
                new_comps[i] = comp_

            new[i] = new_comps[i]
        # new.components = new_comps
        return new

    def __add__(self, other):
        return self.add(other, inplace=False)

    def norms(self, compIDs=None):
        """ if sqrt:  norm = sqrt( integ g*(x)g(x) dx )
            if not sqrt:  norm = sqrt( integ f(x) dx )
        """
        # integ = self.integrate(compIDs)
        # norm_vals = integ._components

        if compIDs is None:
            compIDs = self.componentIDs
            norm_vals = {}

            for compID in compIDs:
                comp = self[compID]
                if isinstance(comp, (float, complex)):
                    val = np.abs(comp ** 2) if self.is_sqrt else comp
                else:
                    val = comp.norm(is_sqrt=self.is_sqrt) if comp is not None else 0.0
                norm_vals[compID] = val if val is not None else 0.0
                # norm_vals[compID] = self.integrate(compIDs=[compID])[compID]
        else:
            norm_vals = []
            # print('norms', self.componentIDs, self.components)
            for compID in compIDs:
                try:
                    comp = self[compID]
                    if isinstance(comp, (float, complex)):
                        val = np.abs(comp ** 2) if self.is_sqrt else comp
                    else:
                        val = comp.norm(is_sqrt=self.is_sqrt) if comp is not None else 0.0
                except KeyError:
                    val = 0.0
                norm_vals += [val if val is not None else 0.0]
        return norm_vals

    def norm(self):
        norms = self.norms()  # takes care of is_sqrt at GTN level
        if self.is_sqrt:
            comp_integs = [val ** 2 if val is not None else 0.0 for k, val in norms.items()]
            return np.sqrt(np.sum(comp_integs))
        else:
            comp_integs = [val for k, val in norms.items()]
            # comp_integs = [val if val is not None else np.nan for k, val in norms.items()]
            return np.sum(comp_integs)

    def frobenius_norms(self):
        """ returns norm**2 """

        norm_vals = []
        # print('norms', self.componentIDs, self.components)
        for compID in self.componentIDs:
            try:
                comp = self[compID]
                if isinstance(comp, (float, complex)):
                    val = np.abs(comp)
                else:
                    val = comp.frobenius_norm() if comp is not None else 0.0
            except KeyError:
                val = 0.0
            norm_vals += [val if val is not None else 0.0]
        return norm_vals

    def frobenius_norm(self):
        norm_vals = self.frobenius_norms()
        return np.sqrt(np.sum(norm_vals ** 2))

    ###########################
    #### make measurements ####
    ###########################

    def meas_comp_total_energy(self, compID, x_axes=None, scale=1.0):
        """ E(x) = integ f(x) x^2 dx
            E(x) = integ g^*(x) g(x) x^2 dx
        """
        comp = self.components[compID]
        v2_mpo = comp.grid.get_x2multiply_mpo(x_axes)
        nrg = comp.meas_expec(v2_mpo, is_sqrt=self.is_sqrt)
        return nrg * scale

    ########################
    #### postprocessing ####
    ########################

    def smooth_data(self, compIDs=None, inplace=True, **gaussian_smooth_kwargs):
        """ apply gaussian smoothing to data
        """

        field = self if inplace else self.copy()

        if compIDs is None:
            compIDs = field.componentIDs
        elif isinstance(compIDs, tuple):
            compIDs = [compIDs]

        for i in compIDs:
            # if self.bcs[i] == 'periodic':  bc_ = 'wrap'
            # else:                          bc_ = 'mirror'
            field[i].gaussian_smooth_data(inplace=True, **gaussian_smooth_kwargs)

        return field

    def apply_absorbing_bc(self, x_ax, v_ax, compIDs=None, inplace=True, compress_level=1):
        """ absorbinb bc:  set x[0] = 0 for v < 0 and x[0] = 0 for v > 0
        """
        field = self if inplace else self.copy()

        if compIDs is None:
            compIDs = field.componentIDs
        elif isinstance(compIDs, int):
            compIDs = [compIDs]

        for i in compIDs:
            field[i].apply_absorbing_bc(x_ax, v_ax, inplace=True, compress=compress_level,
                                        compress_opts=field.compress_config[compress_level])
        return field

    def compress(self, inplace=True, compIDs=None, compress_level=1, compress_opts: Optional[dict] = None,
                 sub_compress_opts: dict['SubCompressConfigType', dict] = None, norm_cutoff=None,
                 verbose=False, use_rdm=False, conservative=False):
        """ compress all components in field according to te_compress_opts
            derivatives and such are calculated with respect to grid.compress_opts
        """
        field = self if inplace else self.copy()

        if isinstance(compIDs, tuple):
            compIDs = [compIDs]
        elif compIDs is None:
            compIDs = field.componentIDs

        if compress_opts is None:
            compress_opts = field.compress_config[compress_level]
        if sub_compress_opts is None:
            sub_compress_opts = field.compress_config.get_all_sub_compress_opts(compress_level)

        if verbose:
            print('field compress', compress_level, compress_opts)
            print('sub compress opts', sub_compress_opts)

        # ref_norm = self.frobenius_norm()
        # print('field ref norm', ref_norm)

        for i in compIDs:
            comp = field.components.get(i, None)
            if comp is None \
                    or isinstance(comp, (int, float, complex, np.number)) \
                    or comp.data is None:  continue
            # print('FIELD COMPRESS', field.compress_config.compress_type)
            if use_rdm:
                comp.compress_rdm(inplace=True, compress_type=field.compress_config.compress_type,
                                  compress_opts=compress_opts, sub_compress_opts=sub_compress_opts,
                                  verbose=verbose)
                if conservative:
                    raise NotImplementedError('rdm compress with conservation not implemented')
            else:
                # if norm_cutoff is None:
                #     norm_cutoff = compress_opts.get('norm_cutoff', None)
                comp.compress(inplace=True, compress_type=field.compress_config.compress_type,
                              compress_opts=compress_opts, sub_compress_opts=sub_compress_opts,
                              norm_cutoff=norm_cutoff,  # ref_norm=ref_norm,
                              verbose=verbose, conservative=conservative,
                              )

        return field

    ###############################
    ## operations between fields ##
    ###############################

    # @profile
    def dot(self, other, comps=None, other_comps=None, zipup=False, check_elem_mult_anc=False,
            compress_level: int = 0, inner_compress_level: int = 0, verbose_plot=False) -> 'ScalarField':
        """ take dot product of self with components (specified by dims) of other field
            defining comps, other comps:  can take dot product between different components
        """
        # assert(self.ncomp >= np.max(list(other_components.keys()))), \
        #         '# of components less than other field'

        if comps is None:         comps = self.componentIDs
        if other_comps is None:   other_comps = comps

        compress_opts = self.compress_config[compress_level]
        inner_compress_opts = self.compress_config[inner_compress_level]
        sub_compress_opts = self.compress_config.get_all_sub_compress_opts(compress_level)
        inner_sub_compress_opts = self.compress_config.get_all_sub_compress_opts(inner_compress_level)

        out = None
        for i in range(len(comps)):

            try:
                comp1 = self.components.get(comps[i], None)
                if isinstance(other, Field):  # other can be Field or Field.components
                    comp2 = other.components.get(other_comps[i], None)
                elif isinstance(other, dict):
                    comp2 = other.get(other_comps[i], None)
                else:
                    raise TypeError(f'other needs to be a Field or dict of GridTNs, not {type(other)}')
            except KeyError:
                continue

            if comp1 is None or comp1.data is None:   continue
            if comp2 is None or comp2.data is None:   continue

            # print('DOT', i, comps[i], other_comps[i])

            add_cc = self.grid.has_basis_k_real() if check_elem_mult_anc else False

            if comp2.data_type == DataType.MPO:
                out_comp = comp1.apply(comp2, inplace=False, zipup=zipup, compress=inner_compress_level,
                                       compress_opts=inner_compress_opts,
                                       sub_compress_opts=inner_sub_compress_opts,
                                       add_cc=add_cc)

            elif comp2.data_type == DataType.MPS:
                # comp2 = comp2.apply_elemental_multiply_op(compress=True)
                out_comp = comp1.elemental_multiply(comp2, inplace=False, zipup=zipup, compress=inner_compress_level,
                                                    compress_opts=inner_compress_opts,
                                                    sub_compress_opts=inner_sub_compress_opts,
                                                    add_cc=add_cc)
            else:
                raise TypeError(f'other needs to be GridTN of MPO or MPS type, not {comp2.data_type}')

            if verbose_plot:

                ax_x, ax_vx, ax_vy, ax_vz = out_comp.grid.axes

                # lorentz_data = lorentz_term.get_comp_data()
                out_data = out_comp.get_data(ax_select={  # ax_x: 0,
                    ax_vx: ax_vx.npts // 2,
                    # ax_vy: ax_vy.npts // 2,
                    ax_vz: ax_vz.npts // 2
                })
                if out_data is not None:
                    plt.figure()
                    plt.imshow(np.real(out_data))
                    plt.title(f'dot out re {i}')
                    plt.xlabel('vy'), plt.ylabel('x')
                    plt.colorbar()

                    plt.figure()
                    plt.imshow(np.imag(out_data))
                    plt.title(f'dot out im {i}')
                    plt.xlabel('vy'), plt.ylabel('x')
                    plt.colorbar()
                    plt.show()

            if out is None:
                out = out_comp
            else:
                # print('dot inner compress level', inner_compress_level)
                out.add(out_comp, inplace=True, compress=False)
                # compress=inner_compress_level,
                # compress_opts=self.compress_config[inner_compress_level])

        if compress_level and out is not None:
            out.compress(compress_opts=compress_opts, sub_compress_opts=sub_compress_opts)

        return self.create_like_scalar(new_component=out)  # scalar field

    def cross_product(self, other, coord_sys: 'CoordinateSystem', comps=None, other_comps=None,
                      check_elem_mult_anc=False, compress_level=0, inner_compress_level=0, zipup=False,
                      verbose_plot=False) -> 'Field':
        """ take cross product of self with components of other field
            other can be Field object or Field.components
            out = self x other
            comps, comps_other:  which components of self and other to take the cross product with,
                               defaults to the first three components
        """
        # if comps is None:           comps = self.componentIDs[:3]  # active dimensions
        # if other_comps is None:     other_comps = comps

        if comps is None:           comps = coord_sys.coords
        if other_comps is None:     other_comps = comps

        compress_opts = self.compress_config[compress_level]
        inner_compress_opts = self.compress_config[inner_compress_level]
        sub_compress_opts = self.compress_config.get_all_sub_compress_opts(compress_level)
        inner_sub_compress_opts = self.compress_config.get_all_sub_compress_opts(inner_compress_level)

        out = {}
        for i in range(3):

            c0 = comps[i]
            c1, c2 = comps[(i + 1) % 3], other_comps[(i + 2) % 3]
            # print('out', c0, 'cross', c1, c2)

            add_cc = self.grid.has_basis_k_real() if check_elem_mult_anc else False

            try:
                out_comp_1 = self[c1].elemental_multiply(other[c2], inplace=False, compress=inner_compress_level,
                                                         compress_opts=inner_compress_opts,
                                                         sub_compress_opts=inner_sub_compress_opts,
                                                         zipup=zipup,
                                                         add_cc=add_cc)

                if verbose_plot:
                    ax_x, ax_y, ax_vx, ax_vy, ax_vz = out_comp_1.grid.axes

                    # other2 = self.grid.pad_gtn_to_grid(other[c2])
                    # out_data = other2.get_data(ax_select={ax_vx: 0, ax_vy: 0, ax_vz: 0})
                    # out_data = self[c1].get_data(ax_select={ax_vx: 0, ax_vy: 0, ax_vz: 0})
                    out_data = out_comp_1.get_data(ax_select={ax_vx: 0, ax_vy: 0, ax_vz: 0})
                    if out_data is not None:
                        plt.figure()
                        # plt.plot(np.real(out_data), label=f'{c0}')
                        # plt.plot(np.imag(out_data), '--', label=f'{c0}')
                        # plt.legend()
                        plt.imshow(out_data)
                        plt.title(f'out 1 {c0}, v{c1}, F{c2}')
                        plt.show()

            except KeyError:
                out_comp_1 = None  ## vector_field[i] = 0

            try:
                out_comp_2 = self[c2].elemental_multiply(other[c1], inplace=False, compress=inner_compress_level,
                                                         compress_opts=inner_compress_opts,
                                                         sub_compress_opts=inner_sub_compress_opts,
                                                         zipup=zipup,
                                                         add_cc=add_cc)

                out_comp_2 = out_comp_2.scalar_multiply(-1, inplace=True)

                if verbose_plot:
                    ax_x, ax_y, ax_vx, ax_vy, ax_vz = out_comp_2.grid.axes

                    # other1 = self.grid.pad_gtn_to_grid(other[c1])
                    # out_data = other1.get_data(ax_select={ax_vx: 0, ax_vy: 0, ax_vz: 0})
                    out_data = out_comp_2.get_data(ax_select={ax_vx: 0, ax_vy: 0, ax_vz: 0})
                    # out_data = self[c2].get_data(ax_select={ax_vx: 0, ax_vy: 0, ax_vz: 0})
                    if out_data is not None:
                        plt.figure()
                        # plt.plot(np.real(out_data), label=f'{c2}')
                        # plt.plot(np.imag(out_data), '--', label=f'{c2}')
                        # plt.legend()
                        plt.imshow(out_data)
                        plt.colorbar()
                        plt.title(f'out 2 {c0}, v{c2}, F{c1}')
                        plt.show()

            except KeyError:
                out_comp_2 = None  ## vector_field[i] = 0

            if out_comp_1 is None and out_comp_2 is None:
                continue
            elif out_comp_1 is None and out_comp_2 is not None:
                out[c0] = out_comp_2
            elif out_comp_1 is not None and out_comp_2 is None:
                out[c0] = out_comp_1
            else:
                # print('out comps', out_comp_1.max_bond(), out_comp_2.max_bond())
                out_comp_1 = out_comp_1.add(out_comp_2, inplace=True, compress=False)
                out[c0] = out_comp_1

                if verbose_plot:
                    ax_x, ax_y, ax_vx, ax_vy, ax_vz = out_comp_1.grid.axes

                    out_data = out_comp_1.get_data(ax_select={ax_y: 0, ax_vx: 0, ax_vy: 0})
                    if out_data is None:
                        print('em term data is None', c0)
                        continue
                    plt.figure()
                    plt.imshow(np.real(out_data))
                    plt.title(f'cross product {c0} re')
                    plt.xlabel('vz')
                    plt.ylabel('x')
                    plt.colorbar()

                    out_data = out_comp_1.get_data(ax_select={ax_y: 0, ax_vx: 0, ax_vy: 0})
                    plt.figure()
                    plt.imshow(np.imag(out_data))
                    plt.title(f'cross product {c0} im')
                    plt.xlabel('vz')
                    plt.ylabel('x')
                    plt.colorbar()
                    plt.show()

        new_field = self.create_like(new_components=out)
        if len(out) > 0:
            new_field.grid = out[next(iter(out))].grid

        if compress_level:
            new_field.compress(inplace=True, compress_opts=compress_opts,
                               sub_compress_opts=sub_compress_opts)

        return new_field

    def get_difference(self, other, compIDs: Sequence = None, return_type='L2'):
        """ return L1-norm or L2-norm or MPS corresponding to difference between self and other fields
            L1, L2, rms, field, array
            rms is the same as L2
        """
        if compIDs is None:
            compIDs = set.union(set(self.componentIDs), set(other.componentIDs))

        other = other.scalar_multiply(-1)

        if return_type in ['mps', 'field']:
            return self.add(other, compress_level=False)

        out = {}
        for i in compIDs:
            try:
                mat1 = self.get_comp_data(i)
            except KeyError:
                mat1 = 0.0
            try:
                mat2 = other.get_comp_data(i)
            except KeyError:
                mat2 = 0.0
            diff = mat1 + mat2
            if return_type in ['rms', 'L2']:
                err = np.linalg.norm(diff) / np.sqrt(diff.size)
            elif return_type in ['L1']:
                err = np.sum(np.abs(diff)) / diff.size
            else:
                err = diff

            out[i] = err

        return self.create_like(out)

    ##############################
    ## operations on the fields ##
    ##############################

    #### probably remove and just call from coord sys? ####

    """ operations on the fields
        comps:  list of (vector) field components to act on. None means act on all components
        compress:  level of compression. 
                   0 = no compression, 
                   1 = final compression, 
                   2 = compression after adds, 
                   3 = compression after application of MPO, etc...
    """

    # @profile
    def gradient(self, compID=None, deriv_axes: Optional[Sequence['Axis']] = None,
                 upwind_axes: Optional[dict['Axis', 'Axis']] = None,
                 ax_deriv_config: dict['Axis', 'DerivativeConfiguration'] = None,
                 compress_level: int = 0) -> 'Field':
        """ df/dx {x} + df/dy {y} + df/dz {z}
        """
        if compID is None:
            compID = next(iter(self.componentIDs))

        comp_data = self[compID]
        if comp_data is None:
            return self.create_like_vector(new_components={})

        if deriv_axes is None:
            deriv_axes = comp_data.grid.axes

        out = {}
        for ax in deriv_axes:
            # print('gradient deriv ax', ax)
            if ax is None:  continue
            # coord_sys = ax.coord_sys
            # out_comp = coord_sys._gtn_gradient(comp_data, [ax], compress=compress, compress_opts=compress_opts)
            # out[ax.axID] = out_comp[ax.axID]
            # # coord_sys = self.grid.ax_coordsys_map[ax.axID]
            # # ax_coord = coord_sys.ax_coords[ax.axID]
            upwind_ax = upwind_axes[ax] if upwind_axes is not None else None
            # print('deriv ax', ax, upwind_ax)
            # print('gradient compress', compress_level, self.compress_config[compress_level])
            out_comp = comp_data.take_firstderivative(ax, upwind_ax=upwind_ax, inplace=False,
                                                      ax_deriv_config=ax_deriv_config, compress=compress_level,
                                                      compress_opts=self.compress_config[compress_level])
            out[ax.coordinate] = out_comp

        return self.create_like_vector(new_components=out)

    ############
    ## use coordinate class ##

    # def laplacian(self, compID=None, deriv_axes: list['Axis'] or None = None, compress_level=0,
    #               inner_compress_level=0) -> 'ScalarField':
    #     """ L(f) = d^2/dx^2 f + d^2/dy^2 f + d^2/dz^2 z
    #         mps_scalar_field:  mps representing the scalar field
    #         bcs:  boundary conditions for the relevant field component
    #     """
    #     if compID is None:
    #         compID = next(iter(self.componentIDs))
    #
    #     gtn = self[compID]
    #     if gtn is None:
    #         self.create_like_scalar(new_component=None)
    #
    #     if deriv_axes is None:
    #         deriv_axes = gtn.grid.axes
    #
    #     ax = deriv_axes[0]
    #     out = gtn.take_secondderivative(ax, None, inplace=False, compress=inner_compress_level,
    #                                     compress_opts=self.compress_config[inner_compress_level])
    #
    #     for ax_k in deriv_axes[1:]:
    #         out2 = gtn.take_secondderivative(ax_k, None, inplace=False, compress=inner_compress_level,
    #                                          compress_opts=self.compress_config[inner_compress_level])
    #         out = out.add(out2, inplace=True, compress=inner_compress_level,
    #                       compress_opts=self.compress_config[inner_compress_level])
    #         # out.compress(inplace=True, compress_opts=self.get_compress_opts(compress + 1))
    #
    #     if compress_level:
    #         out = out.compress(inplace=True, compress_opts=self.compress_config[compress_level])
    #
    #     return self.create_like_scalar(new_component=out)

    def laplacian(self, coord_sys: 'CoordinateSystem', deriv_axes=None, out_comps=None, compress_level=0,
                  inner_compress_level=0, ) \
            -> 'Field':

        if deriv_axes is None:
            deriv_axes = self.grid.axes

        out = coord_sys.vector_laplacian(self, deriv_axes=deriv_axes,
                                         out_comps=out_comps,
                                         compress=compress_level,
                                         compress_opts=self.compress_config[compress_level],
                                         inner_compress=inner_compress_level,
                                         inner_compress_opts=self.compress_config[inner_compress_level])

        return out

    # @profile
    def curl(self, coord_sys: 'CoordinateSystem', out_comps=None, compress_level=0, inner_compress_level=0) \
            -> 'Field':
        """ Curl(F) =  ( d/dy Az - d/dz Ay ) {x} +
                       ( d/dz Ax - d/dx Az ) {y} +
                       ( d/dx Ay - d/dy Ax ) {z}
            axIDs:  axIDs along which to take the curl.  default = first 3 axes in self.grid
            out_comps:  desired components in the output
            compress_level:  determines compress_opts
        """
        # if axIDs is None:        axIDs = coord_sys.axIDs[:3]
        # if out_comps is None:    out_comps = coord_sys.axIDs[:3]

        # assert all(axID in coord_sys.axIDs for axID in axIDs), 'all compIDs must correspond' \
        #                                                        ' to the same coordinate system'

        # out_coords = [coord_sys.coords[axID] for axID in out_comps] if out_comps is not None else None
        # out_coords = out_comps
        return coord_sys.curl(self, out_comps=out_comps,
                              compress=compress_level,
                              compress_opts=self.compress_config[compress_level],
                              inner_compress=inner_compress_level,
                              inner_compress_opts=self.compress_config[inner_compress_level])

    def divergence(self, coord_sys: 'CoordinateSystem', compIDs=None, ax_deriv_configs=None,
                   compress_level=0, inner_compress_level=0) -> 'ScalarField':
        """ d/dx Ax + d/dy Ay + d/dz Az
            gtn_comps:  dictionary of gtn_comps, with CoordinateType/int as keys
            coord_map:  dictionary of axes, with CoordinateType or int as keys
                default: Axis in list gtn.axes indexed by CoordinateType/int
            comps:  components of gtn_comps to include when computing divergence
        """
        # if compIDs is None:  compIDs = self.componentIDs
        # coords = [coord_sys.ax_coords.get(axID,None) for axID in compIDs]
        # coords = coord_sys.coords
        axes = self.grid.axes

        # if ax_deriv_configs is None:
        #     ax_deriv_configs = {}
        #     for compID, comp in self.components.items():
        #         ax = coord_sys.get_axis(compID.type)
        #         ax_deriv_configs[ax] = comp.ax_deriv_configs.get(ax, None)
        #
        # print('ax deriv configs', ax_deriv_configs)

        return coord_sys.divergence(self, axes, ax_deriv_configs,
                                    compress_level, self.compress_config[compress_level],
                                    inner_compress_level, self.compress_config[inner_compress_level])

    def advective_derivative(self, other, coord_sys: 'CoordinateSystem', compress_level=0, inner_compress_level=0) \
            -> 'Field':
        """ solves (other \cdot \grad) self
        """
        raise NotImplementedError

    # @set_io_type
    # def convolve(self,other,comps='all',other_comps='all',compress=1):
    #     """ nonlinear term in k-space is a convolution
    #     """
    #     if other_comps == 'all' or other_comps is None:   other_comps = comps
    #
    #     out = None
    #     for i in range(len(comps)):
    #
    #         try:
    #             field1 = self.components[comps[i]]
    #             field2 = other[other_comps[i]]   # other can be Field or Field.components
    #         except(KeyError):
    #             continue
    #
    #         out_comp = helper.convolve(field1,field2)
    #         helper.compress(out_comp, **self.grid.get_compress_opts(compress+2))
    #
    #         if out is None:   out = out_comp
    #         else:
    #             helper.add_MPS(out,out_comp,inplace=True)
    #             # out.distribute_exponent()
    #             # out_comp.distribute_exponent()
    #             # out.add_MPS(out_comp, inplace=True)
    #             helper.compress(out, **self.grid.get_compress_opts(compress+1))
    #
    #     if compress:  helper.compress(out, **self.grid.get_compress_opts(compress))
    #
    #     raise(NotImplementedError)

    # @set_io_type
    # def convective_term(self,comps=None,dims='all',compress=1,field2=None):
    #     """ compute and return (self \cdot \del) field2
    #     """
    #     if field2 is None:   field2 = self.copy()
    #
    #     # if isinstance(comps,int):               comps = [comps]
    #     # elif comps is None or comps=='all':     comps = self.components.keys()
    #
    #     out = []
    #     for comp in comps:
    #         # grad_vecs = field2.gradient(comps=comp,compress=0)[0]
    #         grad_vecs = self.grid.gradient(field2[comp], field2.bcs[comp], dims=dims,
    #                                        compress=compress+1)
    #         # for k, comp in grad_vecs.items():
    #         #     helper.compress(comp, **self.grid.get_compress_opts(compress+2))
    #
    #         # if compress >= 2:
    #         #     for k, comp in grad_vecs.items():
    #         #         # print('convective term',comp.max_bond())
    #         #         helper.compress(comp, **self.grid.get_compress_opts(2))
    #
    #         # print('convective term, max bond',self.components[0].max_bond(),grad_vecs[0].max_bond(),
    #         #                  self.grid.get_mpo_firstderivative(0).max_bond())
    #
    #         out += [self.dot(grad_vecs,compress=compress)[0]]
    #
    #     return out


SCALAR_COORD = Coordinate('_DUMMY', CoordinateType.SCALAR)


class ScalarField(Field):

    def __init__(self, name: str, grid: 'Grid', data: Optional['GridTN'] = None,
                 compress_config: Optional['CompressionConfiguration'] = None,
                 # deriv_configs: Optional[dict]=None,
                 is_sqrt=False):  # , init_split_opts: dict = None):  # , compress_opts: Optional[dict]=None):
        """
        name (str):  name of the field
        ncomp (int): maximum number of components (eg. 1 if scalar field, ndim if vector field)
        grid:  Grid object on which components live
        data:  dictionary containing MPSs indexed by integer denoting which component of the field
               the MPS represents (eg. 0 if MPS is representing a scalar field)
               keys:  ax.axID if a vector field OR None if a scalar field
        boundary_conditions: nested dict.
                first level: dict of boundary conditions for each field component.
                             define for all possible components
                second level: dictionary of boundary conditions along each axis
                              periodic, antiperiodic, open (default), reflecting, zero gradient
                              (elements are tuples to specify left end/right end of axis)
        coords: list of coordinate systems (predefined)
                coordinate systems take dict
                coords_ax:  dictionary in which the coordinate position (BCType or int) keys the desired axis
        """
        super().__init__(name, grid, {SCALAR_COORD: data}, compress_config, is_sqrt)
        self.compID = SCALAR_COORD

    def _set_component(self, compID: 'Coordinate', new_comp: Optional['GridTN']):
        assert (compID == SCALAR_COORD), 'componentID must be SCALAR_COORD'
        super()._set_component(compID, new_comp)

    @property
    def component(self):
        return self._components[SCALAR_COORD]

    @component.setter
    def component(self, new_comp):
        self._set_component(SCALAR_COORD, new_comp)

    def save_data(self, fstr):
        for compID, gtn in self.components.items():
            if gtn is not None:
                gtn.save_data(fstr)

    def reload_data(self, fstr, compIDs: list['Coordinate'] = None, ax_deriv_configs=None):
        grid = self.grid
        return grid.load_scalar_field_data(fstr, self, ax_deriv_configs=ax_deriv_configs)

    @classmethod
    def load_data(cls, grid: 'Grid', fstr, field_name, compIDs: list['Coordinate'],
                  ax_deriv_configs=None, compress_config=None):
        scalar_field_obj = cls(field_name, grid, data=None, compress_config=compress_config)
        return grid.load_scalar_field_data(fstr, scalar_field_obj, ax_deriv_configs=ax_deriv_configs)

    def create_like(self, new_components: dict['Coordinate', Optional['GridTN']] = None) -> 'ScalarField':
        """ creates new field with the same parameters as self
        """
        if new_components is not None:
            new_component = new_components.get(SCALAR_COORD, None)
        else:
            new_component = None

        new_field = self.__class__(self.name, self.grid, new_component, compress_config=self.compress_config,
                                   is_sqrt=self.is_sqrt)  # , init_split_opts=split_opts)
        # new_field._compress_opts_all = self._compress_opts_all
        return new_field

    def copy(self, deep=True) -> 'ScalarField':
        """ makes a copy of itself
        """
        new_field = self.create_like()
        if self.component is not None:
            # print('comp type', type(self.component))
            new_field.component = self.component.copy(deep=deep) if isinstance(self.component, GridTN) \
                else self.component.copy()
        return new_field

    def get_field_data(self, compIDs=None, ax_select=None) -> 'np.ndarray':
        """ return TN field as real space ndarray
        """
        return super().get_comp_data(SCALAR_COORD, ax_select=ax_select)

    def get_comp_data(self, compID=None, ax_select=None, ax_sum=None) -> 'np.ndarray':
        """ return TN field as real space ndarray
        """
        return super().get_comp_data(SCALAR_COORD, ax_select=ax_select, ax_sum=ax_sum)

    def max_bond(self, compID=SCALAR_COORD):
        return super().max_bond(compID)

    # def xmultiply(self, x_axes: Sequence['Axis'], compID=None, offsets=None, scales=None, compress_level=0) -> 'Field':
    #     """ compute x_i {x_i} * f(x_i) elemental multiplication for all x_i in x_axes
    #         not an inplace operation; generates new field with components x_i
    #     """
    #     return super().xmultiply(x_axes, compID, offsets, scales, compress_level)

    ##############################
    ## operations on the fields ##
    ##############################

    # def integrate(self, compIDs=None, axes=None, new_grid=None, new_ax_deriv_configs=None, inplace=False) \
    #     -> 'ScalarField':
    #     return super().integrate(compIDs, axes, new_grid, new_ax_deriv_configs, inplace)

    def cross_product(self, other, comps=None, other_comps=None, compress_level=0, inner_compress_level=0):
        """ take cross product of self with components of other field
            ohter can be Field object or Field.components
            out = self x other
            comps, comps_other:  which components of self and other to take the cross product with,
                               defaults to the first three components
        """
        raise TypeError('cross product not defined for scalar field')

    def gradient(self, compID=None, deriv_axes: Optional[Sequence['Axis']] = None,
                 upwind_axes: Optional[dict['Axis', 'Axis']] = None,
                 ax_deriv_config: ['Axis', 'DerivativeConfiguration'] = None,
                 compress_level: int = 0) -> 'Field':
        """ df/dx {x} + df/dy {y} + df/dz {z}
        """
        return super().gradient(SCALAR_COORD, deriv_axes, upwind_axes, ax_deriv_config, compress_level)  # Field object

    ############
    ## use coordinate class ##

    # def laplacian(self, compID=None, deriv_axes: list['Axis'] or None = None, compress_level=0,
    #               inner_compress_level=0) -> 'ScalarField':
    #     """ L(f) = d^2/dx^2 f + d^2/dy^2 f + d^2/dz^2 z
    #         mps_scalar_field:  mps representing the scalar field
    #         bcs:  boundary conditions for the relevant field component
    #     """
    #     return super().laplacian(SCALAR_COORD, deriv_axes, compress_level, inner_compress_level )

    def laplacian(self, coord_sys: 'CoordinateSystem', deriv_axes=None, out_comps=None, compress_level=0,
                  inner_compress_level=0, ) \
            -> 'ScalarField':

        if deriv_axes is None:
            deriv_axes = self.grid.axes

        out = coord_sys.laplacian(self, deriv_axes=deriv_axes,
                                  compress=compress_level,
                                  compress_opts=self.compress_config[compress_level],
                                  inner_compress=inner_compress_level,
                                  inner_compress_opts=self.compress_config[inner_compress_level])

        return out

    def curl(self, coord_sys: 'CoordinateSystem', out_comps=None, compress_level=0, inner_compress_level=0):
        """ Curl(F) =  ( d/dy Az - d/dz Ay ) {x} +
                       ( d/dz Ax - d/dx Az ) {y} +
                       ( d/dx Ay - d/dy Ax ) {z}
        """
        raise TypeError('curl not defined for scalar field')

    def divergence(self, coord_sys: 'CoordinateSystem', compIDs=None, ax_deriv_configs=None, compress_level=0,
                   inner_compress_level=0):
        """ d/dx Ax + d/dy Ay + d/dz Az
        """
        raise TypeError('divergence not defined for scalar field')

    #### measurement  ####

    def meas_expec(self, obs_gtn_dict, compIDs=None, integ_axes=None, new_grid=None, new_ax_deriv_configs=None):
        """ integrate over axes specified by axIDs. integrates over all if axIDs=None -> scalar
            otherwise, returns a Field object with reduced dimensionality
            inplace:  if new GridTN object is generated or not
        """
        # if new_grid is None and axes is not None:
        #     new_grid = self.grid.create_like([ax for ax in self.grid.axes if ax not in axes])

        if isinstance(obs_gtn_dict, (GridTN, qtn.TensorNetwork)) or obs_gtn_dict is None:
            out = super().meas_expec({SCALAR_COORD: obs_gtn_dict}, compIDs=[SCALAR_COORD], integ_axes=integ_axes,
                                     new_grid=new_grid, new_ax_deriv_configs=new_ax_deriv_configs)
            out_scalar = out.create_like_scalar(new_component=out.components[SCALAR_COORD])
            return out_scalar
        else:
            out = {}
            for key, obs_gtn in obs_gtn_dict.items():
                out[key] = super().meas_expec({SCALAR_COORD: obs_gtn}, compIDs=[SCALAR_COORD], integ_axes=integ_axes,
                                              new_grid=new_grid, new_ax_deriv_configs=new_ax_deriv_configs)[
                    SCALAR_COORD]
                if new_grid is None:    new_grid = out[key].grid

            return Field(self.name, new_grid, out, compress_config=self.compress_config, is_sqrt=False)
