"""Site-level tensor callables for the local-solver stack.

Provides the per-site operation functions -- applying an effective operator,
adding tensors, and building polynomial/power callables -- that follow the
``(site_tensor, effective_operators, output_to_input_inds)`` calling
convention. Term objects use these callables to define how each term acts on
the current site during an Evaluator sweep.
"""
from setup_.defaults import *
import helper_quimb as helper
import local_solvers.helper_tn as helper_tn
from helper_dmrg import qtn_conjugate_gradient_descent_1site, qtn_conjugate_gradient_squared_1site
from helper_dmrg import qtn_constrained_cgs_1site


""" func arguments should be 
    (site_tensor, effective_operators, output_to_input_inds)
"""


### tensor callables
def apply_effective_op(site_tens: 'qtn.Tensor', eff_ops: Sequence[Union['qtn.Tensor', 'qtn.TensorNetwork']],
                       output_to_input_inds: dict[str, str]):

    tens_list = []
    for eff_op in eff_ops:
        # print('site tens inds', site_tens.inds)
        out_tens = qtn.tensor_contract(*eff_op.tensors, site_tens)
        # print('eval eff op', out_tens.norm())

        # site_tens_copy = site_tens.reindex({v:k for v, k in output_to_input_inds.items()})
        # out_tens = qtn.tensor_contract(*eff_op.tensors, site_tens_copy)
        # print('test eval eff op', out_tens.norm())

        out_tens.modify(apply=lambda x: x * 10 ** eff_op.exponent)
        tens_list += [out_tens]
    out = helper_tn.sum_tens(tens_list)
    out.reindex(output_to_input_inds, inplace=True)
    return out


def add_func(site_tens: 'qtn.Tensor', other_tens: Union['qtn.Tensor', list['qtn.Tensor']],
             output_to_input_inds: dict[str, str]) -> 'qtn.Tensor':

    if not isinstance(other_tens, list):
        other_tens = [other_tens]
    out = helper_tn.sum_tens([site_tens.copy(), *other_tens])
    # out.reindex(output_to_input_inds, inplace=True)
    return out

# ### methods to get tensor callables
# def get_power_func(power: int):
#     def power_func(site_tens: 'qtn.Tensor', eff_ops: Sequence['qtn.TensorNetwork'],
#                    output_to_input_inds: dict[str, str]) -> 'qtn.Tensor':
#
#         out = apply_effective_op(site_tens, eff_ops, output_to_input_inds)
#         for it in range(power - 2):
#             out = apply_effective_op(out, eff_ops, output_to_input_inds)
#         return out
#
#     return power_func


### methods to get tensor callables
def get_power_func(power: int):
    def power_func(site_tens: 'qtn.Tensor', eff_ops: dict[int, Sequence['qtn.TensorNetwork']],
                   output_to_input_inds: dict[str, str], return_intermediates: bool=False
                   ) -> Union['qtn.Tensor', tuple['qtn.Tensor', dict[int,'qtn.Tensor']]]:
        """ intermediate values only needed for intermediate kets
            1: vec_block value
            2: vec_block * op_eff value
            3: vec_block * op_eff[1] * op_eff[2] value
        """
        num_tiers = len(eff_ops)
        intermediates = {0: [site_tens.copy()]} # if num_tiers > 1 else {}

        out = site_tens.copy()
        for it in range(power):
            it_ = min(it, num_tiers - 1)
            out = apply_effective_op(site_tens, eff_ops[it_], output_to_input_inds)
            if it_ + 1 < num_tiers - 1:
                print('power func intermediate', it_ + 1)
                intermediates[it_ + 1] = [out.copy()]
            # out = apply_effective_op(out, eff_ops[it_], output_to_input_inds)

        # print("POwer OUT", out.norm())

        if return_intermediates:
            return out, intermediates

        return out

    return power_func


def get_element_wise_func(func) -> Callable:

    # def tens_func(tens: 'qtn.Tensor', eff_ops: Sequence[Union['qtn.Tensor', 'qtn.TensorNetwork']],
    #               output_to_input_inds: dict[str, str]):
    def tens_func(tens: 'qtn.Tensor', *args, **kwargs):
        tens = tens.copy()
        tens.modify(apply=lambda x: func(x))
        return tens

    return tens_func


def get_euler_func(dt: Numeric, source_func: Optional[Callable]=None) -> Callable:
    """ dx/dt = Ax + b
        returns:  y = (1 + dt * A) x + dt * b
        but term only contains either A or (a portion of) b
    """
    def euler_func(tens: 'qtn.Tensor', eff_ops: Sequence[Union['qtn.Tensor', 'qtn.TensorNetwork']],
                   output_to_input_inds: dict[str, str]):

        if eff_ops is not None and len(eff_ops) > 0:
            deriv = apply_effective_op(tens, eff_ops, output_to_input_inds)
        else:
            deriv = source_func(tens)

        deriv.modify(apply=lambda x: x)
        out = helper.add_tensors(tens, deriv)
        return out

    return euler_func

### for solving linear system
def get_cgd_func(is_H = False, has_constraints=False, **conv_kwargs):

    conv_kwargs.setdefault('max_iter', 500)
    conv_kwargs.setdefault('conv_tol', 1.0e-6)

    def solve_cgd(site_tens: 'qtn.Tensor', eff_ops: Sequence[Union['qtn.TensorNetwork', 'qtn.Tensor']],
                  output_to_input_inds: dict[str, str], return_intermediates=False, init_guess=None,
                  constraint_tns: Sequence[Union['qtn.TensorNetwork', 'qtn.Tensor']] = None,
                  constraint_vals: 'qtn.Tensor'=None):
        """ solve Ax=b
            b_eff: site_tens
            A_eff: sum(eff_ops)
            return x_eff
        """
        init_guess = site_tens.copy() if init_guess is None else init_guess
        if has_constraints:
            out, _, err = qtn_constrained_cgs_1site(eff_ops, site_tens, init_guess,
                                                    output_to_input_inds,
                                                    constraint_tns=constraint_tns, constraint_vals=constraint_vals,
                                                    **conv_kwargs)
        else:
            if is_H:
                out, err = qtn_conjugate_gradient_descent_1site(eff_ops, site_tens, init_guess,
                                                                output_to_input_inds, **conv_kwargs)
            else:
                out, err = qtn_conjugate_gradient_squared_1site(eff_ops, site_tens, init_guess,
                                                                output_to_input_inds, **conv_kwargs)

        if return_intermediates:
            return out, [], err
        else:
            return out, err

    return solve_cgd


def get_lsq_func(**conv_kwargs):

    conv_kwargs.setdefault('max_iter', 500)
    conv_kwargs.setdefault('conv_tol', 1.0e-6)

    def solve_lsq(site_tens: 'qtn.Tensor', eff_ops: Sequence[Union['qtn.TensorNetwork', 'qtn.Tensor']],
                  output_to_input_inds: dict[str, str]):
        """ solve Ax=b
            b_eff: site_tens
            A_eff: sum(eff_ops)
            return x_eff
        """
        shape_ = site_tens.shape
        size_ = int(np.prod(shape_))

        bonds_o = output_to_input_inds.keys()
        bonds_i = [output_to_input_inds[bo] for bo in bonds_o]
        A_eff_tens = helper_tn.sum_eff_TNs(eff_ops, [*bonds_o, *bonds_i])
        explicit_contribution = site_tens.transpose(*bonds_i)

        Amat = A_eff_tens.data.reshape(size_, size_)
        bvec = explicit_contribution.data.reshape(size_)

        out_data = np.linalg.solve(Amat, bvec)

        out = qtn.Tensor(out_data.reshape(shape_), inds=bonds_i)
        return out

    return solve_lsq
