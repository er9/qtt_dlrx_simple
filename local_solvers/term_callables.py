from setup_.defaults import *
import local_solvers.helper_tn as helper_tn
from local_solvers.terms import Term  # Term_DMRG, Term_Cross
import helper_TE

### for time evolution
def get_explicit_TE_func(dt, deriv_combine_func: 'Callable', te_order:int = 4):

    def deriv_func(terms: Sequence['Term'],
                   output_to_input_inds: dict[str, str],
                   left_site_pos: int, nsites: int,
                   site_tens: 'qtn.Tensor' = None) -> 'qtn.Tensor':
        """ for df/dt = Af + b
        """
        eff_Ax = []
        for term in terms:
            eff_Ax += [term.get_evaluated_site(left_site_pos, nsites, site_tens=site_tens)]
            ## dist_submpx exponent should equal 0

        bonds_o = list(output_to_input_inds.keys())
        # bonds_i = [output_to_input_inds[bo] for bo in bonds_o]

        out = deriv_combine_func(eff_Ax)
        out.transpose(*bonds_o, inplace=True)
        out = helper_tn.sum_eff_TNs(eff_Ax, transpose_bonds=bonds_o)
        # out = qtn.tensor_contract(deriv_tens, *dist_submpx_)
        out.reindex({bo: bi for bo, bi in output_to_input_inds.items()}, inplace=True)
        out.modify(apply=lambda x: x * dt)
        return out

    def scale_func(tens: 'qtn.Tensor', scale_val: 'Numeric', inplace=False) -> 'qtn.Tensor':
        out = tens if inplace else tens.copy()
        out.modify(apply=lambda x: x * scale_val)
        return out

    def add_func(tens_list: Sequence['qtn.Tensor']) -> 'qtn.Tensor':
        return helper_tn.sum_tens(tens_list)

    def euler_func(tens, dt, deriv0, **kwargs) -> 'qtn.Tensor':
        return add_func([tens, scale_func(deriv0, dt)])


    def explicit_te():



        if te_order == 4:
            helper_TE.rk4(T1, dt, euler_func, deriv_func, add_func, scale_func)

        return out


### for solving linear system
