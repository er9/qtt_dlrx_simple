import sys
sys.path.append('../')

from setup_.configs import *
# from gate import Gate
# from mps import MatrixProductState, MatrixProductOperator
import quimb.tensor as qtn
import helper_quimb as helper
from axis import Axis
from grid1D import Grid1D
from gridTN_1D import GridTN1D
# from local_solvers_old.local_evaluator import local_dmrg_evaluator
from local_solvers.terms_3 import Term_DMRG, Term_Cross
from local_solvers.local_dmrg_eval import local_dmrg_evaluator
from local_solvers.local_cross_eval import local_cross_evaluator
from local_solvers.terms_mixed import Term_Mixed
from local_solvers.local_mixed_eval import local_mixed_evaluator

L = 10
Ds = [4, 8, 16, 24, 32, None]
legs = []
direction = 1
nsites = 2

x = np.linspace(0,2*np.pi,2**L, endpoint=False)
dx = 2*np.pi/(2**L)
print('dx', dx, x[1] - x[0])
fx = lambda x: 1 + np.sin(5*x)*5.0 * np.exp(-x**2/6) + np.cos(10*x)

x_scale = 0.1

data = fx(x)
np.random.seed(0)
data = data + np.random.rand(len(data))
data = data / np.linalg.norm(data)
print('data_tens', np.linalg.norm(data))

ax_x = Axis(L)
grid_X = Grid1D('GX', [ax_x])


# data_tens = qtn.Tensor( data.reshape((2,)*L), inds=tuple([f'i{i}' for i in range(L)]) )
# init_mps = MatrixProductState.from_vec(data_tens, L, 'i{}', split_opts={'max_bond': D})
init_mps = GridTN1D.from_dense_state(data, grid_X)
init_mps.data.exponent += np.log10(x_scale)
data = data * x_scale
print('init mps exp', init_mps.exponent)
print('init mps max bond', init_mps.max_bond())

plt.figure()
plt.plot(init_mps.get_data())
plt.plot(data, '--')
plt.show()

dmrg_errs = []
cross_errs = []
mixed_errs = []
comp_errs = []

def L1_err(data1: np.ndarray, data2: np.ndarray):
    return np.sum(np.abs(data1 - data2)) # / data1.shape


### distirubte exponent?
# init_mps.data.distribute_exponent()

### test 1:  return self in canonical form
do_test_1 = False
if do_test_1:
    legs += ['scale f']
    errs, errs_2, errs_3, errs_4 = [], [], [], []
    func = lambda x: x * 0.5

    for D in Ds:

        term1d = Term_DMRG(init_mps.data.copy(), mps_coeff=0.5)
        term1c = Term_Cross(init_mps.data.copy(), mps_func=func)  # eval_func=eval_func)

        ## dmrg
        func_mps = local_dmrg_evaluator((term1d,), direction=direction,
                                        nsites=nsites, max_bond=D)
        func_gtn = grid_X.make_gridTN(func_mps)

        func_data = func_gtn.get_data()
        err = np.linalg.norm(func_data - func(data)) / np.linalg.norm(func(data))
        errs += [err]
        print('dmrg err', err)

        ## cross
        func_mps = local_cross_evaluator((term1c,), direction=direction,
                                         nsites=nsites, max_bond=D)
        func_gtn = grid_X.make_gridTN(func_mps)

        print('cross soln', func_gtn.data)
        func_data = func_gtn.get_data()
        err = np.linalg.norm(func_data - func(data)) / np.linalg.norm(func(data))
        errs_3 += [err]
        print('cross err', err)

        ## mixed
        term1m = Term_Mixed(init_mps.data.copy(), mps_func=func)
        func_mps = local_mixed_evaluator((term1m,), direction=direction,
                                         nsites=nsites, max_bond=D)
        func_gtn = grid_X.make_gridTN(func_mps)

        print('cross soln', func_gtn.data)
        func_data = func_gtn.get_data()
        err = np.linalg.norm(func_data - func(data)) / np.linalg.norm(func(data))
        errs_4 += [err]
        print('cross err', err)

        # comp_mps_2 = helper.compress_rdm(init_mps.data.copy(), compress_opts={'max_bond': D},
        #                                  direction=-1)
        # comp_gtn = grid_X.make_gridTN(comp_mps_2)
        # comp_data = comp_gtn.get_data()
        # err = np.linalg.norm(func(comp_data) - func(data)) / np.linalg.norm(func(data))
        # print('err rdm', err)

        ### direct SVD compression
        comp_mps = GridTN1D.from_dense_state(func(data), grid_X, split_opts={'max_bond': D})
        print('comp_mps', comp_mps.max_bond())
        comp_data = comp_mps.get_data()
        err = np.linalg.norm(comp_data - func(data)) / np.linalg.norm(func(data))
        print('comp err', err)
        errs_2 += [err]

        if D is not None and D < 16:
            plt.figure()
            plt.plot(np.real(func_data), label='func dmrg')
            plt.plot(np.real(func(data)), ':', label='exact')
            plt.plot(np.imag(func_data), '--')
            plt.plot(np.imag(func(data)), ':')
            plt.legend()
            plt.title(f'QTT rank = {D}')
            plt.show()

    dmrg_errs += [errs]
    comp_errs += [errs_2]
    cross_errs += [errs_3]
    mixed_errs += [errs_4]


### add init_mps + constant
do_test_2 = True
if do_test_2:
    legs += ['add const.']
    errs, errs_2, errs_3, errs_4 = [], [], [], []
    y_mps = GridTN1D.get_ones_mps(grid_X)
    y_mps.scalar_multiply(0.5, inplace=True)

    init_mps.data.distribute_exponent()
    y_mps.data.distribute_exponent()

    for D in Ds:
        term1d = Term_DMRG(init_mps.data.copy())
        term2d = Term_DMRG(y_mps.data.copy())

        term1c = Term_Cross(init_mps.data.copy())
        term2c = Term_Cross(y_mps.data.copy())

        ## dmrg ##
        func_mps = local_dmrg_evaluator([term1d, term2d], direction=direction,
                                        max_bond=D, nsites=nsites)
        func_gtn = grid_X.make_gridTN(func_mps)

        print('func gtn', func_gtn.data)
        func_gtn.check_orthog()

        func_data = func_gtn.get_data()
        err = np.linalg.norm(func_data - (data + 0.5)) / np.linalg.norm(data + 0.5)
        print('err', err)
        errs += [err]

        ## mixed ##
        term1m = Term_Mixed(init_mps.data.copy())
        term2m = Term_Mixed(y_mps.data.copy())

        func_mps = local_mixed_evaluator([term1m, term2m], direction=direction,
                                        max_bond=D, nsites=nsites)
        func_gtn = grid_X.make_gridTN(func_mps)

        print('func gtn', func_gtn.data)
        func_gtn.check_orthog()

        func_data = func_gtn.get_data()
        err = np.linalg.norm(func_data - (data + 0.5)) / np.linalg.norm(data + 0.5)
        print('err', err)
        errs_4 += [err]

        ## cross ##
        func_mps = local_cross_evaluator([term1c, term2c], direction=direction,
                                          max_bond=D, nsites=nsites)
        func_gtn = grid_X.make_gridTN(func_mps)

        print('max bond', func_gtn.data.max_bond())
        func_data = func_gtn.get_data()
        err = np.linalg.norm(func_data - (data + 0.5)) / np.linalg.norm(data + 0.5)
        print('cross err', err)
        errs_3 += [err]

        if D is not None and D < 16:
            plt.figure()
            plt.plot(np.real(func_data), label='dmrg')
            # plt.plot(np.real(func(data * x_scale)), ':', label='exact')
            plt.plot(np.real(data + 0.5), ':', label='exact')
            plt.plot(np.imag(func_data), '--')
            plt.plot(np.imag(data + 0.5), ':')
            plt.legend()
            plt.title(f'D = {D}')
            plt.show()

        ## svd
        comp_gtn = GridTN1D.from_dense_state(data + 0.5, grid_X, split_opts={'max_bond': D})
        comp_data = comp_gtn.get_data()
        err = np.linalg.norm(comp_data - (data + 0.5)) / np.linalg.norm(data + 0.5)
        print('comp err', err)
        errs_2 += [err]


    dmrg_errs += [errs]
    comp_errs += [errs_2]
    cross_errs += [errs_3]
    mixed_errs += [errs_4]


### add init_mps + other fct
do_test_3 = True
if do_test_3:
    legs += ['add fct']
    fx = lambda x:  x**2 * 4e-4
    y_mps = GridTN1D.from_dense_state(fx(x), grid_X)

    # init_mps.data.distribute_exponent()
    # y_mps.data.distribute_exponent()

    errs, errs_2, errs_3, errs_4 = [], [], [], []
    for D in Ds:

        term1d = Term_DMRG(init_mps.data.copy())
        term2d = Term_DMRG(y_mps.data.copy())

        term1c = Term_Cross(init_mps.data.copy())
        term2c = Term_Cross(y_mps.data.copy())

        ## dmrg
        func_mps = local_dmrg_evaluator([term1d, term2d], direction=direction,
                                        max_bond=D, nsites=nsites)
        func_gtn = grid_X.make_gridTN(func_mps)

        print('func gtn', func_gtn.data.max_bond())
        func_gtn.check_orthog()

        func_data = func_gtn.get_data()
        err = np.linalg.norm(func_data - (data + fx(x))) / np.linalg.norm(data + fx(x))
        print('err', err)
        errs += [err]

        ## cross
        func_mps = local_cross_evaluator([term1c, term2c], direction=direction,
                                         max_bond=D, nsites=nsites)
        func_gtn = grid_X.make_gridTN(func_mps)

        func_data = func_gtn.get_data()
        err = np.linalg.norm(func_data - (data + fx(x))) / np.linalg.norm(data + fx(x))
        print('err x', err, 'max bond', func_gtn.max_bond())
        errs_3 += [err]

        ## mixed ##
        term1m = Term_Mixed(init_mps.data.copy())
        term2m = Term_Mixed(y_mps.data.copy())
        func_mps = local_mixed_evaluator([term1m, term2m], direction=direction,
                                         max_bond=D, nsites=nsites)
        func_gtn = grid_X.make_gridTN(func_mps)

        func_data = func_gtn.get_data()
        err = np.linalg.norm(func_data - (data + fx(x))) / np.linalg.norm(data + fx(x))
        print('err x', err, 'max bond', func_gtn.max_bond())
        errs_4 += [err]

        if D is not None and D < 16:
            plt.figure()
            plt.plot(np.real(func_data), label='dmrg')
            # plt.plot(np.real(func(data * x_scale)), ':', label='exact')
            plt.plot(np.real(data + fx(x)), ':', label='exact')
            plt.plot(np.imag(func_data), '--')
            plt.plot(np.imag(data + fx(x)), ':')
            plt.legend()
            plt.title(f'D = {D}')
            plt.show()

        ## direct SVD compression
        comp_gtn = GridTN1D.from_dense_state(data + fx(x), grid_X, split_opts={'max_bond': D})
        comp_data = comp_gtn.get_data()
        err = np.linalg.norm(comp_data - (data + fx(x))) / np.linalg.norm(data + fx(x))
        print('comp err', err)
        errs_2 += [err]

    dmrg_errs += [errs]
    comp_errs += [errs_2]
    cross_errs += [errs_3]
    mixed_errs += [errs_4]


### mult init_mps * other fct
### note: as with test 5, we get better performance if the contributions to the target are normalized
### (this makes a difference here because norms of f, f*g can be very different in norms)
do_test_7 = True
if do_test_7:
    legs += ['elem mult']
    fx = lambda x:  x**2 * 4e-4
    y_mps = GridTN1D.from_dense_state(fx(x), grid_X)
    # y_mpo = y_mps.apply_elemental_multiply_op()

    # init_mps.data.distribute_exponent()
    # y_mps.data.distribute_exponent()

    errs, errs_2, errs_3, errs_4 = [], [], [], []
    for D in Ds:

        term1d = Term_DMRG(init_mps.data.copy(), operators=[y_mps.data.copy()], num_tiers=2)
        # term1d = Term_DMRG(init_mps.data.copy(), operators=[y_mpo.data.copy()], num_tiers=1)

        # term1c = Term_Cross(init_mps.data.copy(), operators=[y_mps.data.copy()])
        # term1c = Term_Cross(init_mps.data.copy(), operators=[y_mpo.data.copy()])
        term1c = Term_Cross(init_mps.data.copy())
        term2c = Term_Cross(y_mps.data.copy())

        ## dmrg
        func_mps = local_dmrg_evaluator([term1d], direction=direction, max_bond=D, nsites=nsites)
        # func_gtn = grid_X.make_gridTN(term1d.vec_block.bra)
        func_gtn = grid_X.make_gridTN(func_mps)

        print('func gtn', func_gtn.data)
        func_gtn.check_orthog()

        func_data = func_gtn.get_data()
        err = np.linalg.norm(func_data - (data * fx(x))) / np.linalg.norm(data * fx(x))
        # err = np.linalg.norm(func_data - (data)) / np.linalg.norm(data)
        print('err', err, 'max bond', func_mps.max_bond())
        errs += [err]

        if D is not None and D < 16:
            plt.figure()
            plt.plot(np.real(func_data), label='dmrg')
            plt.plot(np.real(data * fx(x)), ':', label='exact')
            # plt.plot(np.real(data ), ':', label='exact')
            plt.plot(np.imag(func_data), '--')
            plt.plot(np.imag(data * fx(x)), ':')
            plt.legend()
            plt.title(f'D = {D}')
            plt.show()



        import local_solvers.helper_tn as helper_tn
        ## cross
        # func_mps = local_cross_evaluator([term1c], direction=direction,
        #                                   max_bond=D, nsites=nsites)
        func_mps = local_cross_evaluator([term1c, term2c], direction=direction,
                                         max_bond=D, nsites=nsites,
                                         combine_terms_func=helper_tn.prod_tens)
        func_gtn = grid_X.make_gridTN(func_mps)

        func_data = func_gtn.get_data()
        err = np.linalg.norm(func_data - (data * fx(x))) / np.linalg.norm(data * fx(x))
        print('err x', err, 'max bond', func_mps.max_bond())
        errs_3 += [err]

        ## mixed ##
        term1m = Term_Mixed(init_mps.data.copy())
        term2m = Term_Mixed(y_mps.data.copy())
        func_mps = local_mixed_evaluator([term1m, term2m], direction=direction,
                                         max_bond=D, nsites=nsites,
                                         combine_terms_func=helper_tn.prod_tens)
        func_gtn = grid_X.make_gridTN(func_mps)

        func_data = func_gtn.get_data()
        err = np.linalg.norm(func_data - (data * fx(x))) / np.linalg.norm(data * fx(x))
        print('err x', err, 'max bond', func_mps.max_bond())
        errs_4 += [err]


        if D is not None and D < 16:
            plt.figure()
            plt.plot(np.real(func_data), label='dmrg')
            plt.plot(np.real(data * fx(x)), ':', label='exact')
            # plt.plot(np.real(data ), ':', label='exact')
            plt.plot(np.imag(func_data), '--')
            plt.plot(np.imag(data * fx(x)), ':')
            plt.legend()
            plt.title(f'D = {D}')
            plt.show()

        # exit()


        ## direct SVD compression
        comp_gtn = GridTN1D.from_dense_state(data * fx(x), grid_X, split_opts={'max_bond': D})
        comp_data = comp_gtn.get_data()
        err = np.linalg.norm(comp_data - (data * fx(x))) / np.linalg.norm(data * fx(x))
        print('comp err', err)
        errs_2 += [err]

        # exit()

    dmrg_errs += [errs]
    comp_errs += [errs_2]
    cross_errs += [errs_3]
    mixed_errs += [errs_4]



### operator * init_mps
do_test_4 = False
nsites = 1
if do_test_4:
    legs += ['mat-vec mult']
    op = grid_X.get_firstderivative_mpo(ax_x)
    op2 = ax_x.get_shift_mpo(5)
    helper.scalar_multiply(op2, 0.1 / ax_x.dx, inplace=True)
    op = helper.add_MPO(op.data, op2, inplace=False)
    op = grid_X.make_gridTN(op)
    # op = grid_X.get_iden_mpo()

    num_tiers = 2

    plt.figure()
    plt.imshow(op.get_data())
    plt.show()

    # init_mps.data.distribute_exponent()
    # op.data.distribute_exponent()

    glob_apply = init_mps.apply(op)
    glob_data = glob_apply.get_data()

    errs, errs_2, errs_3, errs_4 = [], [], [], []
    for D in Ds:
        term1d = Term_DMRG(init_mps.data.copy(), operators=[op.data.copy()], num_tiers=num_tiers)
        # term1d = Term_DMRG(init_mps.data.copy(), operators=[op.data.copy()], num_tiers=2)
        ## performs worse with num_tiers = 2. I suppose this is because the ket projector on the operator
        ## (<b|O|k> spans a smaller space in this case)
        term1c = Term_Cross(init_mps.data.copy(), operators=[op.data.copy()], num_tiers=num_tiers)
        ## num_tiers = 1 doesn't work because ket == out; and out is updated at each iteration

        ## dmrg
        loc_apply = local_dmrg_evaluator([term1d], direction=direction, max_bond=D, nsites=nsites)

        loc_apply = grid_X.make_gridTN(loc_apply)
        loc_data = loc_apply.get_data()
        err = np.linalg.norm(glob_data - loc_data) / np.linalg.norm(glob_data)
        print('err', err, 'max bond', loc_apply.max_bond())
        errs += [err]

        if D is not None and D < 8:
            plt.figure()
            plt.plot(np.real(loc_data), label='dmrg')
            plt.plot(np.real(glob_data), ':', label='exact')
            plt.plot(np.imag(loc_data), '--')
            plt.plot(np.imag(glob_data), ':')
            plt.legend()
            plt.title(f'D = {D}')
            plt.show()

        ## cross
        loc_apply = local_cross_evaluator([term1c], direction=direction, max_bond=D, nsites=nsites)
        loc_apply = grid_X.make_gridTN(loc_apply)
        loc_data = loc_apply.get_data()
        err = np.linalg.norm(glob_data - loc_data) / np.linalg.norm(glob_data)
        print('err x', err, 'max bond', loc_apply.max_bond())
        errs_3 += [err]

        ## mixed ##
        term1m = Term_Mixed(init_mps.data.copy(), operators=[op.data.copy()], num_tiers=num_tiers)
        loc_apply = local_mixed_evaluator([term1m], direction=direction, max_bond=D, nsites=nsites)
        loc_apply = grid_X.make_gridTN(loc_apply)
        loc_data = loc_apply.get_data()
        err = np.linalg.norm(glob_data - loc_data) / np.linalg.norm(glob_data)
        print('err x', err, 'max bond', loc_apply.max_bond())
        errs_4 += [err]

        if D is not None and D < 8:
            plt.figure()
            plt.plot(np.real(loc_data), label='nuxed')
            plt.plot(np.real(glob_data), ':', label='exact')
            plt.plot(np.imag(loc_data), '--')
            plt.plot(np.imag(glob_data), ':')
            plt.legend()
            plt.title(f'D = {D}')
            plt.show()



        ## svd
        comp_gtn = glob_apply.compress(compress_opts={'max_bond': D}, inplace=False)
        comp_data = comp_gtn.get_data()
        err = np.linalg.norm(comp_data - glob_data) / np.linalg.norm(glob_data)
        print('comp err', err)
        errs_2 += [err]



    dmrg_errs += [errs]
    comp_errs += [errs_2]
    cross_errs += [errs_3]
    mixed_errs += [errs_4]


### test 5:  return self in canonical form  (but this method can't include 0 power term)
### note: we get better performance if the contributions to the target are normalized
### (this makes a bigger here because norms of x, x^2, x^3 can be very different
do_test_5 = False
if do_test_5:
    power = 3
    coeff = 1.0  # 0.5

    legs += [fr'$x^{power}$ eval']
    func = lambda x: coeff * (x ** power)

    term1c = Term_Cross(init_mps.data.copy(), mps_func=func)
    term1m = Term_Mixed(init_mps.data.copy(), mps_func=func)
    term1d = Term_DMRG(init_mps.data.copy(), mps_power=power, mps_coeff=coeff)

    errs, errs_2, errs_3, errs_4 = [], [], [], []
    for D in Ds:

        ## dmrg
        term1d_copy = term1d.copy_new()
        func_mps = local_dmrg_evaluator([term1d_copy], direction=direction,
                                        max_bond = D, nsites=nsites)
        func_gtn = grid_X.make_gridTN(func_mps)

        func_gtn.check_orthog()
        print('dmrg max bond', func_gtn.data.max_bond())
        print('dmrg term1', term1d_copy.vec_block.bra.max_bond(), term1d_copy.vec_block.ket.max_bond())

        func_data = func_gtn.get_data()
        err = np.linalg.norm(func_data - func(data)) / np.linalg.norm(func(data))
        print('err', err)
        errs += [err]

        if D is not None and D < 8:
            plt.figure()
            plt.plot(np.real(func_data), label='func dmrg')
            plt.plot(np.real(func(data)), ':', label='exact')
            plt.plot(np.imag(func_data), '--')
            plt.plot(np.imag(func(data)), ':')
            plt.legend()
            plt.title(f'D = {D}, {legs[-1]}')
            plt.show()

        ## cross
        term1c_copy = term1c.copy_new()
        func_mps = local_cross_evaluator([term1c_copy], direction=direction,
                                         max_bond=D, nsites=nsites)
        func_gtn = grid_X.make_gridTN(func_mps)

        print('cross max bond', func_gtn.data.max_bond())
        print('cross term1', term1c_copy.vec_block.bra.max_bond(), term1c_copy.vec_block.ket.max_bond())
        func_data = func_gtn.get_data()
        err = np.linalg.norm(func_data - func(data)) / np.linalg.norm(func(data))
        print('err', err)
        errs_3 += [err]

        ## mixed ##
        term1m_copy = term1m.copy_new()
        func_mps = local_mixed_evaluator([term1m_copy], direction=direction, max_bond=D, nsites=nsites)
        func_gtn = grid_X.make_gridTN(func_mps)

        print('mixed max bond', func_gtn.data.max_bond())
        print('mixed term1', term1m_copy.vec_block.bra.max_bond(), term1m_copy.vec_block.ket.max_bond())
        func_data = func_gtn.get_data()
        err = np.linalg.norm(func_data - func(data)) / np.linalg.norm(func(data))
        print('err', err)
        errs_4 += [err]

        if D is not None and D < 8:
            plt.figure()
            plt.plot(np.real(func_data), label='func dmrg')
            plt.plot(np.real(func(data)), ':', label='exact')
            plt.plot(np.imag(func_data), '--')
            plt.plot(np.imag(func(data)), ':')
            plt.legend()
            plt.title(f'D = {D}, {legs[-1]}')
            plt.show()



        ## svd
        comp_gtn = GridTN1D.from_dense_state(func(data), grid_X, split_opts={'max_bond': D})
        comp_data = comp_gtn.get_data()
        err = np.linalg.norm(comp_data - func(data)) / np.linalg.norm(func(data))
        print('comp err', err)
        errs_2 += [err]


    dmrg_errs += [errs]
    comp_errs += [errs_2]
    cross_errs += [errs_3]
    mixed_errs += [errs_4]

### d/dx f(x)**3
### this doesn't work with bond dimension = None, i think
### this won't work with current implementation -- don't allow for elementwise operators at intermedaite stage
### (the f(x)**3 being acted on by the d/dx
do_test_6 = False
if do_test_6:
    power = 3
    coeff = 0.5

    num_tiers = 2

    legs += ['d/dx f(x)**3']
    op = grid_X.get_firstderivative_mpo(ax_x, recalc=True)
    op.scalar_multiply(2.3)
    print('grid X.ax', grid_X.axes[0] is ax_x)
    plt.figure()
    plt.imshow(op.get_data())
    plt.show()

    func = lambda x: coeff * x ** power

    init_data = init_mps.get_data()
    init_mps3 = grid_X.make_gridTN(init_data ** power)
    glob_apply = init_mps3.apply(op)
    glob_data = glob_apply.get_data() * coeff

    term1d = Term_DMRG(init_mps.data.copy(), operators=[op.data.copy()], mps_power=power, mps_coeff=coeff, num_tiers=num_tiers)
                        #, num_tiers=2)     ## this doesn't work; inconsistent bond dimensions.
    term1c = Term_Cross(init_mps.data.copy(), operators=[op.data.copy()], mps_func=func, num_tiers=num_tiers)
    term1m = Term_Mixed(init_mps.data.copy(), operators=[op.data.copy()], mps_func=func, num_tiers=num_tiers)

    errs, errs_2, errs_3, errs_4 = [], [], [], []
    for D in Ds:
        ## dmrg
        term1d_copy = term1d.copy_new()
        loc_apply = local_dmrg_evaluator([term1d_copy], direction=direction, max_bond=D, nsites=nsites)
        loc_apply = grid_X.make_gridTN(loc_apply)
        # loc_apply = grid_X.make_gridTN(term1d_copy.vec_block.bra)

        loc_data = loc_apply.get_data()
        err = np.linalg.norm(glob_data - loc_data) / np.linalg.norm(glob_data)
        print('dmrg max bond', loc_apply.data.max_bond(), term1d_copy.bra.max_bond(),
              term1d_copy.vec_block.bra.max_bond(), term1d_copy.vec_block.ket.max_bond())
        print('err', err)
        errs += [err]

        ## cross
        term1c_copy = term1c.copy_new()
        loc_apply = local_cross_evaluator([term1c_copy], direction=direction,
                                          max_bond=D, nsites=nsites)
        loc_apply = grid_X.make_gridTN(loc_apply)
        loc_data = loc_apply.get_data()
        err = np.linalg.norm(glob_data - loc_data) / np.linalg.norm(glob_data)
        print('cross max bond', loc_apply.data.max_bond(), term1c_copy.bra.max_bond(),
              term1c_copy.vec_block.bra.max_bond(), term1d_copy.vec_block.ket.max_bond())
        print('err', err)
        errs_3 += [err]

        ## mixed ##
        term1m_copy = term1m.copy_new()
        loc_apply = local_mixed_evaluator([term1m_copy], direction=direction,
                                          max_bond=D, nsites=nsites)
        loc_apply = grid_X.make_gridTN(loc_apply)
        loc_data = loc_apply.get_data()
        err = np.linalg.norm(glob_data - loc_data) / np.linalg.norm(glob_data)
        print('cross max bond', loc_apply.data.max_bond(), term1c_copy.bra.max_bond(),
              term1c_copy.vec_block.bra.max_bond(), term1d_copy.vec_block.ket.max_bond())
        print('err', err)
        errs_4 += [err]


        if D is not None and D < 16:
            plt.figure()
            plt.plot(np.real(loc_data), label='mixed')
            plt.plot(np.real(glob_data), ':', label='exact')
            plt.plot(np.imag(loc_data), '--')
            plt.plot(np.imag(glob_data), ':')
            plt.legend()
            plt.title(f'D = {D}, {legs[-1]}')
            plt.show()

        ## svd
        comp_gtn = glob_apply.compress(compress_opts={'max_bond': D}, inplace=False)
        comp_data = comp_gtn.get_data() * coeff
        err = np.linalg.norm(comp_data - glob_data) / np.linalg.norm(glob_data)
        print('comp err', err)
        errs_2 += [err]


    dmrg_errs += [errs]
    comp_errs += [errs_2]
    cross_errs += [errs_3]
    mixed_errs += [errs_4]


# ### operator^-1 * init_mps
# ### doesn't work very well...
# do_test_8 = False
# if do_test_8:
#     nsites = 1
#     legs += ['solve Ax=b']
#     op = grid_X.get_secondderivative_mpo(ax_x, ax_x)
#     # op = ax_x.get_shift_mpo(5)
#     op3 = ax_x.get_iden_mpo()
#     # helper.scalar_multiply(op2, 0.1 / ax_x.dx, inplace=True)
#     helper.scalar_multiply(op3, 2 / ax_x.dx**2, inplace=True)
#     # op = helper.add_MPO(op.data, op2, inplace=False)
#     op = helper.add_MPO(op.data, op3, inplace=False)
#     op = grid_X.make_gridTN(op)
#     # op = grid_X.get_iden_mpo()
#
#     plt.figure()
#     plt.imshow(op.get_data())
#     plt.show()
#
#     # init_mps.data.distribute_exponent()
#
#     op_mat = op.get_data()
#     op_inv = np.linalg.pinv(op_mat)
#     op_inv = GridTN1D.from_dense_operator(op_inv, grid_X)
#     glob_apply = init_mps.apply(op_inv)
#     glob_data = glob_apply.get_data()
#
#     errs, errs_2, errs_3 = [], [], []
#     for D in Ds:
#         term1d = Term_DMRG(init_mps.data.copy(), operators=[op.data.copy()], mpo_poly=-1, num_tiers=1)
#         # term1d = Term_DMRG(init_mps.data.copy(), operators=[op.data.copy()], num_tiers=2)
#         ## performs worse with num_tiers = 2. I suppose this is because the ket projector on the operator
#         ## (<b|O|k> spans a smaller space in this case)
#         term1c = Term_Cross(init_mps.data.copy(), operators=[op.data.copy()], mpo_poly=-1)
#
#         ## dmrg
#         loc_apply = local_dmrg_evaluator([term1d], direction=direction, max_bond=D, nsites=nsites)
#
#         # tmp_vec = grid_X.make_gridTN(term1d.vec_block.bra)
#         #
#         # vec_data = tmp_vec.get_data()
#         # # err = np.linalg.norm(glob_data - loc_data) / np.linalg.norm(glob_data)
#         # # print('err', err)
#         # # errs += [err]
#         #
#         # if True:  # D is not None and D < 16:
#         #     plt.figure()
#         #     plt.plot(np.real(vec_data), label='dmrg')
#         #     plt.plot(np.real(init_mps.get_data()), ':', label='exact')
#         #     plt.plot(np.imag(vec_data), '--')
#         #     plt.plot(np.imag(glob_data), ':')
#         #     plt.legend()
#         #     plt.title(f'D = {D}')
#
#         loc_apply = grid_X.make_gridTN(loc_apply)
#         loc_data = loc_apply.get_data()
#         err = np.linalg.norm(glob_data - loc_data) / np.linalg.norm(glob_data)
#         print('err', err)
#         errs += [err]
#
#         if True:  # D is not None and D < 16:
#             plt.figure()
#             plt.plot(np.real(loc_data), label='dmrg')
#             plt.plot(np.real(glob_data), ':', label='exact')
#             plt.plot(np.imag(loc_data), '--')
#             plt.plot(np.imag(glob_data), ':')
#             plt.legend()
#             plt.title(f'D = {D}')
#             plt.show()
#
#         ## cross
#         # loc_apply = local_cross_evaluator([term1c], direction=direction, max_bond=D, nsites=nsites)
#         # loc_apply = grid_X.make_gridTN(loc_apply)
#         # loc_data = loc_apply.get_data()
#         # err = np.linalg.norm(glob_data - loc_data) / np.linalg.norm(glob_data)
#         # print('err', err)
#         errs_3 += [err]
#
#         ## svd
#         comp_gtn = glob_apply.compress(compress_opts={'max_bond': D}, inplace=False)
#         comp_data = comp_gtn.get_data()
#         err = np.linalg.norm(comp_data - glob_data) / np.linalg.norm(glob_data)
#         print('comp err', err)
#         errs_2 += [err]
#
#
#
#     dmrg_errs += [errs]
#     comp_errs += [errs_2]
#     cross_errs += [errs_3]


import plot_defaults

Ds[-1] = 100
i = 0
for errs1, errs2, errs3, errs4 in zip(dmrg_errs, cross_errs, mixed_errs, comp_errs):
    plt.figure()
    plt.semilogy(Ds, errs1, 'o-', label='SVD')
    plt.semilogy(Ds, errs2, 'x--', label='CUR')
    plt.semilogy(Ds, errs3, 'v:', label='mixed')
    plt.semilogy(Ds, errs4, 'v:', label='global')
    plt.legend()
    plt.xlabel('QTT rank')
    plt.ylabel('error (L2)')
    plt.title(legs[i])

    i += 1
plt.show()