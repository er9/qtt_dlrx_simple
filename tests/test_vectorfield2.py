"""Unit-style checks for 2-D Field vector calculus.

Exercises vector-field construction and finite-difference vector-calculus
operators in 2-D on a quantized-tensor-train grid.
"""
import sys
sys.path.append('../')

from setup_.defaults import *
import numpy as np
import matplotlib.pyplot as plt

from axis import Axis
from coord.coord_sys import Coordinate
from coord.cartesian import CartesianCoordinateSpace
from grid1D import Grid1D
from field import Field

q = 2
L = 4
npts = q**L

do_test1 = True

TEST_LAYOUT = LayoutType.SEQUENTIAL
TEST_FD = FDType.CENTER

######################
# vector field in 2D #
######################

if do_test1:
    K = 3
    xdata = np.linspace(0, 10, npts)
    data_y, data_x, data_z = np.meshgrid(xdata, xdata, xdata)
    dx = dy = dz = xdata[1] - xdata[0]
    
    ## field 0
    func0    = lambda x,y,z:  (2*(x-4)**3 + (x+2)**3.5)      * y**2 + x*np.exp(-(z/2)**2)
    dxfunc0  = lambda x,y,z:  (6*(x-4)**2 + 3.5*(x+2)**2.5)  * y**2 + np.exp(-(z/2)**2)
    dyfunc0  = lambda x,y,z:  (2*(x-4)**3 + (x+2)**3.5)      * 2*y  
    dzfunc0  = lambda x,y,z:  -1./2*z*x*np.exp(-(z/2)**2)
    dxxfunc0 = lambda x,y,z:  (12*(x-4)   + 3.5*2.5*(x+2)**1.5)  * y**2
    dyyfunc0 = lambda x,y,z:  (2*(x-4)**3 + (x+2)**3.5)      * 2
    dzzfunc0 = lambda x,y,z:  1./4*x*np.exp(-z**2/4)*(z**2-2)

    ## field 1
    func1    = lambda x,y,z:  np.exp(x/10) + y*z
    dxfunc1  = lambda x,y,z:  1./10*np.exp(x/10)
    dyfunc1  = lambda x,y,z:  z
    dzfunc1  = lambda x,y,z:  y
    dxxfunc1 = lambda x,y,z:  1./100*np.exp(x/10)
    dyyfunc1 = lambda x,y,z:  0*y
    dzzfunc1 = lambda x,y,z:  0*z

    ## field 2
    func2    = lambda x,y,z:  np.sin(np.pi/2*y) + np.cos(np.pi/4*x) * z
    dxfunc2  = lambda x,y,z:  -1./4*np.pi*z*np.sin(np.pi*x/4)
    dyfunc2  = lambda x,y,z:  1./2*np.pi*np.cos(np.pi*y/2)
    dzfunc2  = lambda x,y,z:  np.cos(np.pi/4*x)
    dxxfunc2 = lambda x,y,z:  -1./16*np.pi**2 * z * np.cos(np.pi*x/4)
    dyyfunc2 = lambda x,y,z:  -1./4*np.pi**2 * np.sin(np.pi*y/2)
    dzzfunc2 = lambda x,y,z:  0*z
    
    func0_data    = func0(data_x,data_y,data_z)
    dxfunc0_data  = dxfunc0(data_x,data_y,data_z)
    dyfunc0_data  = dyfunc0(data_x,data_y,data_z)
    dzfunc0_data  = dzfunc0(data_x,data_y,data_z)
    dxxfunc0_data = dxxfunc0(data_x,data_y,data_z)
    dyyfunc0_data = dyyfunc0(data_x,data_y,data_z)
    dzzfunc0_data = dzzfunc0(data_x,data_y,data_z)
    
    func1_data    = func1(data_x,data_y,data_z)
    dxfunc1_data  = dxfunc1(data_x,data_y,data_z)
    dyfunc1_data  = dyfunc1(data_x,data_y,data_z)
    dzfunc1_data  = dzfunc1(data_x,data_y,data_z)
    dxxfunc1_data = dxxfunc1(data_x,data_y,data_z)
    dyyfunc1_data = dyyfunc1(data_x,data_y,data_z)
    dzzfunc1_data = dzzfunc1(data_x,data_y,data_z)
    
    func2_data    = func2(data_x,data_y,data_z)
    dxfunc2_data  = dxfunc2(data_x,data_y,data_z)
    dyfunc2_data  = dyfunc2(data_x,data_y,data_z)
    dzfunc2_data  = dzfunc2(data_x,data_y,data_z)
    dxxfunc2_data = dxxfunc2(data_x,data_y,data_z)
    dyyfunc2_data = dyyfunc2(data_x,data_y,data_z)
    dzzfunc2_data = dzzfunc2(data_x,data_y,data_z)
    
    order = 4

    X = Coordinate('X', CoordinateType.X)
    Y = Coordinate('Y', CoordinateType.Y)
    Z = Coordinate('Z', CoordinateType.Z)
    coord_sys = CartesianCoordinateSpace('pos', coords=[X,Y,Z])

    axis0 = Axis(L, q, coordinate=X, xpts=xdata)
    axis1 = Axis(L, q, coordinate=Y, xpts=xdata)
    axis2 = Axis(L, q, coordinate=Z, xpts=xdata)


    cart_cs = CartesianCoordinateSpace('C1', axis0, axis1, axis2)
    grid1 = Grid1D('G1', (axis0, axis1, axis2), layout_type=TEST_LAYOUT)
    field1 = Field('F1', grid1, data={X: func0_data, Y: func1_data, Z: func2_data})
    field2 = Field('F1', grid1, data={X: func1_data, Y: func2_data, Z: func0_data})

    # derivative parameters
    field1.update_comp_deriv_params(left_bc=BCType.OPEN, order=order, fd_type=TEST_FD)
    field2.update_comp_deriv_params(left_bc=BCType.OPEN, order=order, fd_type=TEST_FD)


    a0, a1, a2 = func0_data, func1_data, func2_data
    b0, b1, b2 = func1_data, func2_data, func0_data

    ## field error
    field_tn = field1.get_comp_data(X)
    print('axis_map to state 0', np.linalg.norm(field_tn-a0))
    field_tn = field1.get_comp_data(Y)
    print('axis_map to state 1', np.linalg.norm(field_tn-a1))
    field_tn = field1.get_comp_data(Z)
    print('axis_map to state 2', np.linalg.norm(field_tn-a2))


    ## dot product
    dot_an = a0*b0 + a1*b1 + a2*b2
    dot_tn = field1.dot(field2)
    dot_0 = field1.dot({X: field2.components[X]})
    dot_1 = field1.dot({Y: field2.components[Y]})
    dot_2 = field1.dot({Z: field2.components[Z]})
    # dot_0 = helper.elemental_multiply(field1.components[0],field2.components[0],compress=1,cutoff=1e-30)
    # dot_1 = helper.elemental_multiply(field1.components[1],field2.components[1],compress=1,cutoff=1e-30)
    # dot_2 = helper.elemental_multiply(field1.components[2],field2.components[2],compress=1,cutoff=1e-30)
    print('dot error 0', np.linalg.norm(a0*b0 - dot_0.get_comp_data()))
    print('dot error 1', np.linalg.norm(a1*b1 - dot_1.get_comp_data()))
    print('dot error 2', np.linalg.norm(a2*b2 - dot_2.get_comp_data()))
    dot_tn = dot_tn.get_field_data()
    tn_dot_err = np.linalg.norm(dot_tn - dot_an)/npts**(K/2)
    print('dot product error', tn_dot_err)


    ## elemental multiply
    field_ones = grid1.get_ones_mps()
    field_ones.scalar_multiply(2, inplace=True)

    '''note: b2 (or a0) yields large errors in recontraction. likely due to large range of #s invovled.
    '''
    mult_an = b2*2
    # mult_tn = helper.elemental_multiply(field1.components[1],field_ones.components[0],compress=0)
    # plt.plot(a0.reshape(-1))
    # plt.plot(a1.reshape(-1))
    # plt.plot(a2.reshape(-1))
    # plt.show()

    # mult_tn = field_ones.elemental_multiply(field2[2])
    mult_tn = field2[Z].elemental_multiply(field_ones)
    mult_tn = mult_tn.get_data()
    print('elemental multiply error 1', np.linalg.norm(mult_tn-mult_an)/npts**(K/2))

    mult_an = a1*a2
    mult_tn = field1.elemental_multiply(field1[Z], compIDs=[Y])
    mult_tn = mult_tn.get_comp_data(Y)
    print('elemental multiply error 2', np.linalg.norm(mult_tn-mult_an)/npts**(K/2))

    mult_an = a0*a0
    mult_tn = field1.elemental_multiply(field1[X], compIDs=[X], compress_level=0)
    mult_tn = mult_tn.get_comp_data(X)
    print('elemental multiply error 3', np.linalg.norm(mult_tn-mult_an)/npts**(K/2))

    mult_an = a0*b2
    xdot_1m0 = field1.dot(field2, comps=[X], other_comps=[Z], compress_level=0)
    xdot_1m0 = xdot_1m0.get_comp_data()
    print('elemental multiply error 4', np.linalg.norm(xdot_1m0 - mult_an) / npts ** (K / 2))
    mult_tn = field1.elemental_multiply(field2[Z], compress_level=0)
    mult_tn = mult_tn.get_comp_data(X)
    print('elemental multiply error 5', np.linalg.norm(mult_tn-mult_an)/npts**(K/2))


    ## cross product
    cross_an = [a1*b2 - b1*a2, a2*b0 - b2*a0, a0*b1 - a1*b0]
    cross_tn = field1.cross_product(field2.components, coord_sys, compress_level=0)
    cross_tn = cross_tn.get_field_data(compIDs=[X,Y,Z])
    tn_cross_err = [np.linalg.norm(cross_an[i] - cross_tn[[X,Y,Z][i]])/npts**(K/2) for i in range(3)]
    print('cross product error', tn_cross_err)

    xdot_0p = field1.dot({Y: field2.components[Z]})
    xdot_1p = field1.dot({Z: field2.components[X]})
    xdot_2p = field1.dot({X: field2.components[Y]})

    xdot_0m = field1.dot({Z: field2.components[Y]})
    xdot_1m = field1.dot({X: field2.components[Z]})
    xdot_2m = field1.dot({Y: field2.components[X]})

    print('xdot error 0p', np.linalg.norm(a1*b2 - xdot_0p.get_comp_data())/npts**(K/2))
    print('xdot error 1p', np.linalg.norm(a2*b0 - xdot_1p.get_comp_data())/npts**(K/2))
    print('xdot error 2p', np.linalg.norm(a0*b1 - xdot_2p.get_comp_data())/npts**(K/2))

    print('xdot error 0m', np.linalg.norm(a2*b1 - xdot_0m.get_comp_data())/npts**(K/2))
    print('xdot error 1m', np.linalg.norm(a0*b2 - xdot_1m.get_comp_data())/npts**(K/2))
    print('xdot error 2m', np.linalg.norm(a1*b0 - xdot_2m.get_comp_data())/npts**(K/2))


    ## convective term
    field1 = Field('F1', grid1, data={X: func0_data, Y: func1_data, Z: func2_data})
    field2 = Field('F1', grid1, data={X: func1_data, Y: func2_data, Z: func0_data})

    orders = [x for x in range(1,5)]
    tn_conv_errs = []
    for order in orders:
        dp0 = {'left_bc': BCType.OPEN, 'order': order, 'fd_type': TEST_FD}
        field1.update_comp_deriv_params(compIDs=[X,Y,Z],axes=[axis0,axis1,axis2],**dp0)
        field2.update_comp_deriv_params(**dp0)

        conv_an = [func0_data*dxfunc0_data + func1_data*dyfunc0_data + func2_data*dzfunc0_data,
                   func0_data*dxfunc1_data + func1_data*dyfunc1_data + func2_data*dzfunc1_data,
                   func0_data*dxfunc2_data + func1_data*dyfunc2_data + func2_data*dzfunc2_data]

        print('conv_an')

        grad_an0 = [dxfunc0_data, dyfunc0_data, dzfunc0_data]
        grad_an1 = [dxfunc1_data, dyfunc1_data, dzfunc1_data]
        grad_an2 = [dxfunc2_data, dyfunc2_data, dzfunc2_data]

        grad0_tn, grad1_tn, grad2_tn = [], [], []
        for i in range(3):
            ax = grid1.axes[i]
            gradj_B = cart_cs._gtn_vector_derivative(field1.components, ax, compress=False)
            grad0_tn += [gradj_B[X].get_data()]
            grad1_tn += [gradj_B[Y].get_data()]
            grad2_tn += [gradj_B[Z].get_data()]

        # grad0_tn = field1.gradient(0).get_field_data()
        # grad1_tn = field1.gradient(1).get_field_data()
        # grad2_tn = field1.gradient(2).get_field_data()

        grad_errs0 = [np.log10(np.linalg.norm(grad0_tn[i]-grad_an0[i])/npts**(K/2)) for i in range(3)]
        grad_errs1 = [np.log10(np.linalg.norm(grad1_tn[i]-grad_an1[i])/npts**(K/2)) for i in range(3)]
        grad_errs2 = [np.log10(np.linalg.norm(grad2_tn[i]-grad_an2[i])/npts**(K/2)) for i in range(3)]
        print('grad errs 0', grad_errs0, grad_errs1, grad_errs2)

        conv_tn_ = cart_cs.convective_operator(field1, field1, compress=True, compress_opts={})
        print('conv_tn', [conv_tn_[m].max_bond() for m in [X,Y,Z]])
        conv_tn = conv_tn_.get_field_data(compIDs=[X,Y,Z])
        print('conv_tn data')
        tn_conv_errs += [[np.log10(np.linalg.norm(conv_an[i]-conv_tn[[X,Y,Z][i]])/npts**(K/2)) for i in range(3)]]
        print('tn conv',tn_conv_errs)

    tn_conv_errs = np.array(tn_conv_errs)
    plt.plot(orders,np.array(tn_conv_errs[:,0]),'x',label='tn-0')
    plt.plot(orders,np.array(tn_conv_errs[:,1]),'o',label='tn-1')
    plt.plot(orders,np.array(tn_conv_errs[:,2]),'s',label='tn-2')
    plt.title('convective term')
    plt.xlabel('order')
    plt.ylabel('log10(error)')
    plt.legend()
    plt.show()
