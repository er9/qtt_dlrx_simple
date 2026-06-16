"""Unit-style checks for ScalarField / Field finite-difference vector calculus.

Exercises scalar-field construction and finite-difference derivative operators
(center/forward/backward stencils) on a quantized-tensor-train grid, validating
against findiff references for different MPS layouts.
"""
import sys
sys.path.append('../')

from setup_.configs import *
# from enums import *

import numpy as np
import matplotlib.pyplot as plt
import findiff

from axis import Axis
from coord.coord_sys import Coordinate
from coord.cartesian import CartesianCoordinateSpace
from grid1D import Grid1D
from grid_comb import GridsComb
from field import ScalarField, Field


q = 2
L = 6
npts = q**L

""" choose MPS layout
    see 2D tests for using COMB layout
"""
TEST_LAYOUT = LayoutType.PARALLEL_GROUP
# TEST_LAYOUT = LayoutType.COMB

""" choose finite difference type (center stencil, forward, backward)
"""
TEST_FD = FDType.CENTER

""" choose which test to run
"""
do_test_open1D          = False # cross-checked
do_test_periodic1D      = False # cross-checked
do_test_antiperiodic1D  = False
do_test_bcs             = False # cross-checked  ## limit number of ghost cells in bc
do_test_sym             = True
do_test_2Dopen          = False # cross-checked
do_test_2Dperiodic      = False # cross-checked
do_test_2Dapc           = False


######################
# scalar field in 1D #
######################

if do_test_open1D:
    K = 1
    xdata = np.linspace(0, 10, npts)
    dx = xdata[1]-xdata[0]

    # func   = lambda x:  2*((x-4)**3) + 10*(x+2)**2.5
    # dfunc  = lambda x:  6*(x-4)**2   + 10*2.5*(x+2)**1.5
    # d2func = lambda x:  12*(x-4)     + 10*2.5*1.5*(x+2)**0.5

    func   = lambda x:  2*((x-4)**5) + 10*(x+2)**2.5
    dfunc  = lambda x:  10*(x-4)**4  + 10*2.5*(x+2)**1.5
    d2func = lambda x:  40*(x-4)**3  + 10*2.5*1.5*(x+2)**0.5
    
    """ get f(x) """
    vec_data = func(xdata)

    """ orders of finite difference stencil to test"""
    orders = [x for x in range(1,5)]
    dx_error_mid, dx_error_0, dx_error_L, dx_error_avg = [], [], [], []
    d2_error_mid, d2_error_0, d2_error_L, d2_error_avg = [], [], [], []

    """ initialize scalar field object. 
        coordinate: (optional) defines axis direction 
        axis: contains info about grid points x
        grid: multiple axes together
        (scalar) field: contains value of scalar field f(x)
    """
    X = Coordinate('X')
    coord_sys = CartesianCoordinateSpace('X', coords=[X])
    axis1 = Axis(L, q, coordinate=X, xpts=xdata)
    grid1 = Grid1D('G1', (axis1,), layout_type=TEST_LAYOUT)
    field = ScalarField('F1', grid1, data=vec_data)

    """ specify boundary conditions for the derivative
    """
    field.component.update_deriv_params(axis1, left_bc=BCType.OPEN, right_bc=BCType.OPEN, fd_type=TEST_FD)

    for order in orders:
        field.component.update_deriv_params(axis1, order=order)

        ### center first order diff ###a
        ## take gradient using built-in numpy functions
        diff1 = 1./dx * np.diff(vec_data, n=1)
        diff2 = 1./dx * np.gradient(vec_data)

        ## take gradient via MPS method
        tn_cdiff = field.gradient()
        tn_cdiff_vec = tn_cdiff.get_comp_data(X)

        ## compute errors
        dx_error_mid += [np.log10(np.abs(tn_cdiff_vec[npts//2] - dfunc(xdata[npts // 2])))]
        dx_error_0   += [np.log10(np.abs(tn_cdiff_vec[0] - dfunc(xdata[0])))]
        dx_error_L   += [np.log10(np.abs(tn_cdiff_vec[-1] - dfunc(xdata[-1])))]
        dx_error_avg += [np.log10(np.linalg.norm(tn_cdiff_vec - dfunc(xdata)) / np.sqrt(npts))]
        
        plt.plot(xdata, vec_data)
        plt.plot(xdata[:-1], diff1, ':')
        plt.plot(xdata, diff2, linestyle='-', linewidth=2)
        plt.plot(xdata, dfunc(xdata), 'k--')
        plt.plot(xdata, tn_cdiff_vec, 'r:', linewidth=3)
        plt.show()

        plt.plot(xdata, tn_cdiff_vec - dfunc(xdata), 'k')
        plt.title('error, order %d'%order)
        plt.show()

        ### second order diff ###
        ## built-in numpy method
        diff22 = 1./dx * np.gradient(diff2)

        ## MPS method
        tn_diff2 = field.laplacian(coord_sys=coord_sys)
        tn_diff2_vec = tn_diff2.get_comp_data(X)

        d2_error_mid += [np.log10(np.abs(tn_diff2_vec[npts//2] - d2func(xdata[npts // 2])))]
        d2_error_0   += [np.log10(np.abs(tn_diff2_vec[0] - d2func(xdata[0])))]
        d2_error_L   += [np.log10(np.abs(tn_diff2_vec[-1] - d2func(xdata[-1])))]
        d2_error_avg += [np.log10(np.linalg.norm(tn_diff2_vec - d2func(xdata)) / np.sqrt(npts))]
        
        plt.plot(xdata, diff22, linestyle='-', linewidth=2)
        plt.plot(xdata, d2func(xdata), 'k--')
        plt.plot(xdata, tn_diff2_vec, 'r:', linewidth=3)
        plt.show()

        plt.plot(xdata, tn_diff2_vec - d2func(xdata), 'k')
        plt.title('d2/dx2 error, order %d'%order)
        plt.show()

    plt.plot(orders,dx_error_mid,'bx',label='mid')
    plt.plot(orders,dx_error_0  ,'gx',label='end0')
    plt.plot(orders,dx_error_L  ,'rx',label='endL')
    plt.plot(orders,dx_error_avg,'kx',label='avg')

    res_mid, res_0, res_L, res_avg = np.polyfit([x for x in range(1,5)], 
                           np.array([dx_error_mid,dx_error_0,dx_error_L,dx_error_avg]).T, 1).T
    data_fine = np.linspace(0,5,51)
    plt.plot(data_fine, np.polyval(res_mid,data_fine),'b:',label='p-mid')
    plt.plot(data_fine, np.polyval(res_0  ,data_fine),'g:',label='p-end0')
    plt.plot(data_fine, np.polyval(res_L  ,data_fine),'r:',label='p-endL')
    plt.plot(data_fine, np.polyval(res_avg,data_fine),'k:',label='p-avg')

    print('average error', dx_error_avg)
    print('midpt error', dx_error_mid)
    print('end error', dx_error_L)

    plt.xlabel('order')
    plt.ylabel('log10(error)')
    plt.legend()
    plt.show()


    plt.plot(orders,d2_error_mid,'bx',label='mid')
    plt.plot(orders,d2_error_0  ,'gx',label='end0')
    plt.plot(orders,d2_error_L  ,'rx',label='endL')
    plt.plot(orders,d2_error_avg,'kx',label='avg')

    res_mid, res_0, res_L, res_avg = np.polyfit([x for x in range(1,5)], 
                           np.array([d2_error_mid,d2_error_0,d2_error_L,d2_error_avg]).T, 1).T
    data_fine = np.linspace(0,5,51)
    plt.plot(data_fine, np.polyval(res_mid,data_fine),'b:',label='p-mid')
    plt.plot(data_fine, np.polyval(res_0  ,data_fine),'g:',label='p-end0')
    plt.plot(data_fine, np.polyval(res_L  ,data_fine),'r:',label='p-endL')
    plt.plot(data_fine, np.polyval(res_avg,data_fine),'k:',label='p-avg')

    print('average error', d2_error_avg)
    print('midpt error', d2_error_mid)
    print('end error', d2_error_L)

    plt.xlabel('order')
    plt.ylabel('log10(error)')
    plt.title('d2/dx2 error')
    plt.legend()
    plt.show()


###############################
# periodic scalar field in 1D #
###############################

if do_test_periodic1D:
    K = 1
    xdata = np.linspace(0, 2 * np.pi, npts, endpoint=False)
    dx = xdata[1] - xdata[0]
    
    func   = lambda x:  np.sin(2*x+3*np.pi/5)
    dfunc  = lambda x:  np.cos(2*x+3*np.pi/5) *  2
    d2func = lambda x:  np.sin(2*x+3*np.pi/5) * -4
    
    # vec_data = np.array([func(x) for x in data])
    vec_data = func(xdata)

    orders = [x for x in range(1,5)]
    dx_error_mid, dx_error_0, dx_error_L, dx_error_avg = [], [], [], []
    d2_error_mid, d2_error_0, d2_error_L, d2_error_avg = [], [], [], []

    X = Coordinate('X')
    coord_sys = CartesianCoordinateSpace('X', coords=[X])
    axis1 = Axis(L, q, coordinate=X, xpts=xdata)
    grid1 = Grid1D('G1', (axis1,), layout_type=TEST_LAYOUT)
    field = ScalarField('F1', grid1, data=vec_data)
    field.component.update_deriv_params(axis1, left_bc=BCType.PERIODIC, right_bc=BCType.PERIODIC, fd_type=TEST_FD)

    for order in orders:
        field.component.update_deriv_params(axis1, order=order)

        ### center first order diff ###
        diff1 = 1./dx * np.diff(vec_data, n=1)
        diff2 = 1. / dx * np.gradient(vec_data)
        
        tn_cdiff = field.gradient()
        tn_cdiff_vec = tn_cdiff.get_comp_data(X)

        dx_error_mid += [np.log10(np.abs(tn_cdiff_vec[npts//2] - dfunc(xdata[npts // 2])))]
        dx_error_0   += [np.log10(np.abs(tn_cdiff_vec[0] - dfunc(xdata[0])))]
        dx_error_L   += [np.log10(np.abs(tn_cdiff_vec[-1] - dfunc(xdata[-1])))]
        dx_error_avg += [np.log10(np.linalg.norm(tn_cdiff_vec - dfunc(xdata)) / np.sqrt(npts))]
        
        plt.plot(xdata, vec_data)
        plt.plot(xdata[:-1], diff1, ':')
        plt.plot(xdata, diff2, linestyle='-', linewidth=2)
        plt.plot(xdata, dfunc(xdata), 'k--')
        plt.plot(xdata, tn_cdiff_vec, 'r:', linewidth=3)
        plt.show()

        ### second order diff ###a
        diff22 = 1. / dx * np.gradient(diff2)
        
        tn_diff2 = field.laplacian(coord_sys=coord_sys, compress_level=1)
        tn_diff2_vec = tn_diff2.get_comp_data(X)

        d2_error_mid += [np.log10(np.abs(tn_diff2_vec[npts//2] - d2func(xdata[npts // 2])))]
        d2_error_0   += [np.log10(np.abs(tn_diff2_vec[0] - d2func(xdata[0])))]
        d2_error_L   += [np.log10(np.abs(tn_diff2_vec[-1] - d2func(xdata[-1])))]
        d2_error_avg += [np.log10(np.linalg.norm(tn_diff2_vec - d2func(xdata)) / np.sqrt(npts))]
        
        plt.plot(xdata, diff22, linestyle='-', linewidth=2)
        plt.plot(xdata, d2func(xdata), 'k--')
        plt.plot(xdata, tn_diff2_vec, 'r:', linewidth=3)
        plt.show()

    ## d/dx error
    plt.plot(orders,dx_error_mid,'bx',label='mid')
    plt.plot(orders,dx_error_0  ,'gx',label='end0')
    plt.plot(orders,dx_error_L  ,'rx',label='endL')
    plt.plot(orders,dx_error_avg,'kx',label='avg')

    res_mid, res_0, res_L, res_avg = np.polyfit([x for x in range(1,5)], 
                           np.array([dx_error_mid,dx_error_0,dx_error_L,dx_error_avg]).T, 1).T
    data_fine = np.linspace(0,5,51)
    plt.plot(data_fine, np.polyval(res_mid,data_fine),'b:',label='p-mid')
    plt.plot(data_fine, np.polyval(res_0  ,data_fine),'g:',label='p-end0')
    plt.plot(data_fine, np.polyval(res_L  ,data_fine),'r:',label='p-endL')
    plt.plot(data_fine, np.polyval(res_avg,data_fine),'k:',label='p-avg')

    print('average error', dx_error_avg)
    print('midpt error', dx_error_mid)
    print('end error', dx_error_L)

    plt.xlabel('order')
    plt.ylabel('log10(error)')
    plt.legend()
    plt.show()

    ## d2/dx2 error
    plt.plot(orders,d2_error_mid,'bx',label='mid')
    plt.plot(orders,d2_error_0  ,'gx',label='end0')
    plt.plot(orders,d2_error_L  ,'rx',label='endL')
    plt.plot(orders,d2_error_avg,'kx',label='avg')

    res_mid, res_0, res_L, res_avg = np.polyfit([x for x in range(1,5)], 
                           np.array([d2_error_mid,d2_error_0,d2_error_L,d2_error_avg]).T, 1).T
    data_fine = np.linspace(0,5,51)
    plt.plot(data_fine, np.polyval(res_mid,data_fine),'b:',label='p-mid')
    plt.plot(data_fine, np.polyval(res_0  ,data_fine),'g:',label='p-end0')
    plt.plot(data_fine, np.polyval(res_L  ,data_fine),'r:',label='p-endL')
    plt.plot(data_fine, np.polyval(res_avg,data_fine),'k:',label='p-avg')

    print('average error', d2_error_avg)
    print('midpt error', d2_error_mid)
    print('end error', d2_error_L)

    plt.xlabel('order')
    plt.ylabel('log10(error)')
    plt.title('d2/dx2 error')
    plt.legend()
    plt.show()


###################################
# antiperiodic scalar field in 1D #
###################################

if do_test_antiperiodic1D:
    K = 1
    xdata = np.linspace(0, 2 * np.pi, npts, endpoint=False)
    dx = xdata[1] - xdata[0]
    
    func   = lambda x:  np.sin(3./2*x+3*np.pi/5)
    dfunc  = lambda x:  np.cos(3./2*x+3*np.pi/5) *  3./2
    d2func = lambda x:  np.sin(3./2*x+3*np.pi/5) * -9./4
    
    # vec_data = np.array([func(x) for x in data])
    vec_data = func(xdata)
    
    orders = [x for x in range(1,5)]
    dx_error_mid, dx_error_0, dx_error_L, dx_error_avg = [], [], [], []
    d2_error_mid, d2_error_0, d2_error_L, d2_error_avg = [], [], [], []

    X = Coordinate('X')
    coord_sys = CartesianCoordinateSpace('X', coords=[X])
    axis1 = Axis(L, q, coordinate=X, xpts=xdata)
    grid1 = Grid1D('G1', (axis1,), layout_type=TEST_LAYOUT)
    field = ScalarField('F1', grid1, data=vec_data)
    field.component.update_deriv_params(axis1, left_bc=BCType.ANTIPERIODIC, right_bc=BCType.ANTIPERIODIC,
                                 fd_type=TEST_FD)

    for order in orders:
        field.component.update_deriv_params(axis1, order=order)

        ### center first order diff ###
        diff1 = 1./dx * np.diff(vec_data, n=1)
        diff2 = 1. / dx * np.gradient(vec_data)
        
        tn_cdiff = field.gradient()
        tn_cdiff_vec = tn_cdiff.get_comp_data(X)

        dx_error_mid += [np.log10(np.abs(tn_cdiff_vec[npts//2] - dfunc(xdata[npts // 2])))]
        dx_error_0   += [np.log10(np.abs(tn_cdiff_vec[0] - dfunc(xdata[0])))]
        dx_error_L   += [np.log10(np.abs(tn_cdiff_vec[-1] - dfunc(xdata[-1])))]
        dx_error_avg += [np.log10(np.linalg.norm(tn_cdiff_vec - dfunc(xdata)) / np.sqrt(npts))]
        
        plt.plot(xdata, vec_data)
        plt.plot(xdata[:-1], diff1, ':')
        plt.plot(xdata, diff2, linestyle='-', linewidth=2)
        plt.plot(xdata, dfunc(xdata), 'k--')
        plt.plot(xdata, tn_cdiff_vec, 'r:', linewidth=3)
        plt.show()

        ### second order diff ###a
        diff22 = 1. / dx * np.gradient(diff2)
        
        tn_diff2 = field.laplacian(coord_sys=coord_sys, compress_level=1)
        tn_diff2_vec = tn_diff2.get_comp_data(X)

        d2_error_mid += [np.log10(np.abs(tn_diff2_vec[npts//2] - d2func(xdata[npts // 2])))]
        d2_error_0   += [np.log10(np.abs(tn_diff2_vec[0] - d2func(xdata[0])))]
        d2_error_L   += [np.log10(np.abs(tn_diff2_vec[-1] - d2func(xdata[-1])))]
        d2_error_avg += [np.log10(np.linalg.norm(tn_diff2_vec - d2func(xdata)) / np.sqrt(npts))]
        
        plt.plot(xdata, diff22, linestyle='-', linewidth=2)
        plt.plot(xdata, d2func(xdata), 'k--')
        plt.plot(xdata, tn_diff2_vec, 'r:', linewidth=3)
        plt.show()

    ## d/dx error
    plt.plot(orders,dx_error_mid,'bx',label='mid')
    plt.plot(orders,dx_error_0  ,'gx',label='end0')
    plt.plot(orders,dx_error_L  ,'rx',label='endL')
    plt.plot(orders,dx_error_avg,'kx',label='avg')

    res_mid, res_0, res_L, res_avg = np.polyfit([x for x in range(1,5)], 
                           np.array([dx_error_mid,dx_error_0,dx_error_L,dx_error_avg]).T, 1).T
    data_fine = np.linspace(0,5,51)
    plt.plot(data_fine, np.polyval(res_mid,data_fine),'b:',label='p-mid')
    plt.plot(data_fine, np.polyval(res_0  ,data_fine),'g:',label='p-end0')
    plt.plot(data_fine, np.polyval(res_L  ,data_fine),'r:',label='p-endL')
    plt.plot(data_fine, np.polyval(res_avg,data_fine),'k:',label='p-avg')

    print('average error', dx_error_avg)
    print('midpt error', dx_error_mid)
    print('end error', dx_error_L)

    plt.xlabel('order')
    plt.ylabel('log10(error)')
    plt.legend()
    plt.show()

    ## d2/dx2 error
    plt.plot(orders,d2_error_mid,'bx',label='mid')
    plt.plot(orders,d2_error_0  ,'gx',label='end0')
    plt.plot(orders,d2_error_L  ,'rx',label='endL')
    plt.plot(orders,d2_error_avg,'kx',label='avg')

    res_mid, res_0, res_L, res_avg = np.polyfit([x for x in range(1,5)], 
                           np.array([d2_error_mid,d2_error_0,d2_error_L,d2_error_avg]).T, 1).T
    data_fine = np.linspace(0,5,51)
    plt.plot(data_fine, np.polyval(res_mid,data_fine),'b:',label='p-mid')
    plt.plot(data_fine, np.polyval(res_0  ,data_fine),'g:',label='p-end0')
    plt.plot(data_fine, np.polyval(res_L  ,data_fine),'r:',label='p-endL')
    plt.plot(data_fine, np.polyval(res_avg,data_fine),'k:',label='p-avg')

    print('average error', d2_error_avg)
    print('midpt error', d2_error_mid)
    print('end error', d2_error_L)

    plt.xlabel('order')
    plt.ylabel('log10(error)')
    plt.title('d2/dx2 error')
    plt.legend()
    plt.show()


####################################
# reflecting bc scalar field in 1D #
####################################

if do_test_bcs:
    print('zero gradient/zero value (old reflecting)')

    K = 1
    xdata = np.linspace(0, 2 * np.pi, npts, endpoint=True)
    dx = xdata[1] - xdata[0]

    bc = BCType.ZEROVALUE
    func = lambda x: x*(x-2*np.pi)*(x-1)
    dfunc = lambda x: np.pi*(2-4*x) + x*(3*x-2)
    d2func = lambda x: 6*x - 4*np.pi - 2

    # ## ZG
    # bc = BCType.ZEROGRADIENT
    # # func   = lambda x:  np.cos(3./2*x)
    # # dfunc  = lambda x:  np.sin(3./2*x) * -3./2
    # # d2func = lambda x:  np.cos(3./2*x) * -9./4
    # func   = lambda x: x**2 * (x-2*np.pi)**2
    # dfunc  = lambda x: 4*x*(x**2 - 3*np.pi*x + 2*np.pi**2)
    # d2func = lambda x: 4*(3*x**2 - 6*np.pi*x + 2*np.pi**2)

    # ## R? No. probably open? R needs to be 0 at boundary
    # func   = lambda x: x*(x-2*np.pi) + x*(x-2*np.pi)*np.exp(-x**2/2)
    # dfunc  = lambda x: (2*x-2*np.pi) + (2*x-2*np.pi)*np.exp(-x**2/2) + x*(x-2*np.pi)*(-x*np.exp(-x**2/2))
    # d2func = lambda x: 2 + (x**4 - 5*x**2 - 2*np.pi*(x**2-3)*x + 2)*np.exp(-x**2/2) 
    
    # vec_data = np.array([func(x) for x in data])
    vec_data = func(xdata)
    
    orders = [x for x in range(1,5)]
    dx_error_mid, dx_error_0, dx_error_L, dx_error_avg = [], [], [], []
    d2_error_mid, d2_error_0, d2_error_L, d2_error_avg = [], [], [], []

    X = Coordinate('X')
    coord_sys = CartesianCoordinateSpace('X', coords=[X])
    axis1 = Axis(L, q, coordinate=X, xpts=xdata)
    if TEST_LAYOUT is LayoutType.COMB:
        subgrid1 = Grid1D('SG1', (axis1,), layout_type=LayoutType.SEQUENTIAL)
        grid1 = GridsComb('G1', (subgrid1,), )
    else:
        grid1 = Grid1D('G1', (axis1,), layout_type=TEST_LAYOUT)
    field = ScalarField('F1', grid1, vec_data)
    field.component.update_deriv_params(axis1, left_bc=bc, right_bc=bc, fd_type=TEST_FD)

    for order in orders:
        field.component.update_deriv_params(axis1, order=order)

        ### center first order diff ###
        diff1 = 1. / dx * np.diff(vec_data, n=1)
        diff2 = 1. / dx * np.gradient(vec_data)
        
        tn_cdiff = field.gradient()
        tn_cdiff_vec = tn_cdiff.get_comp_data(X)

        dx_error_mid += [np.log10(np.abs(tn_cdiff_vec[npts//2] - dfunc(xdata[npts // 2])))]
        dx_error_0   += [np.log10(np.abs(tn_cdiff_vec[0] - dfunc(xdata[0])))]
        dx_error_L   += [np.log10(np.abs(tn_cdiff_vec[-1] - dfunc(xdata[-1])))]
        dx_error_avg += [np.log10(np.linalg.norm(tn_cdiff_vec - dfunc(xdata)) / np.sqrt(npts))]
        
        plt.plot(xdata, vec_data)
        plt.plot(xdata[:-1], diff1, ':')
        plt.plot(xdata, diff2, linestyle='-', linewidth=2)
        plt.plot(xdata, dfunc(xdata), 'k--')
        plt.plot(xdata, tn_cdiff_vec, 'r:x', linewidth=3)
        plt.show()

        ### second order diff ###a
        diff22 = 1. / dx * np.gradient(diff2)
        
        tn_diff2 = field.laplacian(coord_sys=coord_sys, compress_level=1)
        tn_diff2_vec = tn_diff2.get_comp_data(0)

        d2_error_mid += [np.log10(np.abs(tn_diff2_vec[npts//2] - d2func(xdata[npts // 2])))]
        d2_error_0   += [np.log10(np.abs(tn_diff2_vec[0] - d2func(xdata[0])))]
        d2_error_L   += [np.log10(np.abs(tn_diff2_vec[-1] - d2func(xdata[-1])))]
        d2_error_avg += [np.log10(np.linalg.norm(tn_diff2_vec - d2func(xdata)) / np.sqrt(npts))]
        
        plt.plot(xdata, diff22, linestyle='-', linewidth=2)
        plt.plot(xdata, d2func(xdata), 'k--')
        plt.plot(xdata, tn_diff2_vec, 'r:', linewidth=3)
        plt.title('laplacian')
        plt.show()

    ## d/dx error
    plt.plot(orders,dx_error_mid,'bx',label='mid')
    plt.plot(orders,dx_error_0  ,'gx',label='end0')
    plt.plot(orders,dx_error_L  ,'rx',label='endL')
    plt.plot(orders,dx_error_avg,'kx',label='avg')

    res_mid, res_0, res_L, res_avg = np.polyfit([x for x in range(1,5)], 
                           np.array([dx_error_mid,dx_error_0,dx_error_L,dx_error_avg]).T, 1).T
    data_fine = np.linspace(0,5,51)
    plt.plot(data_fine, np.polyval(res_mid,data_fine),'b:',label='p-mid')
    plt.plot(data_fine, np.polyval(res_0  ,data_fine),'g:',label='p-end0')
    plt.plot(data_fine, np.polyval(res_L  ,data_fine),'r:',label='p-endL')
    plt.plot(data_fine, np.polyval(res_avg,data_fine),'k:',label='p-avg')

    print('average error', dx_error_avg)
    print('midpt error', dx_error_mid)
    print('end error', dx_error_L)

    plt.xlabel('order')
    plt.ylabel('log10(error)')
    plt.legend()
    plt.show()

    ## d2/dx2 error
    plt.plot(orders,d2_error_mid,'bx',label='mid')
    plt.plot(orders,d2_error_0  ,'gx',label='end0')
    plt.plot(orders,d2_error_L  ,'rx',label='endL')
    plt.plot(orders,d2_error_avg,'kx',label='avg')

    res_mid, res_0, res_L, res_avg = np.polyfit([x for x in range(1,5)], 
                           np.array([d2_error_mid,d2_error_0,d2_error_L,d2_error_avg]).T, 1).T
    data_fine = np.linspace(0,5,51)
    plt.plot(data_fine, np.polyval(res_mid,data_fine),'b:',label='p-mid')
    plt.plot(data_fine, np.polyval(res_0  ,data_fine),'g:',label='p-end0')
    plt.plot(data_fine, np.polyval(res_L  ,data_fine),'r:',label='p-endL')
    plt.plot(data_fine, np.polyval(res_avg,data_fine),'k:',label='p-avg')

    print('average error', d2_error_avg)
    print('midpt error', d2_error_mid)
    print('end error', d2_error_L)

    plt.xlabel('order')
    plt.ylabel('log10(error)')
    plt.title('d2/dx2 error')
    plt.legend()
    plt.show()


##############################
# symmetic and antisymmetric #
##############################

if do_test_sym:
    print('zero gradient/zero value (old reflecting)')

    K = 1
    xdata = np.linspace(0, 10, npts, endpoint=True)
    # xdata = np.linspace(-10, 0, npts, endpoint=True)
    dx = xdata[1] - xdata[0]

    # offset = 0

    # xdata = xdata + dx / 2    # for offset
    # offset = -1

    xdata = xdata - dx / 2    # for offset
    offset = 1
    TEST_FD = FDType.BACKWARD # FORWARD

    # R
    bc = BCType.SYMMETRIC
    func = lambda x: 1. / np.cosh(x) ** 2
    dfunc = lambda x: -2 * np.tanh(x) / np.cosh(x) ** 2
    d2func = lambda x: 2 * (np.cosh(2*x) - 2) / np.cosh(x) ** 4

    # bc = BCType.ANTISYMMETRIC
    # func = lambda x: np.tanh(x)
    # dfunc = lambda x: 1. / np.cosh(x) ** 2
    # d2func = lambda x: -2 * np.tanh(x) / np.cosh(x) ** 2

    deriv_config = DerivativeConfiguration(left_bc=bc, right_bc=BCType.SYMMETRIC, fd_type=TEST_FD, offset=offset)
    # deriv_config = DerivativeConfiguration(left_bc=BCType.SYMMETRIC, right_bc=bc, fd_type=TEST_FD, offset=offset)
    # print('deriv_config', deriv_config.deriv_params)

    plt.figure()
    plt.plot(xdata, func(xdata), label='f')
    plt.plot(xdata, dfunc(xdata), label='df/dx')
    plt.plot(xdata, d2func(xdata), label='d2f/dx2')
    plt.legend()
    plt.show()


    # vec_data = np.array([func(x) for x in data])
    vec_data = func(xdata)

    orders = [x for x in range(1, 5)]
    dx_error_mid, dx_error_0, dx_error_L, dx_error_avg = [], [], [], []
    d2_error_mid, d2_error_0, d2_error_L, d2_error_avg = [], [], [], []

    X = Coordinate('X')
    coord_sys = CartesianCoordinateSpace('X', coords=[X])
    axis1 = Axis(L, q, coordinate=X, xpts=xdata)
    if TEST_LAYOUT is LayoutType.COMB:
        subgrid1 = Grid1D('SG1', (axis1,), layout_type=LayoutType.SEQUENTIAL)
        grid1 = GridsComb('G1', (subgrid1,), )
    else:
        grid1 = Grid1D('G1', (axis1,), layout_type=TEST_LAYOUT)
    field = ScalarField('F1', grid1, vec_data)

    # field.component.update_deriv_params(axis1, left_bc=BCType.ZEROGRADIENT, right_bc=bc, fd_type=TEST_FD)
    field.component.update_deriv_params(axis1, **deriv_config.deriv_params)

    for order in orders:
        field.component.update_deriv_params(axis1, order=order)

        ### center first order diff ###
        diff1 = 1. / dx * np.diff(vec_data, n=1)
        diff2 = 1. / dx * np.gradient(vec_data)

        tn_cdiff = field.gradient()
        tn_cdiff_vec = tn_cdiff.get_comp_data(X)

        dx_error_mid += [np.log10(np.abs(tn_cdiff_vec[npts // 2] - dfunc(xdata[npts // 2])))]
        dx_error_0 += [np.log10(np.abs(tn_cdiff_vec[0] - dfunc(xdata[0])))]
        dx_error_L += [np.log10(np.abs(tn_cdiff_vec[-1] - dfunc(xdata[-1])))]
        dx_error_avg += [np.log10(np.linalg.norm(tn_cdiff_vec - dfunc(xdata)) / np.sqrt(npts))]

        plt.plot(xdata, vec_data, label='f(x)')
        plt.plot(xdata[:-1], diff1, ':', label='np1')
        plt.plot(xdata, diff2, linestyle='-', linewidth=2, label='np2')
        plt.plot(xdata, dfunc(xdata), 'k--', label='df/dx')
        plt.plot(xdata, tn_cdiff_vec, 'r:x', linewidth=3, label='mpo')
        plt.legend()
        plt.show()

        ### second order diff ###a
        diff22 = 1. / dx * np.gradient(diff2)

        tn_diff2 = field.laplacian(coord_sys=coord_sys, compress_level=1)
        tn_diff2_vec = tn_diff2.get_comp_data(0)

        d2_error_mid += [np.log10(np.abs(tn_diff2_vec[npts // 2] - d2func(xdata[npts // 2])))]
        d2_error_0 += [np.log10(np.abs(tn_diff2_vec[0] - d2func(xdata[0])))]
        d2_error_L += [np.log10(np.abs(tn_diff2_vec[-1] - d2func(xdata[-1])))]
        d2_error_avg += [np.log10(np.linalg.norm(tn_diff2_vec - d2func(xdata)) / np.sqrt(npts))]

        plt.plot(xdata, diff22, linestyle='-', linewidth=2)
        plt.plot(xdata, d2func(xdata), 'k--')
        plt.plot(xdata, tn_diff2_vec, 'r:', linewidth=3)
        plt.title('laplacian')
        plt.show()

    ## d/dx error
    plt.plot(orders, dx_error_mid, 'bx', label='mid')
    plt.plot(orders, dx_error_0, 'gx', label='end0')
    plt.plot(orders, dx_error_L, 'rx', label='endL')
    plt.plot(orders, dx_error_avg, 'kx', label='avg')

    res_mid, res_0, res_L, res_avg = np.polyfit([x for x in range(1, 5)],
                                                np.array([dx_error_mid, dx_error_0, dx_error_L, dx_error_avg]).T, 1).T
    data_fine = np.linspace(0, 5, 51)
    plt.plot(data_fine, np.polyval(res_mid, data_fine), 'b:', label='p-mid')
    plt.plot(data_fine, np.polyval(res_0, data_fine), 'g:', label='p-end0')
    plt.plot(data_fine, np.polyval(res_L, data_fine), 'r:', label='p-endL')
    plt.plot(data_fine, np.polyval(res_avg, data_fine), 'k:', label='p-avg')

    print('average error', dx_error_avg)
    print('midpt error', dx_error_mid)
    print('end error', dx_error_L)

    plt.xlabel('order')
    plt.ylabel('log10(error)')
    plt.legend()
    plt.show()

    ## d2/dx2 error
    plt.plot(orders, d2_error_mid, 'bx', label='mid')
    plt.plot(orders, d2_error_0, 'gx', label='end0')
    plt.plot(orders, d2_error_L, 'rx', label='endL')
    plt.plot(orders, d2_error_avg, 'kx', label='avg')

    res_mid, res_0, res_L, res_avg = np.polyfit([x for x in range(1, 5)],
                                                np.array([d2_error_mid, d2_error_0, d2_error_L, d2_error_avg]).T, 1).T
    data_fine = np.linspace(0, 5, 51)
    plt.plot(data_fine, np.polyval(res_mid, data_fine), 'b:', label='p-mid')
    plt.plot(data_fine, np.polyval(res_0, data_fine), 'g:', label='p-end0')
    plt.plot(data_fine, np.polyval(res_L, data_fine), 'r:', label='p-endL')
    plt.plot(data_fine, np.polyval(res_avg, data_fine), 'k:', label='p-avg')

    print('average error', d2_error_avg)
    print('midpt error', d2_error_mid)
    print('end error', d2_error_L)

    plt.xlabel('order')
    plt.ylabel('log10(error)')
    plt.title('d2/dx2 error')
    plt.legend()
    plt.show()

######################
# scalar field in 2D #
######################

if do_test_2Dopen:
    K = 2
    xdata = np.linspace(0, 10, npts)
    data_x, data_y = np.meshgrid(xdata, xdata, indexing='ij')
    dx = dy = xdata[1] - xdata[0]

    func    = lambda x,y:  (2*(x-4)**3 + (x+2)**3.5)      * y**2
    dxfunc  = lambda x,y:  (6*(x-4)**2 + 3.5*(x+2)**2.5)  * y**2
    dyfunc  = lambda x,y:  (2*(x-4)**3 + (x+2)**3.5)      * 2*y
    dxxfunc = lambda x,y:  (12*(x-4)   + 3.5*2.5*(x+2)**1.5)  * y**2
    dxyfunc = lambda x,y:  (6*(x-4)**2 + 3.5*(x+2)**2.5)  * 2*y
    dyyfunc = lambda x,y:  (2*(x-4)**3 + (x+2)**3.5)      * 2

    # func    = lambda x,y:  2*((x-4)**5) + 10*(x+2)**2.5
    # dxfunc  = lambda x,y:  10*(x-4)**4  + 10*2.5*(x+2)**1.5
    # dyfunc  = lambda x,y:  0*x
    # dxxfunc = lambda x,y:  40*(x-4)**3  + 10*2.5*1.5*(x+2)**0.5
    # dxyfunc = lambda x,y:  0*x
    # dyyfunc = lambda x,y:  0*x

    # func    = lambda x,y:  2*((y-4)**5) + 10*(y+2)**2.5
    # dxfunc  = lambda x,y:  0*x
    # dyfunc  = lambda x,y:  10*(y-4)**4  + 10*2.5*(y+2)**1.5
    # dxxfunc = lambda x,y:  0*x
    # dxyfunc = lambda x,y:  0*x
    # dyyfunc = lambda x,y:  40*(y-4)**3  + 10*2.5*1.5*(y+2)**0.5

    # vec_data = np.array([func(x) for x in data])
    func_data = func(data_x,data_y)
    dxfunc_data = dxfunc(data_x,data_y)
    dyfunc_data = dyfunc(data_x,data_y)
    dxxfunc_data = dxxfunc(data_x,data_y)
    dxyfunc_data = dxyfunc(data_x,data_y)
    dyyfunc_data = dyyfunc(data_x,data_y)

    orders = [x for x in range(1,5)]
    dx_err, dy_err = [], []
    dxx_err, dxy_err, dyy_err = [], [], []
    fd_dx_err, fd_dy_err = [], []
    fd_dxx_err, fd_dxy_err, fd_dyy_err = [], [], []

    X = Coordinate('X')
    Y = Coordinate('Y')
    coord_sys = CartesianCoordinateSpace('X', coords=[X, Y])

    axis1 = Axis(L, q, coordinate=X, xpts=xdata)
    axis2 = axis1.create_like(new_coord=Y)

    if TEST_LAYOUT is LayoutType.COMB:
        subgrid1 = Grid1D('SG1', (axis1,), layout_type=LayoutType.SEQUENTIAL)
        subgrid2 = Grid1D('SG2', (axis2,), layout_type=LayoutType.SEQUENTIAL)
        grid2 = GridsComb('G2', (subgrid1,subgrid2,), )
    else:
        grid2 = Grid1D('G2', (axis1,axis2), layout_type=TEST_LAYOUT)
    field = ScalarField('F2', grid2, data=func_data)
    field.component.update_deriv_params(axis1, left_bc=BCType.OPEN, right_bc=BCType.OPEN, fd_type=TEST_FD)
    field.component.update_deriv_params(axis2, left_bc=BCType.OPEN, right_bc=BCType.OPEN, fd_type=TEST_FD)

    for order in orders:
        field.component.update_deriv_params(axis1, order=order)
        field.component.update_deriv_params(axis2, order=order)

        fd_dx = findiff.FinDiff(0, 1, 1, acc=2*order)
        fd_dy = findiff.FinDiff(1, 1, 1, acc=2*order)
        fd_dxx = findiff.FinDiff(0, 1, 2, acc=2*order)
        fd_dyy = findiff.FinDiff(1, 1, 2, acc=2*order)
        fd_dxy = findiff.FinDiff((0,1,1),(1,1,1), acc=2*order)

        ### center first order diff ###a
        # diff1_x = 1./dx * np.gradient(func_data, axis=0)
        # diff1_y = 1./dy * np.gradient(func_data, axis=1)
        diff1_x = 1./dx * fd_dx(func_data)
        diff1_y = 1./dy * fd_dy(func_data)

        tn_diff = field.gradient()
        print('max bond',field.max_bond(), tn_diff.max_bond(X), tn_diff.max_bond(Y))
        tn_diff_x_array = tn_diff.get_comp_data(X)
        tn_diff_y_array = tn_diff.get_comp_data(Y)

        dx_err += [ np.log10( np.linalg.norm( tn_diff_x_array - dxfunc_data )/npts ) ]
        dy_err += [ np.log10( np.linalg.norm( tn_diff_y_array - dyfunc_data )/npts ) ]
        fd_dx_err += [ np.log10( np.linalg.norm( diff1_x - dxfunc_data )/npts ) ]
        fd_dy_err += [ np.log10( np.linalg.norm( diff1_y - dyfunc_data )/npts ) ]

        # ## plot at d/dx f(x,y) at y = yi
        # yi = -1
        # plt.plot(xdata,dxfunc_data[:,yi],'k-')
        # plt.plot(xdata,diff1_x[:,yi],'g--',linewidth=3)
        # plt.plot(xdata,tn_diff_x_array[:,yi],'r:',linewidth=3)
        # plt.title('df/dx')
        # plt.show()
        #
        # ## plot at d/dy f(x,y) at x = 0
        # xi = -1
        # plt.plot(xdata,dyfunc_data[xi,:],'k-')
        # plt.plot(xdata,diff1_y[xi,:],'g--',linewidth=3)
        # plt.plot(xdata,tn_diff_y_array[xi,:],'r:',linewidth=3)
        # plt.title('df/dy')
        # plt.show()

        # fig,ax = plt.subplots(2,3, figsize=(10,10))
        # im00 = ax[0,0].imshow((diff1_y         - dyfunc_data    ))
        # im01 = ax[0,1].imshow((tn_diff_y_array - dyfunc_data    ))
        # im02 = ax[0,2].imshow((diff1_y         - tn_diff_y_array))
        # im10 = ax[1,0].imshow((diff1_x         - dxfunc_data    ))
        # im11 = ax[1,1].imshow((tn_diff_x_array - dxfunc_data    ))
        # im12 = ax[1,2].imshow((diff1_x         - tn_diff_x_array))

        # ims = np.array([[im00,im01,im02],[im10,im11,im12]])
        # diff_strs = ['np-anl','tn-anl','np-tn']
        # dim_strs  = [' df/dy',' df/dx']
        # for idx in np.ndindex((2,3)):
        #     fig.colorbar(ims[idx], ax=ax[idx])
        #     ax[idx].set_xlabel('y')
        #     ax[idx].set_ylabel('x')
        #     ax[idx].set_title( diff_strs[idx[1]] + dim_strs[idx[0]] )

        # fig.subplots_adjust(wspace=0.3)

        # plt.show()

        print('order',order)
        print('difference np d/dx',np.linalg.norm( diff1_x-dxfunc_data ) * 1./np.sqrt(npts**K))
        print('difference np d/dy',np.linalg.norm( diff1_y-dyfunc_data ) * 1./np.sqrt(npts**K))
        print('difference tn d/dx',np.linalg.norm( tn_diff_x_array-dxfunc_data ) * 1./np.sqrt(npts**K))
        print('difference tn d/dy',np.linalg.norm( tn_diff_y_array-dyfunc_data ) * 1./np.sqrt(npts**K))


        ### second order diff ###a
        # diff2_xx = 1./dx * np.gradient(diff1_x, axis=0)
        # diff2_yy = 1./dy * np.gradient(diff1_y, axis=1)
        # diff2_xy = 1./dy * np.gradient(diff1_x, axis=1)
        diff2_xx = 1./dx**2 * fd_dxx(func_data)
        diff2_yy = 1./dy**2 * fd_dyy(func_data)
        diff2_xy = 1./dx/dy * fd_dxy(func_data)

        tn_diff2_xx = field.component.take_secondderivative(axis1, axis1, compress=True)
        tn_diff2_yy = field.component.take_secondderivative(axis2, axis2, compress=True)
        tn_diff2_xy = field.component.take_secondderivative(axis1, axis2, compress=True)
        tn_diff2_xx_array = tn_diff2_xx.get_data()
        tn_diff2_yy_array = tn_diff2_yy.get_data()
        tn_diff2_xy_array = tn_diff2_xy.get_data()

        dxx_err += [ np.log10( np.linalg.norm( tn_diff2_xx_array - dxxfunc_data )/npts ) ]
        dxy_err += [ np.log10( np.linalg.norm( tn_diff2_xy_array - dxyfunc_data )/npts ) ]
        dyy_err += [ np.log10( np.linalg.norm( tn_diff2_yy_array - dyyfunc_data )/npts ) ]
        fd_dxx_err += [ np.log10( np.linalg.norm( diff2_xx - dxxfunc_data )/npts ) ]
        fd_dxy_err += [ np.log10( np.linalg.norm( diff2_xy - dxyfunc_data )/npts ) ]
        fd_dyy_err += [ np.log10( np.linalg.norm( diff2_yy - dyyfunc_data )/npts ) ]

        # ## plot at d^2/dx^2 f(x,y) at y = yi
        # yi = 0
        # plt.plot(data,dxxfunc_data[:,yi],'k-')
        # plt.plot(data,diff2_xx[:,yi],'g--',linewidth=3)
        # plt.plot(data,tn_diff2_xx_array[:,yi],'r:',linewidth=3)
        # plt.show()

        # ## plot at d^2/dy^2 f(x,y) at y = 0
        # yi = -1
        # plt.plot(data,dyyfunc_data[:,yi],'k-')
        # plt.plot(data,diff2_yy[:,yi],'g--',linewidth=3)
        # plt.plot(data,tn_diff2_yy_array[:,yi],'r:',linewidth=3)
        # plt.show()

        # ## plot at d^2/dxdy f(x,y) at x = 0
        # xi = -1
        # plt.plot(data,dxyfunc_data[xi,:],'k-')
        # plt.plot(data,diff2_xy[xi,:],'g--',linewidth=3)
        # plt.plot(data,tn_diff2_xy_array[xi,:],'r:',linewidth=3)
        # plt.show()

        # print('difference np d2/dx2' ,np.linalg.norm( diff2_xx-dxxfunc_data ))
        # print('difference np d2/dy2' ,np.linalg.norm( diff2_yy-dyyfunc_data ))
        # print('difference np d2/dxdy',np.linalg.norm( diff2_xy-dxyfunc_data ))
        # print('difference tn d2/dx2' ,np.linalg.norm( tn_diff2_xx_array-dxxfunc_data ))
        # print('difference tn d2/dy2' ,np.linalg.norm( tn_diff2_yy_array-dyyfunc_data ))
        # print('difference tn d2/dxdy',np.linalg.norm( tn_diff2_xy_array-dxyfunc_data ))

    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()

    ## error
    ax1.plot(orders,dx_err,'bx',label='x')
    ax1.plot(orders,dy_err,'rx',label='y')
    ax2.plot(orders,dxx_err,'bx',label='xx')
    ax2.plot(orders,dxy_err,'gx',label='xy')
    ax2.plot(orders,dyy_err,'rx',label='yy')

    res_x, res_y, res_xx, res_xy, res_yy = np.polyfit([x for x in range(1,5)],
                           np.array([dx_err, dy_err, dxx_err, dxy_err, dyy_err]).T, 1).T
    data_fine = np.linspace(0,5,51)
    ax1.plot(data_fine, np.polyval(res_x ,data_fine),'b:' ,label='p-x')
    ax1.plot(data_fine, np.polyval(res_y ,data_fine),'r:' ,label='p-y')
    ax2.plot(data_fine, np.polyval(res_xx,data_fine),'b:',label='p-xx')
    ax2.plot(data_fine, np.polyval(res_xy,data_fine),'g:',label='p-xy')
    ax2.plot(data_fine, np.polyval(res_yy,data_fine),'r:',label='p-yy')

    ## error
    ax1.plot(orders,fd_dx_err,'bo',label='fd-x')
    ax1.plot(orders,fd_dy_err,'ro',label='fd-y')
    ax2.plot(orders,fd_dxx_err,'bo',label='fd-xx')
    ax2.plot(orders,fd_dxy_err,'go',label='fd-xy')
    ax2.plot(orders,fd_dyy_err,'ro',label='fd-yy')

    res_x, res_y, res_xx, res_xy, res_yy = np.polyfit([x for x in range(1,5)],
                           np.array([fd_dx_err, fd_dy_err, fd_dxx_err, fd_dxy_err, fd_dyy_err]).T, 1).T
    data_fine = np.linspace(0,5,51)
    ax1.plot(data_fine, np.polyval(res_x ,data_fine),'b--' ,label='fdp-x')
    ax1.plot(data_fine, np.polyval(res_y ,data_fine),'r--' ,label='fdp-y')
    ax2.plot(data_fine, np.polyval(res_xx,data_fine),'b--',label='fdp-xx')
    ax2.plot(data_fine, np.polyval(res_xy,data_fine),'g--',label='fdp-xy')
    ax2.plot(data_fine, np.polyval(res_yy,data_fine),'r--',label='fdp-yy')

    ax1.set_xlabel('order')
    ax1.set_ylabel('log10(error)')
    ax2.set_xlabel('order')
    ax2.set_ylabel('log10(error)')
    fig1.legend()
    fig2.legend()
    plt.show()

###############################
# periodic scalar field in 2D #
###############################

if do_test_2Dperiodic:
    K = 2
    # xdata = np.linspace(0, 2 * np.pi, npts, endpoint=False)
    # ydata = np.linspace(0, 2 * np.pi, 2**6, endpoint=False)
    xdata = np.linspace(-np.pi, np.pi, npts, endpoint=False)
    ydata = np.linspace(-np.pi, np.pi, 2 ** 6, endpoint=False)
    data_x, data_y = np.meshgrid(xdata, ydata, indexing='ij')
    dx = dy = xdata[1] - xdata[0]
    
    func    = lambda x,y:  np.sin(x+3*np.pi/5) * np.cos(2*y+0.5)
    dxfunc  = lambda x,y:  np.cos(x+3*np.pi/5) * np.cos(2*y+0.5)
    dyfunc  = lambda x,y:  np.sin(x+3*np.pi/5) * -2*np.sin(2*y+0.5)
    dxxfunc = lambda x,y: -np.sin(x+3*np.pi/5) * np.cos(2*y+0.5)
    dxyfunc = lambda x,y:  np.cos(x+3*np.pi/5) * -2*np.sin(2*y+0.5)
    dyyfunc = lambda x,y:  np.sin(x+3*np.pi/5) * -4*np.cos(2*y+0.5)

    ### note that this fails is fct is not actually periodic in defined grid-space
    
    # vec_data = np.array([func(x) for x in data])
    func_data = func(data_x,data_y)
    dxfunc_data = dxfunc(data_x,data_y)
    dyfunc_data = dyfunc(data_x,data_y)
    dxxfunc_data = dxxfunc(data_x,data_y)
    dxyfunc_data = dxyfunc(data_x,data_y)
    dyyfunc_data = dyyfunc(data_x,data_y)
 
    orders = [x for x in range(1,5)]
    dx_err, dy_err = [], []
    dxx_err, dxy_err, dyy_err = [], [], []

    X = Coordinate('X')
    Y = Coordinate('Y')
    coord_sys = CartesianCoordinateSpace('X', coords=[X, Y])

    axis1 = Axis(L, q, coordinate=X, xpts=xdata, is_flipped=True)
    # axis2 = axis1.create_like(new_coord=Y)
    axis2 = Axis(6, q, coordinate=Y, xpts=ydata)
    # axis3 = Axis(3, q)
    # grid2 = Grid1D('G2', (axis1, axis2, axis3), layout_type=LayoutType.PARALLEL_GROUP)

    if TEST_LAYOUT is LayoutType.COMB:
        subgrid1 = Grid1D('SG1', (axis1,), layout_type=LayoutType.SEQUENTIAL)
        subgrid2 = Grid1D('SG2', (axis2,), layout_type=LayoutType.SEQUENTIAL)
        grid2 = GridsComb('G2', (subgrid1,subgrid2,), )
    else:
        grid2 = Grid1D('G2', (axis1, axis2), layout_type=TEST_LAYOUT)
    # print(grid2.layout.get_inds_in_axis(grid2.axes, axis1))
    # print(grid2.layout.get_inds_in_axis(grid2.axes, axis2))
    # print(grid2.layout.get_inds_in_axis(grid2.axes, axis3))
    # print(grid2.layout.shape(grid2.axes))
    # print(grid2.layout._transpose_sequential_to_parallel_inds((axis1, axis2, axis3)))
    # print(grid2.layout._transpose_parallel_to_sequential_inds((axis1, axis2, axis3)))
    # exit()
    field = ScalarField('F2', grid2, data=func_data)
    field.component.update_deriv_params(axis1, left_bc=BCType.PERIODIC, right_bc=BCType.PERIODIC, fd_type=TEST_FD)
    field.component.update_deriv_params(axis2, left_bc=BCType.PERIODIC, right_bc=BCType.PERIODIC, fd_type=TEST_FD)

    for order in orders:
        field.component.update_deriv_params(axis1, order=order)
        field.component.update_deriv_params(axis2, order=order)


        ### center first order diff ###
        diff1_x = 1. / dx * np.gradient(func_data, axis=0)
        diff1_y = 1. / dy * np.gradient(func_data, axis=1)

        print('taking gradient')
        tn_diff = field.gradient()
        tn_diff_x_array = tn_diff.get_comp_data(X)
        tn_diff_y_array = tn_diff.get_comp_data(Y)

        dx_err += [ np.log10( np.linalg.norm(tn_diff_x_array-dxfunc_data)/npts ) ]
        dy_err += [ np.log10( np.linalg.norm(tn_diff_y_array-dyfunc_data)/npts ) ]
        
        ## plot at d/dx f(x,y) at y = yi
        yi = -1
        # plt.plot(data,dxfunc_data[:,yi],'k-')
        # plt.plot(data,diff1_x[:,yi]-dxfunc_data[:,yi],'g--',linewidth=3)
        plt.plot(xdata, tn_diff_x_array[:, yi] - dxfunc_data[:, yi], 'r:', linewidth=3)
        plt.title('d/dx at y_i')
        plt.show()

        ## plot at d/dy f(x,y) at x = 0
        xi = -1
        # plt.plot(data,dyfunc_data[xi,:],'k-')
        # plt.plot(data,diff1_y[xi,:]-dyfunc_data[xi,:],'g--',linewidth=3)
        plt.plot(ydata, tn_diff_y_array[xi, :] - dyfunc_data[xi, :], 'r:', linewidth=3)
        plt.title('d/dy at x_i')
        plt.show()

        # fig,ax = plt.subplots(2,3, figsize=(10,10))
        # im00 = ax[0,0].imshow((diff1_y         - dyfunc_data    ))
        # im01 = ax[0,1].imshow((tn_diff_y_array - dyfunc_data    ))
        # im02 = ax[0,2].imshow((diff1_y         - tn_diff_y_array))
        # im10 = ax[1,0].imshow((diff1_x         - dxfunc_data    ))
        # im11 = ax[1,1].imshow((tn_diff_x_array - dxfunc_data    ))
        # im12 = ax[1,2].imshow((diff1_x         - tn_diff_x_array))

        # ims = np.array([[im00,im01,im02],[im10,im11,im12]])
        # diff_strs = ['np-anl','tn-anl','np-tn']
        # dim_strs  = [' df/dy',' df/dx']
        # for idx in np.ndindex((2,3)):
        #     fig.colorbar(ims[idx], ax=ax[idx])
        #     ax[idx].set_xlabel('y')
        #     ax[idx].set_ylabel('x')
        #     ax[idx].set_title( diff_strs[idx[1]] + dim_strs[idx[0]] )

        # fig.subplots_adjust(wspace=0.3)

        # plt.show()

        print('difference np d/dx',np.linalg.norm( diff1_x-dxfunc_data ) * 1./np.sqrt(npts**K))
        print('difference np d/dy',np.linalg.norm( diff1_y-dyfunc_data ) * 1./np.sqrt(npts**K))
        print('difference tn d/dx',np.linalg.norm( tn_diff_x_array-dxfunc_data ) * 1./np.sqrt(npts**K))
        print('difference tn d/dy',np.linalg.norm( tn_diff_y_array-dyfunc_data ) * 1./np.sqrt(npts**K))


        ### second order diff ###a
        diff2_xx = 1. / dx * np.gradient(diff1_x, axis=0)
        diff2_yy = 1. / dy * np.gradient(diff1_y, axis=1)
        diff2_xy = 1. / dy * np.gradient(diff1_x, axis=1)
        
        tn_diff2_xx = field.component.take_secondderivative(axis1, axis1, compress=True)
        tn_diff2_yy = field.component.take_secondderivative(axis2, axis2, compress=True)
        tn_diff2_xy = field.component.take_secondderivative(axis1, axis2, compress=True)
        tn_diff2_xx_array = tn_diff2_xx.get_data()
        tn_diff2_yy_array = tn_diff2_yy.get_data()
        tn_diff2_xy_array = tn_diff2_xy.get_data()

        dxx_err += [ np.log10( np.linalg.norm(tn_diff2_xx_array-dxxfunc_data)/npts ) ]
        dxy_err += [ np.log10( np.linalg.norm(tn_diff2_xy_array-dxyfunc_data)/npts ) ]
        dyy_err += [ np.log10( np.linalg.norm(tn_diff2_yy_array-dyyfunc_data)/npts ) ]

        # ## plot at d^2/dx^2 f(x,y) at y = yi
        # yi = 3
        # plt.plot(data,dxxfunc_data[:,yi],'k-')
        # plt.plot(data,diff2_xx[:,yi],'g--',linewidth=3)
        # plt.plot(data,tn_diff2_xx_array[:,yi],'r:',linewidth=3)
        # plt.show()

        # ## plot at d^2/dy^2 f(x,y) at y = 0
        # yi = 3
        # plt.plot(data,dyyfunc_data[:,yi],'k-')
        # plt.plot(data,diff2_yy[:,yi],'g--',linewidth=3)
        # plt.plot(data,tn_diff2_yy_array[:,yi],'r:',linewidth=3)
        # plt.show()

        # ## plot at d^2/dxdy f(x,y) at x = 0
        # xi = 3
        # plt.plot(data,dxyfunc_data[xi,:],'k-')
        # plt.plot(data,diff2_xy[xi,:],'g--',linewidth=3)
        # plt.plot(data,tn_diff2_xy_array[xi,:],'r:',linewidth=3)
        # plt.show()

        print('difference np d2/dx2' ,np.linalg.norm( diff2_xx-dxxfunc_data ))
        print('difference np d2/dy2' ,np.linalg.norm( diff2_yy-dyyfunc_data ))
        print('difference np d2/dxdy',np.linalg.norm( diff2_xy-dxyfunc_data ))
        print('difference tn d2/dx2' ,np.linalg.norm( tn_diff2_xx_array-dxxfunc_data ))
        print('difference tn d2/dy2' ,np.linalg.norm( tn_diff2_yy_array-dyyfunc_data ))
        print('difference tn d2/dxdy',np.linalg.norm( tn_diff2_xy_array-dxyfunc_data ))

    ## error
    plt.plot(orders,dx_err,'bx',label='x')
    plt.plot(orders,dy_err,'rx',label='y')
    plt.plot(orders,dxx_err,'bo',label='xx')
    plt.plot(orders,dxy_err,'go',label='xy')
    plt.plot(orders,dyy_err,'ro',label='yy')

    res_x, res_y, res_xx, res_xy, res_yy = np.polyfit([x for x in range(1,5)], 
                           np.array([dx_err, dy_err, dxx_err, dxy_err, dyy_err]).T, 1).T
    data_fine = np.linspace(0,5,51)
    plt.plot(data_fine, np.polyval(res_x ,data_fine),'b:' ,label='p-x')
    plt.plot(data_fine, np.polyval(res_y ,data_fine),'r:' ,label='p-y')
    plt.plot(data_fine, np.polyval(res_xx,data_fine),'b--',label='p-xx')
    plt.plot(data_fine, np.polyval(res_xy,data_fine),'g--',label='p-xy')
    plt.plot(data_fine, np.polyval(res_yy,data_fine),'r--',label='p-yy')

    plt.xlabel('order')
    plt.ylabel('log10(error)')
    plt.legend()
    plt.show()


##################################
# PBC/antiPBC scalar field in 2D #
##################################

if do_test_2Dapc:
    K = 2
    xdata = np.linspace(0, 2 * np.pi, npts, endpoint=False)
    data_y, data_x = np.meshgrid(xdata, xdata)
    dx = dy = xdata[1] - xdata[0]
    
    func    = lambda x,y:  np.sin(3./2*x) * np.sin(y)
    dxfunc  = lambda x,y:  np.cos(3./2*x) * np.sin(y) * 3./2
    dyfunc  = lambda x,y:  np.sin(3./2*x) * np.cos(y)
    dxxfunc = lambda x,y: -np.sin(3./2*x) * np.sin(y) * 9./4
    dxyfunc = lambda x,y:  np.cos(3./2*x) * np.cos(y) * 3./2
    dyyfunc = lambda x,y:  np.sin(3./2*x) * np.sin(y) * -1

    ### note that this fails is fct is not actually periodic in defined grid-space
    
    # vec_data = np.array([func(x) for x in data])
    func_data = func(data_x,data_y)
    dxfunc_data = dxfunc(data_x,data_y)
    dyfunc_data = dyfunc(data_x,data_y)
    dxxfunc_data = dxxfunc(data_x,data_y)
    dxyfunc_data = dxyfunc(data_x,data_y)
    dyyfunc_data = dyyfunc(data_x,data_y)
 
    orders = [x for x in range(1,5)]
    dx_err, dy_err = [], []
    dxx_err, dxy_err, dyy_err = [], [], []

    X = Coordinate('X')
    Y = Coordinate('Y')
    coord_sys = CartesianCoordinateSpace('X', coords=[X, Y])

    axis1 = Axis(L, q, coordinate=X, xpts=xdata)
    axis2 = axis1.create_like(new_coord=Y)
    if TEST_LAYOUT is LayoutType.COMB:
        subgrid1 = Grid1D('SG1', (axis1,), layout_type=LayoutType.SEQUENTIAL)
        subgrid2 = Grid1D('SG2', (axis2,), layout_type=LayoutType.SEQUENTIAL)
        grid2 = GridsComb('G2', (subgrid1,subgrid2,), )
    else:
        grid2 = Grid1D('G2', (axis1, axis2), layout_type=TEST_LAYOUT)
    field = ScalarField('F2', grid2, func_data)
    field.component.update_deriv_params(axis1, left_bc=BCType.ANTIPERIODIC, fd_type=TEST_FD)
    field.component.update_deriv_params(axis2, left_bc=BCType.PERIODIC, fd_type=TEST_FD)

    for order in orders:
        field.component.update_deriv_params(axis1, order=order)
        field.component.update_deriv_params(axis2, order=order)

        ### center first order diff ###a
        diff1_x = 1. / dx * np.gradient(func_data, axis=0)
        diff1_y = 1. / dy * np.gradient(func_data, axis=1)
        
        tn_diff = field.gradient()
        tn_diff_x_array = tn_diff[X].get_data()
        tn_diff_y_array = tn_diff[Y].get_data()

        dx_err += [ np.log10( np.linalg.norm(tn_diff_x_array-dxfunc_data)/npts ) ]
        dy_err += [ np.log10( np.linalg.norm(tn_diff_y_array-dyfunc_data)/npts ) ]
        
        ## plot at d/dx f(x,y) at y = yi
        yi = -1
        # plt.plot(data,dxfunc_data[:,yi],'k-')
        # plt.plot(data,diff1_x[:,yi]-dxfunc_data[:,yi],'g--',linewidth=3)
        plt.plot(xdata, tn_diff_x_array[:, yi] - dxfunc_data[:, yi], 'r:', linewidth=3)
        plt.title('d/dx at y_i')
        plt.show()

        ## plot at d/dy f(x,y) at x = 0
        xi = -1
        # plt.plot(data,dyfunc_data[xi,:],'k-')
        # plt.plot(data,diff1_y[xi,:]-dyfunc_data[xi,:],'g--',linewidth=3)
        plt.plot(xdata, tn_diff_y_array[xi, :] - dyfunc_data[xi, :], 'r:', linewidth=3)
        plt.title('d/dy at x_i')
        plt.show()

        # fig,ax = plt.subplots(2,3, figsize=(10,10))
        # im00 = ax[0,0].imshow((diff1_y         - dyfunc_data    ))
        # im01 = ax[0,1].imshow((tn_diff_y_array - dyfunc_data    ))
        # im02 = ax[0,2].imshow((diff1_y         - tn_diff_y_array))
        # im10 = ax[1,0].imshow((diff1_x         - dxfunc_data    ))
        # im11 = ax[1,1].imshow((tn_diff_x_array - dxfunc_data    ))
        # im12 = ax[1,2].imshow((diff1_x         - tn_diff_x_array))

        # ims = np.array([[im00,im01,im02],[im10,im11,im12]])
        # diff_strs = ['np-anl','tn-anl','np-tn']
        # dim_strs  = [' df/dy',' df/dx']
        # for idx in np.ndindex((2,3)):
        #     fig.colorbar(ims[idx], ax=ax[idx])
        #     ax[idx].set_xlabel('y')
        #     ax[idx].set_ylabel('x')
        #     ax[idx].set_title( diff_strs[idx[1]] + dim_strs[idx[0]] )

        # fig.subplots_adjust(wspace=0.3)

        # plt.show()

        print('difference np d/dx',np.linalg.norm( diff1_x-dxfunc_data ) * 1./np.sqrt(npts**K))
        print('difference np d/dy',np.linalg.norm( diff1_y-dyfunc_data ) * 1./np.sqrt(npts**K))
        print('difference tn d/dx',np.linalg.norm( tn_diff_x_array-dxfunc_data ) * 1./np.sqrt(npts**K))
        print('difference tn d/dy',np.linalg.norm( tn_diff_y_array-dyfunc_data ) * 1./np.sqrt(npts**K))


        ### second order diff ###a
        diff2_xx = 1. / dx * np.gradient(diff1_x, axis=0)
        diff2_yy = 1. / dy * np.gradient(diff1_y, axis=1)
        diff2_xy = 1. / dy * np.gradient(diff1_x, axis=1)
        
        tn_diff2_xx = field.component.take_secondderivative(axis1, None, compress=True)
        tn_diff2_yy = field.component.take_secondderivative(axis2, None, compress=True)
        tn_diff2_xy = field.component.take_secondderivative(axis1, axis2, compress=True)
        tn_diff2_xx_array = tn_diff2_xx.get_data()
        tn_diff2_yy_array = tn_diff2_yy.get_data()
        tn_diff2_xy_array = tn_diff2_xy.get_data()

        dxx_err += [ np.log10( np.linalg.norm(tn_diff2_xx_array-dxxfunc_data)/npts ) ]
        dxy_err += [ np.log10( np.linalg.norm(tn_diff2_xy_array-dxyfunc_data)/npts ) ]
        dyy_err += [ np.log10( np.linalg.norm(tn_diff2_yy_array-dyyfunc_data)/npts ) ]

        # ## plot at d^2/dx^2 f(x,y) at y = yi
        # yi = 3
        # plt.plot(data,dxxfunc_data[:,yi],'k-')
        # plt.plot(data,diff2_xx[:,yi],'g--',linewidth=3)
        # plt.plot(data,tn_diff2_xx_array[:,yi],'r:',linewidth=3)
        # plt.show()

        # ## plot at d^2/dy^2 f(x,y) at y = 0
        # yi = 3
        # plt.plot(data,dyyfunc_data[:,yi],'k-')
        # plt.plot(data,diff2_yy[:,yi],'g--',linewidth=3)
        # plt.plot(data,tn_diff2_yy_array[:,yi],'r:',linewidth=3)
        # plt.show()

        # ## plot at d^2/dxdy f(x,y) at x = 0
        # xi = 3
        # plt.plot(data,dxyfunc_data[xi,:],'k-')
        # plt.plot(data,diff2_xy[xi,:],'g--',linewidth=3)
        # plt.plot(data,tn_diff2_xy_array[xi,:],'r:',linewidth=3)
        # plt.show()

        print('difference np d2/dx2' ,np.linalg.norm( diff2_xx-dxxfunc_data ))
        print('difference np d2/dy2' ,np.linalg.norm( diff2_yy-dyyfunc_data ))
        print('difference np d2/dxdy',np.linalg.norm( diff2_xy-dxyfunc_data ))
        print('difference tn d2/dx2' ,np.linalg.norm( tn_diff2_xx_array-dxxfunc_data ))
        print('difference tn d2/dy2' ,np.linalg.norm( tn_diff2_yy_array-dyyfunc_data ))
        print('difference tn d2/dxdy',np.linalg.norm( tn_diff2_xy_array-dxyfunc_data ))

    ## error
    plt.plot(orders,dx_err,'bx',label='x')
    plt.plot(orders,dy_err,'rx',label='y')
    plt.plot(orders,dxx_err,'bo',label='xx')
    plt.plot(orders,dxy_err,'go',label='xy')
    plt.plot(orders,dyy_err,'ro',label='yy')

    res_x, res_y, res_xx, res_xy, res_yy = np.polyfit([x for x in range(1,5)], 
                           np.array([dx_err, dy_err, dxx_err, dxy_err, dyy_err]).T, 1).T
    data_fine = np.linspace(0,5,51)
    plt.plot(data_fine, np.polyval(res_x ,data_fine),'b:' ,label='p-x')
    plt.plot(data_fine, np.polyval(res_y ,data_fine),'r:' ,label='p-y')
    plt.plot(data_fine, np.polyval(res_xx,data_fine),'b--',label='p-xx')
    plt.plot(data_fine, np.polyval(res_xy,data_fine),'g--',label='p-xy')
    plt.plot(data_fine, np.polyval(res_yy,data_fine),'r--',label='p-yy')

    plt.xlabel('order')
    plt.ylabel('log10(error)')
    plt.legend()
    plt.show()

