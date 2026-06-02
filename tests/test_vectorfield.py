"""Unit-style checks for 3-D Field vector calculus.

Exercises vector-field construction and finite-difference vector-calculus
operators in 3-D on a quantized-tensor-train grid, validating against findiff
references.
"""
import sys
sys.path.append('../')

from setup_.defaults import *
import numpy as np
import matplotlib.pyplot as plt
import findiff

from axis import Axis
from coord.coord_sys import Coordinate
from coord.cartesian import CartesianCoordinateSpace
from grid1D import Grid1D
from grid_comb import GridsComb
from field import Field


q = 2
L = 4
npts = q**L


TEST_LAYOUT = LayoutType.PARALLEL
TEST_FD = FDType.BACKWARD

######################
# vector field in 3D #
######################

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

orders = [x for x in range(1,5)]
grad_err,    lapl_err,    curl_err,   div_err     = [], [], [], []
fd_grad_err, fd_lapl_err, fd_curl_err, fd_div_err = [], [], [], []

X = Coordinate('X',CoordinateType.X)
Y = Coordinate('Y',CoordinateType.Y)
Z = Coordinate('Z',CoordinateType.Z)
coord_sys = CartesianCoordinateSpace('pos', coords=[X,Y,Z])

axis0 = Axis(L, q, coordinate=X, xpts=xdata)
axis1 = Axis(L, q, coordinate=Y, xpts=xdata)
axis2 = Axis(L, q, coordinate=Z, xpts=xdata)

cart_cs = CartesianCoordinateSpace('C1', axis0, axis1, axis2)
if TEST_LAYOUT is LayoutType.COMB:
    grid_ax0 = Grid1D('g0', (axis0,))
    grid_ax1 = Grid1D('g1', (axis1,))
    grid_ax2 = Grid1D('g2', (axis2,))
    grid1 = GridsComb('G1', (grid_ax0, grid_ax1, grid_ax2), )
else:
    grid1 = Grid1D('G1', (axis0,axis1,axis2), layout_type=TEST_LAYOUT)

field = Field('F1', grid1, data={X: func0_data, Y: func1_data, Z: func2_data})
field.update_comp_deriv_params(left_bc=BCType.OPEN, fd_type=TEST_FD)

for order in orders:
    field.update_comp_deriv_params(order=order)

    ## gradient
    grad0_an = [dxfunc0_data, dyfunc0_data, dzfunc0_data]
    grad1_an = [dxfunc1_data, dyfunc1_data, dzfunc1_data]
    grad2_an = [dxfunc2_data, dyfunc2_data, dzfunc2_data]

    grad0_tn_ = field.gradient(X)
    grad1_tn_ = field.gradient(Y)
    grad2_tn_ = field.gradient(Z)
    grad0_tn = [grad0_tn_[i].get_data() for i in [X, Y, Z]]
    grad1_tn = [grad1_tn_[i].get_data() for i in [X, Y, Z]]
    grad2_tn = [grad2_tn_[i].get_data() for i in [X, Y, Z]]

    grad_op = findiff.Gradient(h=[dx,dy,dz],acc=2*order)
    grad0_fd = grad_op(func0_data)
    grad1_fd = grad_op(func1_data)
    grad2_fd = grad_op(func2_data)

    err0 = np.average( [np.linalg.norm( grad0_an[i]-grad0_tn[i] )/npts**(K/2) for i in range(3)] )
    err1 = np.average( [np.linalg.norm( grad1_an[i]-grad1_tn[i] )/npts**(K/2) for i in range(3)] )
    err2 = np.average( [np.linalg.norm( grad2_an[i]-grad2_tn[i] )/npts**(K/2) for i in range(3)] )
    grad_err += [[np.log10(err0),np.log10(err1),np.log10(err2)]]

    err0 = np.average( [np.linalg.norm( grad0_an[i]-grad0_fd[i] )/npts**(K/2) for i in range(3)] )
    err1 = np.average( [np.linalg.norm( grad1_an[i]-grad1_fd[i] )/npts**(K/2) for i in range(3)] )
    err2 = np.average( [np.linalg.norm( grad2_an[i]-grad2_fd[i] )/npts**(K/2) for i in range(3)] )
    fd_grad_err += [[np.log10(err0),np.log10(err1),np.log10(err2)]]

    ## laplacian
    lap0_an = dxxfunc0_data + dyyfunc0_data + dzzfunc0_data
    lap1_an = dxxfunc1_data + dyyfunc1_data + dzzfunc1_data
    lap2_an = dxxfunc2_data + dyyfunc2_data + dzzfunc2_data

    lap_tn = field.laplacian(coord_sys)
    lap0_tn = lap_tn.get_comp_data(X)
    lap1_tn = lap_tn.get_comp_data(Y)
    lap2_tn = lap_tn.get_comp_data(Z)
    # lap0_tn_ = field.laplacian(coord_sys, out_comps=[X])
    # lap1_tn_ = field.laplacian(coord_sys, out_comps=[Y])
    # lap2_tn_ = field.laplacian(coord_sys, out_comps=[Z])
    # lap0_tn = lap0_tn_.get_comp_data(X)
    # lap1_tn = lap1_tn_.get_comp_data(Y)
    # lap2_tn = lap2_tn_.get_comp_data(Z)

    lap_op = findiff.Laplacian(h=[dx,dy,dz],acc=2*order)
    lap0_fd = lap_op(func0_data)
    lap1_fd = lap_op(func1_data)
    lap2_fd = lap_op(func2_data)

    err0 = np.log10( np.linalg.norm( lap0_tn - lap0_an )/npts**(K/2) )
    err1 = np.log10( np.linalg.norm( lap1_tn - lap1_an )/npts**(K/2) )
    err2 = np.log10( np.linalg.norm( lap2_tn - lap2_an )/npts**(K/2) )
    lapl_err += [[err0,err1,err2]]

    err0 = np.log10( np.linalg.norm( lap0_fd - lap0_an )/npts**(K/2) )
    err1 = np.log10( np.linalg.norm( lap1_fd - lap1_an )/npts**(K/2) )
    err2 = np.log10( np.linalg.norm( lap2_fd - lap2_an )/npts**(K/2) )
    fd_lapl_err += [[err0,err1,err2]]

    ## curl
    curl_an = [dyfunc2_data-dzfunc1_data, dzfunc0_data-dxfunc2_data, dxfunc1_data-dyfunc0_data]

    curl_tn_ = field.curl(cart_cs)
    curl_tn  = np.array([curl_tn_[i].get_data() for i in [X, Y, Z]])

    curl_op = findiff.Curl(h=[dx,dy,dz],acc=2*order)
    curl_fd = curl_op(np.array([func0_data, func1_data, func2_data]))

    curl_err += [np.log10(np.linalg.norm(curl_tn-curl_an)/npts**(K/2))]
    fd_curl_err += [np.log10(np.linalg.norm(curl_fd-curl_an)/npts**(K/2))]

    ## divergence
    div_an = dxfunc0_data + dyfunc1_data + dzfunc2_data

    div_tn_ = field.divergence(cart_cs)
    div_tn  = div_tn_.get_comp_data()

    div_op = findiff.Divergence(h=[dx,dy,dz],acc=2*order)
    div_fd = div_op(np.array([func0_data, func1_data, func2_data]))

    div_err += [np.log10(np.linalg.norm(div_tn-div_an)/npts**(K/2))]
    fd_div_err += [np.log10(np.linalg.norm(div_fd-div_an)/npts**(K/2))]


fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()
fig3, ax3 = plt.subplots()
fig4, ax4 = plt.subplots()

## grad
grad_err = np.array(grad_err)
fd_grad_err = np.array(fd_grad_err)
ax1.plot(orders,np.array(grad_err[:,0]),'bx',label='tn-0')
ax1.plot(orders,np.array(grad_err[:,1]),'bo',label='tn-1')
ax1.plot(orders,np.array(grad_err[:,2]),'bs',label='tn-2')
ax1.plot(orders,np.array(fd_grad_err[:,0]),'rx',label='fd-0')
ax1.plot(orders,np.array(fd_grad_err[:,1]),'ro',label='fd-1')
ax1.plot(orders,np.array(fd_grad_err[:,2]),'rs',label='fd-2')
ax1.set_title('gradient')

## laplacian
lapl_err = np.array(lapl_err)
fd_lapl_err = np.array(fd_lapl_err)
ax2.plot(orders,np.array(lapl_err[:,0]),'bx',label='tn-0')
ax2.plot(orders,np.array(lapl_err[:,1]),'bo',label='tn-1')
ax2.plot(orders,np.array(lapl_err[:,2]),'bs',label='tn-2')
ax2.plot(orders,np.array(fd_lapl_err[:,0]),'rx',label='fd-0')
ax2.plot(orders,np.array(fd_lapl_err[:,1]),'ro',label='fd-1')
ax2.plot(orders,np.array(fd_lapl_err[:,2]),'rs',label='fd-2')
ax2.set_title('laplacian')

## curl
ax3.plot(orders,curl_err,'bx',label='tn')
ax3.plot(orders,fd_curl_err,'rx',label='fd')
ax3.set_title('curl')

## curl
ax4.plot(orders,div_err,'bx',label='tn')
ax4.plot(orders,fd_div_err,'rx',label='fd')
ax4.set_title('divergence')

print('grad', grad_err)
print('lapl', lapl_err)
print('curl', curl_err)
print('div', div_err)

## plot stuff
ax1.set_xlabel('order')
ax1.set_ylabel('log10(error)')
ax2.set_xlabel('order')
ax2.set_ylabel('log10(error)')
ax3.set_xlabel('order')
ax3.set_ylabel('log10(error)')
ax4.set_xlabel('order')
ax4.set_ylabel('log10(error)')
fig1.legend()
fig2.legend()
fig3.legend()
fig4.legend()
plt.show()

