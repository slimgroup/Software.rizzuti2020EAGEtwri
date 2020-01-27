################################################################################
#
# TWRIdual functional/gradient computation routines (python implementation using devito)
#
################################################################################



### Module loading


import numpy as np
import numpy.linalg as npla
from scipy.fftpack import fft, ifft
from sympy import Abs, Pow, sqrt
from PySource import PointSource, Receiver
from devito import Eq, Inc, solve, Function, TimeFunction, Dimension, Operator, clear_cache



### Objective functional


def objTWRIdual_devito(
    model, y,
    src_coords, rcv_coords,
    wav,
    dat, Filter,
    eps,
    mode = "eval",
    objfact = np.float32(1),
    comp_alpha = True, grad_corr = False, weight_fun_pars = None, dt = None, space_order = 8):
    "Evaluate TWRI objective functional/gradients for current (m, y)"

    clear_cache()

    # Setting time sampling
    if dt is None:
        dt = model.critical_dt

    # Computing y in reduced mode (= residual) if not provided
    u0 = None
    y_was_None = y is None
    if y_was_None:
        u0rcv, u0 = forward(model, src_coords, rcv_coords, wav, dt = dt, space_order = space_order, save = (mode == "grad") and grad_corr)
        y = applyfilt(dat-u0rcv, Filter)
        PTy = applyfilt_transp(y, Filter)

    # Normalization constants
    nx = np.float32(model.m.size)
    nt, nr = np.float32(y.shape)
    etaf = npla.norm(wav.reshape(-1))/np.sqrt(nt*nx)
    etad = npla.norm(applyfilt(dat, Filter).reshape(-1))/np.sqrt(nt*nr)

    # Compute wavefield vy = adjoint(F(m))*Py
    norm_vPTy2, vPTy_src, vPTy = adjoint_y(model, PTy, src_coords, rcv_coords, weight_fun_pars = weight_fun_pars, dt = dt, space_order = space_order, save = (mode == "grad"))

    # <PTy, d-F(m)*f> = <PTy, d>-<adjoint(F(m))*PTy, f>
    PTy_dot_r = np.dot(PTy.reshape(-1), dat.reshape(-1))-np.dot(vPTy_src.reshape(-1), wav.reshape(-1))

    # ||y||
    norm_y = npla.norm(y.reshape(-1))

    # Optimal alpha
    c1 = etaf**np.float32(2)/(np.float32(4)*etad**np.float32(2)*nx*nt)
    c2 = np.float32(1)/(etad*nr*nt)
    c3 = eps/np.sqrt(nr*nt)
    alpha = compute_optalpha(c1*norm_vPTy2, c2*PTy_dot_r, c3*norm_y, comp_alpha = comp_alpha)

    # Lagrangian evaluation
    fun = -alpha**np.float32(2)*c1*norm_vPTy2+alpha*c2*PTy_dot_r-np.abs(alpha)*c3*norm_y

    # Gradient computation
    if mode == "grad":

        # Physical parameters
        m, rho, damp = model.m, model.rho, model.damp

        # Create the forward wavefield
        u = TimeFunction(name = "u", grid = model.grid, time_order = 2, space_order = space_order)

        # Set up PDE and rearrange
        ulaplace, rho = laplacian(u, rho)
        if weight_fun_pars is None:
            stencil = damp*(2.0*u-damp*u.backward+dt**2*rho/m*(ulaplace+2.0*c1/c2*alpha*vPTy))
        else:
            weight = weight_fun(weight_fun_pars, model, src_coords)
            stencil = damp*(2.0*u-damp*u.backward+dt**2*rho/m*(ulaplace+2.0*c1/c2*alpha*vPTy/weight**2))
        expression = [Eq(u.forward, stencil)]

        # Setup source with wavelet
        nt = wav.shape[0]
        src = PointSource(name = "src", grid = model.grid, ntime = nt, coordinates = src_coords)
        src.data[:] = wav[:]
        src_term = src.inject(field = u.forward, expr = src*rho*dt**2/m) #######
        expression += src_term

        # Setup data sampling at receiver locations
        rcv = Receiver(name = "rcv", grid = model.grid, ntime = nt, coordinates = rcv_coords)
        rcv_term = rcv.interpolate(expr = u)
        expression += rcv_term

        # Setup gradient wrt m
        gradm = Function(name = "gradm", grid = model.grid)
        expression += [Inc(gradm, alpha*c2*vPTy*u.dt2)]

        # Create operator and run
        subs = model.spacing_map
        subs[u.grid.time_dim.spacing] = dt
        op = Operator(expression, subs = subs, dse = "advanced", dle = "advanced", name = "Grad")
        op()

        # Compute gradient wrt y
        if not y_was_None or grad_corr:
            norm_y = npla.norm(y)
            if norm_y == 0:
                grady_data = alpha*c2*applyfilt(dat-rcv.data, Filter)
            else:
                grady_data = alpha*c2*applyfilt(dat-rcv.data, Filter)-np.abs(alpha)*c3*y/norm_y

        # Correcting for reduced gradient
        if not y_was_None or (y_was_None and not grad_corr):

            gradm_data = gradm.data

        else:

            # Compute wavefield vy_ = adjoint(F(m))*grady
            _, _, vy_ = adjoint_y(model, applyfilt_transp(grady_data, Filter), src_coords, rcv_coords, dt = dt, space_order = space_order, save = True)

            # Setup reduced gradient wrt m
            gradm_corr = Function(name = "gradmcorr", grid = model.grid)
            expression = [Inc(gradm_corr, vy_*u0.dt2)]

            # Create operator and run
            subs = model.spacing_map
            subs[u.grid.time_dim.spacing] = dt
            op = Operator(expression, subs = subs, dse = "advanced", dle = "advanced", name = "GradRed")
            op()

            # Reduced gradient post-processing
            gradm_data = gradm.data+gradm_corr.data

    # Return output
    if mode == "eval":
        return fun/objfact
    elif mode == "grad" and y_was_None:
        return fun/objfact, gradm_data/objfact
    elif mode == "grad" and not y_was_None:
        return fun/objfact, gradm_data/objfact, grady_data/objfact


def compute_optalpha(v1, v2, v3, comp_alpha = True):

    if comp_alpha:
        if v3 < np.abs(v2):
            a = np.sign(v2)*(np.abs(v2)-v3)/(np.float32(2)*v1)
            if np.isinf(a) or np.isnan(a):
                return np.float32(0)
            else:
                return a
        else:
            return np.float32(0)
    else:
        return np.float32(1)



### Forward/Adjoint wavefield propagation


def forward(
    model,
    src_coords, rcv_coords,
    wav,
    dt = None, space_order = 8, save = False):
    "Compute forward wavefield u = A(m)^{-1}*f and related quantities (u(xrcv))"

    clear_cache()

    # Setting time sampling
    if dt is None:
        dt = model.critical_dt

    # Physical parameters
    m, rho, damp = model.m, model.rho, model.damp

    # Setting adjoint wavefield
    nt = wav.shape[0]
    u = TimeFunction(name = "u", grid = model.grid, time_order = 2, space_order = space_order, save = None if not save else nt)

    # Set up PDE expression and rearrange
    ulaplace, rho = laplacian(u, rho)
    stencil = damp*(2.0*u-damp*u.backward+dt**2*rho/m*ulaplace)
    expression = [Eq(u.forward, stencil)]

    # Setup adjoint source injected at receiver locations
    src = PointSource(name = "src", grid = model.grid, ntime = nt, coordinates = src_coords)
    src.data[:] = wav[:]
    src_term = src.inject(field = u.forward, expr = src*rho*dt**2/m)
    expression += src_term

    # Setup adjoint wavefield sampling at source locations
    rcv = Receiver(name = "rcv", grid = model.grid, ntime = nt, coordinates = rcv_coords)
    adj_rcv = rcv.interpolate(expr = u)
    expression += adj_rcv

    # Create operator and run
    subs = model.spacing_map
    subs[u.grid.time_dim.spacing] = dt
    op = Operator(expression, subs = subs, dse = "advanced", dle = "advanced", name = "forward")
    op()

    # Output
    if save:
        return rcv.data, u
    else:
        return rcv.data, None


def adjoint_y(
    model, y,
    src_coords, rcv_coords,
    weight_fun_pars = None, dt = None, space_order = 8, save = False):
    "Compute adjoint wavefield v = adjoint(F(m))*y and related quantities (||v||_w, v(xsrc))"

    clear_cache()

    # Setting time sampling
    if dt is None:
        dt = model.critical_dt

    # Physical parameters
    m, rho, damp = model.m, model.rho, model.damp

    # Setting adjoint wavefield
    nt = y.shape[0]
    v = TimeFunction(name = "v", grid = model.grid, time_order = 2, space_order = space_order, save = None if not save else nt)

    # Set up PDE expression and rearrange
    vlaplace, rho = laplacian(v, rho)
    stencil = damp*(2.0*v-damp*v.forward+dt**2*rho/m*vlaplace)
    expression = [Eq(v.backward, stencil)]

    # Setup adjoint source injected at receiver locations
    rcv = Receiver(name = "rcv", grid = model.grid, ntime = nt, coordinates = rcv_coords)
    rcv.data[:] = y[:]
    adj_src = rcv.inject(field = v.backward, expr = rcv*rho*dt**2/m)
    expression += adj_src

    # Setup adjoint wavefield sampling at source locations
    src = PointSource(name = "src", grid = model.grid, ntime = nt, coordinates = src_coords)
    adj_rcv = src.interpolate(expr = v)
    expression += adj_rcv

    # Setup ||v||_w computation
    norm_vy2_t = Function(name = "nvy2t", grid = model.grid)
    expression += [Inc(norm_vy2_t, Pow(v, 2))]
    i = Dimension(name = "i", )
    norm_vy2 = Function(name = "nvy2", shape = (1, ), dimensions = (i, ), grid = model.grid)
    if weight_fun_pars is None:
        expression += [Inc(norm_vy2[0], norm_vy2_t)]
    else:
        weight = weight_fun(weight_fun_pars, model, src_coords)
        expression += [Inc(norm_vy2[0], norm_vy2_t/weight**2)]

    # Create operator and run
    subs = model.spacing_map
    subs[v.grid.time_dim.spacing] = dt
    op = Operator(expression, subs = subs, dse = "advanced", dle = "advanced", name = "adjoint_y")
    op()

    # Output
    if save:
        return norm_vy2.data[0], src.data, v
    else:
        return norm_vy2.data[0], src.data, None


def laplacian(v, rho):

    if rho is None:
        Lap = v.laplace
        rho = 1
    else:
        if isinstance(rho, Function):
            Lap = sum([first_derivative(first_derivative(v, fd_order=int(v.space_order/2), side=left, dim=d) / rho,
                       fd_order=int(v.space_order/2), dim=d, side=right) for d in v.space_dimensions])
        else:
            Lap = 1 / rho * v.laplace

    return Lap, rho



### Weighted norm symbolic functions


def weight_fun(weight_fun_pars, model, src_coords):

    if weight_fun_pars[0] == "srcfocus":
        return weight_srcfocus(model, src_coords, delta = weight_fun_pars[1])
    elif weight_fun_pars[0] == "depth":
        return weight_depth(model, src_coords, delta = weight_fun_pars[1])


def weight_srcfocus(model, src_coords, delta = 0.01):
    "w(x) = sqrt((||x-xsrc||^2+delta^2)/delta^2)"

    ix, iz = model.grid.dimensions
    isrc = (model.nbpml+src_coords[0, 0]/model.spacing[0], model.nbpml+src_coords[0, 1]/model.spacing[1])
    h = np.sqrt(model.spacing[0]*model.spacing[1])
    return sqrt((ix-isrc[0])**2+(iz-isrc[1])**2+(delta/h)**2)/(delta/h)


def weight_depth(model, src_coords, delta = 0.01):
    "w(x) = sqrt((||z-zsrc||^2+delta^2)/delta^2)"

    _, iz = model.grid.dimensions
    isrc = (model.nbpml+src_coords[0, 0]/model.spacing[0], model.nbpml+src_coords[0, 1]/model.spacing[1])
    h = np.sqrt(model.spacing[0]*model.spacing[1])
    return sqrt((iz-isrc[1])**2+(delta/h)**2)/(delta/h)



### Data filtering


def applyfilt(dat, Filter = None):

    if Filter is None:
        return dat
    else:
        padding = max(dat.shape[0], Filter.size)
        return np.real(ifft(fft(dat, n = padding, axis = 0)*Filter.reshape(-1, 1), axis = 0)[:dat.shape[0], :])


def applyfilt_transp(dat, Filter = None):

    if Filter is None:
        return dat
    else:
        padding = max(dat.shape[0], Filter.size)
        return np.real(ifft(fft(dat, n = padding, axis = 0)*np.conj(Filter).reshape(-1, 1), axis = 0)[:dat.shape[0], :])
