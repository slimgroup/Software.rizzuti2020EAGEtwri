################################################################################
#
# FWI functional/gradient computation routines (python implementation using devito)
#
################################################################################



### Module loading


import numpy as np
import numpy.linalg as npla
from sympy import Abs, Pow, sqrt
from PySource import PointSource, Receiver
from devito import Eq, Inc, solve, Function, TimeFunction, Dimension, Operator, clear_cache
from TWRIdualFun_devito import forward, laplacian, applyfilt, applyfilt_transp



### Objective functional


def objFWI_devito(
    model,
    src_coords, rcv_coords,
    wav,
    dat, Filter = None,
    mode = "eval",
    dt = None, space_order = 8):
    "Evaluate FWI objective functional/gradients for current m"

    clear_cache()

    # Setting time sampling
    if dt is None:
        dt = model.critical_dt

    # Normalization constant
    dat_filt = applyfilt(dat, Filter)
    eta = dt*npla.norm(dat_filt.reshape(-1))**2

    # Computing residual
    dmod, u = forward(model, src_coords, rcv_coords, wav, dt = dt, space_order = space_order, save = (mode == "grad"))
    Pres = applyfilt(dat-dmod, Filter)

    # ||P*r||^2
    norm_Pr2 = dt*npla.norm(Pres.reshape(-1))**2

    # Functional evaluation
    fun = norm_Pr2/eta

    # Gradient computation
    if mode == "grad":

        # Physical parameters
        m, rho, damp = model.m, model.rho, model.damp

        # Create the forward wavefield
        v = TimeFunction(name = "v", grid = model.grid, time_order = 2, space_order = space_order)

        # Set up PDE and rearrange
        vlaplace, rho = laplacian(v, rho)
        stencil = damp*(2.0*v-damp*v.forward+dt**2*rho/m*vlaplace)
        expression = [Eq(v.backward, stencil)]

        # Setup adjoint source with data
        nt = wav.shape[0]
        rcv = Receiver(name = "rcv", grid = model.grid, ntime = nt, coordinates = rcv_coords)
        rcv.data[:] = applyfilt_transp(Pres, Filter)[:]
        adjsrc_term = rcv.inject(field = v.backward, expr = rcv*rho*dt**2/m)
        expression += adjsrc_term

        # Setup gradient wrt m
        gradm = Function(name = "gradm", grid = model.grid)
        expression += [Inc(gradm, 2*dt*v*u.dt2/eta)]

        # Create operator and run
        subs = model.spacing_map
        subs[v.grid.time_dim.spacing] = dt
        op = Operator(expression, subs = subs, dse = "advanced", dle = "advanced", name = "Grad")
        op()

    # Return output
    if mode == "eval":
        return fun
    elif mode == "grad":
        return fun, gradm.data
