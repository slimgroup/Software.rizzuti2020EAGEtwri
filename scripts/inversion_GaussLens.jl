################################################################################
#
# Run inversion for the Gaussian lens model
#
################################################################################



### Module loading


using LinearAlgebra, JLD2, PyPlot
using Optim
using JUDI.TimeModeling
push!(LOAD_PATH, string(pwd(), "/src/"))
using TWRIdual



### Load synthetic data


@load "./data/GaussLens/GaussLens_data.jld"



### Background model


m0 = 1f0./2f0.^2*ones(R, model_true.n)
n = model_true.n
d = model_true.d
o = model_true.o
model0 = Model(n, d, o, m0)



### Set objective functional


# Time sampling
dt_comp = dat.geometry.dt


# Pre- and post-conditioning
mask = BitArray(undef, model0.n); mask .= false
mask[6:end-5, 6:end-5] .= true
function proj_bounds(m, mmin, mmax)
    m[m[:] .< mmin] .= mmin
    m[m[:] .> mmax] .= mmax
    return m
end
mmin = 1f0/4f0^2
mmax = Inf
preproc(x) = proj_bounds(contr2abs(x, mask, m0), mmin, mmax)
postproc(g) = gradprec_contr2abs(g, mask, m0)


# # [FWI]
# @show inv_name = "FWI"
# fun!(F, G, x) = objFWI!(F, G, preproc(x), n, d, o, fsrc, dat; dt_comp = dt_comp, gradmprec_fun = postproc)


# [TWRIdual]
@show inv_name = "TWRIdual"
ε = Array{Float32, 1}(undef, fsrc.nsrc); ε .= 0.01f0
v_bg = sqrt(1/m0[1])
freq_peak = 0.006f0
δ = 0.1f0*R(sqrt(2)/2)*v_bg/freq_peak
weight_fun_pars = ("srcfocus", δ)
grad_corr = true
fun!(F, G, x) = objTWRIdual!(F, G, preproc(x), n, d, o, fsrc, dat, ε; comp_alpha = true, grad_corr = grad_corr, weight_fun_pars = weight_fun_pars, dt_comp = dt_comp, gradmprec_fun = postproc)



### Optimization


# Starting guess
x0 = zeros(R, length(findall(mask .== true)))

# Options
method = LBFGS()
niter = 20
optimopt = Optim.Options(iterations = niter, store_trace = true, show_trace = true, show_every = 1)


# Run
result = optimize(Optim.only_fg!(fun!), x0, method, optimopt)
x0 .= Optim.minimizer(result)
m_inv = preproc(x0)
loss_log = Optim.f_trace(result)
