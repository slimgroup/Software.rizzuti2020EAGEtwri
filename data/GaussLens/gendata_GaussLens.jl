################################################################################
#
# Generate synthetic data for the Gaussian lens model
#
################################################################################



### Module loading


using LinearAlgebra, JUDI.TimeModeling, JLD2, PyPlot
push!(LOAD_PATH, string(pwd(), "/src/"))
using TWRIdual



### Set true model


n = (201, 201)
d = (10f0, 10f0)
o = (0f0, 0f0)
val = 0.5f0
x = reshape((collect(1:n[1]).-1f0).*d[1], :, 1)
z = reshape((collect(1:n[2]).-1f0).*d[2], 1, :)
v = 2f0.-val*exp.(-((x./1000f0.-1f0)/0.25f0).^2f0.-((z./1000f0.-1f0)/0.5f0).^2f0)
m_true = 1f0./v.^2f0
model_true = Model(n, d, o, m_true)


### Acquisition geometry


# Sources
ix_src = 3:14:199
nsrc = length(ix_src)
iz_src = 3
x_src = convertToCell((ix_src.-1)*d[1])
y_src = convertToCell(range(0f0, stop = 0f0, length = nsrc))
z_src = convertToCell(range((iz_src-1)*d[2], stop = (iz_src-1)*d[2], length = nsrc))

# Source wavelet
dt = 1f0
T = 3000f0 # total recording time [ms]
freq_peak = 0.006f0
wavelet = ricker_wavelet(T, dt, freq_peak)

# Receivers
ix_rcv = 3:199
iz_rcv = 199
nrcv = length(ix_rcv)
x_rcv = (ix_rcv.-1)*d[1]
y_rcv = 0f0
z_rcv = range((iz_rcv-1)*d[2], stop = (iz_rcv-1)*d[2], length = nrcv)

# Geometry structures
src_geom = Geometry(x_src, y_src, z_src; dt = dt, t = T)
rcv_geom = Geometry(x_rcv, y_rcv, z_rcv; dt = dt, t = T, nsrc = nsrc)

# Source function
fsrc = judiVector(src_geom, wavelet)



### Modeling synthetic data


dt_comp = Array{Any, 1}(undef, nsrc); dt_comp .= dt
@time dat = gendata(model_true, fsrc, rcv_geom; opt = Options(), dt_comp = dt_comp)



### Saving data


@save string("./data/GaussLens/GaussLens_data.jld") model_true fsrc dat
