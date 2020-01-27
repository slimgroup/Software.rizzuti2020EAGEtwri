################################################################################
#
# TWRI module
#
################################################################################



module TWRIdual

using LinearAlgebra
using JUDI.TimeModeling
using PyCall
using Distributed
using FFTW, DSP
using Optim

const R = Float32; export R
const C = ComplexF32; export C
const SPACE_ORDER = 8
const NB = 100

include("utils.jl")
include("gendata.jl")
include("data_filter.jl")
include("FWIFun.jl")
include("TWRIdualFun.jl")

end
