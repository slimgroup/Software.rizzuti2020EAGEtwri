################################################################################
#
# Routines for the the evaluation and gradient computation of TWRI
#
################################################################################



export objTWRIdual, objTWRIdual!



### Optim.jl-adapted wrapper


function objTWRIdual!(
	F, Gm, m::Array{R, 2},
	n::NTuple{2, Int64}, d::NTuple{2, R}, o::NTuple{2, R},
	wav::judiVector, dat::judiVector,
	eps::Array{R, 1};
	opt = Options(),
	comp_alpha = true, grad_corr = false, weight_fun_pars = nothing, dt_comp::Union{Nothing, Array{Any, 1}} = nothing,
	gradmprec_fun = g->g)

	if Gm == nothing
		mode = "eval"
	else
		mode = "grad"
	end
	argout = objTWRIdual(m, n, d, o, nothing, wav, dat, eps; opt = opt, mode = mode, comp_alpha = comp_alpha, grad_corr = grad_corr, weight_fun_pars = weight_fun_pars, dt_comp = dt_comp, gradmprec_fun = gradmprec_fun)
	if Gm != nothing
		fval = argout[1]
		Gm .= argout[2]
	else
		fval = argout
	end
	return fval

end


function objTWRIdual!(
	F, Gm, m::Array{R, 2},
	y::judiVector,
	n::NTuple{2, Int64}, d::NTuple{2, R}, o::NTuple{2, R},
	wav::judiVector, dat::judiVector,
	eps::Array{R, 1};
	opt = Options(),
	comp_alpha = true, weight_fun_pars = nothing, dt_comp::Union{Nothing, Array{Any, 1}} = nothing,
	gradmprec_fun = g->g)

	if Gm == nothing
		mode = "eval"
	else
		mode = "grad"
	end
	argout = objTWRIdual(m, n, d, o, y, wav, dat, eps; opt = opt, mode = mode, comp_alpha = comp_alpha, grad_corr = false, weight_fun_pars = weight_fun_pars, dt_comp = dt_comp, gradmprec_fun = gradmprec_fun)
	if Gm != nothing
		fval = argout[1]
		Gm .= argout[2]
	else
		fval = argout
	end
	return fval

end



### Objective wrapper for array input


function objTWRIdual(
	m::Array{R, 2}, n::NTuple{2, Int64}, d::NTuple{2, R}, o::NTuple{2, R},
	wav::judiVector, dat::judiVector,
	eps::Array{R, 1};
	opt = Options(),
	mode = "eval",
	comp_alpha = true, grad_corr = false, weight_fun_pars = nothing, dt_comp::Union{Nothing, Array{Any, 1}} = nothing,
	gradmprec_fun = g->g)

	return objTWRIdual(m, n, d, o, nothing, wav, dat, eps; opt = opt, mode = mode, comp_alpha = comp_alpha, grad_corr = grad_corr, weight_fun_pars = weight_fun_pars, dt_comp = dt_comp, gradmprec_fun = gradmprec_fun)

end


function objTWRIdual(
	m::Array{R, 2}, n::NTuple{2, Int64}, d::NTuple{2, R}, o::NTuple{2, R}, y::Union{Nothing, judiVector},
	wav::judiVector, dat::judiVector,
	eps::Array{R, 1};
	opt = Options(),
	mode = "eval",
	comp_alpha = true, grad_corr = false, weight_fun_pars = nothing, dt_comp::Union{Nothing, Array{Any, 1}} = nothing,
	gradmprec_fun = g->g)

	return objTWRIdual(Model(n, d, o, m), y, wav, dat, eps; opt = opt, mode = mode, comp_alpha = comp_alpha, grad_corr = grad_corr, weight_fun_pars = weight_fun_pars, dt_comp = dt_comp, gradmprec_fun = gradmprec_fun)

end



### Objective wrapper for JUDI data types


function objTWRIdual(
	m::Model, y::Union{Nothing, judiVector},
	wav::judiVector, dat::judiVector,
	eps::Array{R, 1};
	opt = Options(),
	mode = "eval",
	comp_alpha = true, grad_corr = false, weight_fun_pars = nothing, dt_comp::Union{Nothing, Array{Any, 1}} = nothing,
	gradmprec_fun = g->g)

	if y == nothing
		return objTWRIdual_rawinput(m, nothing, wav.geometry, dat.geometry, wav.data, dat.data, eps, 1:length(dat.data); opt = opt, mode = mode, comp_alpha = comp_alpha, grad_corr = grad_corr, weight_fun_pars = weight_fun_pars, dt_comp = dt_comp, gradmprec_fun = gradmprec_fun)
	else
		return objTWRIdual_rawinput(m, y.data, wav.geometry, dat.geometry, wav.data, dat.data, eps, 1:length(dat.data); opt = opt, mode = mode, comp_alpha = comp_alpha, grad_corr = grad_corr, weight_fun_pars = weight_fun_pars, dt_comp = dt_comp, gradmprec_fun = gradmprec_fun)
	end

end


function objTWRIdual(
	m::Model,
	wav::judiVector, dat::judiVector,
	eps::Array{R, 1};
	opt = Options(),
	mode = "eval",
	comp_alpha = true, grad_corr = false, weight_fun_pars = nothing, dt_comp::Union{Nothing, Array{Any, 1}} = nothing,
	gradmprec_fun = g->g)

	return objTWRIdual(m, nothing, wav, dat, eps; opt = opt, mode = mode, comp_alpha = comp_alpha, grad_corr = grad_corr, weight_fun_pars = weight_fun_pars, dt_comp = dt_comp, gradmprec_fun = gradmprec_fun)

end



### Objective: serial/parallel implementation


## Parallel implementation

function objTWRIdual_rawinput(
	model::Model, y_data,
	src_geom::Geometry, rcv_geom::Geometry,
	src_data, dat_data,
	eps::Array{R, 1},
	src_idx::UnitRange{Int64};
	opt = Options(),
	mode = "eval",
	comp_alpha = true, grad_corr = false, weight_fun_pars = nothing, dt_comp::Union{Nothing, Array{Any, 1}} = nothing,
	gradmprec_fun = g->g)

	# Initialize output
	nsrc = length(src_idx)
	results = Array{Any}(undef, nsrc)

	# Process shots from source channel
	for j = 1:nsrc

		# Local geometry for current position
		opt_loc = subsample(opt, j)
		src_geom_loc = subsample(src_geom, j)
		rcv_geom_loc = subsample(rcv_geom, j)

		# Selecting variables for current shot index
		src_data_loc = src_data[j]
		if y_data == nothing
			y_data_loc = nothing
		else
			y_data_loc = y_data[j]
		end
		dat_data_loc = dat_data[j]
		eps_loc = eps[j]
		if dt_comp == nothing
			dt_comp_loc = nothing
		else
			dt_comp_loc = dt_comp[j]
		end

		# Local result
		results[j] = objTWRIdual_rawinput(model, y_data_loc, src_geom_loc, rcv_geom_loc, src_data_loc, dat_data_loc, eps_loc, opt = opt_loc; mode = mode, comp_alpha = comp_alpha, grad_corr = grad_corr, weight_fun_pars = weight_fun_pars, dt_comp = dt_comp_loc, gradmprec_fun = gradmprec_fun)

	end

	# Aggregating results
	if mode == "eval"
		argout = fetch(results[1])
		for j = 2:nsrc
			argout += fetch(results[j])
		end
	elseif mode == "grad" && y_data == nothing
		arg = fetch(results[1])
		argout1 = arg[1]
		argout2 = arg[2]
		for j = 2:nsrc
			arg = fetch(results[j])
			argout1 += arg[1]
			argout2 += arg[2]
		end
		argout = (argout1, argout2)
	else
		arg = fetch(results[1])
		argout1 = arg[1]
		argout2 = arg[2]
		argout3 = arg[3]
		for j = 2:nsrc
			arg = fetch(results[j])
			argout1 += arg[1]
			argout2 += arg[2]
			argout3 = [argout3; arg[3]]
		end
		argout = (argout1, argout2, argout3)
	end
	return argout./nsrc

end


## Serial implementation

function objTWRIdual_rawinput(
	model_full::Model, y_data::Union{Nothing, Array{R, 2}},
	src_geom::Geometry, rcv_geom::Geometry,
	src_data::Array{R, 2}, dat_data::Array{R, 2},
	eps::R;
	opt = Options(),
	mode = "eval",
	comp_alpha = true, grad_corr = false, weight_fun_pars = nothing, dt_comp::Union{Nothing, R} = nothing,
	gradmprec_fun = g->g)

	# Setting pre-defined absorbing layer size
	model_full.nb = NB

	# Load full geometry for out-of-core geometry containers
	typeof(rcv_geom) == GeometryOOC && (rcv_geom = Geometry(rcv_geom))
	typeof(src_geom) == GeometryOOC && (src_geom = Geometry(src_geom))
	length(model_full.n) == 3 ? dims = [3, 2, 1] : dims = [2, 1] # model dimensions for Python are (z,y,x) and (z,x)

	# Limit model to area with sources/receivers
	if opt.limit_m == true
		model = deepcopy(model_full)
		model = limit_model_to_receiver_area(src_geom, rcv_geom, model, opt.buffer_size)
	else
		model = model_full
	end

	# Set up Python model structure
	model_py = devito_model(model, "F", 1, opt, 0)

	# Remove receivers outside the modeling domain (otherwise leads to segmentation faults)
	rcv_geom = remove_out_of_bounds_receivers(rcv_geom, model)

	# Call to objective with Julia/Devito interface function
	return objTWRIdual_jldevito(model_py, y_data, model.o, src_geom, rcv_geom, src_data, dat_data, eps; opt = opt, mode = mode, comp_alpha = comp_alpha, grad_corr = grad_corr, weight_fun_pars = weight_fun_pars, dt_comp = dt_comp, gradmprec_fun = gradmprec_fun)

end



### Julia/devito implementation


function objTWRIdual_jldevito(
	model_py::PyCall.PyObject, y_data::Union{Nothing, Array{R, 2}},
	origin,
	src_geom::Geometry, rcv_geom::Geometry,
	src_data::Array{R, 2}, dat_data::Array{R, 2},
	eps::R;
	opt = Options(),
	mode = "eval",
	comp_alpha = true, grad_corr = false, weight_fun_pars = nothing, dt_comp::Union{Nothing, R} = nothing,
	gradmprec_fun = g->g)

	# Loading python modules for devito implementation of objTWRIdual
	devitomod = load_codegen_TWRIdual()

	# Interpolate input data to computational grid
	if dt_comp == nothing
		dt_comp = R(model_py.critical_dt)
	end
	q_in = time_resample(src_data, src_geom, dt_comp)[1]
	if y_data != nothing
		y_in = time_resample(y_data, rcv_geom, dt_comp)[1]
	else
		y_in = nothing
	end
	dat_in = time_resample(dat_data, rcv_geom, dt_comp)[1]
	nt_comp = size(q_in, 1)
	nt_rcv = Int(trunc(rcv_geom.t[1]/dt_comp+1))

	# Set up coordinates with devito dimensions
	src_coords = setup_grid(src_geom, model_py.shape, origin)
	rcv_coords = setup_grid(rcv_geom, model_py.shape, origin)

	# Computing output
	if y_in == nothing
		y_in_ = nothing
	else
		y_in_ = PyReverseDims(copy(transpose(y_in)))
	end
	if mode == "eval"
		argout = pycall(devitomod.objTWRIdual_devito,
						R,
						model_py, y_in_,
						PyReverseDims(copy(transpose(src_coords))), PyReverseDims(copy(transpose(rcv_coords))),
						PyReverseDims(copy(transpose(q_in))), PyReverseDims(copy(transpose(dat_in))),
						eps,
						mode,
						comp_alpha, grad_corr, weight_fun_pars, dt_comp, SPACE_ORDER)
	elseif mode == "grad" && y_in != nothing
		argout = pycall(devitomod.objTWRIdual_devito,
						Tuple{R, Array{R, 2}, Array{R, 2}},
						model_py, y_in_,
						PyReverseDims(copy(transpose(src_coords))), PyReverseDims(copy(transpose(rcv_coords))),
						PyReverseDims(copy(transpose(q_in))), PyReverseDims(copy(transpose(dat_in))),
						eps,
						mode,
						comp_alpha, grad_corr, weight_fun_pars, dt_comp, SPACE_ORDER)
	else
		argout = pycall(devitomod.objTWRIdual_devito,
						Tuple{R, Array{R, 2}},
						model_py, y_in_,
						PyReverseDims(copy(transpose(src_coords))), PyReverseDims(copy(transpose(rcv_coords))),
						PyReverseDims(copy(transpose(q_in))), PyReverseDims(copy(transpose(dat_in))),
						eps,
						mode,
						comp_alpha, grad_corr, weight_fun_pars, dt_comp, SPACE_ORDER)
	end

	# Output post-processing
	if mode == "grad" && y_in != nothing
		gm = remove_padding(argout[2], model_py.nbpml, true_adjoint = opt.sum_padding) # remove AB layers
		gy = time_resample(argout[3], dt_comp, rcv_geom) # resample
		argout = (argout[1], gradmprec_fun(gm), [gy])
	elseif mode == "grad" && y_in == nothing
		gm = remove_padding(argout[2], model_py.nbpml, true_adjoint = opt.sum_padding) # remove AB layers
		argout = (argout[1], gradmprec_fun(gm))
	end

	return argout

end


# Loader for python devito modeling
function load_codegen_TWRIdual()

	pushfirst!(PyVector(pyimport("sys")."path"), string(pwd(), "/src/"))
	return pyimport("TWRIdualFun_devito")

end
