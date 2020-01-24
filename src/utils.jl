################################################################################
#
# Set of utilities for TWRIdual
#
################################################################################



export judiVector2vec, vec2judiVector, model2vec, vec2model, contr2abs, abs2contr, gradprec_contr2abs, devito_model, gradient_desc



### JUDI data types/array format translations


function judiVector2vec(y::judiVector)
	return vec(cat(y.data..., dims = 2))
end


function vec2judiVector(rcv_geom::GeometryIC, y_vec::Array)
	nsrc = length(rcv_geom.xloc)
	y_arr = Array{Array}(undef, nsrc)
	for i = 1:nsrc
		nrcv = length(rcv_geom.xloc[i])
		nt = rcv_geom.nt[i]
		y_arr[i] = reshape(y_vec[nrcv*nt*(i-1)+1:nrcv*nt*i], nt, nrcv)
	end
	return judiVector(rcv_geom, y_arr)
end


function model2vec(m)
	return vec(m.m)
end


function vec2model(n, d, o, m)
	return Model(n, d, o, reshape(m, n))
end



### Utilities for model input preprocessing and gradient postprocessing


function contr2abs(xvec::Array{R, 1}, mask::BitArray{2}, mb::Array{R, 2})
# Preprocessing steps: - effective to global domain
#                      - contrast to absolute properties

    x = zeros(R, size(mb))
    x[mask] .= xvec
    return mb.*(R(1).+x)

end
function abs2contr(m::Array{R, 2}, mask::BitArray{2}, mb::Array{R, 2})

	return ((m.-mb)./mb)[mask]

end


function gradprec_contr2abs(grad::Array{R, 2}, mask::BitArray{2}, mb::Array{R, 2})
# Postprocessing steps: - preconditioning by diagonal background model

    return mb[mask].*grad[mask]

end
function gradprec_contr2abs(grad::Array{R, 2}, mask::BitArray{2}, mb::Model)

    return gradprec_contr2abs(grad, mask, mb.m)

end



### Extra devito utilities


function devito_model(model::Model, op::String, mode::Int64, opt, dm)

	pm = load_pymodel()
	length(model.n) == 3 ? dims = [3, 2, 1] : dims = [2, 1] # model dimensions for Python are (z,y,x) and (z,x)

	# Set up Python model structure
	if op == 'J' && mode == 1
		modelPy = pm.Model(origin = model.o, spacing = model.d, shape = model.n,
		vp = process_physical_parameter(sqrt.(1f0./model.m), dims), nbpml = model.nb, rho = process_physical_parameter(model.rho, dims),
		dm = process_physical_parameter(reshape(dm, model.n), dims), space_order = SPACE_ORDER)
	else
		modelPy = pm.Model(origin = model.o, spacing = model.d, shape = model.n, vp = process_physical_parameter(sqrt.(1f0./model.m), dims), nbpml = model.nb,
		rho = process_physical_parameter(model.rho, dims), space_order = SPACE_ORDER)
	end

end



### Optimization routines


function gradient_desc(fun!, x, fun_proj, niter; alpha = 1f0, nFunEval_max = 1000, tolfun = 1f-10, tolg = 1f-10)

	x = fun_proj(x)
	f = 0f0
	g = zeros(R, size(x))
	nFunEval = 0
	t0 = time()
	for i = 1:niter
		f = fun!(true, g, x); nFunEval += 1
		normg = norm(g)
		println(string(i), ": f = ", string(f), " --- g = ", string(normg), ", alpha = ", string(alpha), ", nFunEval = ", string(nFunEval))
		println("* t = ", string(Int(floor(time()-t0))), " s")
		if f < tolfun || normg < tolg
			return x, f
		end
		f_ = f
		while f_ >= f
			if nFunEval > nFunEval_max
				return x, f
			end
			f_ = fun!(true, nothing, fun_proj(x.-alpha*g)); nFunEval += 1
			if f_ >= f
				println(string("      f = ", string(f_), ", halving step length..."))
				alpha /= 2
			end
		end
		x = fun_proj(x.-alpha*g)
		f = f_
		if f < tolfun
			return x, f
		end
	end
	return x, f

end
