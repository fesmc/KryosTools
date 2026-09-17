#=
Minimal reproduction of a Reactant bug found on 2026-09-17 (Reactant 0.2.285): a raised
KernelAbstractions kernel over a 2D range whose first axis is ≥ 2048 only computes the first
column. Independent of KryosTools.

    julia --project=benchmark/reactant benchmark/reactant/minimal.jl [cpu|gpu] [raise|noraise]
=#

const DEVICE = get(ARGS, 1, "cpu")
const RAISE = get(ARGS, 2, "raise") == "raise"

using Reactant, CUDA, KernelAbstractions

Reactant.set_default_backend(DEVICE)

@kernel function fill_index!(dst)
    i, j = @index(Global, NTuple)
    dst[i, j] = i + 10000 * j
end

function run!(dst)
    fill_index!(get_backend(dst))(dst; ndrange = size(dst))
    return dst
end

println("platform: ", Reactant.XLA.platform_name(Reactant.XLA.default_backend()), ", raise = $RAISE")
for dims in ((2047, 2047), (2048, 2048), (4096, 1024), (1024, 4096), (2048, 2047))
    reference = [Float64(i + 10000 * j) for i in 1:dims[1], j in 1:dims[2]]
    d = Reactant.to_rarray(zeros(dims...))
    f = Reactant.@compile raise = RAISE sync = true run!(d)
    f(d)
    println(dims, ": ", count(Array(d) .!= reference), " wrong of ", prod(dims))
end
