#=
Exploratory comparison: KryosTools regridding natively vs. compiled with Reactant.

    julia --project=benchmark/reactant benchmark/reactant/compare.jl cpu
    julia --project=benchmark/reactant benchmark/reactant/compare.jl gpu

For CPU thread scaling, pin the process so that XLA's thread pool matches Julia's, e.g.
`taskset -c 0-3 julia --project=benchmark/reactant -t 4 benchmark/reactant/compare.jl cpu`.
`KRYOSTOOLS_BENCH_SECONDS` sets the time budget per measurement (default 2).

Not part of the tracked benchmarks. Variants per case:

- native: KryosTools as is (Array on cpu, CuArray on gpu);
- kernel: the same `regrid!` compiled with Reactant, kernels raised to StableHLO
  (`raise = true`), and on gpu also compiled as CUDA kernels inside XLA (`raise = false`);
- array: hand-written reshape/broadcast versions, the style XLA optimises best.

Every Reactant result is checked against the native CPU result and the maximum error is
printed next to the time — never trust a time without it (see minimal.jl).
=#

const DEVICE = get(ARGS, 1, "cpu")
DEVICE in ("cpu", "gpu") || error("Pass `cpu` or `gpu`, got $(repr(DEVICE)).")
const SECONDS = parse(Float64, get(ENV, "KRYOSTOOLS_BENCH_SECONDS", "2"))

using KryosTools, Reactant, CUDA, KernelAbstractions, Chairmarks, Statistics

# ------------------------------------------------------------------------------------------
# Local patches needed for KryosTools kernels to compile under Reactant (2026-09-17,
# Reactant 0.2.285). To be fixed properly: the first in KryosTools, the second upstream.
# ------------------------------------------------------------------------------------------

# Under tracing, eltype(dst) is a traced number type; accumulate in the unwrapped type.
KryosTools.accumtype(::Type{<:Reactant.TracedRNumber{T}}) where {T} = KryosTools.accumtype(T)

# Reactant's read-only kernel arrays (from @Const) only define linear indexing.
const RCE = Base.get_extension(Reactant, :ReactantCUDAExt)
Base.@propagate_inbounds Base.getindex(A::RCE.Const, I::Integer...) = A.a[I...]

# ------------------------------------------------------------------------------------------

const NO_GPU_HINT = "Reactant has no GPU client. If the NVIDIA kernel module and user-space " *
    "libraries have different versions (compare /proc/driver/nvidia/version with the " *
    "libcuda.so version), reboot."
try
    Reactant.set_default_backend(DEVICE)
catch
    DEVICE == "gpu" ? error(NO_GPU_HINT) : rethrow()
end
platform = Reactant.XLA.platform_name(Reactant.XLA.default_backend())
DEVICE == "gpu" && lowercase(platform) == "cpu" && error(NO_GPU_HINT)
DEVICE == "gpu" && (CUDA.functional() || error("CUDA.jl is not functional."))

c(n, dx) = collect(((1:n) .- 0.5) .* dx)
grid(n, dx) = DyadicGrid(c(n, dx), c(n, dx))

# Array-style versions.
function coarsen_array!(dst, src, r)
    nc = size(dst, 1)
    dst .= dropdims(sum(reshape(src, r, nc, r, nc); dims = (1, 3)); dims = (1, 3)) ./ r^2
    return dst
end

function refine_constant_array!(dst, src, r)
    nc = size(src, 1)
    dst .= reshape(reshape(src, 1, nc, 1, nc) .* ones(eltype(dst), r, 1, r, 1), r * nc, r * nc)
    return dst
end

function refine_linear_array!(dst, src, r)
    nc = size(src, 1)
    T = eltype(dst)
    z1, z2 = zeros(T, 1, nc), zeros(T, nc, 1)
    sx = vcat(z1, (src[3:end, :] .- src[1:end-2, :]) ./ 2, z1)
    sy = hcat(z2, (src[:, 3:end] .- src[:, 1:end-2]) ./ 2, z2)
    w = T[(2p - 1 - r) / (2r) for p in 1:r]
    fine = reshape(src, 1, nc, 1, nc) .+ reshape(sx, 1, nc, 1, nc) .* reshape(w, r, 1, 1, 1) .+
           reshape(sy, 1, nc, 1, nc) .* reshape(w, 1, 1, r, 1)
    dst .= reshape(fine, r * nc, r * nc)
    return dst
end

ms(bench) = median(bench).time * 1e3
maxerr(result, reference) = maximum(abs.(Array(result) .- reference))

function time_native(rgd, src, dst)
    if DEVICE == "gpu"
        s, d = CuArray(src), CuArray(dst)
        regrid!(d, s, rgd); CUDA.synchronize()
        return ms(@be (regrid!(d, s, rgd); CUDA.synchronize()) seconds = SECONDS)
    end
    s, d = copy(src), copy(dst)
    return ms(@be regrid!(d, s, rgd) seconds = SECONDS)
end

# Compiles `f(d, s, arg)`, times it, and checks the in-place result on a zeroed destination.
function time_reactant(f, arg, src, dst, reference; compile_kwargs...)
    s, d = Reactant.to_rarray(src), Reactant.to_rarray(zero(dst))
    compiled = try
        Reactant.compile(f, (d, s, arg); sync = true, compile_kwargs...)
    catch e
        return "failed to compile: " * first(split(sprint(showerror, e), '\n'))
    end
    compiled(d, s, arg)
    err = maxerr(d, reference)
    t = ms(@be $compiled($d, $s, $arg) seconds = SECONDS)
    return "$(round(t, sigdigits = 3)) ms (err $(round(err, sigdigits = 2)))"
end

println("Reactant platform: $platform, device: $DEVICE, Julia threads: $(Threads.nthreads()), ",
        "CPUs visible: $(length(Sys.cpu_info()))")
println("Reactant kernels are known to be wrong on CPU when the kernel range's first axis ",
        "is ≥ 2048 (see minimal.jl): check the errors.")

T = Float32
for (n, r) in ((2048, 2), (4096, 2), (4096, 4))
    gf, gc = grid(n, 1000.0), grid(n ÷ r, r * 1000.0)
    fine, coarse = rand(T, n, n), rand(T, n ÷ r, n ÷ r)
    cases = (
        ("AverageCoarsening", AverageCoarsening(gf, gc), fine, zeros(T, n ÷ r, n ÷ r), coarsen_array!),
        ("ConstantRefinement", ConstantRefinement(gc, gf), coarse, zeros(T, n, n), refine_constant_array!),
        ("LinearRefinement", LinearRefinement(gc, gf), coarse, zeros(T, n, n), refine_linear_array!),
    )
    for (name, rgd, src, dst, arrayfn!) in cases
        reference = regrid(src, rgd)
        println("\n$name, $(n)², r = $r")
        println("  native            ", round(time_native(rgd, src, dst), sigdigits = 3), " ms")
        println("  kernel, raised    ", time_reactant(regrid!, rgd, src, dst, reference; raise = true))
        DEVICE == "gpu" &&
            println("  kernel, CUDA      ", time_reactant(regrid!, rgd, src, dst, reference; raise = false))
        println("  array             ", time_reactant(arrayfn!, r, src, dst, reference))
    end
end
