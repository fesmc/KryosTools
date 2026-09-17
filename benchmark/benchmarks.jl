#=
Benchmark cases, loaded by the worker processes of `run.jl` (see there for usage).

Every case is timed with Chairmarks and reported as one row. Arrays are allocated before
timing, and GPU work is synchronized inside the timed expression, so the numbers measure the
computation rather than allocation or kernel launch.
=#

using KryosTools
using Chairmarks: @be
using Statistics: median
import KernelAbstractions
using KernelAbstractions: synchronize

const FINE_SIZES = (256, 1024, 4096)
const RATIOS = (2, 4)
const FLOAT_TYPES = (Float32, Float64)
const SECONDS_PER_CASE = parse(Float64, get(ENV, "KRYOSTOOLS_BENCH_SECONDS", "1.0"))
const FINE_SPACING = 1000.0

# Each entry is a function `(device, backend) -> rows` added by the milestone that introduces
# the benchmarked functionality. `device` turns a host array into a backend array and
# `backend` is the matching KernelAbstractions backend.
const CASES = Function[]

function run_cases(device, backend)
    rows = NamedTuple[]
    for case in CASES
        append!(rows, case(device, backend))
    end
    return rows
end

function result_row(bench, name; fine_size, ratio = 0, eltype, bytes = 0)
    s = median(bench)
    lowest_allocs = minimum(sample.allocs for sample in bench.samples)
    return (; function_name = name, fine_size, ratio, eltype = string(eltype),
            median_s = s.time, allocs = lowest_allocs,
            throughput_GBps = bytes == 0 ? NaN : bytes / s.time / 1e9)
end

# Cell-centre coordinates of an n×n grid with spacing `dx` and its lower-left corner at 0.
centres(n, dx, T = Float64) = T[(i - 0.5) * dx for i in 1:n]

# ------------------------------------------------------------------------------------------
# Grid construction (host-side; timed on the CPU configurations only)
# ------------------------------------------------------------------------------------------

push!(CASES, function (device, backend)
    rows = NamedTuple[]
    backend isa KernelAbstractions.CPU || return rows
    for n in FINE_SIZES, T in FLOAT_TYPES
        x = centres(n, FINE_SPACING, T)
        bench = @be DyadicGrid(x, x) seconds = SECONDS_PER_CASE
        push!(rows, result_row(bench, "DyadicGrid"; fine_size = n, eltype = T,
                               bytes = 2n * sizeof(T)))
    end
    return rows
end)

# ------------------------------------------------------------------------------------------
# regrid!
# ------------------------------------------------------------------------------------------

bench_grid(n, dx) = DyadicGrid(centres(n, dx), centres(n, dx))

# Times `regrid!(dst, src, rgd)` including GPU synchronization. Throughput counts the bytes
# of both fields, the memory the kernel actually traverses.
function regrid_row(rgd, dst, src, backend; fine_size, ratio)
    regrid!(dst, src, rgd)
    synchronize(backend)
    bench = @be (regrid!(dst, src, rgd); synchronize(backend)) seconds = SECONDS_PER_CASE
    T = eltype(dst)
    return result_row(bench, "regrid!($(nameof(typeof(rgd))))"; fine_size, ratio, eltype = T,
                      bytes = (length(src) + length(dst)) * sizeof(T))
end

push!(CASES, function (device, backend)
    rows = NamedTuple[]
    for n in FINE_SIZES, T in FLOAT_TYPES
        g = bench_grid(n, FINE_SPACING)
        src, dst = device(rand(T, n, n)), device(zeros(T, n, n))
        push!(rows, regrid_row(IdentityRegridding(g, g), dst, src, backend;
                               fine_size = n, ratio = 1))
    end
    return rows
end)

push!(CASES, function (device, backend)
    rows = NamedTuple[]
    for n in FINE_SIZES, r in RATIOS, T in FLOAT_TYPES
        rgd = AverageCoarsening(bench_grid(n, FINE_SPACING), bench_grid(n ÷ r, r * FINE_SPACING))
        src, dst = device(rand(T, n, n)), device(zeros(T, n ÷ r, n ÷ r))
        push!(rows, regrid_row(rgd, dst, src, backend; fine_size = n, ratio = r))
    end
    return rows
end)

push!(CASES, function (device, backend)
    rows = NamedTuple[]
    for n in FINE_SIZES, r in RATIOS, T in FLOAT_TYPES
        weights = device(0.8 .+ 0.4 .* rand(n, n))
        rgd = AverageCoarsening(bench_grid(n, FINE_SPACING), bench_grid(n ÷ r, r * FINE_SPACING);
                                weights)
        src, dst = device(rand(T, n, n)), device(zeros(T, n ÷ r, n ÷ r))
        row = regrid_row(rgd, dst, src, backend; fine_size = n, ratio = r)
        push!(rows, merge(row, (; function_name = "regrid!(AverageCoarsening) weighted")))
    end
    return rows
end)

push!(CASES, function (device, backend)
    rows = NamedTuple[]
    for Refinement in (ConstantRefinement, LinearRefinement), n in FINE_SIZES, r in RATIOS,
        T in FLOAT_TYPES
        rgd = Refinement(bench_grid(n ÷ r, r * FINE_SPACING), bench_grid(n, FINE_SPACING))
        src, dst = device(rand(T, n ÷ r, n ÷ r)), device(zeros(T, n, n))
        push!(rows, regrid_row(rgd, dst, src, backend; fine_size = n, ratio = r))
    end
    return rows
end)

push!(CASES, function (device, backend)
    rows = NamedTuple[]
    for n in FINE_SIZES, r in RATIOS, T in FLOAT_TYPES
        rgd = LinearRefinement(bench_grid(n ÷ r, r * FINE_SPACING), bench_grid(n, FINE_SPACING);
                               limiter = MinModLimiter())
        src, dst = device(rand(T, n ÷ r, n ÷ r)), device(zeros(T, n, n))
        row = regrid_row(rgd, dst, src, backend; fine_size = n, ratio = r)
        push!(rows, merge(row, (; function_name = "regrid!(LinearRefinement) MinModLimiter")))
    end
    return rows
end)

# Dispatch overhead of the two-way regridder: compare with the one-way rows above.
push!(CASES, function (device, backend)
    rows = NamedTuple[]
    for n in FINE_SIZES, r in RATIOS, T in FLOAT_TYPES
        rgd = BidirectionalRegridding(bench_grid(n, FINE_SPACING), bench_grid(n ÷ r, r * FINE_SPACING))
        fine, coarse = device(rand(T, n, n)), device(rand(T, n ÷ r, n ÷ r))
        down = regrid_row(rgd, coarse, fine, backend; fine_size = n, ratio = r)
        up = regrid_row(rgd, fine, coarse, backend; fine_size = n, ratio = r)
        push!(rows, merge(down, (; function_name = "regrid!(BidirectionalRegridding) to coarse")))
        push!(rows, merge(up, (; function_name = "regrid!(BidirectionalRegridding) to fine")))
    end
    return rows
end)

push!(CASES, function (device, backend)
    rows = NamedTuple[]
    backend isa KernelAbstractions.CPU || return rows
    for n in FINE_SIZES
        fine, coarse = bench_grid(n, FINE_SPACING), bench_grid(n ÷ 2, 2FINE_SPACING)
        bench = @be BidirectionalRegridding(fine, coarse) seconds = SECONDS_PER_CASE
        push!(rows, result_row(bench, "BidirectionalRegridding"; fine_size = n, ratio = 2,
                               eltype = Float64))
    end
    return rows
end)

# ------------------------------------------------------------------------------------------
# Reference: Interpolations.jl (CPU only)
#
# Interpolations.jl evaluates a point interpolant at the target cell centres; it does not
# average over cells, so neither direction is conservative. These rows only give a sense of
# what the usual alternative costs. The interpolant is built once and only the in-place
# evaluation is timed.
# ------------------------------------------------------------------------------------------

using Interpolations: Interpolations, BSpline, Linear, Constant, Flat

centre_axis(n, dx) = range(dx / 2; step = dx, length = n)

function reference_interpolant(src, dx, degree)
    x = centre_axis(size(src, 1), dx)
    itp = Interpolations.interpolate(src, BSpline(degree))
    return Interpolations.extrapolate(Interpolations.scale(itp, x, x), Flat())
end

function evaluate_at!(dst, etp, x)
    dst .= etp.(x, x')
    return dst
end

const INTERPOLATIONS_CASES = (
    # (row name, degree, source is fine?)
    ("Interpolations.jl Linear to fine", Linear(), false),
    ("Interpolations.jl Constant to fine", Constant(), false),
    ("Interpolations.jl Linear to coarse", Linear(), true),
)

push!(CASES, function (device, backend)
    rows = NamedTuple[]
    backend isa KernelAbstractions.CPU || return rows
    for (name, degree, from_fine) in INTERPOLATIONS_CASES, n in FINE_SIZES, r in RATIOS,
        T in FLOAT_TYPES
        nc = n ÷ r
        src_n, dst_n = from_fine ? (n, nc) : (nc, n)
        src_dx, dst_dx = from_fine ? (FINE_SPACING, r * FINE_SPACING) : (r * FINE_SPACING, FINE_SPACING)
        src, dst = rand(T, src_n, src_n), zeros(T, dst_n, dst_n)
        etp = reference_interpolant(src, src_dx, degree)
        x = centre_axis(dst_n, dst_dx)
        evaluate_at!(dst, etp, x)
        bench = @be evaluate_at!(dst, etp, x) seconds = SECONDS_PER_CASE
        push!(rows, result_row(bench, name; fine_size = n, ratio = r, eltype = T,
                               bytes = (length(src) + length(dst)) * sizeof(T)))
    end
    return rows
end)
