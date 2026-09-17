#=
Speedup of KryosTools over Interpolations.jl, from the benchmark results.

    julia --project=benchmark benchmark/compare.jl

For the latest run in `results.csv` (or `KRYOSTOOLS_BENCH_RESULTS`), prints the Interpolations.jl
time on one CPU thread divided by the KryosTools time on each configuration. The operations
are not equivalent — Interpolations.jl samples point interpolants, KryosTools integrates
over cells — so this is a rough guide to cost, not a like-for-like comparison. Also called at
the end of `run.jl`.
=#

using DelimitedFiles
using Printf

const COMPARISONS = (
    "regrid!(LinearRefinement)" => "Interpolations.jl Linear to fine",
    "regrid!(ConstantRefinement)" => "Interpolations.jl Constant to fine",
    "regrid!(AverageCoarsening)" => "Interpolations.jl Linear to coarse",
)

function compare_with_interpolations(file = get(ENV, "KRYOSTOOLS_BENCH_RESULTS",
                                                joinpath(@__DIR__, "results", "results.csv")))
    data, header = readdlm(file, ',', String; header = true)
    col = Dict(name => j for (j, name) in enumerate(vec(header)))
    rows = [Dict(name => data[i, j] for (name, j) in col) for i in axes(data, 1)]
    isempty(rows) && return println("No benchmark results to compare.")

    latest = maximum(r["run_date"] for r in rows)
    rows = filter(r -> r["run_date"] == latest, rows)
    time(r) = parse(Float64, r["median_s"])
    config(r) = r["backend"] == "cpu" ? "cpu×$(r["threads"])" : r["backend"]
    case(r) = (parse(Int, r["fine_size"]), parse(Int, r["ratio"]), r["eltype"])

    configs = sort!(unique(config(r) for r in rows if !startswith(r["function"], "Interpolations")))
    println("\nSpeedup over Interpolations.jl on 1 CPU thread (run $latest, commit $(first(rows[1]["commit"], 7)))")
    @printf("%-30s %-18s", "KryosTools", "size, r, type")
    foreach(c -> @printf("%10s", c), configs)
    println()
    for (ours, theirs) in COMPARISONS
        reference = Dict(case(r) => time(r) for r in rows
                         if r["function"] == theirs && config(r) == "cpu×1")
        for key in sort!(collect(keys(reference)))
            @printf("%-30s %-18s", ours, "$(key[1])², $(key[2]), $(key[3])")
            for c in configs
                match = filter(r -> r["function"] == ours && config(r) == c && case(r) == key, rows)
                isempty(match) ? @printf("%10s", "—") :
                                 @printf("%9.1f×", reference[key] / time(only(match)))
            end
            println()
        end
    end
end

abspath(PROGRAM_FILE) == (@__FILE__) && compare_with_interpolations()
