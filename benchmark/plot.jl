#=
Plots the benchmark history in `benchmark/results/results.csv`.

    julia --project=benchmark benchmark/plot.jl [--metric=time|throughput] [--include-dirty]

Writes one figure per benchmarked function to `benchmark/results/<function>.png`: one panel
per (fine size, ratio, eltype), commits on the x-axis in commit-date order, and one line per
(host, backend, threads). When a commit was measured several times on the same
configuration, the latest run is shown. Rows from dirty trees are left out unless
`--include-dirty` is given.
=#

using CairoMakie
using DelimitedFiles

const RESULTS_FILE = get(ENV, "KRYOSTOOLS_BENCH_RESULTS", joinpath(@__DIR__, "results", "results.csv"))

function load_rows(include_dirty)
    data, header = readdlm(RESULTS_FILE, ',', String; header = true)
    col = Dict(name => j for (j, name) in enumerate(vec(header)))
    rows = [Dict(name => data[i, j] for (name, j) in col) for i in axes(data, 1)]
    include_dirty || filter!(r -> r["dirty"] == "false", rows)
    return rows
end

function plot_history(; metric = "time", include_dirty = false)
    rows = load_rows(include_dirty)
    isempty(rows) && return println("No rows to plot.")

    ylabel, value = metric == "throughput" ?
        ("throughput [GB/s]", r -> parse(Float64, r["throughput_GBps"])) :
        ("median time [s]", r -> parse(Float64, r["median_s"]))

    # Commits ordered by commit date define the shared x-axis.
    commit_dates = Dict(r["commit"] => r["commit_date"] for r in rows)
    commits = sort!(collect(keys(commit_dates)); by = c -> commit_dates[c])
    xpos = Dict(c => i for (i, c) in enumerate(commits))
    xticks = (1:length(commits), first.(commits, 7))

    for fname in sort!(unique(r["function"] for r in rows))
        frows = filter(r -> r["function"] == fname && isfinite(value(r)), rows)
        isempty(frows) && continue      # e.g. no throughput for constructors
        panels = sort!(unique((parse(Int, r["fine_size"]), parse(Int, r["ratio"]), r["eltype"])
                              for r in frows))
        ncols = min(length(panels), 4)
        fig = Figure(size = (380 * ncols, 300 * cld(length(panels), ncols)))
        for (k, (n, ratio, T)) in enumerate(panels)
            ax = Axis(fig[fldmod1(k, ncols)...];
                      title = "$(n)²" * (ratio > 0 ? ", r = $ratio" : "") * ", $T",
                      xticks, xticklabelrotation = π / 4, ylabel, yscale = log10,
                      ytickformat = vs -> string.(round.(vs; sigdigits = 3)))
            prows = filter(r -> (parse(Int, r["fine_size"]), parse(Int, r["ratio"]),
                                 r["eltype"]) == (n, ratio, T), frows)
            for key in sort!(unique((r["hostname"], r["backend"], r["threads"]) for r in prows))
                latest = Dict{String,Dict{String,String}}()
                for r in prows
                    (r["hostname"], r["backend"], r["threads"]) == key || continue
                    prev = get(latest, r["commit"], nothing)
                    (prev === nothing || r["run_date"] > prev["run_date"]) &&
                        (latest[r["commit"]] = r)
                end
                pts = sort!([(xpos[c], value(r)) for (c, r) in latest])
                host, backend, threads = key
                scatterlines!(ax, first.(pts), last.(pts);
                              label = "$host $backend" * (backend == "cpu" ? "×$threads" : ""))
            end
            k == 1 && axislegend(ax; position = :lb, labelsize = 10)
        end
        Label(fig[0, :], fname; fontsize = 18, font = :bold)
        out = joinpath(dirname(RESULTS_FILE), replace(fname, r"[^A-Za-z0-9]+" => "_") * ".png")
        save(out, fig)
        println("Wrote $out")
    end
end

metric_arg = findfirst(a -> startswith(a, "--metric="), ARGS)
plot_history(;
    metric = metric_arg === nothing ? "time" : split(ARGS[metric_arg], '=')[2],
    include_dirty = "--include-dirty" in ARGS)
