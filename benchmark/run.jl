#=
Benchmark driver. Runs every case in `benchmarks.jl` on 1 CPU thread, 4 CPU threads and one
CUDA GPU, and appends the results to `benchmark/results/results.csv`.

    julia --project=benchmark benchmark/run.jl [--allow-dirty] [--configs=cpu1,cpu4,cuda]

Environment overrides for smoke tests: `KRYOSTOOLS_BENCH_RESULTS` (results file) and
`KRYOSTOOLS_BENCH_SECONDS` (time budget per case, default 1).

The run is refused on a dirty git tree unless `--allow-dirty` is given, in which case its
rows are flagged `dirty = true`. Only tracked files count, and changes under
`benchmark/results/` do not, so uncommitted results do not block the next run.

At the end, the speedups over Interpolations.jl are printed (see `compare.jl`).

Julia fixes its thread count at startup, so each configuration runs in its own worker
process (this same file, invoked with `--worker`).
=#

using Dates

const IS_WORKER = get(ARGS, 1, "") == "--worker"
IS_WORKER && get(ARGS, 2, "") == "cuda" && @eval using CUDA
IS_WORKER && include(joinpath(@__DIR__, "benchmarks.jl"))
IS_WORKER || include(joinpath(@__DIR__, "compare.jl"))

# Overridable for smoke tests, so that trial runs never touch the tracked history.
const RESULTS_FILE = get(ENV, "KRYOSTOOLS_BENCH_RESULTS", joinpath(@__DIR__, "results", "results.csv"))
const HEADER = ["commit", "dirty", "commit_date", "run_date", "hostname", "cpu_model",
                "gpu_model", "julia_version", "backend", "threads", "function", "fine_size",
                "ratio", "eltype", "median_s", "allocs", "throughput_GBps"]
const CONFIGS = Dict("cpu1" => ("cpu", 1), "cpu4" => ("cpu", 4), "cuda" => ("cuda", 1))

csvfield(x) = replace(string(x), ',' => ';', '\n' => ' ')

# ------------------------------------------------------------------------------------------
# Worker: time every case on one backend and write the rows to `outfile`.
# ------------------------------------------------------------------------------------------

function worker(backend_name, outfile)
    if backend_name == "cuda"
        if !CUDA.functional()
            println("CUDA is not functional on this machine; skipping the GPU configuration.")
            return
        end
        device, backend = CUDA.CuArray, CUDA.CUDABackend()
        gpu_model = CUDA.name(CUDA.device())
    else
        device, backend = Array, KernelAbstractions.CPU()
        gpu_model = ""
    end

    rows = run_cases(device, backend)
    open(outfile, "w") do io
        for row in rows
            fields = (backend_name, Threads.nthreads(), gpu_model, row.function_name,
                      row.fine_size, row.ratio, row.eltype, row.median_s, row.allocs,
                      row.throughput_GBps)
            println(io, join(csvfield.(fields), ','))
        end
    end
    return
end

# ------------------------------------------------------------------------------------------
# Driver: check the tree, run the workers, append their rows with the run metadata.
# ------------------------------------------------------------------------------------------

function driver(args)
    allow_dirty = "--allow-dirty" in args
    configs = ["cpu1", "cpu4", "cuda"]
    for arg in args
        startswith(arg, "--configs=") && (configs = split(arg[length("--configs=")+1:end], ','))
    end
    unknown = setdiff(configs, keys(CONFIGS))
    isempty(unknown) || error("Unknown configuration(s) $(unknown); choose from $(sort!(collect(keys(CONFIGS)))).")

    root = dirname(@__DIR__)
    git(cmd...) = readchomp(Cmd(`git -C $root $cmd`))
    # Only tracked files count: untracked folders (e.g. a nested roadmap repository) cannot
    # affect the code being measured.
    dirty = !isempty(git("status", "--porcelain", "--untracked-files=no", "--", ".",
                         ":!benchmark/results"))
    if dirty && !allow_dirty
        error("The working tree has uncommitted changes. Commit them first, or pass " *
              "--allow-dirty to record flagged results anyway.")
    end

    meta = (commit = git("rev-parse", "HEAD"), dirty,
            commit_date = git("show", "-s", "--format=%cI", "HEAD"),
            run_date = string(now()),
            hostname = gethostname(),
            cpu_model = strip(Sys.cpu_info()[1].model),
            julia_version = string(VERSION))

    if !isfile(RESULTS_FILE)
        mkpath(dirname(RESULTS_FILE))
        write(RESULTS_FILE, join(HEADER, ',') * "\n")
    end

    appended = 0
    for config in configs
        backend_name, threads = CONFIGS[config]
        outfile = tempname()
        cmd = `$(Base.julia_cmd()) --project=$(@__DIR__) --threads=$threads $(@__FILE__) --worker $backend_name $outfile`
        println("▶ $config: running…")
        run(cmd)
        isfile(outfile) || continue
        open(RESULTS_FILE, "a") do io
            for line in eachline(outfile)
                backend, nthreads, gpu_model, rest = split(line, ',', limit = 4)
                fields = (meta.commit, meta.dirty, meta.commit_date, meta.run_date,
                          meta.hostname, meta.cpu_model, gpu_model, meta.julia_version,
                          backend, nthreads)
                println(io, join(csvfield.(fields), ','), ',', rest)
                appended += 1
            end
        end
        rm(outfile)
    end
    shown = startswith(RESULTS_FILE, root) ? relpath(RESULTS_FILE, root) : RESULTS_FILE
    println("Appended $appended rows to $shown.")
    compare_with_interpolations(RESULTS_FILE)
    return
end

IS_WORKER ? worker(ARGS[2], ARGS[3]) : driver(ARGS)
