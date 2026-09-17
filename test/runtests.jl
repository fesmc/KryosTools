using KryosTools
using Test
using Aqua
using JET
using Random

include("properties.jl")

@info "Test threads" Threads.nthreads()

@testset "KryosTools.jl" begin
    @testset "Code quality (Aqua.jl)" begin
        # Declared ahead of the convolution and IO modules; remove each entry as its
        # dependency starts being used.
        planned_deps = [:AbstractFFTs, :DelimitedFiles, :Downloads, :FFTW, :Statistics]
        Aqua.test_all(KryosTools; stale_deps = (ignore = planned_deps,))
    end
    @testset "Code linting (JET.jl)" begin
        # Package-wide analysis sees abstract argument types, for which a KernelAbstractions
        # launch may target a GPU whose methods only exist once a GPU package is loaded, so
        # it is limited to typos here. Concrete calls get the full analysis in
        # regridding.jl.
        JET.test_package(KryosTools; target_modules = (KryosTools,), mode = :typo)
    end
    include("grids.jl")
    include("regridding.jl")
    test_properties(Array)
end
