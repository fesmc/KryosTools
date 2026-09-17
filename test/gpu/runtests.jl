#=
CUDA run of the property suite. Local only — GitHub-hosted runners have no GPU.

    julia --project=test/gpu test/gpu/runtests.jl
=#

using KryosTools
using Test
using Random
using CUDA

include(joinpath(@__DIR__, "..", "properties.jl"))

CUDA.functional() || error("CUDA is not functional on this machine; cannot run the GPU suite.")
CUDA.allowscalar(false)   # any scalar indexing into a CuArray is a bug in a kernel path

@testset "KryosTools.jl on CUDA" begin
    test_properties(CuArray)
end
