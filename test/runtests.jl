using KryosTools
using Test
using Aqua
using JET

@testset "KryosTools.jl" begin
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(KryosTools)
    end
    @testset "Code linting (JET.jl)" begin
        JET.test_package(KryosTools; target_defined_modules = true)
    end
    # Write your tests here.
end
