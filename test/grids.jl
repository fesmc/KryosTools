using KryosTools: origin, _same_geometry

@testset "DyadicGrid" begin
    x = 500.0:1000.0:7500.0

    @testset "construction and accessors" begin
        g = DyadicGrid(x, collect(x))
        @test size(g) == (8, 8)
        @test size(g, 2) == 8
        @test ndims(g) == 2
        @test spacing(g) == 1000.0
        @test origin(g) == (0.0, 0.0)
        @test centers(g) == (x, x)
        @test faces(g) == (0.0:1000.0:8000.0, 0.0:1000.0:8000.0)
        @test all(length.(faces(g)) .== length.(centers(g)) .+ 1)
        @test isbits(g)
        @test cellarea(g, 3, 4) == 1.0e6
        @test sprint(show, g) == "8×8 DyadicGrid with spacing 1000.0 and origin (0.0, 0.0)"

        # any dimension, any real coordinate type
        @test size(DyadicGrid(0.5:1.0:3.5)) == (4,)
        @test sprint(show, DyadicGrid(0.5:1.0:3.5)) == "4-cell DyadicGrid with spacing 1.0 and origin (0.0,)"
        @test size(DyadicGrid(x, x, x)) == (8, 8, 8)
        @test DyadicGrid(500:1000:7500, 500:1000:7500).spacing == 1000.0
        @test size(DyadicGrid(x, 500.0:1000.0:2500.0)) == (8, 3)
    end

    @testset "Float32 coordinates far from the origin" begin
        x64 = collect(-3.0395e6 + 500:1000.0:3.0395e6)
        g64 = DyadicGrid(x64, x64)
        g32 = DyadicGrid(Float32.(x64), Float32.(x64))
        # The rounding of Float32 is tolerated and simplified away entirely.
        @test g32.spacing == g64.spacing == 1000.0
        @test g32.origin == g64.origin == (-3.0395e6, -3.0395e6)
        @test g32.atol > g64.atol
        @test _same_geometry(g32, g64)
    end

    @testset "cell area" begin
        area = fill(1.0e6, 8, 8)
        area[2, 3] = 1.1e6
        g = DyadicGrid(x, x; area)
        @test cellarea(g, 2, 3) == 1.1e6
        @test cellarea(g, 1, 1) == 1.0e6
        @test g.area == area && g.area !== area            # stored as an independent copy
        @test eltype(DyadicGrid(x, x; area = Float32.(area)).area) == Float64
        @test !isbits(g)
        @test endswith(sprint(show, g), ", with cell areas")

        # a dimensionless factor passed as area is almost certainly a mistake
        @test_logs (:warn, r"distortion factor") DyadicGrid(x, x; area = fill(1.02, 8, 8))
        @test_logs DyadicGrid(x, x; area = fill(1.15e6, 8, 8))
    end

    @testset "geometric identity" begin
        g = DyadicGrid(x, x)
        @test _same_geometry(g, DyadicGrid(collect(x), collect(x)))
        @test _same_geometry(g, DyadicGrid(x, x; area = fill(1.0e6, 8, 8)))  # area ignored
        @test !_same_geometry(g, DyadicGrid(x .+ 1000, x))                   # shifted
        @test !_same_geometry(g, DyadicGrid(x, 500.0:1000.0:6500.0))         # smaller
        @test !_same_geometry(g, DyadicGrid(1000.0:2000.0:15000.0, 1000.0:2000.0:15000.0))
        @test !_same_geometry(g, DyadicGrid(x))                              # other N
        # indistinguishable at the first cell, beyond tolerance at the far edge
        stretched = DyadicGrid(range(0.5; step = 1.000005, length = 1000))
        @test stretched.spacing == 1.000005
        @test !_same_geometry(DyadicGrid(0.5:1.0:999.5), stretched)
    end

    @testset "invalid coordinates" begin
        err(f) = try
            f()
            ""
        catch e
            e isa ArgumentError ? sprint(showerror, e) : rethrow()
        end
        @test contains(err(() -> DyadicGrid([0.0], x)), "at least 2")
        @test contains(err(() -> DyadicGrid(reverse(x), x)), "not increasing")
        @test contains(err(() -> DyadicGrid([0.0, 1000, 2000, 3005, 4000])), "not uniformly spaced")
        @test contains(err(() -> DyadicGrid(x, 1000.0:2000.0:15000.0)), "isotropic")
        @test contains(err(() -> DyadicGrid(0.0:1005.0:8000.0, 0.0:1000.0:8000.0)), "isotropic")
        @test contains(err(() -> DyadicGrid([0.0, NaN])), "non-finite")
        @test contains(err(() -> DyadicGrid()), "at least one")
        @test contains(err(() -> DyadicGrid(x, x; area = ones(8, 7))), "size (8, 7)")
        @test contains(err(() -> DyadicGrid(x, x; area = -ones(8, 8))), "strictly positive")
        @test contains(err(() -> DyadicGrid(x, x; area = fill(Inf, 8, 8))), "strictly positive")
    end
end
