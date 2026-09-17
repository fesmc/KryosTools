using KryosTools: _nesting

@testset "Regridder construction" begin
    x(n, dx) = collect(((1:n) .- 0.5) .* dx)
    grid(n, dx; shift = 0.0) = DyadicGrid(x(n, dx) .+ shift, x(n, dx))
    fine, coarse = grid(16, 1000.0), grid(8, 2000.0)

    err(f) = try
        f()
        ""
    catch e
        e isa ArgumentError ? sprint(showerror, e) : rethrow()
    end

    @testset "AverageCoarsening" begin
        rgd = AverageCoarsening(fine, coarse)
        @test grids(rgd) == (fine, coarse)
        @test ratio(rgd) == 2
        @test isconservative(rgd) && islinear(rgd)
        @test rgd isa AbstractCoarsening{2,Center}
        @test sprint(show, rgd) == "AverageCoarsening(16×16 → 8×8, ratio 2)"
        @test ratio(AverageCoarsening(grid(32, 500.0), coarse)) == 4
        @test ratio(AverageCoarsening(fine, fine)) == 1       # valid, just not the fast path

        @test contains(err(() -> AverageCoarsening(coarse, fine)),
                       "expects grid1 to be the finer grid")
        @test contains(err(() -> AverageCoarsening(coarse, fine)), "Did you mean ConstantRefinement")
        @test contains(err(() -> AverageCoarsening(grid(24, 1000.0), grid(8, 3000.0))), "not a power of 2")
        @test contains(err(() -> AverageCoarsening(grid(16, 1000.0), grid(10, 1500.0))), "integer ratio")
        @test contains(err(() -> AverageCoarsening(fine, grid(8, 2000.0; shift = 1000.0))), "corners")
        @test contains(err(() -> AverageCoarsening(fine, grid(7, 2000.0))), "refined 2×")
        @test contains(err(() -> AverageCoarsening(fine, coarse; location = XFace())), "not implemented")
        @test_throws MethodError AverageCoarsening(fine, DyadicGrid(x(8, 2000.0)))

        # weights
        w = ones(16, 16)
        @test AverageCoarsening(fine, coarse; weights = w).weights == w
        @test AverageCoarsening(fine, coarse; weights = w).weights !== w
        @test contains(err(() -> AverageCoarsening(fine, coarse; weights = ones(8, 8))), "weights have size")
        @test contains(err(() -> AverageCoarsening(fine, coarse; weights = -w)), "non-negative")
        @test contains(err(() -> AverageCoarsening(fine, coarse; weights = fill(NaN, 16, 16))), "non-negative")
        holey = copy(w)
        holey[3:4, 5:6] .= 0                       # one whole coarse cell
        @test contains(err(() -> AverageCoarsening(fine, coarse; weights = holey)), "zero total weight")
        holey[3:4, 5] .= 1
        @test AverageCoarsening(fine, coarse; weights = holey) isa AverageCoarsening
    end

    @testset "IdentityRegridding" begin
        rgd = IdentityRegridding(fine, grid(16, 1000.0))
        @test ratio(rgd) == 1
        @test isconservative(rgd) && islinear(rgd)
        @test sprint(show, rgd) == "IdentityRegridding(16×16 → 16×16)"
        @test IdentityRegridding(fine, fine; location = XFace()) isa IdentityRegridding{2,XFace}
        @test contains(err(() -> IdentityRegridding(fine, coarse)), "differ")
    end

    @testset "$Refinement" for Refinement in (ConstantRefinement, LinearRefinement)
        rgd = Refinement(coarse, fine)
        @test grids(rgd) == (coarse, fine)
        @test ratio(rgd) == 2
        @test isconservative(rgd) && islinear(rgd)
        @test rgd isa AbstractRefinement{2,Center}
        @test sprint(show, rgd) == "$(nameof(Refinement))(8×8 → 16×16, ratio 2)"
        @test ratio(Refinement(coarse, grid(32, 500.0))) == 4
        @test contains(err(() -> Refinement(fine, coarse)),
                       "expects grid2 to be the finer grid")
        @test contains(err(() -> Refinement(fine, coarse)), "Did you mean AverageCoarsening")
        @test contains(err(() -> Refinement(grid(8, 3000.0), grid(24, 1000.0))), "not a power of 2")
        @test contains(err(() -> Refinement(coarse, fine; location = YFace())), "not implemented")
    end
    @test LinearRefinement(coarse, fine).limiter === nothing
    limited = LinearRefinement(coarse, fine; limiter = MinModLimiter())
    @test limited.limiter === MinModLimiter()
    @test isconservative(limited) && !islinear(limited)
    @test_throws TypeError LinearRefinement(coarse, fine; limiter = :minmod)

    @testset "BidirectionalRegridding" begin
        rgd = BidirectionalRegridding(fine, coarse)
        @test grids(rgd) == (fine, coarse)
        @test rgd.to1 isa LinearRefinement && rgd.to2 isa AverageCoarsening
        @test grids(rgd.to1) == (coarse, fine) && grids(rgd.to2) == (fine, coarse)
        @test ratio(rgd) == 2
        @test isconservative(rgd) && islinear(rgd)
        @test sprint(show, rgd) ==
              "BidirectionalRegridding(16×16 ↔ 8×8: to1 = LinearRefinement, to2 = AverageCoarsening)"

        # either order
        swapped = BidirectionalRegridding(coarse, fine)
        @test swapped.to1 isa AverageCoarsening && swapped.to2 isa LinearRefinement

        # identical grids copy both ways
        same = BidirectionalRegridding(fine, grid(16, 1000.0))
        @test same.to1 isa IdentityRegridding && same.to2 isa IdentityRegridding
        @test ratio(same) == 1

        # refinement type and forwarded keywords
        @test BidirectionalRegridding(fine, coarse; refinement = ConstantRefinement).to1 isa ConstantRefinement
        limited = BidirectionalRegridding(coarse, fine; limiter = MinModLimiter())
        @test limited.to2.limiter === MinModLimiter() && !islinear(limited)
        @test_throws MethodError BidirectionalRegridding(fine, coarse; refinement = ConstantRefinement,
                                                         limiter = MinModLimiter())
        @test_throws MethodError BidirectionalRegridding(fine, coarse; limitter = MinModLimiter())

        # weights belong to the finer grid, whichever number it has
        w = rand(16, 16)
        @test BidirectionalRegridding(fine, coarse; weights = w).to2.weights == w
        @test BidirectionalRegridding(coarse, fine; weights = w).to1.weights == w
        @test contains(err(() -> BidirectionalRegridding(fine, coarse; weights = ones(8, 8))),
                       "finer grid has 16×16 cells")

        # incompatible grids name the composite
        message = err(() -> BidirectionalRegridding(fine, grid(8, 3000.0)))
        @test startswith(message, "ArgumentError: BidirectionalRegridding between")
        @test contains(message, "not a power of 2")
        @test contains(err(() -> BidirectionalRegridding(fine, grid(16, 1000.0; shift = 500.0))), "corners")

        # direction detection needs one field on each grid
        @test_throws DimensionMismatch regrid!(zeros(16, 16), zeros(16, 16), rgd)
        @test_throws DimensionMismatch regrid!(zeros(8, 8), zeros(8, 8), rgd)
        @test_throws DimensionMismatch regrid!(zeros(8, 8), zeros(4, 4), rgd)
        @test_throws DimensionMismatch regrid(zeros(4, 4), rgd)
    end

    @testset "JET on concrete calls" begin
        for rgd in (AverageCoarsening(fine, coarse), IdentityRegridding(fine, fine),
                    AverageCoarsening(fine, coarse; weights = rand(16, 16)),
                    ConstantRefinement(coarse, fine), LinearRefinement(coarse, fine),
                    LinearRefinement(coarse, fine; limiter = MinModLimiter()),
                    BidirectionalRegridding(fine, coarse))
            g1, g2 = grids(rgd)
            src, dst = rand(Float32, size(g1)), zeros(Float32, size(g2))
            JET.@test_call target_modules = (KryosTools,) regrid!(dst, src, rgd)
            JET.@test_call target_modules = (KryosTools,) regrid(src, rgd)
            JET.@test_call target_modules = (KryosTools,) regrid!(src, dst, rgd)
        end
    end

    @testset "field checks" begin
        rgd = AverageCoarsening(fine, coarse)
        @test_throws DimensionMismatch regrid!(zeros(8, 8), zeros(8, 8), rgd)
        @test_throws DimensionMismatch regrid!(zeros(16, 16), zeros(16, 16), rgd)
        @test_throws DimensionMismatch regrid(zeros(8, 8), rgd)
    end
end
