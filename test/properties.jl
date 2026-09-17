#=
Property suite shared by every backend.

`test_properties(AT)` runs the same checks with arrays created by the constructor `AT` —
`Array` from runtests.jl, `CuArray` from gpu/runtests.jl — so CPU and GPU are held to
identical standards.
=#

function test_properties(AT; float_types = (Float16, Float32, Float64))
    @testset "Properties ($(nameof(AT)))" begin
        test_grid_properties(AT, float_types)
        test_identity_properties(AT, float_types)
        test_coarsening_properties(AT, float_types)
        test_weighted_coarsening_properties(AT, float_types)
        test_refinement_properties(AT, float_types)
        test_limiter_properties(AT, float_types)
        test_bidirectional_properties(AT, float_types)
    end
    return nothing
end

# ------------------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------------------

const TEST_DX = 1000.0
const TEST_RATIOS = (1, 2, 4, 8)
const TEST_DIMS = (1, 2)

# An N-dimensional grid of n cells per axis with its corner at the origin.
test_grid(n, dx, N) = DyadicGrid(ntuple(_ -> collect(((1:n) .- 0.5) .* dx), N)...)

# A coarse grid and the grid r times finer over the same domain.
function test_pair(r, N)
    nc = N == 1 ? 7 : 5
    return test_grid(nc * r, TEST_DX, N), test_grid(nc, r * TEST_DX, N)
end

# Relative accuracy of a mean over r^N values accumulated in accumtype(T) and stored as T.
mean_rtol(T, r, N) = max(r^N, 4) * eps(KryosTools.accumtype(T)) + eps(T)

approx(a, b; rtol, atol = 0.0) = isapprox(Float64.(Array(a)), Float64.(Array(b)); rtol, atol)

# Block means computed naively in Float64 on the host: the reference for every backend.
function reference_block_mean(fine::AbstractArray{<:Any,N}, r) where {N}
    coarse = zeros(Float64, size(fine) .÷ r)
    for I in CartesianIndices(fine)
        coarse[CartesianIndex(ntuple(d -> (I[d] - 1) ÷ r + 1, N))] += fine[I]
    end
    return coarse ./ r^N
end

# Weighted block means computed naively in Float64 on the host.
function reference_weighted_mean(fine::AbstractArray{<:Any,N}, weights, r) where {N}
    num = zeros(Float64, size(fine) .÷ r)
    den = zeros(Float64, size(fine) .÷ r)
    for I in CartesianIndices(fine)
        J = CartesianIndex(ntuple(d -> (I[d] - 1) ÷ r + 1, N))
        num[J] += weights[I] * fine[I]
        den[J] += weights[I]
    end
    return num ./ den
end

# Conservative linear refinement computed naively in Float64 on the host, including the
# zero-slope policy in the outermost coarse cells.
function reference_linear_refinement(coarse::AbstractArray{<:Any,N}, r) where {N}
    fine = zeros(Float64, size(coarse) .* r)
    for I in CartesianIndices(fine)
        J = CartesianIndex(ntuple(d -> (I[d] - 1) ÷ r + 1, N))
        value = Float64(coarse[J])
        for d in 1:N
            j, n = J[d], size(coarse, d)
            1 < j < n || continue
            e = CartesianIndex(ntuple(k -> k == d ? 1 : 0, N))
            slope = (Float64(coarse[J + e]) - Float64(coarse[J - e])) / 2
            p = (I[d] - 1) % r + 1
            value += slope * (2p - 1 - r) / (2r)
        end
        fine[I] = value
    end
    return fine
end

# Bytes allocated by a warmed-up call, measured behind a function barrier.
function allocated(f, args...)
    f(args...)
    return @allocated f(args...)
end

function test_grid_properties(AT, float_types)
    @testset "DyadicGrid from device arrays" begin
        x = collect(500.0:1000.0:7500.0)
        reference = DyadicGrid(x, x)
        for T in float_types
            g = DyadicGrid(AT(T.(x)), AT(T.(x)))
            @test g.spacing == reference.spacing
            @test g.origin == reference.origin
        end

        area = AT(fill(1.0f6, 8, 8))
        g = DyadicGrid(x, x; area)
        @test g.area isa AT{Float64}           # stays on the device, stored as Float64
        @test Array(g.area) == fill(1.0e6, 8, 8)
    end
end

function test_identity_properties(AT, float_types)
    @testset "IdentityRegridding" begin
        rng = Random.Xoshiro(1)
        for N in TEST_DIMS, T in float_types
            g = test_grid(6, TEST_DX, N)
            rgd = IdentityRegridding(g, g)
            src = AT(rand(rng, T, size(g)))
            dst = AT(zeros(T, size(g)))
            @test regrid!(dst, src, rgd) === dst
            @test Array(dst) == Array(src)
            @test regrid!(src, src, rgd) === src          # aliasing is a no-op
            @test Array(regrid(src, rgd)) == Array(src)
            wide = AT(zeros(Float64, size(g)))             # mixed element types
            @test Array(regrid!(wide, src, rgd)) == Float64.(Array(src))
            @inferred regrid!(dst, src, rgd)
            @inferred regrid(src, rgd)
        end
    end
end

function test_coarsening_properties(AT, float_types)
    @testset "AverageCoarsening" begin
        rng = Random.Xoshiro(2)
        for N in TEST_DIMS, r in TEST_RATIOS, T in float_types
            fine_grid, coarse_grid = test_pair(r, N)
            rgd = AverageCoarsening(fine_grid, coarse_grid)
            rtol = mean_rtol(T, r, N)
            coarse = AT(zeros(T, size(coarse_grid)))

            # constants are preserved exactly
            regrid!(coarse, AT(fill(T(0.75), size(fine_grid))), rgd)
            @test all(==(T(0.75)), Array(coarse))

            # block means, against a naive Float64 reference
            fine = AT(rand(rng, T, size(fine_grid)))
            @test regrid!(coarse, fine, rgd) === coarse
            @test approx(coarse, reference_block_mean(Array(fine), r); rtol, atol = eps(T))

            # conservation
            @test isapprox(sum(Float64, Array(coarse)) * r^N, sum(Float64, Array(fine));
                           rtol = rtol)

            # a field constant on each coarse block is reproduced
            blocks = rand(rng, T, size(coarse_grid))
            refined = AT(repeat(blocks, inner = ntuple(_ -> r, N)))
            @test approx(regrid(refined, rgd), blocks; rtol, atol = eps(T))

            @inferred regrid!(coarse, fine, rgd)
            @inferred regrid(fine, rgd)
        end
    end

    @testset "AverageCoarsening allocations" begin
        # KernelAbstractions allocates a small, fixed amount per launch (a few hundred bytes
        # serially, a few KB when spawning tasks). It must not grow with the field size.
        pair(n) = AverageCoarsening(test_grid(2n, TEST_DX, 2), test_grid(n, 2TEST_DX, 2))
        # Below 1024 elements the CPU backend runs a single task, so compare two fields that
        # are both above that threshold.
        bytes_medium = allocated(regrid!, AT(zeros(64, 64)), AT(rand(128, 128)), pair(64))
        bytes_large = allocated(regrid!, AT(zeros(512, 512)), AT(rand(1024, 1024)), pair(512))
        @info "regrid! allocations on $(nameof(AT)), $(Threads.nthreads()) thread(s)" bytes_medium bytes_large
        # On CPU the cost is exactly constant; the CUDA launcher's host-side cost steps with
        # its block configuration but stays bounded.
        AT === Array && @test bytes_large == bytes_medium
        @test bytes_large <= 4096 * Threads.nthreads()
    end
end

function test_weighted_coarsening_properties(AT, float_types)
    @testset "AverageCoarsening with weights" begin
        rng = Random.Xoshiro(3)
        for N in TEST_DIMS, r in TEST_RATIOS, T in float_types
            fine_grid, coarse_grid = test_pair(r, N)
            rtol = mean_rtol(T, r, N) + 8 * eps(Float32)
            fine = AT(rand(rng, T, size(fine_grid)))
            coarse = AT(zeros(T, size(coarse_grid)))

            # area-like weights: weighted means, and conservation of Σ w x
            weights = 0.8 .+ 0.4 .* rand(rng, size(fine_grid)...)
            rgd = AverageCoarsening(fine_grid, coarse_grid; weights = AT(weights))
            @test rgd.weights isa AT{Float64}
            regrid!(coarse, fine, rgd)
            @test approx(coarse, reference_weighted_mean(Array(fine), weights, r); rtol, atol = eps(T))
            coarse_weights = reference_block_mean(weights, r) .* r^N
            @test isapprox(sum(Float64.(Array(coarse)) .* coarse_weights),
                           sum(Float64.(Array(fine)) .* weights); rtol)

            # constants are preserved
            regrid!(coarse, AT(fill(T(0.75), size(fine_grid))), rgd)
            @test approx(coarse, fill(0.75, size(coarse_grid)); rtol)

            # uniform weights reproduce the unweighted mean, whatever their scale
            uniform = AverageCoarsening(fine_grid, coarse_grid; weights = AT(fill(3.0e6, size(fine_grid))))
            @test approx(regrid(fine, uniform), regrid(fine, AverageCoarsening(fine_grid, coarse_grid)); rtol, atol = eps(T))

            # a mask: zero-weight cells are ignored as long as every block keeps one cell
            mask = Float64.(rand(rng, size(fine_grid)...) .< 0.5)
            mask[CartesianIndex(ntuple(_ -> 1, N))] = 1.0
            for J in CartesianIndices(size(coarse_grid))
                mask[CartesianIndex(ntuple(d -> (J[d] - 1) * r + 1, N))] = 1.0
            end
            masked = AverageCoarsening(fine_grid, coarse_grid; weights = AT(mask))
            @test approx(regrid(fine, masked), reference_weighted_mean(Array(fine), mask, r); rtol, atol = eps(T))

            @inferred regrid!(coarse, fine, rgd)
        end
    end
end

function test_refinement_properties(AT, float_types)
    @testset "$Refinement" for Refinement in (ConstantRefinement, LinearRefinement)
        rng = Random.Xoshiro(4)
        for N in TEST_DIMS, r in TEST_RATIOS, T in float_types
            fine_grid, coarse_grid = test_pair(r, N)
            up = Refinement(coarse_grid, fine_grid)
            down = AverageCoarsening(fine_grid, coarse_grid)
            rtol = mean_rtol(T, r, N)
            fine = AT(zeros(T, size(fine_grid)))

            # constants are preserved exactly
            regrid!(fine, AT(fill(T(0.75), size(coarse_grid))), up)
            @test all(==(T(0.75)), Array(fine))

            # against a naive Float64 reference
            coarse = AT(rand(rng, T, size(coarse_grid)))
            @test regrid!(fine, coarse, up) === fine
            reference = Refinement === ConstantRefinement ?
                repeat(Float64.(Array(coarse)), inner = ntuple(_ -> r, N)) :
                reference_linear_refinement(Array(coarse), r)
            @test approx(fine, reference; rtol, atol = 4 * eps(T))

            # conservation, and coarsening undoes refinement
            @test isapprox(sum(Float64, Array(fine)), sum(Float64, Array(coarse)) * r^N; rtol)
            @test approx(regrid(fine, down), coarse; rtol, atol = 4 * eps(T))

            @inferred regrid!(fine, coarse, up)
            @inferred regrid(coarse, up)
        end
    end

    @testset "ConstantRefinement is the adjoint of AverageCoarsening" begin
        rng = Random.Xoshiro(5)
        for N in TEST_DIMS, r in TEST_RATIOS
            fine_grid, coarse_grid = test_pair(r, N)
            x = AT(rand(rng, size(fine_grid)...))
            y = AT(rand(rng, size(coarse_grid)...))
            lhs = sum(Array(regrid(x, AverageCoarsening(fine_grid, coarse_grid))) .* Array(y)) * r^N
            rhs = sum(Array(x) .* Array(regrid(y, ConstantRefinement(coarse_grid, fine_grid))))
            @test lhs ≈ rhs rtol = 1e-12
        end
    end

    @testset "LinearRefinement reproduces linear fields" begin
        for N in TEST_DIMS, r in TEST_RATIOS, T in float_types
            fine_grid, coarse_grid = test_pair(r, N)
            # small coefficients keep Float16 values exactly representable at every centre
            coefficients = (0.25, -0.5, 0.125)[1:N]
            linear(g) = [sum(coefficients[d] * (I[d] - 0.5) * g.spacing / TEST_DX for d in 1:N) + 1
                         for I in CartesianIndices(size(g))]
            fine = regrid(AT(T.(linear(coarse_grid))), LinearRefinement(coarse_grid, fine_grid))
            nc = size(coarse_grid, 1)
            # fine cells whose parent is an interior cell along every axis
            interior = ntuple(_ -> r + 1:(nc - 1) * r, N)
            @test approx(Array(fine)[interior...], linear(fine_grid)[interior...];
                         rtol = 0, atol = 4 * eps(T))
        end
    end
end

function test_limiter_properties(AT, float_types)
    @testset "LinearRefinement with MinModLimiter" begin
        rng = Random.Xoshiro(6)
        for N in TEST_DIMS, r in TEST_RATIOS, T in float_types
            fine_grid, coarse_grid = test_pair(r, N)
            up = LinearRefinement(coarse_grid, fine_grid; limiter = MinModLimiter())
            down = AverageCoarsening(fine_grid, coarse_grid)
            rtol = mean_rtol(T, r, N)
            coarse = AT(rand(rng, T, size(coarse_grid)))
            fine = regrid(coarse, up)

            # still conservative, and still undone by coarsening
            @test isapprox(sum(Float64, Array(fine)), sum(Float64, Array(coarse)) * r^N; rtol)
            @test approx(regrid(fine, down), coarse; rtol, atol = 4 * eps(T))

            # no new extrema: every fine value lies within its parent's axis stencil
            host, finehost = Float64.(Array(coarse)), Float64.(Array(fine))
            within = true
            for I in CartesianIndices(finehost)
                J = CartesianIndex(ntuple(d -> (I[d] - 1) ÷ r + 1, N))
                stencil = [host[J]]
                for d in 1:N, step in (-1, 1)
                    K = CartesianIndex(ntuple(k -> k == d ? clamp(J[k] + step, 1, size(host, k)) : J[k], N))
                    push!(stencil, host[K])
                end
                tol = 4 * eps(T) * maximum(abs, stencil)
                within &= minimum(stencil) - tol <= finehost[I] <= maximum(stencil) + tol
            end
            @test within
            @inferred regrid!(fine, coarse, up)
        end

        # a positive field with a sharp margin: the unlimited scheme undershoots below zero,
        # the limited one does not
        for T in float_types
            grid_c = test_grid(6, 2TEST_DX, 1)
            grid_f = test_grid(12, TEST_DX, 1)
            margin = AT(T[0, 0, 0, 5, 5, 5])
            @test minimum(Array(regrid(margin, LinearRefinement(grid_c, grid_f)))) < 0
            limited = Array(regrid(margin, LinearRefinement(grid_c, grid_f; limiter = MinModLimiter())))
            @test minimum(limited) == 0
            @test maximum(limited) == 5
        end
    end
end

function test_bidirectional_properties(AT, float_types)
    @testset "BidirectionalRegridding" begin
        rng = Random.Xoshiro(7)
        for N in TEST_DIMS, r in TEST_RATIOS, T in float_types
            fine_grid, coarse_grid = test_pair(r, N)
            fine = AT(rand(rng, T, size(fine_grid)))
            coarse = AT(rand(rng, T, size(coarse_grid)))
            for rgd in (BidirectionalRegridding(fine_grid, coarse_grid),
                        BidirectionalRegridding(coarse_grid, fine_grid))
                # each direction is bit-identical to the matching one-way regridder
                down = regrid!(AT(zeros(T, size(coarse_grid))), fine, rgd)
                up = regrid!(AT(zeros(T, size(fine_grid))), coarse, rgd)
                @test Array(down) == Array(regrid(fine, AverageCoarsening(fine_grid, coarse_grid)))
                reference_up = r == 1 ? coarse : regrid(coarse, LinearRefinement(coarse_grid, fine_grid))
                @test Array(up) == Array(reference_up)
                @test Array(regrid(fine, rgd)) == Array(down)
                @test Array(regrid(coarse, rgd)) == Array(up)
                @inferred regrid!(down, fine, rgd)
                @inferred regrid!(up, coarse, rgd)
                @inferred regrid(fine, rgd)
            end
        end
    end
end
