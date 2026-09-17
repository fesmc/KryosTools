#=
Dyadic regridding.

Two grids are regridded onto each other when their spacings differ by a power of 2 and their
corners coincide, so that every coarse cell contains a whole number of fine cells with
perfect overlap. The index map between them is then affine, `parent = (i - 1) ÷ r + 1`, so
no weights or index lists are ever stored.

The two directions are not symmetric:

  - Coarsening is the exact cell integral — the block mean — so it is unique and takes no
    scheme.
  - Refinement reconstructs sub-cell values, which is a modelling choice.

Kernels are memory-bandwidth bound: each workitem reads its block and writes once, with no
atomics and no temporaries. Grids never enter kernels.

The ratio `r` is a type parameter of every regridder. Block loops then have constant bounds,
which the compiler unrolls and vectorises (and turns `* r` into shifts); with a run-time `r`
the same kernels are several times slower on the CPU. The price is that a regridder's
concrete type depends on its grids, so constructors are not inferable: build regridders once,
at setup, and store them in parametric fields.
=#

# ------------------------------------------------------------------------------------------
# Field location
# ------------------------------------------------------------------------------------------

"""
    AbstractLocation

Where a field lives within a grid cell. Only [`Center`](@ref) is implemented so far; the other
locations are declared so that staggered (Arakawa C-grid) fields can be added without an API
break.
"""
abstract type AbstractLocation end

"Cell-centred field, such as ice thickness or temperature."
struct Center <: AbstractLocation end

"Field on x-normal cell faces. Declared, not yet implemented."
struct XFace <: AbstractLocation end

"Field on y-normal cell faces. Declared, not yet implemented."
struct YFace <: AbstractLocation end

"Field on cell corners. Declared, not yet implemented."
struct Corner <: AbstractLocation end

function _require_center(location::AbstractLocation, name)
    location isa Center || throw(ArgumentError(
        "$name: location $(nameof(typeof(location)))() is declared but not implemented " *
        "yet; only Center() is supported."))
    return location
end

# ------------------------------------------------------------------------------------------
# Limiters
# ------------------------------------------------------------------------------------------

"""
    AbstractLimiter

Slope limiter for [`LinearRefinement`](@ref). A limiter replaces the centred slope with one
that keeps the reconstruction within the range of the neighbouring coarse values, so that
refinement creates no new extrema (and, for instance, no negative thickness).

Limiting changes only the slopes, and the refinement is conservative whatever the slopes are,
so limited refinement is still exactly conservative. It is no longer linear, though, and
limiters are not differentiable where they switch branches: gradients through a limited
refinement are zero wherever the limiter clips the slope, for example at ice divides and
sharp margins. Limiters are written without branches, so they run on GPUs and trace under
compilers such as Reactant.
"""
abstract type AbstractLimiter end

"""
    MinModLimiter()

The minmod limiter: the one-sided difference of smaller magnitude when both have the same
sign, and zero at local extrema. The most diffusive classic limiter, and the one that bounds
the reconstruction most strictly.
"""
struct MinModLimiter <: AbstractLimiter end

# ------------------------------------------------------------------------------------------
# Type tree
# ------------------------------------------------------------------------------------------

"""
    AbstractRegridding{N,L}

An operator that moves `N`-dimensional fields located at `L` from `grid1` to `grid2`, applied
with [`regrid!`](@ref). Every concrete subtype provides:

- `regrid!(dst, src, rgd)`, writing into `dst` without allocating;
- the fields `grid1` and `grid2`, as returned by [`grids`](@ref);
- [`ratio`](@ref).

Regridders hold no scratch space and are built once, at setup.
"""
abstract type AbstractRegridding{N,L<:AbstractLocation} end

"""
    AbstractCoarsening{N,L}

A regridding from a finer `grid1` to a coarser (or equal) `grid2`.
"""
abstract type AbstractCoarsening{N,L} <: AbstractRegridding{N,L} end

"""
    AbstractRefinement{N,L}

A regridding from a coarser `grid1` to a finer (or equal) `grid2`.
"""
abstract type AbstractRefinement{N,L} <: AbstractRegridding{N,L} end

"""
    IdentityRegridding(grid1, grid2; location = Center())

Regridding between two geometrically identical grids: a plain `copyto!`. Throws unless the
grids have the same size, spacing and origin; cell areas are ignored.
"""
struct IdentityRegridding{N,L,G1<:DyadicGrid{N},G2<:DyadicGrid{N}} <: AbstractRegridding{N,L}
    grid1::G1
    grid2::G2
    location::L
end

function IdentityRegridding(grid1::DyadicGrid{N}, grid2::DyadicGrid{N};
                            location::AbstractLocation = Center()) where {N}
    _same_geometry(grid1, grid2) || throw(ArgumentError(
        "IdentityRegridding: the grids differ ($grid1 vs $grid2)."))
    return IdentityRegridding(grid1, grid2, location)
end

"""
    AverageCoarsening(grid1, grid2; weights = nothing, location = Center())

Coarsening from `grid1` onto a coarser `grid2`: each coarse value is the mean of the fine
cells it contains. This is the exact cell integral, so it is conservative and needs no
scheme.

`grid1` must be finer than or equal to `grid2`, by a power-of-2 ratio, with coincident
corners. For identical grids the operator is correct, but [`IdentityRegridding`](@ref) is the
fast path.

# Weights

With `weights` — a non-negative array on `grid1` — each coarse value is the weighted mean
`Σ wᵢxᵢ / Σ wᵢ` over its block, which conserves `Σ wᵢxᵢ`. Typical weights are true cell
areas (`weights = grid1.area`, for area conservation on a distorted projection) or a mask
(zeros exclude cells). Constant factors cancel. Every coarse cell needs a positive total
weight. Weights are stored as `Float64` on the device of the array passed in, and must live
on the same backend as the fields. They are never applied implicitly.
"""
struct AverageCoarsening{N,L,R,G1<:DyadicGrid{N},G2<:DyadicGrid{N},W,A} <: AbstractCoarsening{N,L}
    grid1::G1
    grid2::G2
    location::L
    weights::W
    inv_weightsum::A

    function AverageCoarsening{N,L,R}(grid1::G1, grid2::G2, location::L, weights::W,
                                      inv_weightsum::A) where {N,L,R,G1,G2,W,A}
        return new{N,L,R,G1,G2,W,A}(grid1, grid2, location, weights, inv_weightsum)
    end
end

function AverageCoarsening(grid1::DyadicGrid{N}, grid2::DyadicGrid{N}; weights = nothing,
                           location::AbstractLocation = Center()) where {N}
    name = "AverageCoarsening"
    _require_center(location, name)
    r = _nesting(grid1, grid2, name, "grid1", "ConstantRefinement or LinearRefinement")
    stored, inv_weightsum = _coarsening_weights(weights, grid1, grid2, Val(r))
    return AverageCoarsening{N,typeof(location),r}(grid1, grid2, location, stored, inv_weightsum)
end

_coarsening_weights(::Nothing, grid1, grid2, ratio) = (nothing, nothing)

function _coarsening_weights(weights::AbstractArray, grid1, grid2, ratio::Val{R}) where {R}
    size(weights) == size(grid1) || throw(ArgumentError(
        "AverageCoarsening: weights have size $(size(weights)), but the finer grid has " *
        "$(join(size(grid1), "×")) cells."))
    stored = Float64.(weights)
    all(w -> isfinite(w) && w >= 0, stored) || throw(ArgumentError(
        "AverageCoarsening: weights must be finite and non-negative."))
    backend = get_backend(stored)
    sums = similar(stored, size(grid2))
    _block_sum_kernel!(backend)(sums, stored, ratio; ndrange = size(sums))
    synchronize(backend)
    all(>(0), sums) || throw(ArgumentError(
        "AverageCoarsening: at least one coarse cell has zero total weight, so its mean is " *
        "undefined."))
    sums .= inv.(sums)
    return stored, sums
end

@kernel function _block_sum_kernel!(dst, @Const(src), ratio)
    I = @index(Global, NTuple)
    @inbounds dst[I...] = _block_sum(_Values(src), I, ratio, zero(eltype(dst)))
end

"""
    ConstantRefinement(grid1, grid2; location = Center())

Refinement from `grid1` onto a finer `grid2` that copies each coarse value into all the fine
cells it covers. Exactly conservative, and exactly undone by [`AverageCoarsening`](@ref);
blocky.

`grid2` must be finer than or equal to `grid1`, by a power-of-2 ratio, with coincident
corners.
"""
struct ConstantRefinement{N,L,R,G1<:DyadicGrid{N},G2<:DyadicGrid{N}} <: AbstractRefinement{N,L}
    grid1::G1
    grid2::G2
    location::L

    function ConstantRefinement{N,L,R}(grid1::G1, grid2::G2,
                                       location::L) where {N,L,R,G1,G2}
        return new{N,L,R,G1,G2}(grid1, grid2, location)
    end
end

function ConstantRefinement(grid1::DyadicGrid{N}, grid2::DyadicGrid{N};
                            location::AbstractLocation = Center()) where {N}
    name = "ConstantRefinement"
    _require_center(location, name)
    r = _nesting(grid2, grid1, name, "grid2", "AverageCoarsening")
    return ConstantRefinement{N,typeof(location),r}(grid1, grid2, location)
end

"""
    LinearRefinement(grid1, grid2; limiter = nothing, location = Center())

Refinement from `grid1` onto a finer `grid2` by conservative piecewise-linear
reconstruction. Within each coarse cell the field is `T_c + Σ_d s_d ξ_d`, where `ξ_d` is the
position within the cell in units of the coarse spacing (from -1/2 to 1/2) and `s_d` is the
centred difference of the neighbouring coarse values. Each fine cell receives the mean of that
reconstruction over its extent.

The fine-cell offsets within a coarse cell sum to zero, so the block mean of the refined field
is exactly `T_c` *whatever the slopes are*: the scheme is conservative and exactly undone by
[`AverageCoarsening`](@ref). It reproduces fields that are linear in each coordinate; it has
no cross term, so bilinear fields are not reproduced.

# Boundaries

The centred slope needs both neighbours, so in the outermost coarse cells along each axis
the slope is zero and the scheme reduces to [`ConstantRefinement`](@ref) along that axis.

# Limiter

With `limiter = nothing` (the default) the operator is exactly linear in the field. The
reconstruction can then overshoot near sharp gradients — e.g. produce negative ice thickness
next to a margin. A limiter (see [`AbstractLimiter`](@ref)) bounds the overshoot at the cost
of linearity; conservation is unaffected.

`grid2` must be finer than or equal to `grid1`, by a power-of-2 ratio, with coincident
corners.
"""
struct LinearRefinement{N,L,R,G1<:DyadicGrid{N},G2<:DyadicGrid{N},Lim} <: AbstractRefinement{N,L}
    grid1::G1
    grid2::G2
    location::L
    limiter::Lim

    function LinearRefinement{N,L,R}(grid1::G1, grid2::G2, location::L,
                                     limiter::Lim) where {N,L,R,G1,G2,Lim}
        return new{N,L,R,G1,G2,Lim}(grid1, grid2, location, limiter)
    end
end

function LinearRefinement(grid1::DyadicGrid{N}, grid2::DyadicGrid{N};
                          limiter::Union{Nothing,AbstractLimiter} = nothing,
                          location::AbstractLocation = Center()) where {N}
    name = "LinearRefinement"
    _require_center(location, name)
    r = _nesting(grid2, grid1, name, "grid2", "AverageCoarsening")
    return LinearRefinement{N,typeof(location),r}(grid1, grid2, location, limiter)
end

"""
    BidirectionalRegridding(grid1, grid2; refinement = LinearRefinement, weights = nothing,
                            location = Center(), refinement_kwargs...)

Regridding in both directions between two grids, as a coupler needs it: `regrid!(x2, x1,
rgd)` writes onto `grid2` and `regrid!(x1, x2, rgd)` onto `grid1`. The direction is detected
from the sizes of both fields.

The grids may be passed in either order. For each direction the fitting one-way regridder is
built — [`AverageCoarsening`](@ref) towards the coarser grid, a `refinement` towards the finer
one, and [`IdentityRegridding`](@ref) (a plain copy) if the grids are identical — and stored
in the fields `to1` and `to2`, named after their destination grid.

- `refinement` is a refinement *type*; any further keyword (e.g.
  `limiter = MinModLimiter()`) is passed to its constructor.
- `weights` belong to the finer grid, whichever number it has, and only affect coarsening.

Coarsening undoes refinement exactly, but not the other way round: refining a coarsened
field does not restore the fine field.

Which regridders are stored depends on the grid spacings, so the constructor's return type
cannot be inferred. Build it once, at setup, and store it in a parametric field; `regrid!`
itself is type-stable.
"""
struct BidirectionalRegridding{N,L,G1<:DyadicGrid{N},G2<:DyadicGrid{N},
                               T1<:AbstractRegridding{N,L},T2<:AbstractRegridding{N,L}} <:
       AbstractRegridding{N,L}
    grid1::G1
    grid2::G2
    to1::T1
    to2::T2
end

function BidirectionalRegridding(grid1::DyadicGrid{N}, grid2::DyadicGrid{N};
                                 refinement::Type{<:AbstractRefinement} = LinearRefinement,
                                 weights = nothing, location::AbstractLocation = Center(),
                                 refinement_kwargs...) where {N}
    try
        to2 = _regridding(grid1, grid2; refinement, weights, location, refinement_kwargs...)
        to1 = _regridding(grid2, grid1; refinement, weights, location, refinement_kwargs...)
        return BidirectionalRegridding(grid1, grid2, to1, to2)
    catch e
        e isa ArgumentError || rethrow()
        throw(ArgumentError("BidirectionalRegridding between $grid1 and $grid2: $(e.msg)"))
    end
end

"""
    _regridding(src_grid, dst_grid; refinement, weights, location, refinement_kwargs...)

The one-way regridder from `src_grid` to `dst_grid`: identity, coarsening or refinement,
depending on the grids. Its return type depends on the grid values.
"""
function _regridding(src_grid::DyadicGrid, dst_grid::DyadicGrid; refinement, weights,
                     location, refinement_kwargs...)
    _same_geometry(src_grid, dst_grid) && return IdentityRegridding(src_grid, dst_grid; location)
    if src_grid.spacing < dst_grid.spacing
        return AverageCoarsening(src_grid, dst_grid; weights, location)
    end
    return refinement(src_grid, dst_grid; location, refinement_kwargs...)
end

# ------------------------------------------------------------------------------------------
# Pair validation
# ------------------------------------------------------------------------------------------

"""
    _nesting(fine, coarse, name, fine_label, alternative) -> r

Validate that `coarse` is `fine` coarsened by a power-of-2 ratio `r` with coincident
corners, and return `r`. `name`, `fine_label` (which argument should be the finer grid) and
`alternative` phrase the errors, e.g. when the grids are passed the wrong way round.
"""
function _nesting(fine::DyadicGrid{N}, coarse::DyadicGrid{N}, name, fine_label,
                  alternative) where {N}
    atol = max(fine.atol, coarse.atol)
    ρ = coarse.spacing / fine.spacing
    n = maximum(size(fine))
    if ρ < 1 && !_same_coordinate(n * fine.spacing, n * coarse.spacing, atol)
        throw(ArgumentError(
            "$name expects $fine_label to be the finer grid, but its spacing is " *
            "$(fine.spacing / coarse.spacing)× larger ($(fine.spacing) vs " *
            "$(coarse.spacing)). Did you mean $alternative?"))
    end
    r = max(round(Int, ρ), 1)
    extent = coarse.spacing * maximum(size(coarse))
    _same_coordinate(extent, r * fine.spacing * maximum(size(coarse)), atol) || throw(ArgumentError(
        "$name: spacings $(fine.spacing) and $(coarse.spacing) are not related by an " *
        "integer ratio (ratio $ρ)."))
    ispow2(r) || throw(ArgumentError(
        "$name: $(fine.spacing) → $(coarse.spacing) is a ratio of $r, not a power of 2."))
    for d in 1:N
        _same_coordinate(fine.origin[d], coarse.origin[d], atol) || throw(ArgumentError(
            "$name: the grid corners do not coincide along axis $d " *
            "($(fine.origin[d]) vs $(coarse.origin[d]))."))
    end
    size(fine) == size(coarse) .* r || throw(ArgumentError(
        "$name: a $(join(size(coarse), "×")) grid refined $r× has " *
        "$(join(size(coarse) .* r, "×")) cells, but the finer grid has " *
        "$(join(size(fine), "×"))."))
    return r
end

# ------------------------------------------------------------------------------------------
# Accessors and traits
# ------------------------------------------------------------------------------------------

"""
    grids(rgd) -> (grid1, grid2)

The two grids of a regridder, in the order they were passed.
"""
grids(rgd::AbstractRegridding) = (rgd.grid1, rgd.grid2)

"""
    ratio(rgd)

Spacing ratio between the coarser and the finer grid of `rgd`: a power of 2, and 1 for
identical grids.
"""
ratio(::AverageCoarsening{N,L,R}) where {N,L,R} = R
ratio(::ConstantRefinement{N,L,R}) where {N,L,R} = R
ratio(::LinearRefinement{N,L,R}) where {N,L,R} = R
ratio(::IdentityRegridding) = 1
ratio(rgd::BidirectionalRegridding) = ratio(rgd.to2)

"""
    isconservative(rgd)

Whether `rgd` preserves the integral of the field. True for every regridder in KryosTools.
"""
isconservative(::AbstractRegridding) = true
isconservative(rgd::BidirectionalRegridding) = isconservative(rgd.to1) && isconservative(rgd.to2)

"""
    islinear(rgd)

Whether `rgd` is a linear map of the field. Linear regridders have exact, cheap adjoints, so
gradients through them are exact.
"""
islinear(::IdentityRegridding) = true
islinear(::AverageCoarsening) = true
islinear(::ConstantRefinement) = true
islinear(rgd::LinearRefinement) = rgd.limiter === nothing
islinear(rgd::BidirectionalRegridding) = islinear(rgd.to1) && islinear(rgd.to2)

function Base.show(io::IO, rgd::BidirectionalRegridding)
    g1, g2 = grids(rgd)
    print(io, "BidirectionalRegridding(", join(size(g1), "×"), " ↔ ", join(size(g2), "×"),
          ": to1 = ", nameof(typeof(rgd.to1)), ", to2 = ", nameof(typeof(rgd.to2)), ")")
end

function Base.show(io::IO, rgd::AbstractRegridding)
    g1, g2 = grids(rgd)
    print(io, nameof(typeof(rgd)), "(", join(size(g1), "×"), " → ", join(size(g2), "×"))
    ratio(rgd) == 1 || print(io, ", ratio ", ratio(rgd))
    print(io, ")")
end

# ------------------------------------------------------------------------------------------
# Precision and index helpers
# ------------------------------------------------------------------------------------------

"""
    accumtype(T)

Element type used to accumulate a block sum. Adding many `Float16` values in sequence loses
accuracy well before a block is exhausted, so half precision accumulates in `Float32` and
narrows on write.
"""
accumtype(::Type{Float16}) = Float32
accumtype(::Type{T}) where {T} = T

# What a block sum adds up for each fine cell: the value itself, or weight × value. Callable
# structs rather than closures, so kernels capture nothing implicitly.
struct _Values{S}
    src::S
end
@inline (t::_Values)(idx::Vararg{Int}) = @inbounds t.src[idx...]

struct _WeightedValues{A,S,W}
    src::S
    weights::W
end
_WeightedValues{A}(src::S, weights::W) where {A,S,W} = _WeightedValues{A,S,W}(src, weights)
@inline (t::_WeightedValues{A})(idx::Vararg{Int}) where {A} =
    @inbounds A(t.weights[idx...]) * t.src[idx...]

"""
    _block_sum(term, I, Val(r), acc)

`acc` plus `term(i...)` summed over the `r^N` fine cells `i` covered by coarse cell `I`.
Loops are written out for 1 to 3 dimensions so that their bounds are constants; other
dimensions use a generic, slower loop. Callers guarantee that the fine arrays behind `term`
are `r` times larger than the coarse field.
"""
@inline function _block_sum(term, (i,)::NTuple{1,Int}, ::Val{R}, acc) where {R}
    i0 = (i - 1) * R
    for p in 1:R
        acc += term(i0 + p)
    end
    return acc
end

@inline function _block_sum(term, (i, j)::NTuple{2,Int}, ::Val{R}, acc) where {R}
    i0, j0 = (i - 1) * R, (j - 1) * R
    for q in 1:R, p in 1:R
        acc += term(i0 + p, j0 + q)
    end
    return acc
end

@inline function _block_sum(term, (i, j, k)::NTuple{3,Int}, ::Val{R}, acc) where {R}
    i0, j0, k0 = (i - 1) * R, (j - 1) * R, (k - 1) * R
    for s in 1:R, q in 1:R, p in 1:R
        acc += term(i0 + p, j0 + q, k0 + s)
    end
    return acc
end

@inline function _block_sum(term, I::NTuple{N,Int}, ::Val{R}, acc) where {N,R}
    base = ntuple(d -> (I[d] - 1) * R, Val(N))
    for Δ in CartesianIndices(ntuple(_ -> R, Val(N)))
        acc += term((base .+ Tuple(Δ))...)
    end
    return acc
end

"""
    _block_fill!(dst, fill, I, Val(r))

Write `fill(p...)` into the `r^N` fine cells of `dst` covered by coarse cell `I`, where `p`
is the position of the fine cell within the block (from 1 to `r` along each axis). Loops are
written out for 1 to 3 dimensions so that their bounds are constants. Callers guarantee that
`dst` is `r` times larger than the coarse field.
"""
@inline function _block_fill!(dst::AbstractArray{<:Any,1}, fill, (i,)::NTuple{1,Int},
                              ::Val{R}) where {R}
    i0 = (i - 1) * R
    for p in 1:R
        @inbounds dst[i0 + p] = fill(p)
    end
    return nothing
end

@inline function _block_fill!(dst::AbstractArray{<:Any,2}, fill, (i, j)::NTuple{2,Int},
                              ::Val{R}) where {R}
    i0, j0 = (i - 1) * R, (j - 1) * R
    for q in 1:R, p in 1:R
        @inbounds dst[i0 + p, j0 + q] = fill(p, q)
    end
    return nothing
end

@inline function _block_fill!(dst::AbstractArray{<:Any,3}, fill, (i, j, k)::NTuple{3,Int},
                              ::Val{R}) where {R}
    i0, j0, k0 = (i - 1) * R, (j - 1) * R, (k - 1) * R
    for s in 1:R, q in 1:R, p in 1:R
        @inbounds dst[i0 + p, j0 + q, k0 + s] = fill(p, q, s)
    end
    return nothing
end

@inline function _block_fill!(dst::AbstractArray{<:Any,N}, fill, I::NTuple{N,Int},
                              ::Val{R}) where {N,R}
    base = ntuple(d -> (I[d] - 1) * R, Val(N))
    for Δ in CartesianIndices(ntuple(_ -> R, Val(N)))
        p = Tuple(Δ)
        @inbounds dst[(base .+ p)...] = fill(p...)
    end
    return nothing
end

# What a refinement writes into each fine cell of a block: the coarse value, or the coarse
# value plus the linear reconstruction offsets.
struct _ConstantFill{V}
    value::V
end
@inline (f::_ConstantFill)(::Vararg{Int}) = f.value

struct _LinearFill{R,N,A}
    value::A
    slopes::NTuple{N,A}
end
_LinearFill{R}(value::A, slopes::NTuple{N,A}) where {R,N,A} = _LinearFill{R,N,A}(value, slopes)

@inline function (f::_LinearFill{R,N,A})(p::Vararg{Int,N}) where {R,N,A}
    return f.value + _tuplesum(ntuple(@inline(d -> f.slopes[d] * _child_offset(p[d], R, A)), Val(N)))
end

# Centre of child `p` of `r` relative to its parent's centre, in units of the parent's width.
# Exact in binary for power-of-2 `r`, and summing to zero over `p = 1:r`.
@inline _child_offset(p::Int, r::Int, ::Type{A}) where {A} = A(2p - 1 - r) / A(2r)

@inline _tuplesum(t::Tuple) = sum(t)
@inline _tuplesum(t::Tuple{Any}) = t[1]
@inline _tuplesum(t::Tuple{Any,Any}) = t[1] + t[2]
@inline _tuplesum(t::Tuple{Any,Any,Any}) = t[1] + t[2] + t[3]

"""
    _slope(src, I, Val(d), limiter, A)

Change of `src` across coarse cell `I` along axis `d`, in units of one coarse cell: the
(optionally limited) centred difference, or zero in the outermost cells of the axis. The axis
is a compile-time constant and the edge test is branchless (neighbour indices are clamped and
the result masked), which is several times faster on the CPU and GPU-friendly.
"""
@inline function _slope(src, I::NTuple{N,Int}, ::Val{d}, limiter, ::Type{A}) where {N,d,A}
    i = I[d]
    n = size(src, d)
    @inbounds here = A(src[I...])
    @inbounds left = A(src[Base.setindex(I, max(i - 1, 1), d)...])
    @inbounds right = A(src[Base.setindex(I, min(i + 1, n), d)...])
    return ifelse(1 < i < n, _limited_slope(limiter, right - here, here - left), zero(A))
end

@inline _limited_slope(::Nothing, Δright, Δleft) = (Δright + Δleft) / 2
@inline _limited_slope(::MinModLimiter, Δright, Δleft) =
    (sign(Δright) + sign(Δleft)) / 2 * min(abs(Δright), abs(Δleft))

@inline _slopes(src, I::NTuple{N,Int}, limiter, ::Type{A}) where {N,A} =
    ntuple(@inline(d -> _slope(src, I, Val(d), limiter, A)), Val(N))

# ------------------------------------------------------------------------------------------
# regrid!
# ------------------------------------------------------------------------------------------

"""
    regrid!(dst, src, rgd) -> dst

Regrid the field `src`, defined on `grid1` of `rgd`, onto `grid2`, writing into `dst`.
Allocates nothing. `src` and `dst` may have different element types, but must live on the
same KernelAbstractions backend.

Like any KernelAbstractions kernel, the call is asynchronous on GPUs: call
`KernelAbstractions.synchronize(backend)` before timing it or reading `dst` on the host.
"""
function regrid! end

"""
    regrid(src, rgd) -> dst

Allocating form of [`regrid!`](@ref): returns a new array on `grid2` with the element type
and backend of `src`. A convenience; use `regrid!` in hot loops.
"""
function regrid(src::AbstractArray, rgd::AbstractRegridding)
    dst = similar(src, size(rgd.grid2))
    return regrid!(dst, src, rgd)
end

function _check_fields(dst, src, rgd)
    g1, g2 = grids(rgd)
    size(src) == size(g1) || throw(DimensionMismatch(
        "$(nameof(typeof(rgd))): the source field has size $(size(src)), but grid1 is " *
        "$(join(size(g1), "×"))."))
    size(dst) == size(g2) || throw(DimensionMismatch(
        "$(nameof(typeof(rgd))): the destination field has size $(size(dst)), but grid2 " *
        "is $(join(size(g2), "×"))."))
    typeof(get_backend(dst)) == typeof(get_backend(src)) || throw(ArgumentError(
        "$(nameof(typeof(rgd))): source and destination live on different backends " *
        "($(typeof(get_backend(src))) and $(typeof(get_backend(dst))))."))
    return nothing
end

function regrid!(dst::AbstractArray, src::AbstractArray, rgd::IdentityRegridding)
    _check_fields(dst, src, rgd)
    dst === src || copyto!(dst, src)
    return dst
end

function regrid!(dst::AbstractArray, src::AbstractArray, rgd::AverageCoarsening)
    _check_fields(dst, src, rgd)
    _average_coarsening!(get_backend(dst), dst, src, rgd, rgd.weights)
    return dst
end

function _average_coarsening!(backend, dst, src, rgd::AverageCoarsening{N,L,R},
                              ::Nothing) where {N,L,R}
    inv_volume = one(accumtype(eltype(dst))) / R^N
    kernel! = _average_coarsening_kernel!(backend)
    kernel!(dst, src, Val(R), inv_volume; ndrange = size(dst))
    return nothing
end

function _average_coarsening!(backend, dst, src, rgd::AverageCoarsening{N,L,R},
                              weights::AbstractArray) where {N,L,R}
    typeof(get_backend(weights)) == typeof(backend) || throw(ArgumentError(
        "AverageCoarsening: the weights live on $(typeof(get_backend(weights))), but the " *
        "fields on $(typeof(backend))."))
    kernel! = _weighted_coarsening_kernel!(backend)
    kernel!(dst, src, weights, rgd.inv_weightsum, Val(R), zero(accumtype(eltype(dst)));
            ndrange = size(dst))
    return nothing
end

@kernel function _average_coarsening_kernel!(dst, @Const(src), ratio, inv_volume)
    I = @index(Global, NTuple)
    @inbounds dst[I...] = _block_sum(_Values(src), I, ratio, zero(inv_volume)) * inv_volume
end

@kernel function _weighted_coarsening_kernel!(dst, @Const(src), @Const(weights),
                                              @Const(inv_weightsum), ratio, acc)
    I = @index(Global, NTuple)
    A = typeof(acc)
    total = _block_sum(_WeightedValues{A}(src, weights), I, ratio, acc)
    @inbounds dst[I...] = total * A(inv_weightsum[I...])
end

function regrid!(dst::AbstractArray, src::AbstractArray,
                 rgd::ConstantRefinement{N,L,R}) where {N,L,R}
    _check_fields(dst, src, rgd)
    kernel! = _constant_refinement_kernel!(get_backend(dst))
    kernel!(dst, src, Val(R); ndrange = size(src))
    return dst
end

function regrid!(dst::AbstractArray, src::AbstractArray,
                 rgd::LinearRefinement{N,L,R}) where {N,L,R}
    _check_fields(dst, src, rgd)
    kernel! = _linear_refinement_kernel!(get_backend(dst))
    kernel!(dst, src, Val(R), rgd.limiter, zero(accumtype(eltype(dst)));
            ndrange = size(src))
    return dst
end

# Refinement kernels run one workitem per *coarse* cell, writing its r^N children: the
# children of different coarse cells never overlap, and the slopes are computed once per
# block rather than once per fine cell.
@kernel function _constant_refinement_kernel!(dst, @Const(src), ratio)
    I = @index(Global, NTuple)
    @inbounds _block_fill!(dst, _ConstantFill(src[I...]), I, ratio)
end

@kernel function _linear_refinement_kernel!(dst, @Const(src), ratio::Val{R}, limiter,
                                            acc) where {R}
    I = @index(Global, NTuple)
    A = typeof(acc)
    @inbounds value = A(src[I...])
    _block_fill!(dst, _LinearFill{R}(value, _slopes(src, I, limiter, A)), I, ratio)
end

# The pair of field sizes decides the direction: checking only one field would silently
# accept two fields from the same grid. Nested grids of equal size are identical, so the
# check is unambiguous; for identical grids both directions are copies.
function regrid!(dst::AbstractArray, src::AbstractArray, rgd::BidirectionalRegridding)
    n1, n2 = size(rgd.grid1), size(rgd.grid2)
    size(src) == n2 && size(dst) == n1 && return regrid!(dst, src, rgd.to1)
    size(src) == n1 && size(dst) == n2 && return regrid!(dst, src, rgd.to2)
    throw(_direction_mismatch(size(dst), size(src), rgd))
end

function regrid(src::AbstractArray, rgd::BidirectionalRegridding)
    size(src) == size(rgd.grid2) && return regrid(src, rgd.to1)
    size(src) == size(rgd.grid1) && return regrid(src, rgd.to2)
    throw(_direction_mismatch(nothing, size(src), rgd))
end

function _direction_mismatch(dst_size, src_size, rgd)
    n1, n2 = join(size(rgd.grid1), "×"), join(size(rgd.grid2), "×")
    fields = dst_size === nothing ? "a source field of size $src_size" :
             "fields of size $dst_size (destination) and $src_size (source)"
    return DimensionMismatch(
        "BidirectionalRegridding between $n1 and $n2 cells cannot regrid $fields: " *
        "one field must be on each grid.")
end
