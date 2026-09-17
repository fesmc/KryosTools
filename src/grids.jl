#=
Grids.

A grid is metadata: sizes, spacing and coordinates. It never enters a kernel, so it is free
to hold whatever is convenient on the host, and its coordinate precision is independent of
the precision of the fields defined on it.
=#

"""
    AbstractGrid{N,T}

An `N`-dimensional grid with coordinates of type `T`.
"""
abstract type AbstractGrid{N,T} end

"""
    DyadicGrid(x, y; area = nothing)
    DyadicGrid(coords::AbstractVector...; area = nothing)

A regular, isotropic Cartesian grid built from the **cell-centre** coordinate vectors that a
coupled module already has.

"Dyadic" refers to how such grids are paired: two grids can be regridded onto each other when
their spacings differ by a power of 2 and their corners coincide, so that every coarse cell
contains a whole number of fine cells with perfect overlap. Those are properties of a *pair*
and are checked when a regridder is built. A single grid is validated for what it can know
about itself:

- at least 2 points per axis;
- strictly increasing coordinates (descending axes throw — flipping an axis also means
  flipping the data, which is the caller's job);
- uniform spacing, and the same spacing along every axis.

Coordinates may be of any real type and may live on a GPU. Validation allows for the
rounding of the input type — `Float32` coordinates at several thousand kilometres are
accepted — and the grid then stores exact `Float64` ranges fitted to the input. `centers(g)`
may therefore differ from the vectors passed in by rounding noise.

# Cell area

`area`, if given, is the true area of each cell in the coordinate unit squared (m² for metre
coordinates), for example to account for the distortion of a polar stereographic
projection. It must match the grid size and be strictly positive. It is stored as `Float64`
on the device of the array passed in.

The area is never used implicitly. To coarsen area-conservatively, pass it explicitly as
weights. Since regridding only uses *relative* areas, any constant factor cancels.

A warning is emitted if the mean of `area / spacing^N` lies outside [0.5, 2]. That usually
means a dimensionless distortion factor was passed where true areas were expected.

# Examples
```jldoctest
julia> x = 500.0:1000.0:7500.0;  # 8 cell centres, 1 km apart

julia> g = DyadicGrid(x, x)
8×8 DyadicGrid with spacing 1000.0 and origin (0.0, 0.0)

julia> spacing(g), size(g)
(1000.0, (8, 8))
```
"""
struct DyadicGrid{N,T<:AbstractFloat,C,F,A} <: AbstractGrid{N,T}
    size::NTuple{N,Int}
    spacing::T
    origin::NTuple{N,T}      # lower-left corner
    centers::C               # NTuple{N,<:AbstractRange{T}}, length size[d]
    faces::F                 # NTuple{N,<:AbstractRange{T}}, length size[d] + 1
    area::A                  # nothing (uniform spacing^N) | array of true cell areas
    atol::T                  # absolute coordinate tolerance, from the input's precision

    # The only way to build a grid: centres and faces are always derived, so an inconsistent
    # grid cannot exist.
    function DyadicGrid(dims::NTuple{N,Int}, spacing::T, origin::NTuple{N,T}, area,
                        atol::T) where {N,T<:AbstractFloat}
        all(≥(1), dims) || throw(ArgumentError("DyadicGrid: every size must be at least 1, got $dims."))
        spacing > 0 || throw(ArgumentError("DyadicGrid: spacing must be positive, got $spacing."))
        centers = ntuple(d -> range(origin[d] + spacing / 2; step = spacing, length = dims[d]), N)
        faces = ntuple(d -> range(origin[d]; step = spacing, length = dims[d] + 1), N)
        return new{N,T,typeof(centers),typeof(faces),typeof(area)}(
            dims, spacing, origin, centers, faces, area, atol)
    end
end

# Relative tolerance on positions, as a fraction of the spacing. Dyadic grids are spaced
# hundreds of metres apart, so any genuine misalignment or stretching is a large fraction of
# a cell; 1e-3 of a cell is far below that and far above floating-point noise.
const COORDINATE_RTOL = 1e-3

# How many units of the input's machine epsilon (relative to its largest magnitude) are
# tolerated on top, so that e.g. Float32 coordinates far from the origin still validate.
const COORDINATE_EPS_FACTOR = 8

_coordinate_eps(::Type{T}) where {T<:AbstractFloat} = Float64(eps(T))
_coordinate_eps(::Type{<:Real}) = 0.0

"""
    _same_coordinate(a, b, atol)

The one comparison used by every geometric check — grid uniformity, identity, corner
alignment and ratios — so that those checks can never disagree about whether two positions
coincide.
"""
_same_coordinate(a, b, atol) = abs(a - b) <= atol

function DyadicGrid(coords::AbstractVector...; area = nothing)
    N = length(coords)
    N ≥ 1 || throw(ArgumentError("DyadicGrid: at least one coordinate vector is required."))

    # Coordinates are few; validate them on the host whatever device they live on.
    host = ntuple(d -> Float64.(Array(coords[d])), N)
    for d in 1:N
        length(host[d]) ≥ 2 || throw(ArgumentError(
            "DyadicGrid: axis $d has $(length(host[d])) point(s); at least 2 are needed " *
            "to determine the spacing."))
        all(isfinite, host[d]) || throw(ArgumentError(
            "DyadicGrid: axis $d contains non-finite coordinates."))
    end
    maxabs = maximum(h -> maximum(abs, h), host)
    in_eps = maximum(d -> _coordinate_eps(eltype(coords[d])), 1:N)
    tolerance(s) = max(COORDINATE_RTOL * s, COORDINATE_EPS_FACTOR * in_eps * maxabs)

    # 1. Each axis on its own: increasing and uniform.
    axis_spacing = ntuple(d -> _fit_spacing(host[d]), N)
    for d in 1:N
        h, s = host[d], axis_spacing[d]
        s > 0 || throw(ArgumentError(
            "DyadicGrid: axis $d is not increasing (fitted spacing $s). Reverse the axis — " *
            "and the data along it — before building the grid."))
        worst, iworst = _worst_residual(h, _mean_first_centre(h, s), s)
        _same_coordinate(worst, 0.0, tolerance(s)) || throw(ArgumentError(
            "DyadicGrid: axis $d is not uniformly spaced: point $iworst is $worst away " *
            "from the best uniform fit with spacing $s (tolerance $(tolerance(s)))."))
    end

    # 2. All axes together: one shared spacing (isotropy).
    fitted_dx = _fit_common_spacing(host)
    atol = tolerance(fitted_dx)
    fits(dx) = all(h -> _same_coordinate(first(_worst_residual(h, _mean_first_centre(h, dx), dx)),
                                         0.0, atol), host)
    fits(fitted_dx) || throw(ArgumentError(
        "DyadicGrid: the axes have different spacings $axis_spacing; a dyadic grid is " *
        "isotropic (dx == dy)."))

    # 3. The simplest spacing and origin that still fit every point, so that e.g. Float32
    #    coordinates yield exactly the same grid as their Float64 counterparts.
    dx = _simplest(fitted_dx, atol, fits)
    origin = ntuple(N) do d
        h = host[d]
        _simplest(_mean_first_centre(h, dx) - dx / 2, atol,
                  o -> _same_coordinate(first(_worst_residual(h, o + dx / 2, dx)), 0.0, atol))
    end

    dims = ntuple(d -> length(host[d]), N)
    return DyadicGrid(dims, dx, origin, _validate_area(area, dims, dx), atol)
end

function _fit_spacing(h)
    n = length(h)
    ī = (n + 1) / 2
    h̄ = sum(h) / n
    return sum((i - ī) * (h[i] - h̄) for i in 1:n) / sum((i - ī)^2 for i in 1:n)
end

# Joint least-squares spacing over all axes.
function _fit_common_spacing(host)
    num = 0.0
    den = 0.0
    for h in host
        n = length(h)
        ī = (n + 1) / 2
        h̄ = sum(h) / n
        num += sum((i - ī) * (h[i] - h̄) for i in 1:n)
        den += sum((i - ī)^2 for i in 1:n)
    end
    return num / den
end

# First centre of an axis for a given spacing, averaged over all points so that rounding in
# the input does not bias it.
_mean_first_centre(h, s) = sum(h[i] - (i - 1) * s for i in eachindex(h)) / length(h)

# Largest deviation from the uniform axis `c + (i-1)s`, and where it occurs.
_worst_residual(h, c, s) = findmax(i -> abs(h[i] - (c + (i - 1) * s)), eachindex(h))

# `x` rounded to the coarsest decimal unit for which `fits` still holds.
function _simplest(x, atol, fits)
    kmax = floor(Int, log10(max(abs(x), atol))) + 1
    for k in kmax:-1:-15
        y = round(x; digits = -k)
        fits(y) && return y
    end
    return x
end

_validate_area(::Nothing, dims, spacing) = nothing

function _validate_area(area::AbstractArray, dims, spacing)
    size(area) == dims || throw(ArgumentError(
        "DyadicGrid: area has size $(size(area)) but the grid has $dims cells."))
    stored = Float64.(area)
    all(a -> isfinite(a) && a > 0, stored) || throw(ArgumentError(
        "DyadicGrid: area must be finite and strictly positive everywhere."))
    relative = sum(stored) / length(stored) / spacing^length(dims)
    if !(0.5 ≤ relative ≤ 2)
        @warn "DyadicGrid: the mean cell area is $(relative) × spacing^$(length(dims)). " *
              "`area` should hold true cell areas (e.g. m²), not a dimensionless " *
              "distortion factor."
    end
    return stored
end

Base.size(g::DyadicGrid) = g.size
Base.size(g::DyadicGrid, d::Integer) = g.size[d]
Base.ndims(::DyadicGrid{N}) where {N} = N

"""
    spacing(g::DyadicGrid)

Cell size, identical along every axis.
"""
spacing(g::DyadicGrid) = g.spacing

"""
    origin(g::DyadicGrid)

Coordinates of the grid's lower-left corner.
"""
origin(g::DyadicGrid) = g.origin

"""
    centers(g::DyadicGrid)

Cell-centre coordinates along each axis, as a tuple of ranges.
"""
centers(g::DyadicGrid) = g.centers

"""
    faces(g::DyadicGrid)

Cell-face coordinates along each axis, as a tuple of ranges one element longer than
[`centers`](@ref).
"""
faces(g::DyadicGrid) = g.faces

"""
    cellarea(g::DyadicGrid, I...)

True area of cell `I`: `spacing(g)^N` when the grid was built without `area`, and the stored
area otherwise. A host-side convenience; it indexes the area array directly.
"""
cellarea(g::DyadicGrid, I...) = _cellarea(g.area, g, I...)
_cellarea(::Nothing, g::DyadicGrid{N}, I...) where {N} = g.spacing^N
_cellarea(area::AbstractArray, g::DyadicGrid, I...) = area[I...]

function Base.show(io::IO, g::DyadicGrid)
    dims = length(g.size) == 1 ? "$(only(g.size))-cell" : join(g.size, "×")
    print(io, dims, " DyadicGrid with spacing ", g.spacing,
          " and origin ", g.origin)
    g.area === nothing || print(io, ", with cell areas")
end

# Grids are compared geometrically, never by the arrays they hold.
function _same_geometry(g1::DyadicGrid, g2::DyadicGrid)
    ndims(g1) == ndims(g2) && size(g1) == size(g2) || return false
    atol = max(g1.atol, g2.atol)
    # Equal spacing means the far edges coincide, not just the first step.
    n = maximum(size(g1))
    _same_coordinate(n * g1.spacing, n * g2.spacing, atol) || return false
    return all(d -> _same_coordinate(g1.origin[d], g2.origin[d], atol), 1:ndims(g1))
end
