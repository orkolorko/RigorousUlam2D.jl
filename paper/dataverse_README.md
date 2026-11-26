# Certified stationary densities of random Hénon and contracting Lorenz maps

Data for M. J. Pacifico, R. Bioni Liberalquino, I. Nisoli, *On the existence, regularity and
rigorous computation of stationary densities of systems with additive BV noise*: the computed
stationary densities `f̃` of Theorem E, one for each row of its Table 1, and the records every
number of that table was read from.  The code that wrote every file is RigorousUlam2D.jl,
https://github.com/orkolorko/RigorousUlam2D.jl, at the commits named below.

## The systems

Three maps of the torus, each composed with uniform noise on the square `[-a, a]²` (periodic);
the files name each by its key:

- `henon`: `H(X, Y) = (1 − (7/5) X² + Y, (3/10) X)` in the coordinates `X = 3x/2`, `Y = y/2`, on
  the domain `[-1, 1]²`;
- `lorenz`: on `[-1/2, 1/2]²`, for `x > 0`, `F(x, y) = (1/2 − (13/10) x^(1/2), y x³ − 1/4)`, for
  `x < 0`, `F(x, y) = ((13/10) |x|^(1/2) − 1/2, −y |x|³ + 1/4)`;
- `mlorenz_rho18-5`: on `[-1/2, 1/2]²`, for `x > 0`, `F(x, y) = ((18/5) x² − 1/2, y x⁶ − 1/4)`, for
  `x < 0`, `F(x, y) = (1/2 − (18/5) x², −y x⁶ + 1/4)`.

The noise half-widths `a` are the dyadic rationals 77/256, 51/256, 51/512 and 13/256.

## The densities

Each density is given on the uniform grid of depth 12 of the domain: 4096 × 4096 square cells,
of side `2^-11` for `henon` and `2^-12` for the other two.  It is stored as the vector of the `2^24`
masses of `f̃` on the cells, as IEEE 754 double-precision numbers; on each cell `f̃` is constant,
equal to the mass of the cell divided by its area.  The mass of the cell in column `ix` (along
`x`) and row `iy` (along `y`), counted from 1 at the lower left corner, is entry
`(ix − 1) · 4096 + iy` of the vector.

Each density is given in two files with the same masses, bit for bit:

- `densities/figures.density_<map>_a<p>-<q>_d12.<commit>.<host>.h5`: HDF5, the dataset `masses`
  (Float64, length `2^24`) and the attributes `map`, `depth`, `half_width` (`"p/q"`),
  `half_width_numerator`, `half_width_denominator`, `domain` (`[xmin, xmax, ymin, ymax]`),
  `ordering` (the rule above, in words), `source` (the name of the `.jls` file) and `header` (the
  header of the run that computed it).  Readable with any HDF5 library, e.g. `h5py` or HDF5.jl.
- `densities/stationary_density.<map>_d12_a<p>-<q>.<commit>.<host>.jls`: the record as computed,
  written with the `Serialization` standard library of Julia 1.13: a named tuple with fields
  `map`, `depth`, `a` (a `Rational{Int}`), `density` (the masses) and `header`.  `Serialization`
  does not guarantee that another Julia version can read it; the HDF5 file is the archival copy.

## The records

`records/` holds the CertifiedRecords records (tab-separated text) the table of bounds was read
from: the residuals `ε₁ ≥ ‖P f̃ − f̃‖₁` and `ε₂ ≥ |Σ f̃ − 1|` of each density
(`stationary_density.*.tsv`), the bounds on the norms of the powers of the coarse operator on the
functions of zero average (`l1_mixing.*.tsv`) and the L¹ error bounds (`l1_transfer.*.tsv`).
Each bound in a record is written as the exact decimal expansion of a double.  Each record starts with `#` lines naming the driver, the machine, the date, the commit of
RigorousUlam2D.jl, the Julia version and the thread count.  The flag `(working tree dirty)` in
these headers is set when `git status --porcelain` lists anything, untracked files included.

`environments/` holds the `Manifest.toml` of a run, for every row of a record that names one in
its column `environment`.  The records written before that column existed name only the commit;
the environment of such a run is the `Manifest.toml` of the package at that commit.

`figures/` holds the table of bounds as LaTeX (`figures.table_bounds.*.tex`), the figures of the
densities (`figures.density_*.pdf`) and the list of the records the table and the figures were
read from (`figures.sources.*.txt`), with the SHA-256 of every density file.

<!-- MATRICES -->

## Checksums

`MANIFEST.sha256` gives the SHA-256 of every file, by its path in the dataset; `sha256sum -c
MANIFEST.sha256` run at the root of a download checks them.

<!-- GENERATED -->
