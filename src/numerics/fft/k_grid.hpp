#ifndef NUMERICS_FFT_K_GRID_HPP
#define NUMERICS_FFT_K_GRID_HPP

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <limits>
#include <utility>
#include <vector>

#include "configuration.hpp"
#include "nda/nda.hpp"
#include "numerics/fft/nda.hpp"
#include "utilities/check.hpp"

namespace math::fft {

// Transform a complete, possibly shifted and permuted k grid to the R ordering
// used by utils::k_to_R_coefficients(range(nk), ..., grid, ...).
class k_grid_forward {
public:
  template<::nda::ArrayOfRank<2> KPoints, ::nda::ArrayOfRank<1> Grid>
  k_grid_forward(KPoints const& kpts, ::nda::stack_array<double, 3, 3> const& lattv,
                 Grid const& grid, long columns)
      : dims{long(grid(0)), long(grid(1)), long(grid(2))},
        k_to_slot(kpts.shape(0)), phase(kpts.shape(0)), ncols(columns) {
    utils::check(dims[0] > 0 && dims[1] > 0 && dims[2] > 0 &&
                     dims[0] <= std::numeric_limits<long>::max() / dims[1] &&
                     dims[0] * dims[1] <= std::numeric_limits<long>::max() / dims[2],
                 "k_grid_forward: invalid grid dimensions");
    long nk = dims[0] * dims[1] * dims[2];
    utils::check(kpts.shape(0) == nk && kpts.shape(1) == 3,
                 "k_grid_forward: k-points do not match grid dimensions");
    for (long d = 0; d < 3; ++d)
      for (long c = 0; c < 3; ++c)
        utils::check(std::isfinite(lattv(d, c)), "k_grid_forward: non-finite lattice vector");
    for (long ik = 0; ik < nk; ++ik)
      for (long c = 0; c < 3; ++c)
        utils::check(std::isfinite(kpts(ik, c)), "k_grid_forward: non-finite k-point {}", ik);

    constexpr double two_pi = 6.283185307179586476925286766559;
    auto scaled_coordinate = [&](long ik, long d) {
      double crystal = 0.0;
      double magnitude = 0.0;
      for (long c = 0; c < 3; ++c) {
        double term = lattv(d, c) * kpts(ik, c) / two_pi;
        crystal += term;
        magnitude += std::abs(term);
      }
      double scaled = dims[d] * crystal;
      double scaled_magnitude = dims[d] * magnitude;
      utils::check(std::isfinite(scaled) && std::isfinite(scaled_magnitude),
                   "k_grid_forward: non-finite crystal coordinate at k-point {}", ik);
      return std::pair{scaled, scaled_magnitude};
    };
    std::array<double, 3> offset{};
    std::array<double, 3> offset_scale{};
    for (long d = 0; d < 3; ++d) {
      auto [scaled, magnitude] = scaled_coordinate(0, d);
      offset[d] = scaled - std::nearbyint(scaled);
      offset_scale[d] = magnitude;
    }

    std::vector<bool> occupied(nk, false);
    for (long ik = 0; ik < nk; ++ik) {
      std::array<long, 3> index{};
      for (long d = 0; d < 3; ++d) {
        auto [coordinate, magnitude] = scaled_coordinate(ik, d);
        double scaled = coordinate - offset[d];
        double nearest_real = std::nearbyint(scaled);
        // Allow only coordinate-conversion roundoff, not a displaced k-point.
        // The cap conservatively rejects large coordinates whose roundoff is ambiguous.
        double tolerance = std::min(1e-10, 16 * std::numeric_limits<double>::epsilon() *
                                             (1 + magnitude + offset_scale[d]));
        if (std::abs(scaled - nearest_real) > tolerance ||
            std::abs(nearest_real) > double(std::numeric_limits<long>::max() / 2)) {
          compatible_ = false;
          return;
        }
        long nearest = static_cast<long>(nearest_real);
        index[d] = (nearest % dims[d] + dims[d]) % dims[d];
      }
      long slot = (index[0] * dims[1] + index[1]) * dims[2] + index[2];
      if (occupied[slot]) {
        compatible_ = false;
        return;
      }
      occupied[slot] = true;
      k_to_slot[ik] = slot;
    }

    for (long r = 0; r < nk; ++r) {
      std::array<long, 3> index{r / (dims[1] * dims[2]),
                                (r / dims[2]) % dims[1], r % dims[2]};
      double argument = 0.0;
      for (long d = 0; d < 3; ++d) {
        if (index[d] > dims[d] / 2) index[d] -= dims[d];
        argument += offset[d] * index[d] / dims[d];
      }
      phase[r] = std::exp(ComplexType(0.0, -two_pi * argument));
    }

    if (ncols > 0) {
      scratch.resize(std::array<long, 4>{ncols, dims[0], dims[1], dims[2]});
      plan = create_plan_many(scratch, FFT_MEASURE | FFT_DESTROY_INPUT);
    }
  }

  k_grid_forward(k_grid_forward const&) = delete;
  k_grid_forward& operator=(k_grid_forward const&) = delete;

  ~k_grid_forward() { destroy_plan(plan); }

  bool can_fft() const noexcept { return compatible_; }

  void transform(::nda::MemoryArrayOfRank<2> auto&& values) {
    utils::check(compatible_, "k_grid_forward: mesh is not FFT-compatible");
    utils::check(values.shape(0) == long(k_to_slot.size()) && values.shape(1) == ncols,
                 "k_grid_forward: input shape mismatch");
    if (ncols == 0) return;

    auto packed = ::nda::reshape(scratch, std::array<long, 2>{ncols, long(k_to_slot.size())});
    // Bound the number of simultaneously touched FFT columns during the transpose.
    // This is a cache tile, independent of the MPI/node topology.
    constexpr long column_tile = 32;
    for (long first = 0; first < ncols; first += column_tile)
      for (long ik = 0; ik < long(k_to_slot.size()); ++ik)
        for (long col = first; col < std::min(first + column_tile, ncols); ++col)
          packed(col, k_to_slot[ik]) = values(ik, col);

    fwdfft(plan, scratch);

    for (long first = 0; first < ncols; first += column_tile)
      for (long r = 0; r < long(phase.size()); ++r)
        for (long col = first; col < std::min(first + column_tile, ncols); ++col)
          values(r, col) = phase[r] * packed(col, r);
  }

private:
  std::array<long, 3> dims;
  std::vector<long> k_to_slot;
  std::vector<ComplexType> phase;
  long ncols;
  bool compatible_ = true;
  ::nda::array<ComplexType, 4> scratch;
  fftplan_t plan{};
};

} // namespace math::fft

#endif
