#include <array>
#include <cmath>
#include <complex>

#include "catch2/catch.hpp"
#include "configuration.hpp"
#include "nda/nda.hpp"
#include "numerics/fft/k_grid.hpp"
#include "utilities/kpoint_utils.hpp"
#include "utilities/test_common.hpp"

namespace {

void compare_grid(std::array<long, 3> dims, std::array<double, 3> shift) {
  constexpr double two_pi = 6.283185307179586476925286766559;
  nda::stack_array<double, 3, 3> lattv{{2.0, 0.3, 0.0},
                                         {0.0, 3.0, 0.2},
                                         {0.0, 0.0, 4.0}};
  nda::stack_array<long, 3> grid{dims[0], dims[1], dims[2]};
  long nk = dims[0] * dims[1] * dims[2];
  constexpr long columns = 35;
  nda::array<double, 2> kpts(nk, 3);
  nda::array<long, 2> rpts(nk, 3);

  for (long ik = 0; ik < nk; ++ik) {
    long slot = (5 * ik + 1) % nk; // A permutation for both grids below.
    std::array<long, 3> index{slot / (dims[1] * dims[2]),
                              (slot / dims[2]) % dims[1], slot % dims[2]};
    for (long d = 2; d >= 0; --d) {
      if (index[d] > dims[d] / 2) index[d] -= dims[d];
      double component = two_pi * (index[d] + shift[d]) / dims[d];
      for (long c = d + 1; c < 3; ++c) component -= lattv(d, c) * kpts(ik, c);
      kpts(ik, d) = component / lattv(d, d);
    }
  }
  for (long r = 0; r < nk; ++r) {
    std::array<long, 3> index{r / (dims[1] * dims[2]),
                              (r / dims[2]) % dims[1], r % dims[2]};
    for (long d = 0; d < 3; ++d) {
      if (index[d] > dims[d] / 2) index[d] -= dims[d];
      rpts(r, d) = index[d];
    }
  }

  nda::matrix<ComplexType> f_Rk(nk, nk);
  utils::k_to_R_coefficients(rpts, kpts, lattv, f_Rk);
  math::fft::k_grid_forward fft(kpts, lattv, grid, columns);
  REQUIRE(fft.can_fft());

  for (long repeat = 0; repeat < 2; ++repeat) {
    nda::matrix<ComplexType> gp(nk, columns), gn(nk, columns);
    for (long ik = 0; ik < nk; ++ik)
      for (long col = 0; col < columns; ++col) {
        gp(ik, col) = {0.25 + 0.1 * ik + 0.2 * col + repeat,
                       -0.3 + 0.15 * ik - 0.05 * col};
        gn(ik, col) = {-0.1 + 0.07 * ik - 0.2 * col,
                       0.4 - 0.04 * ik + 0.1 * repeat};
      }

    nda::matrix<ComplexType> gp_ref(nk, columns), gn_ref(nk, columns);
    nda::blas::gemm(ComplexType(1.0), f_Rk, gp, ComplexType(0.0), gp_ref);
    nda::blas::gemm(ComplexType(1.0), f_Rk, gn, ComplexType(0.0), gn_ref);
    fft.transform(gp);
    fft.transform(gn);
    utils::ARRAY_EQUAL(gp, gp_ref);
    utils::ARRAY_EQUAL(gn, gn_ref);

    nda::matrix<ComplexType> pi_fft(nk, columns), pi_ref(nk, columns);
    for (long r = 0; r < nk; ++r)
      for (long col = 0; col < columns; ++col) {
        pi_fft(r, col) = -2.0 * gp(r, col) * std::conj(gn(r, col));
        pi_ref(r, col) = -2.0 * gp_ref(r, col) * std::conj(gn_ref(r, col));
      }
    utils::ARRAY_EQUAL(pi_fft, pi_ref);
  }
}

} // namespace

TEST_CASE("k_grid_forward", "[fft][k_grid]") {
  compare_grid({2, 2, 2}, {0.0, 0.0, 0.0});
  compare_grid({3, 2, 1}, {0.5, 0.25, 0.0});

  nda::stack_array<double, 3, 3> lattv{{1.0, 0.0, 0.0},
                                         {0.0, 1.0, 0.0},
                                         {0.0, 0.0, 1.0}};
  nda::stack_array<long, 3> grid{2, 1, 1};
  nda::array<double, 2> kpts{{0.0, 0.0, 0.0}, {3.141592653589793, 0.0, 0.0}};
  nda::matrix<ComplexType> empty(2, 0);
  math::fft::k_grid_forward no_columns(kpts, lattv, grid, 0);
  REQUIRE(no_columns.can_fft());
  no_columns.transform(empty);
}

TEST_CASE("k_grid_forward_falls_back_for_off_grid_points", "[fft][k_grid]") {
  constexpr double two_pi = 6.283185307179586476925286766559;
  nda::stack_array<double, 3, 3> lattv{{1.0, 0.0, 0.0},
                                         {0.0, 1.0, 0.0},
                                         {0.0, 0.0, 1.0}};
  nda::stack_array<long, 3> grid{3, 1, 1};
  nda::array<double, 2> kpts{{0.0, 0.0, 0.0},
                             {two_pi * (1.0 + 5e-7) / 3.0, 0.0, 0.0},
                             {-two_pi / 3.0, 0.0, 0.0}};
  math::fft::k_grid_forward off_grid(kpts, lattv, grid, 1);
  REQUIRE_FALSE(off_grid.can_fft());

  // The original coefficients retain the measured k-point coordinate. A rounded
  // FFT slot would lose this phase for the nonzero-R impulse below.
  nda::array<long, 2> rpts{{0L, 0L, 0L}, {1L, 0L, 0L}, {-1L, 0L, 0L}};
  nda::matrix<ComplexType> f_Rk(3, 3), impulse(3, 1), reference(3, 1);
  utils::k_to_R_coefficients(rpts, kpts, lattv, f_Rk);
  impulse() = ComplexType(0.0);
  impulse(1, 0) = {0.7, -0.2};
  nda::blas::gemm(ComplexType(1.0), f_Rk, impulse, ComplexType(0.0), reference);
  for (long r = 0; r < 3; ++r) {
    REQUIRE(std::isfinite(reference(r, 0).real()));
    REQUIRE(std::isfinite(reference(r, 0).imag()));
  }

  kpts(1, 0) = two_pi / 3.0;
  math::fft::k_grid_forward regular(kpts, lattv, grid, 1);
  REQUIRE(regular.can_fft());
  regular.transform(impulse);
  REQUIRE(std::abs(reference(1, 0) - impulse(1, 0)) > 1e-8);

  kpts(2, 0) = kpts(1, 0);
  math::fft::k_grid_forward duplicate(kpts, lattv, grid, 1);
  REQUIRE_FALSE(duplicate.can_fft());

  nda::array<double, 2> shifted_permuted{
      {two_pi * 1.25 / 3.0, 0.0, 0.0},
      {two_pi * 0.25 / 3.0, 0.0, 0.0},
      {two_pi * -0.75 / 3.0, 0.0, 0.0}};
  math::fft::k_grid_forward shifted(shifted_permuted, lattv, grid, 0);
  REQUIRE(shifted.can_fft());
}
