#include <cmath>
#include <cstdio>
#include <memory>

#include "catch2/catch.hpp"

#include "mean_field/default_MF.hpp"
#include "methods/ERI/mb_eri_context.h"
#include "methods/ERI/eri_utils.hpp"
#include "methods/SCF/scf_driver.hpp"
#include "methods/SCF/simple_dyson.h"
#include "numerics/fft/k_grid.hpp"
#include "utilities/mpi_context.h"
#include "utilities/test_common.hpp"

TEST_CASE("thc_rpa_off_grid_gemm_fallback", "[methods][thc][rpa][fft_fallback]") {
  auto& mpi_context = utils::make_unit_test_mpi_context();
  auto [outdir, prefix] = utils::utest_filename("qe_lih222");
  mf::qe::qe_readonly source(mpi_context, outdir, prefix);
  auto system = source.get_sys();

  // Change only the in-memory Cartesian coordinate, after the QE fixture has
  // established the k/q maps. This is a dispatch probe, not a new physical
  // reference calculation; the checked-in fixture remains untouched.
  constexpr double two_pi = 6.283185307179586476925286766559;
  auto& bz = const_cast<mf::bz_symm&>(system.bz());
  bz.kpts(1, 0) += two_pi * 5e-7 / system.latt(0, 0);
  auto mf = std::make_shared<mf::MF>(mf::qe::qe_readonly(std::move(system)));
  math::fft::k_grid_forward dispatch(mf->kpts(), mf->lattv(), mf->kp_grid(), 1);
  REQUIRE_FALSE(dispatch.can_fft());

  imag_axes_ft::IAFT ft(1000, 1.2, imag_axes_ft::ir_basis, "high");
  methods::solvers::hf_t hf;
  methods::solvers::gw_t gw(&ft, "gygi_smallest_q");
  methods::simple_dyson dyson(mf.get(), &ft);
  methods::thc_reader_t thc(mf, methods::make_thc_reader_ptree(
      mf->nbnd() * 20, "", "incore", "", "bdft", 1e-10, mf->ecutrho(), 1, 1024));
  auto eri = methods::mb_eri_t(thc, thc);
  methods::MBState mb_state(mpi_context, ft, "bdft");
  double e_rpa = methods::rpa_loop(
      mb_state, dyson, eri, ft, methods::solvers::mb_solver_t(&hf, &gw));
  REQUIRE(std::isfinite(e_rpa));
  mpi_context->comm.barrier();
  if (mpi_context->comm.root()) {
    std::remove("./thc_eri.h5");
    std::remove("./bdft.mbpt.h5");
  }
  mpi_context->comm.barrier();
}
