/**
 * ==========================================================================
 * CoQuí: Correlated Quantum ínterface
 *
 * Copyright (c) 2022-2026 Simons Foundation & The CoQuí developer team
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ==========================================================================
 */


#undef NDEBUG

#include "catch2/catch.hpp"

#include <cmath>
#include <complex>
#include <cstdlib>
#include <filesystem>
#include <string>

#include "mpi3/environment.hpp"
#include "mpi3/communicator.hpp"
#include "mpi3/shared_communicator.hpp"
#include "utilities/mpi_context.h"

#include "utilities/test_common.hpp"
#include "methods/tests/test_common.hpp"

#include "h5/h5.hpp"
#include "nda/nda.hpp"
#include "nda/h5.hpp"

#include "mean_field/MF.hpp"
#include "mean_field/mf_utils.hpp"
#include "mean_field/default_MF.hpp"
#include "mean_field/model_hamiltonian/model_readonly.hpp"
#include "mean_field/symmetry/bz_symmetry.hpp"
#include "methods/ERI/cholesky.h"
#include "methods/ERI/chol_reader_t.hpp"
#include "methods/ERI/eri_utils.hpp"
#include "utilities/symmetry.hpp"

namespace bdft_tests
{
  using namespace methods;

  TEST_CASE("chol_reader", "[methods]") {
    auto& mpi = utils::make_unit_test_mpi_context();

    auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi, mf::pyscf_source));

    chol_reader_t chol(mf, methods::make_chol_reader_ptree(1e-6, mf->ecutrho(), 32, "./", "chol_info.h5"));

    auto V = chol.V(0, 0, 0);
    REQUIRE(V.shape() == shape_t<3>{(long)chol.Np(), (long)chol.nbnd(), (long)chol.nbnd()});
    std::cout << "Reading type = " << chol.chol_read_type() << std::endl;
    std::cout << "Writing type = " << chol.chol_write_type() << std::endl;
    mpi->comm.barrier();

    chol.set_read_type() = methods::chol_reading_type_e::each_q;
    [[maybe_unused]] auto Vq = chol.V(0, 0, 1);
    REQUIRE(V.shape() == shape_t<3>{(long)chol.Np(), (long)chol.nbnd(), (long)chol.nbnd()});
    std::cout << "Reading type = " << chol.chol_read_type() << std::endl;
    mpi->comm.barrier();

    if(mpi->comm.root()) {
      remove("chol_info.h5");
      for (size_t iq = 0; iq < mf->nkpts(); ++iq) remove(("Vq"+std::to_string(iq)+".h5").c_str());
    }
  }

  TEST_CASE("chol_reader_single_write", "[methods]") {
    auto& mpi = utils::make_unit_test_mpi_context();

    auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi, mf::qe_source));

    chol_reader_t chol(mf, methods::make_chol_reader_ptree(1e-6, mf->ecutrho(), 32, "./",
                                                           "chol_info.h5", each_q, single_file));

    auto V = chol.V(0, 0, 0);
    REQUIRE(V.shape() == shape_t<3>{(long)chol.Np(), (long)chol.nbnd(), (long)chol.nbnd()});
    std::cout << "Reading type = " << chol.chol_read_type() << std::endl;
    std::cout << "Writing type = " << chol.chol_write_type() << std::endl;

    chol.set_read_type() = methods::chol_reading_type_e::each_q;
    [[maybe_unused]] auto Vq = chol.V(0, 0, 1);
    REQUIRE(V.shape() == shape_t<3>{(long)chol.Np(), (long)chol.nbnd(), (long)chol.nbnd()});
    std::cout << "Reading type = " << chol.chol_read_type() << std::endl;

    if(mpi->comm.root())
      remove("chol_info.h5");
  }

  TEST_CASE("make_cholesky", "[methods]") {
    auto& mpi = utils::make_unit_test_mpi_context();

    auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi, mf::qe_source));
    ptree pt;
    auto chol = make_cholesky(mf, pt);

    auto V = chol.V(0, 0, 0);
    REQUIRE(V.shape() == shape_t<3>{(long)chol.Np(), (long)chol.nbnd(), (long)chol.nbnd()});
    std::cout << "Reading type = " << chol.chol_read_type() << std::endl;
    mpi->comm.barrier();

    chol.set_read_type() = methods::chol_reading_type_e::each_q;
    [[maybe_unused]] auto Vq = chol.V(0, 0, 1);
    REQUIRE(V.shape() == shape_t<3>{(long)chol.Np(), (long)chol.nbnd(), (long)chol.nbnd()});
    std::cout << "Reading type = " << chol.chol_read_type() << std::endl;
    
    mpi->comm.barrier();
    if(mpi->comm.root()) {
      remove("chol_info.h5");
      for (size_t iq = 0; iq < mf->nkpts(); ++iq) remove(("Vq"+std::to_string(iq)+".h5").c_str());
    }
  }

  // M-L6d2: external libmuffintin scalar Cholesky fixture through live chol_reader_t.
  // libmuffintin commit 89ff8f8c80711eb6ded36efba688c8a7fd640bf9 (published M-L6d1).
  // CoQui base wg-dev @ a19774d03fb979bd852fae4f7f95c045a4cbca78.
  // Unversioned/private bounded scalar gate: not q-dependent THC/MLDUMP compatibility,
  // no singular Gamma head, not spinor/core/GW/material acceptance.
  //
  // Catch2 v2.13.10 has no SKIP(). Unset LIBMUFFINTIN_COQUI_CHOLESKY_FILE returns
  // after WARN so the ordinary suite stays green. When set, missing/wrong schema
  // hard-fails; this test never writes a CoQui-generated ERI file.
  //
  // Type check (run notes, not a committed binary):
  //   h5dump -H -d /Interaction/Vq0 -a /Interaction/Vq0/__complex__ \
  //     "$LIBMUFFINTIN_COQUI_CHOLESKY_FILE"
  // Published 89ff8f8 hydrogen window is 1-band / Np=1, Vq shape [1,1,2,1,1,2].
  TEST_CASE("chol_reader_libmuffintin_fixture", "[methods][libmuffintin][ml6d2]") {
    auto& mpi = utils::make_unit_test_mpi_context();

    const char* env = std::getenv("LIBMUFFINTIN_COQUI_CHOLESKY_FILE");
    if (env == nullptr || env[0] == '\0') {
      WARN("LIBMUFFINTIN_COQUI_CHOLESKY_FILE unset; skipping M-L6d2 libmuffintin "
           "chol_reader fixture (Catch2 v2.13.10 has no SKIP()).");
      return;
    }

    std::filesystem::path artifact(env);
    CAPTURE(artifact);
    REQUIRE(std::filesystem::exists(artifact));
    REQUIRE(std::filesystem::is_regular_file(artifact));

    h5::file file(artifact.string(), 'r');
    h5::group root(file);
    REQUIRE(root.has_subgroup("Interaction"));
    h5::group interaction = root.open_group("Interaction");

    int np = 0, nspin = 0, nspin_in_basis = 0, nkpts = 0, nbnd = 0, nbnd_aux = -1;
    h5::h5_read(interaction, "Np", np);
    h5::h5_read(interaction, "nspin", nspin);
    h5::h5_read(interaction, "nspin_in_basis", nspin_in_basis);
    h5::h5_read(interaction, "nkpts", nkpts);
    h5::h5_read(interaction, "nbnd", nbnd);
    h5::h5_read(interaction, "nbnd_aux", nbnd_aux);

    // Bounded published 89ff8f8 hydrogen Cholesky header (not the 2-band planning sketch).
    REQUIRE(np == 1);
    REQUIRE(nspin == 1);
    REQUIRE(nspin_in_basis == 1);
    REQUIRE(nkpts == 2);
    REQUIRE(nbnd == 1);
    REQUIRE(nbnd_aux == 0);

    nda::array<double, 2> kpts;
    nda::array<double, 2> qpts;
    nda::array<int, 2> qk_to_kmq;
    nda::h5_read(interaction, "kpts", kpts);
    nda::h5_read(interaction, "qpts", qpts);
    nda::h5_read(interaction, "qk_to_kmq", qk_to_kmq);
    REQUIRE(kpts.shape() == shape_t<2>{2, 3});
    REQUIRE(qpts.shape() == shape_t<2>{2, 3});
    REQUIRE(qk_to_kmq.shape() == shape_t<2>{2, 2});
    REQUIRE(std::abs(qpts(0, 0)) + std::abs(qpts(0, 1)) + std::abs(qpts(0, 2)) < 1e-12);
    REQUIRE(std::abs(qpts(1, 0)) > 1e-8);
    REQUIRE(qk_to_kmq(0, 0) == 0);
    REQUIRE(qk_to_kmq(0, 1) == 1);
    REQUIRE(qk_to_kmq(1, 0) == 1);
    REQUIRE(qk_to_kmq(1, 1) == 0);

    auto read_raw_vq = [&](int iq) {
      std::string name = "Vq" + std::to_string(iq);
      REQUIRE(interaction.has_dataset(name));
      auto info = h5::array_interface::get_dataset_info(interaction, name);
      REQUIRE(info.lengths.size() == 6);
      REQUIRE(info.lengths[0] == static_cast<size_t>(np));
      REQUIRE(info.lengths[1] == 1);
      REQUIRE(info.lengths[2] == static_cast<size_t>(nkpts));
      REQUIRE(info.lengths[3] == static_cast<size_t>(nbnd));
      REQUIRE(info.lengths[4] == static_cast<size_t>(nbnd));
      REQUIRE(info.lengths[5] == 2);
      nda::array<ComplexType, 5> raw;
      nda::h5_read(interaction, name, raw);
      REQUIRE(raw.shape() == shape_t<5>{np, 1, nkpts, nbnd, nbnd});
      return raw;
    };
    auto raw0 = read_raw_vq(0);
    auto raw1 = read_raw_vq(1);

    constexpr double alat = 8.0; // published hydrogen SnapshotV2 lattice, Bohr
    nda::array<double, 2> latt = {{alat, 0.0, 0.0}, {0.0, alat, 0.0}, {0.0, 0.0, alat}};
    double bvec = 2.0 * std::acos(-1.0) / alat;
    nda::array<double, 2> recv = {{bvec, 0.0, 0.0}, {0.0, bvec, 0.0}, {0.0, 0.0, bvec}};
    nda::array<double, 1> kp_grid(3);
    kp_grid(0) = 2.0;
    kp_grid(1) = 1.0;
    kp_grid(2) = 1.0;
    nda::array<double, 2> R = {{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};
    nda::array<double, 1> ft = {0.0, 0.0, 0.0};
    std::vector<utils::symm_op> slist{utils::symm_op{R, R, ft}};
    mf::bz_symm symm(mpi->comm, true, latt, recv, kp_grid, kpts, slist, false);

    auto h = nda::array<ComplexType, 4>::zeros({1, nkpts, nbnd, nbnd});
    auto ov = nda::array<ComplexType, 4>::zeros({1, nkpts, nbnd, nbnd});
    auto dm = nda::array<ComplexType, 4>::zeros({1, nkpts, nbnd, nbnd});
    auto fock = nda::array<ComplexType, 4>::zeros({1, nkpts, nbnd, nbnd});
    for (int ik = 0; ik < nkpts; ++ik) ov(0, ik, 0, 0) = 1.0;

    mf::model::model_system model(mpi, "./", "ml6d2_libmuffintin_dummy",
                                  std::move(symm), 1, 1, 1.0, h, ov, dm, fock);
    auto mf = std::make_shared<mf::MF>(mf::model::model_readonly(model));
    REQUIRE(mf->nkpts() == 2);
    REQUIRE(mf->nqpts() == 2);
    REQUIRE(mf->nbnd() == 1);
    REQUIRE(mf->nspin() == 1);
    REQUIRE(mf->nspin_in_basis() == 1);
    ARRAY_EQUAL(mf->kpts(), kpts, 1e-12, 1e-12);
    ARRAY_EQUAL(mf->Qpts(), qpts, 1e-12, 1e-12);
    for (int iq = 0; iq < 2; ++iq) {
      for (int ik = 0; ik < 2; ++ik) {
        REQUIRE(mf->qk_to_k2(iq, ik) == qk_to_kmq(iq, ik));
      }
    }

    std::string eri_dir = artifact.parent_path().string();
    if (eri_dir.empty()) eri_dir = ".";
    std::string eri_name = artifact.filename().string();
    chol_reader_t chol(mf, eri_dir, eri_name, each_q, single_file);
    REQUIRE(chol.chol_read_type() == each_q);
    REQUIRE(chol.chol_write_type() == single_file);
    REQUIRE(chol.Np() == np);
    REQUIRE(chol.nbnd() == nbnd);

    auto expected_from_raw = [&](nda::array<ComplexType, 5> const& raw, int ik) {
      nda::array<ComplexType, 3> expected(np, nbnd, nbnd);
      for (int Q = 0; Q < np; ++Q)
        for (int i = 0; i < nbnd; ++i)
          for (int j = 0; j < nbnd; ++j)
            expected(Q, i, j) = raw(Q, 0, ik, i, j);
      return expected;
    };

    auto contract_conj = [](nda::array<ComplexType, 3> const& Lp,
                            nda::array<ComplexType, 3> const& Ls) {
      // GF2 build_int: (pr|qs) = sum_Q L_Qpr conj(L_Qsq); nbnd=1 => (00|00).
      ComplexType acc = 0.0;
      long nQ = Lp.extent(0);
      for (long Q = 0; Q < nQ; ++Q)
        acc += Lp(Q, 0, 0) * std::conj(Ls(Q, 0, 0));
      return acc;
    };
    auto contract_plain = [](nda::array<ComplexType, 3> const& Lp,
                             nda::array<ComplexType, 3> const& Ls) {
      ComplexType acc = 0.0;
      long nQ = Lp.extent(0);
      for (long Q = 0; Q < nQ; ++Q)
        acc += Lp(Q, 0, 0) * Ls(Q, 0, 0);
      return acc;
    };

    nda::array<ComplexType, 5> const* raws[2] = {&raw0, &raw1};
    for (int iq = 0; iq < 2; ++iq) {
      for (int ik = 0; ik < nkpts; ++ik) {
        auto V = chol.V(iq, 0, ik);
        REQUIRE(V.shape() == shape_t<3>{(long)np, (long)nbnd, (long)nbnd});
        auto expected = expected_from_raw(*raws[iq], ik);
        ARRAY_EQUAL(V, expected, 1e-12, 1e-12);
      }
    }

    for (int iq = 0; iq < 2; ++iq) {
      auto L0 = expected_from_raw(*raws[iq], 0);
      auto L1 = expected_from_raw(*raws[iq], 1);
      auto R0 = nda::make_regular(chol.V(iq, 0, 0));
      auto R1 = nda::make_regular(chol.V(iq, 0, 1));
      auto from_raw = contract_conj(L0, L1);
      auto from_reader = contract_conj(R0, R1);
      VALUE_EQUAL(from_reader, from_raw, 1e-12, 1e-12);
    }

    {
      auto L0 = expected_from_raw(raw1, 0);
      auto L1 = expected_from_raw(raw1, 1);
      auto R0 = nda::make_regular(chol.V(1, 0, 0));
      auto R1 = nda::make_regular(chol.V(1, 0, 1));
      auto correct = contract_conj(R0, R1);
      auto plain = contract_plain(R0, R1);
      auto swapped = contract_conj(R1, R0);
      REQUIRE(std::abs(correct - contract_conj(L0, L1)) < 1e-12);
      REQUIRE(std::abs(correct - plain) > 1e-8);
      REQUIRE(std::abs(correct - swapped) > 1e-8);
    }

    mpi->comm.barrier();
  }

} // bdft_tests
