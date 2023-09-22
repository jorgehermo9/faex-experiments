// #define RRR_NO_OPT
// Try with this flag...
#include <algorithm>
#include <fstream>
#include <iostream>
#include <iterator>
#include <random>
#include <sdsl/bit_vectors.hpp>
#include <sdsl/wavelet_trees.hpp>
#include <string>
#include <vector>
#include "nlohmann/json.hpp"

const size_t BITVEC_SIZE = 100000000;
const size_t NUM_REPEATS = 1;

using namespace std;
using namespace sdsl;
using namespace std::chrono;
using namespace nlohmann;

struct MultipleQueriesBenchmark {
  size_t num_queries;
  vector<double> total_time;
};

struct ParametrizableRankSelectBenchmark {
  size_t parameter;
  size_t size_in_bits;
  vector<double> construction;
  MultipleQueriesBenchmark rank;
  MultipleQueriesBenchmark select;
};

struct NonParametrizableRankBenchmark {
  size_t size_in_bits;
  vector<double> construction;
  MultipleQueriesBenchmark rank;
};

struct NonParametrizableSelectBenchmark {
  size_t size_in_bits;
  vector<double> construction;
  MultipleQueriesBenchmark select;
};

struct SamplingBenchmark {
  uint64_t original_size;
  vector<ParametrizableRankSelectBenchmark> dense_sampling;
  vector<ParametrizableRankSelectBenchmark> sparse_sampling;
  vector<ParametrizableRankSelectBenchmark> bit_vector_il;
  vector<NonParametrizableRankBenchmark> rank_support_v;
  vector<NonParametrizableRankBenchmark> rank_support_v5;
  vector<NonParametrizableSelectBenchmark> select_support_mcl;
};

struct CompressedBenchmark {
  uint64_t original_size;
  vector<ParametrizableRankSelectBenchmark> rrr_faex_15;
  vector<ParametrizableRankSelectBenchmark> rrr_faex_31;
  vector<ParametrizableRankSelectBenchmark> rrr_faex_63;
  vector<ParametrizableRankSelectBenchmark> rrr_sdsl_15;
  vector<ParametrizableRankSelectBenchmark> rrr_sdsl_31;
  vector<ParametrizableRankSelectBenchmark> rrr_sdsl_63;
};

struct ParametrizableRankSelectAccessBenchmark {
  size_t parameter;
  size_t size_in_bits;
  vector<double> construction;
  MultipleQueriesBenchmark rank;
  MultipleQueriesBenchmark select;
  MultipleQueriesBenchmark access;
};

struct NonParametrizableRankAccessBenchmark {
  size_t size_in_bits;
  vector<double> construction;
  MultipleQueriesBenchmark rank;
  MultipleQueriesBenchmark access;
};

struct WtBenchmark {
  uint64_t original_size;
  vector<ParametrizableRankSelectAccessBenchmark> dense_sampling;
  vector<ParametrizableRankSelectAccessBenchmark> sparse_sampling;
  vector<ParametrizableRankSelectAccessBenchmark> bit_vector_il;
  vector<NonParametrizableRankAccessBenchmark> rank_support_v;
  vector<NonParametrizableRankAccessBenchmark> rank_support_v5;
  vector<NonParametrizableSelectBenchmark> select_support_mcl;
  vector<ParametrizableRankSelectAccessBenchmark> rrr_faex_63;
  vector<ParametrizableRankSelectAccessBenchmark> rrr_sdsl_63;
};
struct WtBenchmarkDatasets {
  WtBenchmark english;
  WtBenchmark proteins;
};

struct CompressedBenchmarkDatasets {
  CompressedBenchmark five;
  CompressedBenchmark fifty;
  CompressedBenchmark ninety;
};

struct SamplingBenchmarkDatasets {
  SamplingBenchmark five;
  SamplingBenchmark fifty;
  SamplingBenchmark ninety;
};

struct Benchmark {
  SamplingBenchmarkDatasets sampling;
  CompressedBenchmarkDatasets compressed;
  WtBenchmarkDatasets wt;
};

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(MultipleQueriesBenchmark,
                                   num_queries,
                                   total_time)
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(ParametrizableRankSelectBenchmark,
                                   parameter,
                                   size_in_bits,
                                   construction,
                                   rank,
                                   select)
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(NonParametrizableRankBenchmark,
                                   size_in_bits,
                                   construction,
                                   rank)

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(NonParametrizableSelectBenchmark,
                                   size_in_bits,
                                   construction,
                                   select)
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(SamplingBenchmark,
                                   original_size,
                                   dense_sampling,
                                   sparse_sampling,
                                   bit_vector_il,
                                   rank_support_v,
                                   rank_support_v5,
                                   select_support_mcl)
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(CompressedBenchmark,
                                   original_size,
                                   rrr_faex_15,
                                   rrr_faex_31,
                                   rrr_faex_63,
                                   rrr_sdsl_15,
                                   rrr_sdsl_31,
                                   rrr_sdsl_63)

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(ParametrizableRankSelectAccessBenchmark,
                                   parameter,
                                   size_in_bits,
                                   construction,
                                   rank,
                                   select,
                                   access)
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(NonParametrizableRankAccessBenchmark,
                                   size_in_bits,
                                   construction,
                                   rank,
                                   access)

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(WtBenchmark,
                                   original_size,
                                   dense_sampling,
                                   sparse_sampling,
                                   bit_vector_il,
                                   rank_support_v,
                                   rank_support_v5,
                                   select_support_mcl,
                                   rrr_faex_63,
                                   rrr_sdsl_63)
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(WtBenchmarkDatasets, english, proteins)
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(CompressedBenchmarkDatasets,
                                   five,
                                   fifty,
                                   ninety)
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(SamplingBenchmarkDatasets,
                                   five,
                                   fifty,
                                   ninety)
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(Benchmark, sampling, compressed, wt)

// Function to prevent optimization
template <typename T>
void black_box(const T& value) {
  volatile T dummy = value;
  // You can also use the value in other
  // ways to make it observable For example:
  // std::cout << value;
}

// template <typename T>
// void black_box(const T& value) {
//     asm volatile("" : : "r,m"(value) :
//     "memory");
// }

bit_vector bitvec_with_distribution(size_t size, double density) {
  bit_vector bv(size, 0);

  // Create a bit vector with the exact given density.
  // Do not rely on randomness to get the exact density,
  // but the positions should be random.

  random_device rd;
  mt19937 rng(rd());
  size_t num_ones = static_cast<size_t>(size * density);

  vector<size_t> all_positions(size);
  iota(all_positions.begin(), all_positions.end(), 0);

  shuffle(all_positions.begin(), all_positions.end(), rng);
  vector<size_t> ones_positions(all_positions.begin(),
                                all_positions.begin() + num_ones);

  for (const size_t i : ones_positions) {
    bv[i] = 1;
  }

  return bv;
}

template <uint32_t k>
ParametrizableRankSelectBenchmark bit_vector_il_benchmark(
    bit_vector& bv,
    vector<size_t>& random_rank_positions,
    vector<size_t>& random_select_values,
    int num_rank_queries,
    int num_select_queries) {
  const uint32_t t_bs = 64 * k;
  ParametrizableRankSelectBenchmark benchmark;
  benchmark.parameter = k;
  bit_vector_il<t_bs> bit_vector_il_structure(bv);

  benchmark.size_in_bits =
      (size_in_bytes(bit_vector_il_structure) - size_in_bytes(bv)) * 8;
  typename bit_vector_il<t_bs>::rank_1_type rank_il(&bit_vector_il_structure);
  benchmark.rank.num_queries = num_rank_queries;
  benchmark.select.num_queries = num_select_queries;

  cout << "Starting bit_vector_il repeats" << endl;
  for (int r = 0; r < NUM_REPEATS; r++) {
    cout << "Starting bit_vector_il construction" << endl;
    auto start_construction = high_resolution_clock::now();
    bit_vector_il<t_bs> bit_vector_il_structure(bv);
    auto end_construction = high_resolution_clock::now();
    auto elapsed_construction =
        duration_cast<duration<double>>(end_construction - start_construction)
            .count();

    benchmark.construction.push_back(elapsed_construction);
    typename bit_vector_il<t_bs>::rank_1_type rank_il(&bit_vector_il_structure);
    typename bit_vector_il<t_bs>::select_1_type select_il(
        &bit_vector_il_structure);

    cout << "Starting bit_vector_il rank repeat " << r << endl;
    auto start_rank = high_resolution_clock::now();
    for (auto i : random_rank_positions) {
      black_box(rank_il.rank(i));
    }
    auto end_rank = high_resolution_clock::now();
    auto elapsed_rank =
        duration_cast<duration<double>>(end_rank - start_rank).count();

    benchmark.rank.total_time.push_back(elapsed_rank);
    cout << "elapsed_rank: " << elapsed_rank << endl;

    cout << "Starting bit_vector_il select repeat " << r << endl;
    auto start_select = high_resolution_clock::now();
    for (auto i : random_select_values) {
      black_box(select_il.select(i));
    }
    auto end_select = high_resolution_clock::now();
    auto elapsed_select =
        duration_cast<duration<double>>(end_select - start_select).count();

    benchmark.select.total_time.push_back(elapsed_select);
    cout << "elapsed_select: " << elapsed_select << endl;
  }

  return benchmark;
}

SamplingBenchmark sampling_benchmark(bit_vector bv,
                                     SamplingBenchmark sampling_benchmark,
                                     double select_ratio) {
  NonParametrizableRankBenchmark concrete_rank_support_v;
  rank_support_v<> rank_counter(&bv);
  int num_ones = rank_counter.rank(bv.size());

  random_device rd;
  mt19937 rng(rd());

  vector<size_t> random_rank_positions(bv.size() + 1);
  iota(random_rank_positions.begin(), random_rank_positions.end(), 0);
  shuffle(random_rank_positions.begin(), random_rank_positions.end(), rng);

  vector<size_t> random_select_values_aux(num_ones + 1);
  iota(random_select_values_aux.begin(), random_select_values_aux.end(), 1);
  shuffle(random_select_values_aux.begin(), random_select_values_aux.end(),
          rng);
  vector<size_t> random_select_values(
      random_select_values_aux.begin(),
      random_select_values_aux.begin() +
          random_select_values_aux.size() * select_ratio);

  int num_rank_queries = random_rank_positions.size();
  int num_select_queries = random_select_values.size();

  cout << "Starting rank_support_v construction" << endl;
  rank_support_v<> rank_support_v_structure(&bv);
  concrete_rank_support_v.size_in_bits =
      size_in_bytes(rank_support_v_structure) * 8;
  concrete_rank_support_v.rank.num_queries = num_rank_queries;

  cout << "Starting rank_support_v repeats" << endl;
  for (int r = 0; r < NUM_REPEATS; r++) {
    cout << "Starting rank_support_v construction" << endl;
    auto start_construction = high_resolution_clock::now();
    rank_support_v<> rank_support_v_structure(&bv);
    auto end_construction = high_resolution_clock::now();
    auto elapsed_construction =
        duration_cast<duration<double>>(end_construction - start_construction)
            .count();

    concrete_rank_support_v.construction.push_back(elapsed_construction);

    auto start_rank = high_resolution_clock::now();
    cout << "Starting rank_support_v rank repeat " << r << endl;
    for (auto i : random_rank_positions) {
      black_box(rank_support_v_structure.rank(i));
    }

    auto end_rank = high_resolution_clock::now();
    auto elapsed_rank =
        duration_cast<duration<double>>(end_rank - start_rank).count();

    concrete_rank_support_v.rank.total_time.push_back(elapsed_rank);
    cout << "elapsed_rank: " << elapsed_rank << endl;
  }
  sampling_benchmark.rank_support_v.push_back(concrete_rank_support_v);

  NonParametrizableRankBenchmark concrete_rank_support_v5;

  cout << "Starting rank_support_v5 construction" << endl;
  rank_support_v5<> rank_support_v5_structure(&bv);
  concrete_rank_support_v5.size_in_bits =
      size_in_bytes(rank_support_v5_structure) * 8;
  concrete_rank_support_v5.rank.num_queries = num_rank_queries;

  cout << "Starting rank_support_v5 repeats" << endl;
  for (int r = 0; r < NUM_REPEATS; r++) {
    cout << "Starting rank_support_v5 construction" << endl;
    auto start_construction_v5 = high_resolution_clock::now();
    rank_support_v5<> rank_support_v5_structure(&bv);
    auto end_construction_v5 = high_resolution_clock::now();
    auto elapsed_construction_v5 =
        duration_cast<duration<double>>(end_construction_v5 -
                                        start_construction_v5)
            .count();

    concrete_rank_support_v5.construction.push_back(elapsed_construction_v5);

    cout << "Starting rank_support_v5 rank repeat " << r << endl;
    auto start_rank = high_resolution_clock::now();
    for (auto i : random_rank_positions) {
      black_box(rank_support_v5_structure.rank(i));
    }
    auto end_rank = high_resolution_clock::now();
    auto elapsed_rank =
        duration_cast<duration<double>>(end_rank - start_rank).count();

    concrete_rank_support_v5.rank.total_time.push_back(elapsed_rank);
    cout << "elapsed_rank: " << elapsed_rank << endl;
  }
  sampling_benchmark.rank_support_v5.push_back(concrete_rank_support_v5);

  cout << "Starting bit_vector_il 4" << endl;
  sampling_benchmark.bit_vector_il.push_back(bit_vector_il_benchmark<4>(
      bv, random_rank_positions, random_select_values, num_rank_queries,
      num_select_queries));
  cout << "Starting bit_vector_il 8" << endl;
  sampling_benchmark.bit_vector_il.push_back(bit_vector_il_benchmark<8>(
      bv, random_rank_positions, random_select_values, num_rank_queries,
      num_select_queries));
  cout << "Starting bit_vector_il 16" << endl;
  sampling_benchmark.bit_vector_il.push_back(bit_vector_il_benchmark<16>(
      bv, random_rank_positions, random_select_values, num_rank_queries,
      num_select_queries));
  cout << "Starting bit_vector_il 32" << endl;
  sampling_benchmark.bit_vector_il.push_back(bit_vector_il_benchmark<32>(
      bv, random_rank_positions, random_select_values, num_rank_queries,
      num_select_queries));
  cout << "Starting bit_vector_il 64" << endl;
  sampling_benchmark.bit_vector_il.push_back(bit_vector_il_benchmark<64>(
      bv, random_rank_positions, random_select_values, num_rank_queries,
      num_select_queries));
  cout << "Starting bit_vector_il 128" << endl;
  sampling_benchmark.bit_vector_il.push_back(bit_vector_il_benchmark<128>(
      bv, random_rank_positions, random_select_values, num_rank_queries,
      num_select_queries));

  NonParametrizableSelectBenchmark concrete_select_support_mcl;

  select_support_mcl<> select_support_mcl_structure(&bv);
  select_support_mcl<0> select_support_mcl0_structure(&bv);

  concrete_select_support_mcl.size_in_bits =
      size_in_bytes(select_support_mcl_structure) * 8 +
      size_in_bytes(select_support_mcl0_structure) * 8;

  concrete_select_support_mcl.select.num_queries = num_select_queries;

  cout << "Starting select_support_mcl repeats" << endl;
  for (int r = 0; r < NUM_REPEATS; r++) {
    cout << "Starting select_support_mcl construction" << endl;
    auto start_construction_mcl = high_resolution_clock::now();
    select_support_mcl<> select_support_mcl_structure(&bv);
    auto end_construction_mcl = high_resolution_clock::now();
    auto elapsed_construction_mcl =
        duration_cast<duration<double>>(end_construction_mcl -
                                        start_construction_mcl)
            .count();
    concrete_select_support_mcl.construction.push_back(
        elapsed_construction_mcl);

    cout << "Starting select_support_mcl select repeat " << r << endl;
    auto start_select = high_resolution_clock::now();
    for (auto i : random_select_values) {
      black_box(select_support_mcl_structure.select(i));
    }
    auto end_select = high_resolution_clock::now();
    auto elapsed_select =
        duration_cast<duration<double>>(end_select - start_select).count();

    concrete_select_support_mcl.select.total_time.push_back(elapsed_select);
    cout << "elapsed_select: " << elapsed_select << endl;
  }
  sampling_benchmark.select_support_mcl.push_back(concrete_select_support_mcl);

  return sampling_benchmark;
}

template <uint16_t t_bs, uint16_t k>
ParametrizableRankSelectBenchmark rrr_vector_benchmark(
    bit_vector& bv,
    vector<size_t>& random_rank_positions,
    vector<size_t>& random_select_values,
    int num_rank_queries,
    int num_select_queries) {
  ParametrizableRankSelectBenchmark benchmark;
  benchmark.parameter = k;
  rrr_vector<t_bs, int_vector<>, k> rrr_vector_structure(bv);
  typename rrr_vector<t_bs, int_vector<>, k>::rank_1_type rank_rrr(
      &rrr_vector_structure);
  benchmark.size_in_bits = size_in_bytes(rrr_vector_structure) * 8;
  benchmark.rank.num_queries = num_rank_queries;
  benchmark.select.num_queries = num_select_queries;

  cout << "Starting rrr_vector " << t_bs << " " << k << " repeats" << endl;
  for (int r = 0; r < NUM_REPEATS; r++) {
    cout << "Starting rrr_vector " << t_bs << " " << k << " construction"
         << endl;

    auto start_construction = high_resolution_clock::now();
    rrr_vector<t_bs, int_vector<>, k> rrr_vector_structure(bv);
    auto end_construction = high_resolution_clock::now();
    auto elapsed_construction =
        duration_cast<duration<double>>(end_construction - start_construction)
            .count();
    benchmark.construction.push_back(elapsed_construction);
    typename rrr_vector<t_bs, int_vector<>, k>::rank_1_type rank_rrr(
        &rrr_vector_structure);
    typename rrr_vector<t_bs, int_vector<>, k>::select_1_type select_rrr(
        &rrr_vector_structure);

    cout << "Starting rrr_vector " << t_bs << " " << k << "rank repeat " << r
         << endl;
    auto start_rank = high_resolution_clock::now();
    for (auto i : random_rank_positions) {
      black_box(rank_rrr.rank(i));
    }
    auto end_rank = high_resolution_clock::now();
    auto elapsed_rank =
        duration_cast<duration<double>>(end_rank - start_rank).count();

    benchmark.rank.total_time.push_back(elapsed_rank);
    cout << "elapsed_rank: " << elapsed_rank << endl;

    cout << "Starting rrr_vector " << t_bs << " " << k << "select repeat " << r
         << endl;
    auto start_select = high_resolution_clock::now();
    for (auto i : random_select_values) {
      black_box(select_rrr.select(i));
    }
    auto end_select = high_resolution_clock::now();
    auto elapsed_select =
        duration_cast<duration<double>>(end_select - start_select).count();

    benchmark.select.total_time.push_back(elapsed_select);
    cout << "elapsed_select: " << elapsed_select << endl;
  }

  return benchmark;
}

CompressedBenchmark compressed_benchmark(
    bit_vector bv,
    CompressedBenchmark compressed_benchmark,
    double select_ratio) {
  rank_support_v<> rank_support_v_structure(&bv);
  int num_ones = rank_support_v_structure.rank(bv.size());

  random_device rd;
  mt19937 rng(rd());

  vector<size_t> random_rank_positions(bv.size() + 1);
  iota(random_rank_positions.begin(), random_rank_positions.end(), 0);
  shuffle(random_rank_positions.begin(), random_rank_positions.end(), rng);

  vector<size_t> random_select_values_aux(num_ones + 1);
  iota(random_select_values_aux.begin(), random_select_values_aux.end(), 1);
  shuffle(random_select_values_aux.begin(), random_select_values_aux.end(),
          rng);
  vector<size_t> random_select_values(
      random_select_values_aux.begin(),
      random_select_values_aux.begin() +
          random_select_values_aux.size() * select_ratio);

  int num_rank_queries = random_rank_positions.size();
  int num_select_queries = random_select_values.size();

  cout << "Starting rrr_vector 15" << endl;
  compressed_benchmark.rrr_sdsl_15.push_back(rrr_vector_benchmark<15, 4>(
      bv, random_rank_positions, random_select_values, num_rank_queries,
      num_select_queries));
  compressed_benchmark.rrr_sdsl_15.push_back(rrr_vector_benchmark<15, 8>(
      bv, random_rank_positions, random_select_values, num_rank_queries,
      num_select_queries));
  compressed_benchmark.rrr_sdsl_15.push_back(rrr_vector_benchmark<15, 16>(
      bv, random_rank_positions, random_select_values, num_rank_queries,
      num_select_queries));
  compressed_benchmark.rrr_sdsl_15.push_back(rrr_vector_benchmark<15, 32>(
      bv, random_rank_positions, random_select_values, num_rank_queries,
      num_select_queries));
  compressed_benchmark.rrr_sdsl_15.push_back(rrr_vector_benchmark<15, 64>(
      bv, random_rank_positions, random_select_values, num_rank_queries,
      num_select_queries));
  compressed_benchmark.rrr_sdsl_15.push_back(rrr_vector_benchmark<15, 128>(
      bv, random_rank_positions, random_select_values, num_rank_queries,
      num_select_queries));

  cout << "Starting rrr_vector 31" << endl;
  compressed_benchmark.rrr_sdsl_31.push_back(rrr_vector_benchmark<31, 4>(
      bv, random_rank_positions, random_select_values, num_rank_queries,
      num_select_queries));
  compressed_benchmark.rrr_sdsl_31.push_back(rrr_vector_benchmark<31, 8>(
      bv, random_rank_positions, random_select_values, num_rank_queries,
      num_select_queries));
  compressed_benchmark.rrr_sdsl_31.push_back(rrr_vector_benchmark<31, 16>(
      bv, random_rank_positions, random_select_values, num_rank_queries,
      num_select_queries));
  compressed_benchmark.rrr_sdsl_31.push_back(rrr_vector_benchmark<31, 32>(
      bv, random_rank_positions, random_select_values, num_rank_queries,
      num_select_queries));
  compressed_benchmark.rrr_sdsl_31.push_back(rrr_vector_benchmark<31, 64>(
      bv, random_rank_positions, random_select_values, num_rank_queries,
      num_select_queries));
  compressed_benchmark.rrr_sdsl_31.push_back(rrr_vector_benchmark<31, 128>(
      bv, random_rank_positions, random_select_values, num_rank_queries,
      num_select_queries));

  cout << "Starting rrr_vector 63" << endl;
  compressed_benchmark.rrr_sdsl_63.push_back(rrr_vector_benchmark<63, 4>(
      bv, random_rank_positions, random_select_values, num_rank_queries,
      num_select_queries));
  compressed_benchmark.rrr_sdsl_63.push_back(rrr_vector_benchmark<63, 8>(
      bv, random_rank_positions, random_select_values, num_rank_queries,
      num_select_queries));
  compressed_benchmark.rrr_sdsl_63.push_back(rrr_vector_benchmark<63, 16>(
      bv, random_rank_positions, random_select_values, num_rank_queries,
      num_select_queries));
  compressed_benchmark.rrr_sdsl_63.push_back(rrr_vector_benchmark<63, 32>(
      bv, random_rank_positions, random_select_values, num_rank_queries,
      num_select_queries));
  compressed_benchmark.rrr_sdsl_63.push_back(rrr_vector_benchmark<63, 64>(
      bv, random_rank_positions, random_select_values, num_rank_queries,
      num_select_queries));
  compressed_benchmark.rrr_sdsl_63.push_back(rrr_vector_benchmark<63, 128>(
      bv, random_rank_positions, random_select_values, num_rank_queries,
      num_select_queries));

  return compressed_benchmark;
}

template <uint32_t k>
ParametrizableRankSelectAccessBenchmark wt_il_benchmark(
    string dataset,
    vector<pair<size_t, uint64_t>>& random_rank_queries,
    vector<pair<size_t, uint64_t>>& random_select_queries,
    vector<size_t>& random_access_queries,
    int num_rank_queries,
    int num_select_queries,
    int num_access_queries) {
  const uint32_t t_bs = 64 * k;
  ParametrizableRankSelectAccessBenchmark benchmark;
  benchmark.parameter = k;
  typedef bit_vector_il<t_bs> t_bv;
  typedef typename t_bv::rank_1_type t_rank;
  typedef typename t_bv::select_1_type t_select;
  typedef typename t_bv::select_0_type t_select0;

  wt_int<t_bv, t_rank, t_select, t_select0> wt_il;
  construct(wt_il, dataset, 4);
  benchmark.size_in_bits = size_in_bytes(wt_il) * 8;
  benchmark.rank.num_queries = num_rank_queries;
  benchmark.select.num_queries = num_select_queries;
  benchmark.access.num_queries = num_access_queries;

  cout << "Starting wt_il repeats" << endl;
  for (int r = 0; r < NUM_REPEATS; r++) {
    cout << "Starting wt_il construction" << endl;
    auto start_construction = high_resolution_clock::now();
    wt_int<t_bv, t_rank, t_select, t_select0> wt_il;
    construct(wt_il, dataset, 4);
    auto end_construction = high_resolution_clock::now();
    auto elapsed_construction =
        duration_cast<duration<double>>(end_construction - start_construction)
            .count();

    benchmark.construction.push_back(elapsed_construction);

    cout << "Starting wt_il rank repeat " << r << endl;
    auto start_rank = high_resolution_clock::now();
    for (auto i : random_rank_queries) {
      black_box(wt_il.rank(i.first, i.second));
    }
    auto end_rank = high_resolution_clock::now();
    auto elapsed_rank =
        duration_cast<duration<double>>(end_rank - start_rank).count();

    benchmark.rank.total_time.push_back(elapsed_rank);
    cout << "elapsed_rank: " << elapsed_rank << endl;

    cout << "Starting wt_il select repeat " << r << endl;
    auto start_select = high_resolution_clock::now();
    for (auto i : random_select_queries) {
      black_box(wt_il.select(i.first, i.second));
    }
    auto end_select = high_resolution_clock::now();
    auto elapsed_select =
        duration_cast<duration<double>>(end_select - start_select).count();

    benchmark.select.total_time.push_back(elapsed_select);
    cout << "elapsed_select: " << elapsed_select << endl;

    cout << "Starting wt_il access repeat " << r << endl;
    auto start_access = high_resolution_clock::now();
    for (auto i : random_access_queries) {
      black_box(wt_il[i]);
    }
    auto end_access = high_resolution_clock::now();
    auto elapsed_access =
        duration_cast<duration<double>>(end_access - start_access).count();

    benchmark.access.total_time.push_back(elapsed_access);
    cout << "elapsed_access: " << elapsed_access << endl;
  }

  return benchmark;
}

template <uint16_t t_bs, uint16_t k>
ParametrizableRankSelectAccessBenchmark wt_rrr_vector_benchmark(
    string dataset,
    vector<pair<size_t, uint64_t>>& random_rank_queries,
    vector<pair<size_t, uint64_t>>& random_select_queries,
    vector<size_t>& random_access_queries,
    int num_rank_queries,
    int num_select_queries,
    int num_access_queries) {
  ParametrizableRankSelectAccessBenchmark benchmark;
  benchmark.parameter = k;
  typedef rrr_vector<t_bs, int_vector<>, k> t_rrr;
  typedef typename t_rrr::rank_1_type t_rank;
  typedef typename t_rrr::select_1_type t_select;
  typedef typename t_rrr::select_0_type t_select0;

  wt_int<t_rrr, t_rank, t_select, t_select0> wt_rrr;
  construct(wt_rrr, dataset, 4);
  benchmark.size_in_bits = size_in_bytes(wt_rrr) * 8;
  benchmark.rank.num_queries = num_rank_queries;
  benchmark.select.num_queries = num_select_queries;
  benchmark.access.num_queries = num_access_queries;

  cout << "Starting wt_rrr repeats" << endl;
  for (int r = 0; r < NUM_REPEATS; r++) {
    cout << "Starting wt_rrr construction" << endl;
    auto start_construction = high_resolution_clock::now();
    wt_int<t_rrr, t_rank, t_select, t_select0> wt_rrr;
    construct(wt_rrr, dataset, 4);
    auto end_construction = high_resolution_clock::now();
    auto elapsed_construction =
        duration_cast<duration<double>>(end_construction - start_construction)
            .count();

    benchmark.construction.push_back(elapsed_construction);

    cout << "Starting wt_rrr rank repeat " << r << endl;
    auto start_rank = high_resolution_clock::now();
    for (auto i : random_rank_queries) {
      black_box(wt_rrr.rank(i.first, i.second));
    }
    auto end_rank = high_resolution_clock::now();
    auto elapsed_rank =
        duration_cast<duration<double>>(end_rank - start_rank).count();

    benchmark.rank.total_time.push_back(elapsed_rank);
    cout << "elapsed_rank: " << elapsed_rank << endl;

    cout << "Starting wt_rrr select repeat " << r << endl;
    auto start_select = high_resolution_clock::now();
    for (auto i : random_select_queries) {
      black_box(wt_rrr.select(i.first, i.second));
    }
    auto end_select = high_resolution_clock::now();
    auto elapsed_select =
        duration_cast<duration<double>>(end_select - start_select).count();

    benchmark.select.total_time.push_back(elapsed_select);
    cout << "elapsed_select: " << elapsed_select << endl;

    cout << "Starting wt_rrr access repeat " << r << endl;
    auto start_access = high_resolution_clock::now();
    for (auto i : random_access_queries) {
      black_box(wt_rrr[i]);
    }
    auto end_access = high_resolution_clock::now();
    auto elapsed_access =
        duration_cast<duration<double>>(end_access - start_access).count();

    benchmark.access.total_time.push_back(elapsed_access);
    cout << "elapsed_access: " << elapsed_access << endl;
  }

  return benchmark;
}

WtBenchmark wt_benchmark(string dataset_name, WtBenchmark wt_benchmark) {
  wt_int<bit_vector, rank_support_v<>, select_support_scan<1>,
         select_support_scan<0>>
      wt;
  construct(wt, dataset_name, 4);
  set<uint64_t> alphabet_set;
  for (auto i = 0; i < wt.size(); ++i) {
    alphabet_set.insert(wt[i]);
  }
  vector<uint64_t> alphabet(alphabet_set.begin(), alphabet_set.end());

  unordered_map<uint64_t, uint64_t> alphabet_occs;
  for (auto i : alphabet) {
    alphabet_occs[i] = wt.rank(wt.size(), i);
  }

  random_device rd;
  mt19937 rng(rd());
  vector<uint64_t> random_chars(wt.size() / 10 + 1);
  // generate random number between 0 and alphabet.size()
  uniform_int_distribution<uint64_t> dist(0, alphabet.size() - 1);
  for (auto i = 0; i < random_chars.size(); ++i) {
    random_chars[i] = alphabet[dist(rng)];
  }

  vector<size_t> random_rank_positions(random_chars.size());
  iota(random_rank_positions.begin(), random_rank_positions.end(), 0);
  shuffle(random_rank_positions.begin(), random_rank_positions.end(), rng);

  vector<pair<size_t, uint64_t>> random_rank_queries(random_chars.size());
  for (int i = 0; i < random_rank_queries.size(); i++) {
    random_rank_queries[i] = {random_rank_positions[i], random_chars[i]};
  }

  vector<pair<size_t, uint64_t>> random_select_queries(random_chars.size() / 2);
  for (int i = 0; i < random_select_queries.size(); i++) {
    uint64_t random_char = random_chars[i];
    uniform_int_distribution<uint64_t> dist(1, alphabet_occs[random_char]);
    random_select_queries[i] = {dist(rng), random_char};
  }

  vector<size_t> random_access_queries(random_chars.size() / 2);
  iota(random_access_queries.begin(), random_access_queries.end(), 0);
  shuffle(random_access_queries.begin(), random_access_queries.end(), rng);
  int num_rank_queries = random_rank_queries.size();
  int num_select_queries = random_select_queries.size();
  int num_access_queries = random_access_queries.size();

  NonParametrizableRankAccessBenchmark concrete_rank_support_v;
  wt_int<bit_vector, rank_support_v<>, select_support_scan<1>,
         select_support_scan<0>>
      wt_rank_support_v;
  construct(wt_rank_support_v, dataset_name, 4);
  concrete_rank_support_v.size_in_bits = size_in_bytes(wt_rank_support_v) * 8;
  concrete_rank_support_v.rank.num_queries = num_rank_queries;
  concrete_rank_support_v.access.num_queries = num_access_queries;

  cout << "Starting rank_support_v repeats" << endl;
  for (int r = 0; r < NUM_REPEATS; r++) {
    cout << "Starting rank_support_v construction" << endl;
    auto start_construction = high_resolution_clock::now();
    wt_int<bit_vector, rank_support_v<>, select_support_scan<1>,
           select_support_scan<0>>
        wt_rank_support_v;
    construct(wt_rank_support_v, dataset_name, 4);
    auto end_construction = high_resolution_clock::now();
    auto elapsed_construction =
        duration_cast<duration<double>>(end_construction - start_construction)
            .count();

    concrete_rank_support_v.construction.push_back(elapsed_construction);

    cout << "Starting rank_support_v rank repeat " << r << endl;
    auto start_rank = high_resolution_clock::now();
    for (auto i : random_rank_queries) {
      black_box(wt_rank_support_v.rank(i.first, i.second));
    }
    auto end_rank = high_resolution_clock::now();
    auto elapsed_rank =
        duration_cast<duration<double>>(end_rank - start_rank).count();

    concrete_rank_support_v.rank.total_time.push_back(elapsed_rank);
    cout << "elapsed_rank: " << elapsed_rank << endl;

    cout << "Starting rank_support_v access repeat " << r << endl;
    auto start_access = high_resolution_clock::now();
    for (auto i : random_access_queries) {
      black_box(wt_rank_support_v[i]);
    }
    auto end_access = high_resolution_clock::now();
    auto elapsed_access =
        duration_cast<duration<double>>(end_access - start_access).count();

    concrete_rank_support_v.access.total_time.push_back(elapsed_access);
    cout << "elapsed_access: " << elapsed_access << endl;
  }
  wt_benchmark.rank_support_v.push_back(concrete_rank_support_v);

  NonParametrizableRankAccessBenchmark concrete_rank_support_v5;
  wt_int<bit_vector, rank_support_v5<>, select_support_scan<1>,
         select_support_scan<0>>
      wt_rank_support_v5;
  construct(wt_rank_support_v5, dataset_name, 4);
  concrete_rank_support_v5.size_in_bits = size_in_bytes(wt_rank_support_v5) * 8;
  concrete_rank_support_v5.rank.num_queries = num_rank_queries;
  concrete_rank_support_v5.access.num_queries = num_access_queries;

  cout << "Starting rank_support_v5 repeats" << endl;
  for (int r = 0; r < NUM_REPEATS; r++) {
    cout << "Starting rank_support_v5 construction" << endl;
    auto start_construction = high_resolution_clock::now();
    wt_int<bit_vector, rank_support_v5<>, select_support_scan<1>,
           select_support_scan<0>>
        wt_rank_support_v5;
    construct(wt_rank_support_v5, dataset_name, 4);
    auto end_construction = high_resolution_clock::now();
    auto elapsed_construction =
        duration_cast<duration<double>>(end_construction - start_construction)
            .count();

    concrete_rank_support_v5.construction.push_back(elapsed_construction);

    cout << "Starting rank_support_v5 rank repeat " << r << endl;
    auto start_rank = high_resolution_clock::now();
    for (auto i : random_rank_queries) {
      black_box(wt_rank_support_v5.rank(i.first, i.second));
    }
    auto end_rank = high_resolution_clock::now();
    auto elapsed_rank =
        duration_cast<duration<double>>(end_rank - start_rank).count();

    concrete_rank_support_v5.rank.total_time.push_back(elapsed_rank);
    cout << "elapsed_rank: " << elapsed_rank << endl;

    cout << "Starting rank_support_v5 access repeat " << r << endl;
    auto start_access = high_resolution_clock::now();
    for (auto i : random_access_queries) {
      black_box(wt_rank_support_v5[i]);
    }
    auto end_access = high_resolution_clock::now();
    auto elapsed_access =
        duration_cast<duration<double>>(end_access - start_access).count();

    concrete_rank_support_v5.access.total_time.push_back(elapsed_access);
    cout << "elapsed_access: " << elapsed_access << endl;
  }
  wt_benchmark.rank_support_v5.push_back(concrete_rank_support_v5);

  cout << "Starting wt_il 4" << endl;
  wt_benchmark.bit_vector_il.push_back(wt_il_benchmark<4>(
      dataset_name, random_rank_queries, random_select_queries,
      random_access_queries, num_rank_queries, num_select_queries,
      num_access_queries));
  cout << "Starting wt_il 8" << endl;
  wt_benchmark.bit_vector_il.push_back(wt_il_benchmark<8>(
      dataset_name, random_rank_queries, random_select_queries,
      random_access_queries, num_rank_queries, num_select_queries,
      num_access_queries));
  cout << "Starting wt_il 16" << endl;
  wt_benchmark.bit_vector_il.push_back(wt_il_benchmark<16>(
      dataset_name, random_rank_queries, random_select_queries,
      random_access_queries, num_rank_queries, num_select_queries,
      num_access_queries));
  cout << "Starting wt_il 32" << endl;
  wt_benchmark.bit_vector_il.push_back(wt_il_benchmark<32>(
      dataset_name, random_rank_queries, random_select_queries,
      random_access_queries, num_rank_queries, num_select_queries,
      num_access_queries));
  cout << "Starting wt_il 64" << endl;
  wt_benchmark.bit_vector_il.push_back(wt_il_benchmark<64>(
      dataset_name, random_rank_queries, random_select_queries,
      random_access_queries, num_rank_queries, num_select_queries,
      num_access_queries));
  cout << "Starting wt_il 128" << endl;
  wt_benchmark.bit_vector_il.push_back(wt_il_benchmark<128>(
      dataset_name, random_rank_queries, random_select_queries,
      random_access_queries, num_rank_queries, num_select_queries,
      num_access_queries));

  NonParametrizableSelectBenchmark concrete_select_support_mcl;
  wt_int<bit_vector, rank_support_scan<>, select_support_mcl<1>,
         select_support_mcl<0>>
      wt_select_support_mcl;

  construct(wt_select_support_mcl, dataset_name, 4);

  concrete_select_support_mcl.size_in_bits =
      size_in_bytes(wt_select_support_mcl) * 8;

  concrete_select_support_mcl.select.num_queries = num_select_queries;

  cout << "Starting select_support_mcl repeats" << endl;
  for (int r = 0; r < NUM_REPEATS; r++) {
    cout << "Starting select_support_mcl construction" << endl;
    auto start_construction = high_resolution_clock::now();
    wt_int<bit_vector, rank_support_scan<>, select_support_mcl<1>,
           select_support_mcl<0>>
        wt_select_support_mcl;
    construct(wt_select_support_mcl, dataset_name, 4);
    auto end_construction = high_resolution_clock::now();
    auto elapsed_construction =
        duration_cast<duration<double>>(end_construction - start_construction)
            .count();
    concrete_select_support_mcl.construction.push_back(elapsed_construction);

    cout << "Starting select_support_mcl select repeat " << r << endl;
    auto start_select = high_resolution_clock::now();
    for (auto i : random_select_queries) {
      black_box(wt_select_support_mcl.select(i.first, i.second));
    }
    auto end_select = high_resolution_clock::now();
    auto elapsed_select =
        duration_cast<duration<double>>(end_select - start_select).count();

    concrete_select_support_mcl.select.total_time.push_back(elapsed_select);
    cout << "elapsed_select: " << elapsed_select << endl;
  }
  wt_benchmark.select_support_mcl.push_back(concrete_select_support_mcl);

  cout << "Starting wt_rrr_vector 63 4" << endl;
  wt_benchmark.rrr_sdsl_63.push_back(wt_rrr_vector_benchmark<63, 4>(
      dataset_name, random_rank_queries, random_select_queries,
      random_access_queries, num_rank_queries, num_select_queries,
      num_access_queries));
  cout << "Starting wt_rrr_vector 63 8" << endl;
  wt_benchmark.rrr_sdsl_63.push_back(wt_rrr_vector_benchmark<63, 8>(
      dataset_name, random_rank_queries, random_select_queries,
      random_access_queries, num_rank_queries, num_select_queries,
      num_access_queries));
  cout << "Starting wt_rrr_vector 63 16" << endl;
  wt_benchmark.rrr_sdsl_63.push_back(wt_rrr_vector_benchmark<63, 16>(
      dataset_name, random_rank_queries, random_select_queries,
      random_access_queries, num_rank_queries, num_select_queries,
      num_access_queries));
  cout << "Starting wt_rrr_vector 63 32" << endl;
  wt_benchmark.rrr_sdsl_63.push_back(wt_rrr_vector_benchmark<63, 32>(
      dataset_name, random_rank_queries, random_select_queries,
      random_access_queries, num_rank_queries, num_select_queries,
      num_access_queries));
  cout << "Starting wt_rrr_vector 63 64" << endl;
  wt_benchmark.rrr_sdsl_63.push_back(wt_rrr_vector_benchmark<63, 64>(
      dataset_name, random_rank_queries, random_select_queries,
      random_access_queries, num_rank_queries, num_select_queries,
      num_access_queries));
  cout << "Starting wt_rrr_vector 63 128" << endl;
  wt_benchmark.rrr_sdsl_63.push_back(wt_rrr_vector_benchmark<63, 128>(
      dataset_name, random_rank_queries, random_select_queries,
      random_access_queries, num_rank_queries, num_select_queries,
      num_access_queries));

  return wt_benchmark;
}

void save_benchmarks(Benchmark benchmark) {
  //   // Serialize benchmark to ../benchmarks
  ofstream o("../benchmarks.json");
  o << json(benchmark).dump(2) << std::endl;
}

int main() {
  ifstream f("../benchmarks.json");
  nlohmann::json j;
  f >> j;
  Benchmark benchmark = j.get<Benchmark>();

  // // Create bit vectors
  bit_vector bitvec_05 = bitvec_with_distribution(BITVEC_SIZE, 0.05);
  bit_vector bitvec_50 = bitvec_with_distribution(BITVEC_SIZE, 0.5);
  bit_vector bitvec_90 = bitvec_with_distribution(BITVEC_SIZE, 0.9);

  // Sampling benchmark
  benchmark.sampling.five =
      sampling_benchmark(bitvec_05, benchmark.sampling.five, 1.0);
  save_benchmarks(benchmark);

  benchmark.sampling.fifty =
      sampling_benchmark(bitvec_50, benchmark.sampling.fifty, 0.2);
  save_benchmarks(benchmark);

  benchmark.sampling.ninety =
      sampling_benchmark(bitvec_90, benchmark.sampling.ninety, 0.1);
  save_benchmarks(benchmark);

  // Compressed benchmark
  benchmark.compressed.five =
      compressed_benchmark(bitvec_05, benchmark.compressed.five, 1.0);
  save_benchmarks(benchmark);

  benchmark.compressed.fifty =
      compressed_benchmark(bitvec_50, benchmark.compressed.fifty, 0.2);
  save_benchmarks(benchmark);

  benchmark.compressed.ninety =
      compressed_benchmark(bitvec_90, benchmark.compressed.ninety, 0.1);
  save_benchmarks(benchmark);

  // Wt benchmark
  benchmark.wt.proteins =
      wt_benchmark("../datasets/proteins.200MB.bin", benchmark.wt.proteins);
  save_benchmarks(benchmark);

  benchmark.wt.english =
      wt_benchmark("../datasets/english.200MB.bin", benchmark.wt.english);
  save_benchmarks(benchmark);

  return 0;
}