#include <iostream>
#include <sdsl/bit_vectors.hpp>
#include <sdsl/wavelet_trees.hpp>

using namespace std;
using namespace sdsl;

int main() {
  wt_int<bit_vector, rank_support_v<>> wt;
  //   wt_blcd<bit_vector, rank_support_v<>> wt;

  //   std::mt19937_64 rng;
  //   std::uniform_int_distribution<uint64_t> distribution(0, 256);
  //   auto dice = bind(distribution, rng);
  //   int_vector<8> text(100000000, 0);
  //   // fill text with random values
  //   for (int i = 0; i < 100000000; i++) {
  //     text[i] = dice() % 256;
  //   }
  //   //   Measure time
  auto start = std::chrono::high_resolution_clock::now();

  //   construct_im(wt, text);
  construct(wt, "/home/jorge/tmp/einstein.en.txt", 1);
  auto end = std::chrono::high_resolution_clock::now();
  auto elapsed_seconds =
      std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
  cout << "Construction elapsed time: " << elapsed_seconds.count() << "s\n";

  // Measure time
  start = std::chrono::high_resolution_clock::now();
  uint64_t sum_ranks = 0;
  for (int j = 0; j < wt.size(); ++j) {
    sum_ranks += wt.rank(j, wt[j]);
  }
  end = std::chrono::high_resolution_clock::now();
  elapsed_seconds =
      std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
  cout << "Rank elapsed time: " << elapsed_seconds.count() << "s\n";

  cout << "sum_ranks: " << sum_ranks << endl;
  cout << "size: " << size_in_bytes(wt) / 1024 << endl;

  // Do random queries for all characters that do not occur in the string
  //   for (size_type j = 0; j < cnt.size(); ++j) {
  //     if (cnt[j] == 0) {
  //       for (size_type k = 0; k < 1000; ++k) {
  //         size_type pos = dice();
  //         ASSERT_EQ((size_type)0, wt.rank(pos, (unsigned char)j))
  //             << " pos=" << pos;
  //       }
  //     }
  //   }
  //   // Test rank(size(), c) for each character c
  //   for (size_type c = 0; c < 256; ++c) {
  //     ASSERT_EQ(cnt[c], wt.rank(wt.size(), (unsigned char)c)) << " c=" << c;
  //   }
}