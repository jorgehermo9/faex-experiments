#include <array>
#include <queue>
#include <sdsl/wavelet_trees.hpp>
#include <string>
#include <vector>

using namespace sdsl;
using namespace std;

int main(int argc, char* argv[]) {
  //    string s = "barbarabierbarbarbar";
  auto file = "test.txt";
  wt_int<bit_vector, rank_support_v<>> wt;
  //   construct_im(wt, text);
  //   construct(wt, "/home/jorge/tmp/einstein.en.txt", 1);
  construct(wt, "/home/jorge/uni/tfg/experiments/datasets/proteins.50MB.bin",
            4);
  set<uint64_t> sigma;
  cout << "size: " << wt.size() << endl;
  for (auto i = 0; i < wt.size(); ++i) {
    sigma.insert(wt[i]);
  }
  cout << "sigma: " << sigma.size() << endl;
}