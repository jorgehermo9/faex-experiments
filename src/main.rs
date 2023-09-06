use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Read, Write};
use std::time::Instant;

use faex::bit_vectors::rank_select::{DenseSamplingRank, Select, SparseSamplingRank};
use faex::bit_vectors::RRRBitVec;
use faex::bit_vectors::{rank_select::Rank, BitVec};
use faex::character_sequence::wavelet_tree::WaveletTree;
use faex::character_sequence::{CharacterAccess, CharacterRank, CharacterSelect};
use faex::profiling::HeapSize;
use faex::Build;
use rand::seq::SliceRandom;
use rand::Rng;
use serde::{Deserialize, Serialize};
#[derive(Debug, Default, Serialize, Deserialize)]
struct Benchmark {
    sampling: SamplingBenchmarkDatasets,
    compressed: CompressedBenchmarkDatasets,
    wt: WtBenchmarkDatasets,
}

#[derive(Debug, Default, Serialize, Deserialize)]
struct SamplingBenchmarkDatasets {
    five: SamplingBenchmark,
    fifty: SamplingBenchmark,
    ninety: SamplingBenchmark,
}
#[derive(Debug, Default, Serialize, Deserialize)]
struct CompressedBenchmarkDatasets {
    five: CompressedBenchmark,
    fifty: CompressedBenchmark,
    ninety: CompressedBenchmark,
}
#[derive(Debug, Default, Serialize, Deserialize)]
struct WtBenchmarkDatasets {
    english: WtBenchmark,
    proteins: WtBenchmark,
}

#[derive(Debug, Default, Serialize, Deserialize)]
struct SamplingBenchmark {
    original_size: usize,
    dense_sampling: Vec<ParametrizableRankSelectBenchmark>,
    sparse_sampling: Vec<ParametrizableRankSelectBenchmark>,
    bit_vector_il: Vec<ParametrizableRankSelectBenchmark>,
    rank_support_v: Vec<NonParametrizableRankBenchmark>,
    rank_support_v5: Vec<NonParametrizableRankBenchmark>,
    select_support_mcl: Vec<NonParametrizableSelectBenchmark>,
}

#[derive(Debug, Default, Serialize, Deserialize)]
struct ParametrizableRankSelectBenchmark {
    parameter: usize,
    size_in_bits: usize,
    construction: Vec<f64>,
    rank: MultipleQueriesBenchmark,
    select: MultipleQueriesBenchmark,
}

#[derive(Debug, Default, Serialize, Deserialize)]
struct NonParametrizableRankBenchmark {
    size_in_bits: usize,
    construction: Vec<f64>,
    rank: MultipleQueriesBenchmark,
}

#[derive(Debug, Default, Serialize, Deserialize)]
struct NonParametrizableSelectBenchmark {
    size_in_bits: usize,
    construction: Vec<f64>,
    select: MultipleQueriesBenchmark,
}

#[derive(Debug, Default, Serialize, Deserialize)]
struct MultipleQueriesBenchmark {
    num_queries: usize,
    total_time: Vec<f64>,
}

#[derive(Debug, Default, Serialize, Deserialize)]
struct CompressedBenchmark {
    original_size: usize,
    rrr_faex_15: Vec<ParametrizableRankSelectBenchmark>,
    rrr_faex_31: Vec<ParametrizableRankSelectBenchmark>,
    rrr_faex_63: Vec<ParametrizableRankSelectBenchmark>,
    rrr_sdsl_15: Vec<ParametrizableRankSelectBenchmark>,
    rrr_sdsl_31: Vec<ParametrizableRankSelectBenchmark>,
    rrr_sdsl_63: Vec<ParametrizableRankSelectBenchmark>,
}

#[derive(Debug, Default, Serialize, Deserialize)]
struct ParametrizableRankSelectAccessBenchmark {
    parameter: usize,
    size_in_bits: usize,
    construction: Vec<f64>,
    rank: MultipleQueriesBenchmark,
    select: MultipleQueriesBenchmark,
    access: MultipleQueriesBenchmark,
}

#[derive(Debug, Default, Serialize, Deserialize)]
struct NonParametrizableRankAccessBenchmark {
    size_in_bits: usize,
    construction: Vec<f64>,
    rank: MultipleQueriesBenchmark,
    access: MultipleQueriesBenchmark,
}

#[derive(Debug, Default, Serialize, Deserialize)]
struct WtBenchmark {
    original_size: usize,
    dense_sampling: Vec<ParametrizableRankSelectAccessBenchmark>,
    sparse_sampling: Vec<ParametrizableRankSelectAccessBenchmark>,
    bit_vector_il: Vec<ParametrizableRankSelectAccessBenchmark>,
    rank_support_v: Vec<NonParametrizableRankAccessBenchmark>,
    rank_support_v5: Vec<NonParametrizableRankAccessBenchmark>,
    select_support_mcl: Vec<NonParametrizableSelectBenchmark>,
    rrr_faex_63: Vec<ParametrizableRankSelectAccessBenchmark>,
    rrr_sdsl_63: Vec<ParametrizableRankSelectAccessBenchmark>,
}

fn bitvec_with_distribution(size: usize, density: f64) -> BitVec {
    let mut bitvec = BitVec::from_value(false, size);

    // create a bit vector with the exact given density.
    // do not rely on randomness to get the exact density,
    // but the positions should be random

    let mut rng = rand::thread_rng();
    let num_ones = (size as f64 * density) as usize;

    let all_positions: Vec<usize> = (0..size).collect();
    let ones_positions = all_positions.choose_multiple(&mut rng, num_ones);

    for i in ones_positions {
        bitvec.set(*i, true);
    }

    bitvec
}

fn sampling_benchmark(bv: BitVec, select_ratio: f64) -> SamplingBenchmark {
    let mut sampling_benchmark = SamplingBenchmark {
        original_size: bv.len(),
        ..Default::default()
    };

    let num_ones = bv.rank(bv.len()).unwrap();
    let mut rng = rand::thread_rng();
    let mut random_rank_positions = (0..=bv.len()).collect::<Vec<_>>();
    random_rank_positions.shuffle(&mut rng);
    let mut random_select_values = (0..=num_ones).collect::<Vec<_>>();
    random_select_values.shuffle(&mut rng);

    let random_select_values = random_select_values
        .iter()
        .take((random_select_values.len() as f64 * select_ratio) as usize)
        .cloned()
        .collect::<Vec<_>>();

    let num_rank_queries = random_rank_positions.len();
    let num_select_queries = random_select_values.len();

    for k in [4, 8, 16, 32, 64, 128] {
        // Dense Sampling Rank

        let mut concrete_dense_sampling = ParametrizableRankSelectBenchmark {
            parameter: k,
            ..Default::default()
        };
        let spec = DenseSamplingRank::spec(k);

        let structure = spec.build(bv.clone());
        concrete_dense_sampling.size_in_bits = structure.rank_support().heap_size_in_bits();
        concrete_dense_sampling.rank.num_queries = num_rank_queries;
        concrete_dense_sampling.select.num_queries = num_select_queries;

        for r in 0..NUM_REPEATS {
            println!("Dense Sampling k = {}, repeat = {}", k, r);
            let bv_clone = bv.clone();
            let start_build = Instant::now();
            let structure = spec.build(bv_clone);
            concrete_dense_sampling
                .construction
                .push(start_build.elapsed().as_secs_f64());
            let start_rank = Instant::now();
            for i in random_rank_positions.clone() {
                std::hint::black_box(structure.rank(i));
            }
            concrete_dense_sampling
                .rank
                .total_time
                .push(start_rank.elapsed().as_secs_f64());
            dbg!(concrete_dense_sampling.rank.total_time.last().unwrap());
            let start_select = Instant::now();
            for i in random_select_values.clone() {
                std::hint::black_box(structure.select(i));
            }
            concrete_dense_sampling
                .select
                .total_time
                .push(start_select.elapsed().as_secs_f64());
            dbg!(concrete_dense_sampling.select.total_time.last().unwrap());
        }

        sampling_benchmark
            .dense_sampling
            .push(concrete_dense_sampling);

        // Sparse Sampling Rank
        let mut concrete_sparse_sampling = ParametrizableRankSelectBenchmark {
            parameter: k,
            ..Default::default()
        };

        let spec = SparseSamplingRank::spec(k);
        let structure = spec.build(bv.clone());
        concrete_sparse_sampling.size_in_bits = structure.rank_support().heap_size_in_bits();
        concrete_sparse_sampling.rank.num_queries = num_rank_queries;
        concrete_sparse_sampling.select.num_queries = num_select_queries;

        for r in 0..NUM_REPEATS {
            println!("Sparse Sampling k = {}, repeat = {}", k, r);
            let bv_clone = bv.clone();
            let start_build = Instant::now();
            let structure = spec.build(bv_clone);
            concrete_sparse_sampling
                .construction
                .push(start_build.elapsed().as_secs_f64());

            let start_rank = Instant::now();
            for i in random_rank_positions.clone() {
                std::hint::black_box(structure.rank(i));
            }

            concrete_sparse_sampling
                .rank
                .total_time
                .push(start_rank.elapsed().as_secs_f64());
            dbg!(concrete_sparse_sampling.rank.total_time.last().unwrap());

            let start_select = Instant::now();
            for i in random_select_values.clone() {
                std::hint::black_box(structure.select(i));
            }
            concrete_sparse_sampling
                .select
                .total_time
                .push(start_select.elapsed().as_secs_f64());
            dbg!(concrete_sparse_sampling.select.total_time.last().unwrap());
        }

        sampling_benchmark
            .sparse_sampling
            .push(concrete_sparse_sampling);
    }
    sampling_benchmark
}

fn compressed_benchmark(bv: BitVec, select_ratio: f64) -> CompressedBenchmark {
    let mut compressed_benchmark = CompressedBenchmark {
        original_size: bv.len(),
        ..Default::default()
    };

    let num_ones = bv.rank(bv.len()).unwrap();
    let mut rng = rand::thread_rng();
    let mut random_rank_positions = (0..=bv.len()).collect::<Vec<_>>();
    random_rank_positions.shuffle(&mut rng);
    let mut random_select_values = (0..=num_ones).collect::<Vec<_>>();
    random_select_values.shuffle(&mut rng);

    let random_select_values = random_select_values
        .iter()
        .take((random_select_values.len() as f64 * select_ratio) as usize)
        .cloned()
        .collect::<Vec<_>>();

    let num_rank_queries = random_rank_positions.len();
    let num_select_queries = random_select_values.len();
    for k in [4, 8, 16, 32, 64, 128] {
        // RRRBitVec 15
        let mut concrete_rrr_faex_15 = ParametrizableRankSelectBenchmark {
            parameter: k,
            ..Default::default()
        };
        let spec = RRRBitVec::spec(15, k);
        let structure = spec.build(bv.clone());
        concrete_rrr_faex_15.size_in_bits = structure.heap_size_in_bits();
        concrete_rrr_faex_15.rank.num_queries = num_rank_queries;
        concrete_rrr_faex_15.select.num_queries = num_select_queries;

        for r in 0..NUM_REPEATS {
            println!("RRRBitVec 15 k = {}, repeat = {}", k, r);
            let bv_clone = bv.clone();
            let start_build = Instant::now();
            let structure = spec.build(bv_clone);
            concrete_rrr_faex_15
                .construction
                .push(start_build.elapsed().as_secs_f64());

            let start_rank = Instant::now();
            for i in random_rank_positions.clone() {
                std::hint::black_box(structure.rank(i));
            }
            concrete_rrr_faex_15
                .rank
                .total_time
                .push(start_rank.elapsed().as_secs_f64());
            dbg!(concrete_rrr_faex_15.rank.total_time.last().unwrap());

            let start_select = Instant::now();
            for i in random_select_values.clone() {
                std::hint::black_box(structure.select(i));
            }
            concrete_rrr_faex_15
                .select
                .total_time
                .push(start_select.elapsed().as_secs_f64());
            dbg!(concrete_rrr_faex_15.select.total_time.last().unwrap());
        }

        compressed_benchmark.rrr_faex_15.push(concrete_rrr_faex_15);

        // RRRBitVec 31

        let mut concrete_rrr_faex_31 = ParametrizableRankSelectBenchmark {
            parameter: k,
            ..Default::default()
        };

        let spec = RRRBitVec::spec(31, k);
        let structure = spec.build(bv.clone());
        concrete_rrr_faex_31.size_in_bits = structure.heap_size_in_bits();
        concrete_rrr_faex_31.rank.num_queries = num_rank_queries;
        concrete_rrr_faex_31.select.num_queries = num_select_queries;

        for r in 0..NUM_REPEATS {
            println!("RRRBitVec 31 k = {}, repeat = {}", k, r);
            let bv_clone = bv.clone();
            let start_build = Instant::now();
            let structure = spec.build(bv_clone);
            concrete_rrr_faex_31
                .construction
                .push(start_build.elapsed().as_secs_f64());

            let start_rank = Instant::now();
            for i in random_rank_positions.clone() {
                std::hint::black_box(structure.rank(i));
            }
            concrete_rrr_faex_31
                .rank
                .total_time
                .push(start_rank.elapsed().as_secs_f64());
            dbg!(concrete_rrr_faex_31.rank.total_time.last().unwrap());

            let start_select = Instant::now();
            for i in random_select_values.clone() {
                std::hint::black_box(structure.select(i));
            }
            concrete_rrr_faex_31
                .select
                .total_time
                .push(start_select.elapsed().as_secs_f64());
            dbg!(concrete_rrr_faex_31.select.total_time.last().unwrap());
        }

        compressed_benchmark.rrr_faex_31.push(concrete_rrr_faex_31);

        // RRRBitVec 63

        let mut concrete_rrr_faex_63 = ParametrizableRankSelectBenchmark {
            parameter: k,
            ..Default::default()
        };

        let spec = RRRBitVec::spec(63, k);
        let structure = spec.build(bv.clone());
        concrete_rrr_faex_63.size_in_bits = structure.heap_size_in_bits();
        concrete_rrr_faex_63.rank.num_queries = num_rank_queries;
        concrete_rrr_faex_63.select.num_queries = num_select_queries;

        for r in 0..NUM_REPEATS {
            println!("RRRBitVec 63 k = {}, repeat = {}", k, r);
            let bv_clone = bv.clone();
            let start_build = Instant::now();
            let structure = spec.build(bv_clone);
            concrete_rrr_faex_63
                .construction
                .push(start_build.elapsed().as_secs_f64());

            let start_rank = Instant::now();
            for i in random_rank_positions.clone() {
                std::hint::black_box(structure.rank(i));
            }
            concrete_rrr_faex_63
                .rank
                .total_time
                .push(start_rank.elapsed().as_secs_f64());
            dbg!(concrete_rrr_faex_63.rank.total_time.last().unwrap());

            let start_select = Instant::now();
            for i in random_select_values.clone() {
                std::hint::black_box(structure.select(i));
            }
            concrete_rrr_faex_63
                .select
                .total_time
                .push(start_select.elapsed().as_secs_f64());
            dbg!(concrete_rrr_faex_63.select.total_time.last().unwrap());
        }

        compressed_benchmark.rrr_faex_63.push(concrete_rrr_faex_63);
    }

    compressed_benchmark
}

fn wt_benchmark(dataset_name: &str) -> WtBenchmark {
    let mut wt_benchmark = WtBenchmark::default();
    // read text as lossy string
    let mut dataset = File::open(format!("datasets/{dataset_name}")).unwrap();
    let mut buffer = Vec::new();
    dataset.read_to_end(&mut buffer).unwrap();
    let text = String::from_utf8_lossy(&buffer).to_string();

    // Sdsl needs the file in binary format...
    let bin_file = File::create(format!("datasets/{dataset_name}.bin")).unwrap();
    let mut bin_file_buffered = BufWriter::new(bin_file);
    for c in text.chars() {
        let c = c as u32;
        bin_file_buffered.write_all(&c.to_le_bytes()).unwrap();
    }
    bin_file_buffered.flush().unwrap();

    let dense_sampling_spec = DenseSamplingRank::spec(4);
    let wt_spec = WaveletTree::spec(dense_sampling_spec);

    let structure = wt_spec.build(&text);
    // n * ceil(log_2 sima)
    let alphabet = structure.alphabet();
    wt_benchmark.original_size = structure.len() * (alphabet.len() as f64).log2().ceil() as usize;
    let alphabet_occs = structure
        .alphabet()
        .iter()
        .map(|a| (*a, structure.rank(*a, structure.len()).unwrap()))
        .collect::<HashMap<_, _>>();

    let mut rng = rand::thread_rng();
    let random_chars = (0..=structure.len() / 10)
        .map(|_| *alphabet.choose(&mut rng).unwrap())
        .collect::<Vec<_>>();

    let mut random_rank_positions = (0..random_chars.len()).collect::<Vec<_>>();
    random_rank_positions.shuffle(&mut rng);

    let random_rank_queries = random_chars
        .iter()
        .cloned()
        .zip(random_rank_positions.iter().cloned())
        .collect::<Vec<_>>();

    let random_select_queries = random_chars
        .iter()
        .take(random_chars.len() / 2)
        .map(|c| (*c, rng.gen_range(0..=*alphabet_occs.get(c).unwrap())))
        .collect::<Vec<_>>();

    let mut random_access_queries = (0..random_chars.len() / 2).collect::<Vec<_>>();
    random_access_queries.shuffle(&mut rng);

    let num_rank_queries = random_rank_queries.len();
    let num_select_queries = random_select_queries.len();
    let num_access_queries = random_access_queries.len();

    for k in [4, 8, 16, 32, 64, 128] {
        // Dense Sampling Rank
        let mut concrete_dense_sampling = ParametrizableRankSelectAccessBenchmark {
            parameter: k,
            ..Default::default()
        };

        let dense_sampling_spec = DenseSamplingRank::spec(k);
        let wt_spec = WaveletTree::spec(dense_sampling_spec);
        let structure = wt_spec.build(&text);
        concrete_dense_sampling.size_in_bits = structure.heap_size_in_bits();
        concrete_dense_sampling.rank.num_queries = num_rank_queries;
        concrete_dense_sampling.select.num_queries = num_select_queries;
        concrete_dense_sampling.access.num_queries = num_access_queries;

        for r in 0..NUM_REPEATS {
            println!("Wt Dense Sampling k = {}, repeat = {}", k, r);
            let start_build = Instant::now();
            let structure = wt_spec.build(&text);
            concrete_dense_sampling
                .construction
                .push(start_build.elapsed().as_secs_f64());

            println!("Wt Dense Sampling rank starting");
            let start_rank = Instant::now();
            for (c, i) in random_rank_queries.clone() {
                std::hint::black_box(structure.rank(c, i));
            }
            concrete_dense_sampling
                .rank
                .total_time
                .push(start_rank.elapsed().as_secs_f64());
            dbg!(concrete_dense_sampling.rank.total_time.last().unwrap());

            println!("Wt Dense Sampling select starting");
            let start_select = Instant::now();
            for (c, i) in random_select_queries.clone() {
                std::hint::black_box(structure.select(c, i));
            }
            concrete_dense_sampling
                .select
                .total_time
                .push(start_select.elapsed().as_secs_f64());
            dbg!(concrete_dense_sampling.select.total_time.last().unwrap());

            println!("Wt Dense Sampling access starting");
            let start_access = Instant::now();
            for i in random_access_queries.clone() {
                std::hint::black_box(structure.access(i));
            }
            concrete_dense_sampling
                .access
                .total_time
                .push(start_access.elapsed().as_secs_f64());
            dbg!(concrete_dense_sampling.access.total_time.last().unwrap());
        }
        wt_benchmark.dense_sampling.push(concrete_dense_sampling);

        // Sparse Sampling Rank
        let mut concrete_sparse_sampling = ParametrizableRankSelectAccessBenchmark {
            parameter: k,
            ..Default::default()
        };

        let sparse_sampling_spec = SparseSamplingRank::spec(k);
        let wt_spec = WaveletTree::spec(sparse_sampling_spec);

        let structure = wt_spec.build(&text);
        concrete_sparse_sampling.size_in_bits = structure.heap_size_in_bits();
        concrete_sparse_sampling.rank.num_queries = num_rank_queries;
        concrete_sparse_sampling.select.num_queries = num_select_queries;
        concrete_sparse_sampling.access.num_queries = num_access_queries;

        for r in 0..NUM_REPEATS {
            println!("Wt Sparse Sampling k = {}, repeat = {}", k, r);
            let start_build = Instant::now();
            let structure = wt_spec.build(&text);
            concrete_sparse_sampling
                .construction
                .push(start_build.elapsed().as_secs_f64());

            println!("Wt Sparse Sampling rank starting");
            let start_rank = Instant::now();
            for (c, i) in random_rank_queries.clone() {
                std::hint::black_box(structure.rank(c, i));
            }
            concrete_sparse_sampling
                .rank
                .total_time
                .push(start_rank.elapsed().as_secs_f64());
            dbg!(concrete_sparse_sampling.rank.total_time.last().unwrap());

            println!("Wt Sparse Sampling select starting");
            let start_select = Instant::now();
            for (c, i) in random_select_queries.clone() {
                std::hint::black_box(structure.select(c, i));
            }
            concrete_sparse_sampling
                .select
                .total_time
                .push(start_select.elapsed().as_secs_f64());
            dbg!(concrete_sparse_sampling.select.total_time.last().unwrap());

            println!("Wt Sparse Sampling access starting");
            let start_access = Instant::now();
            for i in random_access_queries.clone() {
                std::hint::black_box(structure.access(i));
            }
            concrete_sparse_sampling
                .access
                .total_time
                .push(start_access.elapsed().as_secs_f64());
            dbg!(concrete_sparse_sampling.access.total_time.last().unwrap());
        }
        wt_benchmark.sparse_sampling.push(concrete_sparse_sampling);

        // rrr_faex_63

        let mut concrete_rrr_faex_63 = ParametrizableRankSelectAccessBenchmark {
            parameter: k,
            ..Default::default()
        };

        let rrr_spec = RRRBitVec::spec(63, k);
        let wt_spec = WaveletTree::spec(rrr_spec);

        let structure = wt_spec.build(&text);

        concrete_rrr_faex_63.size_in_bits = structure.heap_size_in_bits();
        concrete_rrr_faex_63.rank.num_queries = num_rank_queries;
        concrete_rrr_faex_63.select.num_queries = num_select_queries;
        concrete_rrr_faex_63.access.num_queries = num_access_queries;

        for r in 0..NUM_REPEATS {
            println!("Wt RRRBitVec 63 k = {}, repeat = {}", k, r);
            let start_build = Instant::now();
            let structure = wt_spec.build(&text);
            concrete_rrr_faex_63
                .construction
                .push(start_build.elapsed().as_secs_f64());

            println!("Wt RRRBitVec 63 rank starting");
            let start_rank = Instant::now();
            for (c, i) in random_rank_queries.clone() {
                std::hint::black_box(structure.rank(c, i));
            }
            concrete_rrr_faex_63
                .rank
                .total_time
                .push(start_rank.elapsed().as_secs_f64());
            dbg!(concrete_rrr_faex_63.rank.total_time.last().unwrap());

            println!("Wt RRRBitVec 63 select starting");
            let start_select = Instant::now();
            for (c, i) in random_select_queries.clone() {
                std::hint::black_box(structure.select(c, i));
            }
            concrete_rrr_faex_63
                .select
                .total_time
                .push(start_select.elapsed().as_secs_f64());
            dbg!(concrete_rrr_faex_63.select.total_time.last().unwrap());

            println!("Wt RRRBitVec 63 access starting");
            let start_access = Instant::now();
            for i in random_access_queries.clone() {
                std::hint::black_box(structure.access(i));
            }
            concrete_rrr_faex_63
                .access
                .total_time
                .push(start_access.elapsed().as_secs_f64());
            dbg!(concrete_rrr_faex_63.access.total_time.last().unwrap());
        }
        wt_benchmark.rrr_faex_63.push(concrete_rrr_faex_63);
    }
    wt_benchmark
}

const BIT_VEC_SIZE: usize = 100_000_000;
const NUM_REPEATS: usize = 1;
fn main() {
    let default = Benchmark::default();
    let serialized = serde_json::to_string_pretty(&default).unwrap();
    std::fs::write("benchmarks.json", serialized).unwrap();

    // Warm CPU
    println!("Warming CPU");
    {
        let warm_bv = bitvec_with_distribution(BIT_VEC_SIZE, 0.05);
        std::hint::black_box(sampling_benchmark(warm_bv.clone(), 1.0));
    }
    let sampling = SamplingBenchmarkDatasets {
        five: sampling_benchmark(bitvec_with_distribution(BIT_VEC_SIZE, 0.05), 1.0),
        fifty: sampling_benchmark(bitvec_with_distribution(BIT_VEC_SIZE, 0.5), 0.2),
        ninety: sampling_benchmark(bitvec_with_distribution(BIT_VEC_SIZE, 0.9), 0.1),
    };

    let compressed = CompressedBenchmarkDatasets {
        five: compressed_benchmark(bitvec_with_distribution(BIT_VEC_SIZE, 0.05), 1.0),
        fifty: compressed_benchmark(bitvec_with_distribution(BIT_VEC_SIZE, 0.5), 0.2),
        ninety: compressed_benchmark(bitvec_with_distribution(BIT_VEC_SIZE, 0.9), 0.1),
    };

    let wt = WtBenchmarkDatasets {
        english: wt_benchmark("english.200MB"),
        proteins: wt_benchmark("proteins.200MB"),
    };

    let benchmark = Benchmark {
        sampling,
        compressed,
        wt,
    };

    // serialize pretty
    let serialized = serde_json::to_string_pretty(&benchmark).unwrap();
    std::fs::write("benchmarks.json", serialized).unwrap();
}
