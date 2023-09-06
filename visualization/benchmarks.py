import json
from dataclasses import dataclass
from typing import List
import numpy as np


@dataclass
class MultipleQueriesBenchmark:
    num_queries: int
    total_time: List[float]

    def from_json(data):
        return MultipleQueriesBenchmark(
            num_queries=data["num_queries"], total_time=data["total_time"]
        )

    def time(self):
        return np.min(self.total_time) / self.num_queries

    def time_in_millis(self):
        return self.time() * 1000

    def time_in_micros(self):
        return self.time_in_millis() * 1000

    def time_in_nanos(self):
        return self.time_in_micros() * 1000


@dataclass
class ParametrizableRankSelectBenchmark:
    parameter: int
    size_in_bits: int
    construction: List[float]
    rank: MultipleQueriesBenchmark
    select: MultipleQueriesBenchmark

    def from_json(data):
        return ParametrizableRankSelectBenchmark(
            parameter=data["parameter"],
            size_in_bits=data["size_in_bits"],
            construction=data["construction"],
            rank=MultipleQueriesBenchmark.from_json(data["rank"]),
            select=MultipleQueriesBenchmark.from_json(data["select"]),
        )


@dataclass
class NonParametrizableRankBenchmark:
    size_in_bits: int
    construction: List[float]
    rank: MultipleQueriesBenchmark

    def from_json(data):
        return NonParametrizableRankBenchmark(
            size_in_bits=data["size_in_bits"],
            construction=data["construction"],
            rank=MultipleQueriesBenchmark.from_json(data["rank"]),
        )


@dataclass
class NonParametrizableSelectBenchmark:
    size_in_bits: int
    construction: List[float]
    select: MultipleQueriesBenchmark

    def from_json(data):
        return NonParametrizableSelectBenchmark(
            size_in_bits=data["size_in_bits"],
            construction=data["construction"],
            select=MultipleQueriesBenchmark.from_json(data["select"]),
        )


@dataclass
class ParametrizableRankSelectAccessBenchmark:
    parameter: int
    size_in_bits: int
    construction: List[float]
    rank: MultipleQueriesBenchmark
    select: MultipleQueriesBenchmark
    access: MultipleQueriesBenchmark

    def from_json(data):
        return ParametrizableRankSelectAccessBenchmark(
            parameter=data["parameter"],
            size_in_bits=data["size_in_bits"],
            construction=data["construction"],
            rank=MultipleQueriesBenchmark.from_json(data["rank"]),
            select=MultipleQueriesBenchmark.from_json(data["select"]),
            access=MultipleQueriesBenchmark.from_json(data["access"]),
        )


@dataclass
class NonParametrizableRankAccessBenchmark:
    size_in_bits: int
    construction: List[float]
    rank: MultipleQueriesBenchmark
    access: MultipleQueriesBenchmark

    def from_json(data):
        return NonParametrizableRankAccessBenchmark(
            size_in_bits=data["size_in_bits"],
            construction=data["construction"],
            rank=MultipleQueriesBenchmark.from_json(data["rank"]),
            access=MultipleQueriesBenchmark.from_json(data["access"]),
        )


@dataclass
class SamplingBenchmark:
    original_size: int
    dense_sampling: List[ParametrizableRankSelectBenchmark]
    sparse_sampling: List[ParametrizableRankSelectBenchmark]
    bit_vector_il: List[ParametrizableRankSelectBenchmark]
    rank_support_v: List[NonParametrizableRankBenchmark]
    rank_support_v5: List[NonParametrizableRankBenchmark]
    select_support_mcl: List[NonParametrizableSelectBenchmark]

    def from_json(data):
        return SamplingBenchmark(
            original_size=data["original_size"],
            dense_sampling=[
                ParametrizableRankSelectBenchmark.from_json(x)
                for x in data["dense_sampling"]
            ],
            sparse_sampling=[
                ParametrizableRankSelectBenchmark.from_json(x)
                for x in data["sparse_sampling"]
            ],
            bit_vector_il=[
                ParametrizableRankSelectBenchmark.from_json(x)
                for x in data["bit_vector_il"]
            ],
            rank_support_v=[
                NonParametrizableRankBenchmark.from_json(x)
                for x in data["rank_support_v"]
            ],
            rank_support_v5=[
                NonParametrizableRankBenchmark.from_json(x)
                for x in data["rank_support_v5"]
            ],
            select_support_mcl=[
                NonParametrizableSelectBenchmark.from_json(x)
                for x in data["select_support_mcl"]
            ],
        )


@dataclass
class CompressedBenchmark:
    original_size: int
    rrr_faex_15: List[ParametrizableRankSelectBenchmark]
    rrr_faex_31: List[ParametrizableRankSelectBenchmark]
    rrr_faex_63: List[ParametrizableRankSelectBenchmark]
    rrr_sdsl_15: List[ParametrizableRankSelectBenchmark]
    rrr_sdsl_31: List[ParametrizableRankSelectBenchmark]
    rrr_sdsl_63: List[ParametrizableRankSelectBenchmark]

    def from_json(data):
        return CompressedBenchmark(
            original_size=data["original_size"],
            rrr_faex_15=[
                ParametrizableRankSelectBenchmark.from_json(x)
                for x in data["rrr_faex_15"]
            ],
            rrr_faex_31=[
                ParametrizableRankSelectBenchmark.from_json(x)
                for x in data["rrr_faex_31"]
            ],
            rrr_faex_63=[
                ParametrizableRankSelectBenchmark.from_json(x)
                for x in data["rrr_faex_63"]
            ],
            rrr_sdsl_15=[
                ParametrizableRankSelectBenchmark.from_json(x)
                for x in data["rrr_sdsl_15"]
            ],
            rrr_sdsl_31=[
                ParametrizableRankSelectBenchmark.from_json(x)
                for x in data["rrr_sdsl_31"]
            ],
            rrr_sdsl_63=[
                ParametrizableRankSelectBenchmark.from_json(x)
                for x in data["rrr_sdsl_63"]
            ],
        )


@dataclass
class WtBenchmark:
    original_size: int
    dense_sampling: List[ParametrizableRankSelectAccessBenchmark]
    sparse_sampling: List[ParametrizableRankSelectAccessBenchmark]
    bit_vector_il: List[ParametrizableRankSelectAccessBenchmark]
    rank_support_v: List[NonParametrizableRankAccessBenchmark]
    rank_support_v5: List[NonParametrizableRankAccessBenchmark]
    select_support_mcl: List[NonParametrizableSelectBenchmark]
    rrr_faex_63: List[ParametrizableRankSelectAccessBenchmark]
    rrr_sdsl_63: List[ParametrizableRankSelectAccessBenchmark]

    def from_json(data):
        return WtBenchmark(
            original_size=data["original_size"],
            dense_sampling=[
                ParametrizableRankSelectAccessBenchmark.from_json(x)
                for x in data["dense_sampling"]
            ],
            sparse_sampling=[
                ParametrizableRankSelectAccessBenchmark.from_json(x)
                for x in data["sparse_sampling"]
            ],
            bit_vector_il=[
                ParametrizableRankSelectAccessBenchmark.from_json(x)
                for x in data["bit_vector_il"]
            ],
            rank_support_v=[
                NonParametrizableRankAccessBenchmark.from_json(x)
                for x in data["rank_support_v"]
            ],
            rank_support_v5=[
                NonParametrizableRankAccessBenchmark.from_json(x)
                for x in data["rank_support_v5"]
            ],
            select_support_mcl=[
                NonParametrizableSelectBenchmark.from_json(x)
                for x in data["select_support_mcl"]
            ],
            rrr_faex_63=[
                ParametrizableRankSelectAccessBenchmark.from_json(x)
                for x in data["rrr_faex_63"]
            ],
            rrr_sdsl_63=[
                ParametrizableRankSelectAccessBenchmark.from_json(x)
                for x in data["rrr_sdsl_63"]
            ],
        )


@dataclass
class WtBenchmarkDatasets:
    english: WtBenchmark
    proteins: WtBenchmark

    def from_json(data):
        return WtBenchmarkDatasets(
            english=WtBenchmark.from_json(data["english"]),
            proteins=WtBenchmark.from_json(data["proteins"]),
        )


@dataclass
class CompressedBenchmarkDatasets:
    five: CompressedBenchmark
    fifty: CompressedBenchmark
    ninety: CompressedBenchmark

    def from_json(data):
        return CompressedBenchmarkDatasets(
            five=CompressedBenchmark.from_json(data["five"]),
            fifty=CompressedBenchmark.from_json(data["fifty"]),
            ninety=CompressedBenchmark.from_json(data["ninety"]),
        )


@dataclass
class SamplingBenchmarkDatasets:
    five: SamplingBenchmark
    fifty: SamplingBenchmark
    ninety: SamplingBenchmark

    def from_json(data):
        return SamplingBenchmarkDatasets(
            five=SamplingBenchmark.from_json(data["five"]),
            fifty=SamplingBenchmark.from_json(data["fifty"]),
            ninety=SamplingBenchmark.from_json(data["ninety"]),
        )


@dataclass
class Benchmark:
    sampling: SamplingBenchmarkDatasets
    compressed: CompressedBenchmarkDatasets
    wt: WtBenchmark

    def from_json(data):
        return Benchmark(
            sampling=SamplingBenchmarkDatasets.from_json(data["sampling"]),
            compressed=CompressedBenchmarkDatasets.from_json(data["compressed"]),
            wt=WtBenchmarkDatasets.from_json(data["wt"]),
        )


# Define a function to load the data from a JSON file
def load_data_from_json(json_file):
    with open(json_file, "r") as file:
        data = json.load(file)

    return Benchmark.from_json(data)


def load_benchmarks():
    return load_data_from_json("../benchmarks.json")
