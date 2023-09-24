#!/usr/bin/env python3
from benchmarks import load_benchmarks
import os
import matplotlib.pyplot as plt
import matplotlib
import scienceplots

dense_sampling_rank_style = {
    "marker": "^",
    "fillstyle": "none",
    "color": "blue",
    "label": "dense_sampling_rank",
}

sparse_sampling_rank_style = {
    "marker": "^",
    "fillstyle": "none",
    "color": "red",
    "label": "sparse_sampling_rank",
}

bit_vector_ilstyle = {
    "marker": "x",
    "fillstyle": "none",
    "color": "green",
    "label": "bit_vector_il",
}

rank_support_v_style = {
    "marker": "x",
    "fillstyle": "none",
    "color": "brown",
    "label": "rank_support_v",
}

rank_support_v5_style = {
    "marker": "x",
    "fillstyle": "none",
    "color": "orange",
    "label": "rank_support_v5",
}

select_support_mcl_style = {
    "marker": "x",
    "fillstyle": "none",
    "color": "magenta",
    "label": "select_support_mcl",
}

rrr_faex_15_style = {
    "marker": "^",
    "fillstyle": "none",
    "color": "blue",
    "label": "rrr_faex_15",
}

rrr_faex_31_style = {
    "marker": "^",
    "fillstyle": "none",
    "color": "red",
    "label": "rrr_faex_31",
}

rrr_faex_63_style = {
    "marker": "^",
    "fillstyle": "none",
    "color": "purple",
    "label": "rrr_faex_63",
}

rrr_sdsl_15_style = {
    "marker": "x",
    "fillstyle": "none",
    "color": "blue",
    "label": "rrr_sdsl_15",
}

rrr_sdsl_31_style = {
    "marker": "x",
    "fillstyle": "none",
    "color": "red",
    "label": "rrr_sdsl_31",
}

rrr_sdsl_63_style = {
    "marker": "x",
    "fillstyle": "none",
    "color": "purple",
    "label": "rrr_sdsl_63",
}


def plot_sampling_benchmark(sampling_benchmark):
    # do not take last sampling rank since parameter 128 is not working well

    # create a 3 column 1 row figure
    datasets = [
        sampling_benchmark.five,
        sampling_benchmark.fifty,
        sampling_benchmark.ninety,
    ]
    datasets_names = ["5\%", "50\%", "90\%"]
    # Rank plots
    fig, axs = plt.subplots(1, 3, figsize=(8, 4))
    for ax, dataset, dataset_name in zip(axs, datasets, datasets_names):
        original_size = dataset.original_size

        dense_sampling_x = [
            x.size_in_bits / original_size * 100 for x in dataset.dense_sampling[:-1]
        ]
        dense_sampling_rank_y = [
            x.rank.time_in_nanos() for x in dataset.dense_sampling[:-1]
        ]
        ax.plot(
            dense_sampling_x,
            dense_sampling_rank_y,
            **dense_sampling_rank_style,
        )

        sparse_sampling_x = [
            x.size_in_bits / original_size * 100 for x in dataset.sparse_sampling
        ]
        sparse_sampling_rank_y = [
            x.rank.time_in_nanos() for x in dataset.sparse_sampling
        ]

        ax.plot(
            sparse_sampling_x,
            sparse_sampling_rank_y,
            **sparse_sampling_rank_style,
        )

        bit_vector_il_x = [
            x.size_in_bits / original_size * 100 for x in dataset.bit_vector_il
        ]
        bit_vector_il_rank_y = [x.rank.time_in_nanos() for x in dataset.bit_vector_il]

        ax.plot(
            bit_vector_il_x,
            bit_vector_il_rank_y,
            **bit_vector_ilstyle,
        )

        rank_support_v_x = [
            x.size_in_bits / original_size * 100 for x in dataset.rank_support_v
        ]
        rank_support_v_rank_y = [x.rank.time_in_nanos() for x in dataset.rank_support_v]

        ax.plot(
            rank_support_v_x,
            rank_support_v_rank_y,
            **rank_support_v_style,
        )

        rank_support_v5_x = [
            x.size_in_bits / original_size * 100 for x in dataset.rank_support_v5
        ]
        rank_support_v5_rank_y = [
            x.rank.time_in_nanos() for x in dataset.rank_support_v5
        ]

        ax.plot(
            rank_support_v5_x,
            rank_support_v5_rank_y,
            **rank_support_v5_style,
        )
        ax.set_title(f"{dataset_name} de densidad")
    fig.supxlabel("sobrecoste en espacio (\% de la secuencia original)")
    fig.supylabel("tiempo medio de rank (ns)")
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.05),
        ncol=5,
        fancybox=True,
        shadow=True,
    )
    # show plot
    # plt.show()
    # Save plot
    plt.tight_layout()
    plt.savefig("output/rank_sampling.pdf")

    # Select plots
    fig, axs = plt.subplots(1, 3, figsize=(8, 4))
    for ax, dataset, dataset_name in zip(axs, datasets, datasets_names):
        original_size = dataset.original_size

        dense_sampling_x = [
            x.size_in_bits / original_size * 100 for x in dataset.dense_sampling[:-1]
        ]
        dense_sampling_select_y = [
            x.select.time_in_nanos() for x in dataset.dense_sampling[:-1]
        ]
        ax.plot(
            dense_sampling_x,
            dense_sampling_select_y,
            **dense_sampling_rank_style,
        )

        sparse_sampling_x = [
            x.size_in_bits / original_size * 100 for x in dataset.sparse_sampling
        ]
        sparse_sampling_select_y = [
            x.select.time_in_nanos() for x in dataset.sparse_sampling
        ]

        ax.plot(
            sparse_sampling_x,
            sparse_sampling_select_y,
            **sparse_sampling_rank_style,
        )

        bit_vector_il_x = [
            x.size_in_bits / original_size * 100 for x in dataset.bit_vector_il
        ]
        bit_vector_il_select_y = [
            x.select.time_in_nanos() for x in dataset.bit_vector_il
        ]

        ax.plot(
            bit_vector_il_x,
            bit_vector_il_select_y,
            **bit_vector_ilstyle,
        )

        select_support_mcl_x = [
            x.size_in_bits / original_size * 100 for x in dataset.select_support_mcl
        ]
        select_support_mcl_select_y = [
            x.select.time_in_nanos() for x in dataset.select_support_mcl
        ]

        ax.plot(
            select_support_mcl_x,
            select_support_mcl_select_y,
            **select_support_mcl_style,
        )

        ax.set_title(f"{dataset_name} de densidad")
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.05),
        ncol=5,
        fancybox=True,
        shadow=True,
    )
    fig.supxlabel("sobrecoste en espacio (\% de la secuencia original)")
    fig.supylabel("tiempo medio de select (ns)")

    # show plot
    # plt.show()
    # Save plot
    plt.tight_layout()
    plt.savefig("output/select_sampling.pdf")


def plot_compressed_benchmark(compressed_benchmark):
    # create a 3 column 1 row figure
    datasets = [
        compressed_benchmark.five,
        compressed_benchmark.fifty,
        compressed_benchmark.ninety,
    ]
    datasets_names = ["5\%", "50\%", "90\%"]
    # Rank plots
    fig, axs = plt.subplots(1, 3, figsize=(8, 4))
    for ax, dataset, dataset_name in zip(axs, datasets, datasets_names):
        original_size = dataset.original_size

        rrr_faex_15_x = [
            x.size_in_bits / original_size * 100 for x in dataset.rrr_faex_15
        ]
        rrr_faex_15_rank_y = [x.rank.time_in_nanos() for x in dataset.rrr_faex_15]
        ax.plot(
            rrr_faex_15_x,
            rrr_faex_15_rank_y,
            **rrr_faex_15_style,
        )

        rrr_sdsl_15_x = [
            x.size_in_bits / original_size * 100 for x in dataset.rrr_sdsl_15
        ]
        rrr_sdsl_15_rank_y = [x.rank.time_in_nanos() for x in dataset.rrr_sdsl_15]
        ax.plot(
            rrr_sdsl_15_x,
            rrr_sdsl_15_rank_y,
            **rrr_sdsl_15_style,
        )

        rrr_faex_31_x = [
            x.size_in_bits / original_size * 100 for x in dataset.rrr_faex_31
        ]
        rrr_faex_31_rank_y = [x.rank.time_in_nanos() for x in dataset.rrr_faex_31]
        ax.plot(
            rrr_faex_31_x,
            rrr_faex_31_rank_y,
            **rrr_faex_31_style,
        )

        rrr_sdsl_31_x = [
            x.size_in_bits / original_size * 100 for x in dataset.rrr_sdsl_31
        ]
        rrr_sdsl_31_rank_y = [x.rank.time_in_nanos() for x in dataset.rrr_sdsl_31]
        ax.plot(
            rrr_sdsl_31_x,
            rrr_sdsl_31_rank_y,
            **rrr_sdsl_31_style,
        )

        rrr_faex_63_x = [
            x.size_in_bits / original_size * 100 for x in dataset.rrr_faex_63
        ]
        rrr_faex_63_rank_y = [x.rank.time_in_nanos() for x in dataset.rrr_faex_63]
        ax.plot(
            rrr_faex_63_x,
            rrr_faex_63_rank_y,
            **rrr_faex_63_style,
        )

        rrr_sdsl_63_x = [
            x.size_in_bits / original_size * 100 for x in dataset.rrr_sdsl_63
        ]
        rrr_sdsl_63_rank_y = [x.rank.time_in_nanos() for x in dataset.rrr_sdsl_63]
        ax.plot(
            rrr_sdsl_63_x,
            rrr_sdsl_63_rank_y,
            **rrr_sdsl_63_style,
        )
        # Vertical line
        ax.axvline(x=100, color="grey", linestyle="--")
        minx = min(
            rrr_faex_15_x
            + rrr_sdsl_15_x
            + rrr_faex_31_x
            + rrr_sdsl_31_x
            + rrr_faex_63_x
            + rrr_sdsl_63_x
            + [100]
        )
        ax.set_xlim(left=minx - 5)
        ax.set_title(f"{dataset_name} de densidad")

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.05),
        ncol=6,
        fancybox=True,
        shadow=True,
    )
    fig.supxlabel("espacio ocupado (\% de la secuencia original)")
    fig.supylabel("tiempo medio de rank (ns)")

    # show plot
    # plt.show()
    # Save plot
    plt.tight_layout()
    plt.savefig("output/rank_compressed.pdf")

    # Select plots
    fig, axs = plt.subplots(1, 3, figsize=(8, 4))
    for ax, dataset, dataset_name in zip(axs, datasets, datasets_names):
        original_size = dataset.original_size

        rrr_faex_15_x = [
            x.size_in_bits / original_size * 100 for x in dataset.rrr_faex_15
        ]
        rrr_faex_15_select_y = [x.select.time_in_nanos() for x in dataset.rrr_faex_15]
        ax.plot(
            rrr_faex_15_x,
            rrr_faex_15_select_y,
            **rrr_faex_15_style,
        )

        rrr_sdsl_15_x = [
            x.size_in_bits / original_size * 100 for x in dataset.rrr_sdsl_15
        ]
        rrr_sdsl_15_select_y = [x.select.time_in_nanos() for x in dataset.rrr_sdsl_15]
        ax.plot(
            rrr_sdsl_15_x,
            rrr_sdsl_15_select_y,
            **rrr_sdsl_15_style,
        )

        rrr_faex_31_x = [
            x.size_in_bits / original_size * 100 for x in dataset.rrr_faex_31
        ]
        rrr_faex_31_select_y = [x.select.time_in_nanos() for x in dataset.rrr_faex_31]
        ax.plot(
            rrr_faex_31_x,
            rrr_faex_31_select_y,
            **rrr_faex_31_style,
        )

        rrr_sdsl_31_x = [
            x.size_in_bits / original_size * 100 for x in dataset.rrr_sdsl_31
        ]
        rrr_sdsl_31_select_y = [x.select.time_in_nanos() for x in dataset.rrr_sdsl_31]
        ax.plot(
            rrr_sdsl_31_x,
            rrr_sdsl_31_select_y,
            **rrr_sdsl_31_style,
        )

        rrr_faex_63_x = [
            x.size_in_bits / original_size * 100 for x in dataset.rrr_faex_63
        ]
        rrr_faex_63_select_y = [x.select.time_in_nanos() for x in dataset.rrr_faex_63]
        ax.plot(
            rrr_faex_63_x,
            rrr_faex_63_select_y,
            **rrr_faex_63_style,
        )

        rrr_sdsl_63_x = [
            x.size_in_bits / original_size * 100 for x in dataset.rrr_sdsl_63
        ]
        rrr_sdsl_63_select_y = [x.select.time_in_nanos() for x in dataset.rrr_sdsl_63]
        ax.plot(
            rrr_sdsl_63_x,
            rrr_sdsl_63_select_y,
            **rrr_sdsl_63_style,
        )

        # Vertical line
        ax.axvline(x=100, color="grey", linestyle="--")
        minx = min(
            rrr_faex_15_x
            + rrr_sdsl_15_x
            + rrr_faex_31_x
            + rrr_sdsl_31_x
            + rrr_faex_63_x
            + rrr_sdsl_63_x
            + [100]
        )
        ax.set_xlim(left=minx - 5)
        ax.set_title(f"{dataset_name} de densidad")

    fig.supxlabel("espacio ocupado (\% de la secuencia original)")
    fig.supylabel("tiempo medio de select (ns)")
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.05),
        ncol=6,
        fancybox=True,
        shadow=True,
    )
    # show plot
    # plt.show()
    # Save plot
    plt.tight_layout()
    plt.savefig("output/select_compressed.pdf")


def plot_wt_benchmark(wt_benchmark):
    datasets = [wt_benchmark.english, wt_benchmark.proteins]

    datasets_names = ["english", "proteins"]

    for dataset, dataset_name in zip(datasets, datasets_names):
        # Create a 3 column 1 row figure for rank,select and access
        fig, axs = plt.subplots(1, 3, figsize=(8, 4))
        original_size = dataset.original_size
        # Rank plots
        queries = ["rank", "select", "access"]
        queries_functions = [lambda x: x.rank, lambda x: x.select, lambda x: x.access]

        for ax, query, query_function in zip(axs, queries, queries_functions):
            wt_dense_sampling_x = [
                x.size_in_bits / original_size * 100 for x in dataset.dense_sampling
            ]
            wt_dense_sampling_y = [
                query_function(x).time_in_nanos() for x in dataset.dense_sampling
            ]
            ax.plot(
                wt_dense_sampling_x,
                wt_dense_sampling_y,
                **dense_sampling_rank_style,
            )

            wt_sparse_sampling_x = [
                x.size_in_bits / original_size * 100 for x in dataset.sparse_sampling
            ]
            wt_sparse_sampling_y = [
                query_function(x).time_in_nanos() for x in dataset.sparse_sampling
            ]
            ax.plot(
                wt_sparse_sampling_x,
                wt_sparse_sampling_y,
                **sparse_sampling_rank_style,
            )

            wt_bit_vector_il_x = [
                x.size_in_bits / original_size * 100 for x in dataset.bit_vector_il
            ]
            wt_bit_vector_il_y = [
                query_function(x).time_in_nanos() for x in dataset.bit_vector_il
            ]
            ax.plot(
                wt_bit_vector_il_x,
                wt_bit_vector_il_y,
                **bit_vector_ilstyle,
            )

            if query != "select":
                wt_rank_support_v_x = [
                    x.size_in_bits / original_size * 100 for x in dataset.rank_support_v
                ]
                wt_rank_support_v_y = [
                    query_function(x).time_in_nanos() for x in dataset.rank_support_v
                ]
                ax.plot(
                    wt_rank_support_v_x,
                    wt_rank_support_v_y,
                    **rank_support_v_style,
                )

                wt_rank_support_v5_x = [
                    x.size_in_bits / original_size * 100
                    for x in dataset.rank_support_v5
                ]
                wt_rank_support_v5_y = [
                    query_function(x).time_in_nanos() for x in dataset.rank_support_v5
                ]
                ax.plot(
                    wt_rank_support_v5_x,
                    wt_rank_support_v5_y,
                    **rank_support_v5_style,
                )

            

            wt_rrr_faex_63_x = [
                x.size_in_bits / original_size * 100 for x in dataset.rrr_faex_63
            ]
            wt_rrr_faex_63_y = [
                query_function(x).time_in_nanos() for x in dataset.rrr_faex_63
            ]
            ax.plot(
                wt_rrr_faex_63_x,
                wt_rrr_faex_63_y,
                **rrr_faex_63_style,
            )

            wt_rrr_sdsl_63_x = [
                x.size_in_bits / original_size * 100 for x in dataset.rrr_sdsl_63
            ]
            wt_rrr_sdsl_63_y = [
                query_function(x).time_in_nanos() for x in dataset.rrr_sdsl_63
            ]
            ax.plot(
                wt_rrr_sdsl_63_x,
                wt_rrr_sdsl_63_y,
                **rrr_sdsl_63_style,
            )
            
            if query == "select":
                wt_select_support_mcl_x = [
                    x.size_in_bits / original_size * 100
                    for x in dataset.select_support_mcl
                ]
                wt_select_support_mcl_y = [
                    query_function(x).time_in_nanos()
                    for x in dataset.select_support_mcl
                ]
                ax.plot(
                    wt_select_support_mcl_x,
                    wt_select_support_mcl_y,
                    **select_support_mcl_style,
                )

            # Vertical line
            ax.axvline(x=100, color="grey", linestyle="--")
            minx = min(
                wt_dense_sampling_x
                + wt_sparse_sampling_x
                + wt_bit_vector_il_x
                + wt_rrr_faex_63_x
                + wt_rrr_sdsl_63_x
                + [100]
            )
            if query != "select":
                minx = min(minx, min(wt_rank_support_v_x + wt_rank_support_v5_x))
            if query == "select":
                minx = min(minx, min(wt_select_support_mcl_x))
                handles, labels = ax.get_legend_handles_labels()
                select_support_mcl_style_handle = handles[-1]
                select_support_mcl_style_label = labels[-1]
                    

            ax.set_xlim(left=minx - 5)

            ax.set_ylabel(f"tiempo medio de {query} (ns)")
            ax.set_yscale("log")

        handles, labels = ax.get_legend_handles_labels()
        handles+= [select_support_mcl_style_handle]
        labels+= [select_support_mcl_style_label]
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.1),
            ncol=5,
            fancybox=True,
            shadow=True,
        )
        fig.supxlabel("espacio ocupado (\% de la secuencia original)")
        # fig.suptitle(f"WT {dataset_name}")
        # show plot
        # plt.show()
        # Save plot
        plt.tight_layout()
        plt.savefig(f"output/{dataset_name}_wt.pdf")


benchmarks = load_benchmarks()


plt.style.use("science")

if not os.path.exists("output"):
    os.makedirs("output")


plot_sampling_benchmark(benchmarks.sampling)
plot_compressed_benchmark(benchmarks.compressed)
plot_wt_benchmark(benchmarks.wt)
