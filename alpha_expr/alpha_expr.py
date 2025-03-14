from expr_util import Experiment, plot_all_expr, random_vectors
from diskannpy import defaults
import numpy as np
from pprint import pprint

colors = [
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#7f7f7f",  # gray
    "#bcbd22",  # olive
    "#17becf"  # cyan
]


def alpha_experiment(data_vector_gen,  # the data distribution
                     query_vector_gen,  # the query distribution
                     alpha_values,  # a list of alpha values to try
                     size_configs,  # (abbreviation, num_row, dimension)
                     num_query=1000,
                     query_batch_size=25,
                     k=10,
                     seed=1028):
    for abb, num_row, dimension in size_configs:
        latency_recall_experiments_plot_config = []
        throughput_recall_experiments_plot_config = []
        sampled_vectors = data_vector_gen(num_row, dimension, np.float32, seed)
        for idx, alpha_value in enumerate(alpha_values):
            expr_name = f"expr_{abb}_alpha{alpha_value}"
            expr = Experiment.construct(expr_name, sampled_vectors, seed, overwrite=True)
            expr.build_index(
                initial_search_complexity=defaults.COMPLEXITY,
                complexity=defaults.COMPLEXITY,
                graph_degree=defaults.GRAPH_DEGREE,
                alpha=alpha_value,
            )
            expr.run_queries(
                query_vector_gen,
                num=num_query,
                batch_size=query_batch_size,
                k=k,
                complexity=defaults.COMPLEXITY
            )
            summary, _ = expr.save_perf_counters()
            avg_latency = summary[Experiment.PERF_COUNTER_NAMES[Experiment.PERF_COUNTER_QUERY_LATENCY_SECOND]]["average"]
            avg_throughput = summary[Experiment.PERF_COUNTER_NAMES[Experiment.PERF_COUNTER_QUERY_THROUGHPUT_PS]]["average"]
            avg_recall = summary[Experiment.PERF_COUNTER_NAMES[Experiment.PERF_COUNTER_QUERY_RECALL_PERCENT]]["average"]
            latency_recall_experiments_plot_config.append(((avg_latency, avg_recall), colors[idx], f"alpha={alpha_value}"))
            throughput_recall_experiments_plot_config.append(((avg_throughput, avg_recall), colors[idx], f"alpha={alpha_value}"))
            seed += 157  # randomize different runs
        plot_all_expr(
            f"latency_recall_{abb}.png",
            latency_recall_experiments_plot_config,
            fig_name=f"{abb}: Latency vs {k}-recall@{k}",
            fig_x_name="Latency (seconds)",
            fig_y_name=f"Recall@{k}"
        )
        plot_all_expr(
            f"throughput_recall_{abb}.png",
            throughput_recall_experiments_plot_config,
            fig_name=f"{abb}: Throughput vs {k}-recall@{k}",
            fig_x_name="Throughput (Query Per Second)",
            fig_y_name=f"Recall@{k}"
        )


if __name__ == "__main__":
    alpha_experiment(
        data_vector_gen=random_vectors,
        query_vector_gen=random_vectors,
        alpha_values=[1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
        size_configs=[
            ("10k", 10_000, 128),
            # ("100k", 100_000, 128),
            # ("1m", 1_000_000, 128),
        ],
        num_query=5000,
        query_batch_size=-1,  # use point query
        k=10,
        seed=1028,
    )
