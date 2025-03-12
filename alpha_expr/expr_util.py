import diskannpy
import numpy as np
import os
import shutil
from sklearn.neighbors import NearestNeighbors
import time
from enum import Enum
from matplotlib import pyplot as plt
import json
import datetime
import traceback


def dominates(a, b, objectives):
    """
    Returns True if point a dominates point b according to the objectives.
    Each element in 'objectives' should be either 'min' or 'max'.
    """
    strictly_better = False
    for i, obj in enumerate(objectives):
        if obj == 'min':
            # For "min", lower is better.
            if a[i] > b[i]:
                return False
            elif a[i] < b[i]:
                strictly_better = True
        elif obj == 'max':
            # For "max", higher is better.
            if a[i] < b[i]:
                return False
            elif a[i] > b[i]:
                strictly_better = True
        else:
            raise ValueError("Objective must be either 'min' or 'max'.")
    return strictly_better


def compute_pareto(points, objectives):
    """
    Computes the Pareto frontier from a list of points.

    Parameters:
    - points: list of tuples, where each tuple contains metric values.
    - objectives: list of strings ('min' or 'max') corresponding to each metric.

    Returns:
    - List of points that are on the Pareto frontier.
    """
    pareto_points = []
    for point in points:
        if not any(dominates(other, point, objectives) for other in points if other != point):
            pareto_points.append(point)
    return pareto_points


def random_vectors(rows: int, dimensions: int, dtype, seed: int = 12345) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if dtype == np.float32:
        vectors = rng.random((rows, dimensions), dtype=dtype)
    elif dtype == np.uint8:
        vectors = rng.integers(
            low=0, high=256, size=(rows, dimensions), dtype=dtype
        )  # low is inclusive, high is exclusive
    elif dtype == np.int8:
        vectors = rng.integers(
            low=-128, high=128, size=(rows, dimensions), dtype=dtype
        )  # low is inclusive, high is exclusive
    else:
        raise RuntimeError("Only np.float32, np.int8, and np.uint8 are supported")
    return vectors


class Timed:
    def __init__(self, name=""):
        self.name_ = name

    def __enter__(self):
        self.start_ = time.perf_counter()
        self.end_ = None
        self.time_elapsed_ = None
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{current_time}] {self.name_} started")
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.end_ = time.perf_counter()
        self.time_elapsed_ = self.end_ - self.start_
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if exc_type is not None:
            print(f"[{current_time}] exception in {self.name_}: elapsed {self.time_elapsed_:.5f} seconds")
            print("".join(traceback.format_exception(exc_type, exc_value, exc_tb)))
            exit(1)
        else:
            print(f"[{current_time}] {self.name_} finished: elapsed {self.time_elapsed_:.5f} seconds")

    def time_elapsed(self):
        return self.time_elapsed_


class Experiment:
    PERF_COUNTER_NAMES = [
        "index.build.second",
        "query.latency.second",
        "query.throughput.per_second",
        "query.k-recall@k.percent"
    ]
    PERF_COUNTER_BUILD_INDEX_SECOND = 0
    PERF_COUNTER_QUERY_LATENCY_SECOND = 1
    PERF_COUNTER_QUERY_THROUGHPUT_PS = 2
    PERF_COUNTER_QUERY_RECALL_PERCENT = 3

    @classmethod
    def construct(cls, name, vectors, seed, overwrite=True, **kwargs):
        # overwrite experiments
        if os.path.exists(name):
            if overwrite:
                shutil.rmtree(name)
            else:
                print(f"Experiment {name} already_exists, skipping")
                return None
        os.makedirs(name)
        return cls(name, name, vectors, seed, **kwargs)

    def __init__(self, name, expr_dir, vectors, seed, metric="l2", max_recall_at=100):
        self.name_ = name
        self.expr_dir_ = expr_dir
        self.expr_index_dir_ = os.path.join(self.expr_dir_, "diskann_index")
        os.makedirs(self.expr_index_dir_)
        self.vectors_ = vectors
        self.dimension_ = self.vectors_.shape[1]
        self.dtype_ = self.vectors_.dtype
        self.seed_ = seed
        self.metric_ = metric
        # the index to be build
        self.diskann_index_ = None
        # create a flatmap index (for measuring recall)
        self.knn_ = NearestNeighbors(n_neighbors=max_recall_at, algorithm="auto", metric=metric)
        self.knn_.fit(self.vectors_)
        # performance counters
        self.counters_ = [[] for _ in Experiment.PERF_COUNTER_NAMES]

    def _calculate_recalls(self, queries, result_set_indices, recall_at):
        _, truth_set_indices = self.knn_.kneighbors(queries)
        found = 0
        for i in range(0, result_set_indices.shape[0]):
            result_set_set = set(result_set_indices[i][0:recall_at])
            truth_set_set = set(truth_set_indices[i][0:recall_at])
            found += len(result_set_set.intersection(truth_set_set))
        return found / (result_set_indices.shape[0] * recall_at)

    def build_index(self, complexity, graph_degree, alpha,
                    initial_search_complexity,
                    num_threads_build=0, num_threads_query=0,
                    build_index_args=None, index_args=None):
        if build_index_args is None:
            build_index_args = {}
        if index_args is None:
            index_args = {}
        with Timed("create_index") as build_index_time:
            diskannpy.build_memory_index(
                data=self.vectors_,
                distance_metric=self.metric_,
                index_directory=self.expr_index_dir_,
                alpha=alpha,
                complexity=complexity,
                graph_degree=graph_degree,
                num_threads=num_threads_build,
                **build_index_args
            )
            self.diskann_index_ = diskannpy.StaticMemoryIndex(
                initial_search_complexity=initial_search_complexity,
                index_directory=self.expr_index_dir_,
                num_threads=num_threads_query,
                **index_args
            )
        self.counters_[Experiment.PERF_COUNTER_BUILD_INDEX_SECOND].append(
            build_index_time.time_elapsed())

    def _batch_query(self, idx, queries, k, complexity, num_threads_query=0):
        with Timed(f"{idx} batch_query") as batch_query_time:
            batch_response: diskannpy.QueryResponseBatch = self.diskann_index_.batch_search(
                queries, k_neighbors=k, complexity=complexity, num_threads=num_threads_query
            )
        # calculate recall
        recall = self._calculate_recalls(queries, batch_response.identifiers, recall_at=k)
        latency = batch_query_time.time_elapsed() / queries.shape[0]
        throughput = queries.shape[0] / batch_query_time.time_elapsed()
        self.counters_[Experiment.PERF_COUNTER_QUERY_LATENCY_SECOND].append(latency)
        self.counters_[Experiment.PERF_COUNTER_QUERY_THROUGHPUT_PS].append(throughput)
        self.counters_[Experiment.PERF_COUNTER_QUERY_RECALL_PERCENT].append(recall)

    def _query(self, idx, query, k, complexity, num_threads_query=0):
        with Timed(f"{idx} query") as query_time:
            response: diskannpy.QueryResponse = self.diskann_index_.search(
                query[0], k_neighbors=k, complexity=complexity
            )
        # calculate recall
        result_ = response.identifiers.reshape(1, -1)
        recall = self._calculate_recalls(query, result_, recall_at=k)
        latency = query_time.time_elapsed() / 1
        throughput = 1 / query_time.time_elapsed()
        self.counters_[Experiment.PERF_COUNTER_QUERY_LATENCY_SECOND].append(latency)
        self.counters_[Experiment.PERF_COUNTER_QUERY_THROUGHPUT_PS].append(throughput)
        self.counters_[Experiment.PERF_COUNTER_QUERY_RECALL_PERCENT].append(recall)

    def run_queries(self, query_sample_gen, num, batch_size=-1, **query_kwargs):
        if batch_size <= 0:
            for query_idx in range(0, num):
                query_vectors = query_sample_gen(1, self.dimension_, self.dtype_, self.seed_)
                self.seed_ += 157  # randomize different runs
                self._query(f"{query_idx}/{num}", query_vectors, **query_kwargs)
        else:
            for start_idx in range(0, num, batch_size):
                end_idx = min(start_idx + batch_size, num)
                curr_batch_size = end_idx - start_idx
                query_vectors = query_sample_gen(curr_batch_size, self.dimension_, self.dtype_, self.seed_)
                self.seed_ += 17  # randomize different runs
                self._batch_query(f"[{start_idx}, {end_idx}]/{num}", query_vectors, **query_kwargs)

    def dump_perf_counters_summary(self):
        summary = {}
        for name, values in zip(Experiment.PERF_COUNTER_NAMES, self.counters_):
            p50 = np.percentile(values, 50)
            p95 = np.percentile(values, 95)
            p99 = np.percentile(values, 99)
            average = np.mean(values)
            summary[name] = {"P50": p50, "P95": p95, "p99": p99, "average": average, "count": len(values)}
        return summary

    def dump_perf_counters_raw(self):
        raw = {name: values for name, values in zip(Experiment.PERF_COUNTER_NAMES, self.counters_)}
        return raw

    def save_perf_counters(self):
        perf_summary_file = os.path.join(self.expr_dir_, "perf_summary.json")
        summary = self.dump_perf_counters_summary()
        perf_raw_file = os.path.join(self.expr_dir_, "perf_raw.json")
        raw = self.dump_perf_counters_raw()
        with open(perf_summary_file, "w") as f:
            json.dump(summary, f, indent=4)
        with open(perf_raw_file, "w") as f:
            json.dump(raw, f, indent=4)
        return summary, raw


def plot_all_expr(fig_save_name, args, fig_name,
                  fig_x_name, fig_y_name, fig_size=(6, 6),
                  fig_use_legend=False):
    fig, ax = plt.subplots(figsize=fig_size)
    x_values = [arg[0][0] for arg in args]
    y_values = [arg[0][1] for arg in args]
    ax.scatter(x_values, y_values)
    for (x, y), color, label in args:
        plt.annotate(label, (x, y), color=color, textcoords="offset points", xytext=(5, 5), fontsize=10)
    ax.set_xlabel(fig_x_name)
    ax.set_ylabel(fig_y_name)
    ax.set_title(fig_name)
    if fig_use_legend:
        ax.legend()
    plt.savefig(fig_save_name, dpi=300)
    print(f"Plotted {fig_save_name}.")
