# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing cluster_graph, apply_clustering and run_layout methods definition."""

import logging
import numpy as np

import networkx as nx
from graspologic.partition import hierarchical_leiden

from graphrag.index.utils.stable_lcc import stable_largest_connected_component

from sklearn.cluster import KMeans
from sklearn.manifold import SpectralEmbedding
from node2vec import Node2Vec

Communities = list[tuple[int, int, int, list[str]]]


logger = logging.getLogger(__name__)


def cluster_graph(
    graph: nx.Graph,
    max_cluster_size: int,
    use_lcc: bool,
    seed: int | None = None
) -> Communities:
    """Apply a hierarchical clustering algorithm to a graph."""
    if len(graph.nodes) == 0:
        logger.warning("Graph has no nodes")
        return []

    node_id_to_community_map, parent_mapping = _compute_leiden_communities(
        graph=graph,
        max_cluster_size=max_cluster_size,
        use_lcc=use_lcc,
        seed=seed
    )

    levels = sorted(node_id_to_community_map.keys())

    clusters: dict[int, dict[int, list[str]]] = {}
    for level in levels:
        result = {}
        clusters[level] = result
        for node_id, raw_community_id in node_id_to_community_map[level].items():
            community_id = raw_community_id
            if community_id not in result:
                result[community_id] = []
            result[community_id].append(node_id)

    results: Communities = []
    for level in clusters:
        for cluster_id, nodes in clusters[level].items():
            results.append((level, cluster_id, parent_mapping[cluster_id], nodes))
    return results


def initialize_partition(
    G: nx.Graph,
    embedding_dim: int = 32,
    avg_community_size: int = 64,
    density_threshold: float = 0.1,
    clustering_threshold: float = 0.2,
    correlation_threshold: float = 0.1,
    verbose: bool = False
):
    """
    Implements the Embedding-based Community-aware Leiden (EC-Leiden) algorithm.

    Args:
        G (nx.Graph): The input graph as a networkx.Graph object.
        embedding_dim (int): The dimensionality of the node embedding space.
        avg_community_size (int): The assumed average size of a community, used to set k for k-means.
        density_threshold (float): Threshold to determine if the graph is dense.
        clustering_threshold (float): Threshold for the average clustering coefficient.
        correlation_threshold (float): Threshold for the degree correlation coefficient (assortativity).
        verbose (bool): If True, logs status updates during execution.

    Returns:
        - initial_partition (dict): A dictionary mapping original node labels to their final community ID.
    """
    if not isinstance(G, nx.Graph):
        raise TypeError("Input must be a networkx.Graph object.")

    n_nodes = G.number_of_nodes()
    if n_nodes == 0:
        raise ValueError("Graph is empty. Cannot create initial partition.")

    # --- Step 0: Pre-processing for Consistency ---
    node_list = sorted(list(G.nodes()))
    H = G.copy()  # Work with a copy but keep original node labels
    
    # --- Step 1: Compute Statistics and Embed Nodes ---
    if verbose: logger.info("--- Step 1: Analyzing graph and generating node embeddings ---")
    
    # Calculate graph statistics
    n_edges = H.number_of_edges()
    density = (2 * n_edges) / (n_nodes * (n_nodes - 1)) if n_nodes > 1 else 0
    try:
        avg_cluster_coeff = nx.average_clustering(H)
    except nx.NetworkXError: # Handles graphs with nodes of degree 0 or 1
        avg_cluster_coeff = 0.0
    try:
        degree_corr = nx.degree_assortativity_coefficient(H)
        degree_corr = 0.0 if np.isnan(degree_corr) else degree_corr
    except (nx.NetworkXError, ZeroDivisionError): # Handles graphs with no degree variance
        degree_corr = 0.0
    
    logger.info(f"Graph Stats for ECCD: Density={density:.4f}, Avg. Clustering={avg_cluster_coeff:.4f}, Assortativity={degree_corr:.4f}")

    # Logic to select appropriate embedding algorithm
    use_le = (
        density > density_threshold and
        avg_cluster_coeff < clustering_threshold and
        abs(degree_corr) < correlation_threshold
    )

    embeddings = None
    if use_le:
        if verbose: logger.info("Condition met: Using Laplacian Eigenmaps (LE) for embedding.")
        adj_matrix = nx.to_numpy_array(H, nodelist=sorted(H.nodes()))
        le_dim = min(embedding_dim, n_nodes - 2) if n_nodes > 2 else 1
        se = SpectralEmbedding(n_components=le_dim, affinity='precomputed')
        embeddings = se.fit_transform(adj_matrix)
    else:
        if verbose: logger.info("Condition not met: Using node2vec for embedding.")
        if n_edges > 0:
            n2v = Node2Vec(H, dimensions=embedding_dim, quiet=not verbose, workers=4)
            model = n2v.fit()
            embeddings = np.array([model.wv[node] for node in sorted(H.nodes())])
        else: # Handle graph with no edges
            if verbose: logger.info("Warning: Graph has no edges. Using random embeddings.")
            embeddings = np.random.rand(n_nodes, embedding_dim)

    # --- Step 2: K-Means Clustering ---
    if verbose: logger.info("\n--- Step 2: Running K-Means to create initial partition ---")
    k = max(1, int(np.ceil(n_nodes / avg_community_size)))
    k = min(k, n_nodes) # k cannot be larger than the number of samples

    if verbose:
        logger.info(f"n_nodes: {n_nodes}, avg_community_size: {avg_community_size}")
        logger.info(f"Targeting k = {k} initial clusters.")
    
    if k > 1:
        kmeans = KMeans(n_clusters=k)
        initial_partition_labels = kmeans.fit_predict(embeddings)
    else:
        initial_partition_labels = np.zeros(n_nodes, dtype=int)
            
    # Map community IDs back to original node labels
    initial_partition = {node_list[i]: int(initial_partition_labels[i]) for i in range(n_nodes)}
    return initial_partition


# Taken from graph_intelligence & adapted
def _compute_leiden_communities(
    graph: nx.Graph | nx.DiGraph,
    max_cluster_size: int,
    use_lcc: bool,
    seed: int | None = None,
    starting_communities: dict[str, int] | None = None,
) -> tuple[dict[int, dict[str, int]], dict[int, int]]:
    """Return Leiden root communities and their hierarchy mapping."""
    logger.info(f"max_cluster_size: {max_cluster_size}, use_lcc: {use_lcc}, seed: {seed}")

    if use_lcc:
        # logger.info(f"graph prior to stable_largest_connected_component: {list(graph.nodes)[:20]}")
        graph = stable_largest_connected_component(graph)
        # logger.info(f"graph after stable_largest_connected_component: {list(graph.nodes)[:20]}")

    # Comment out for no starting_communities (default)
    # Uncomment to pass starting communities to hierarchical_leiden (ECCD)
    starting_communities = initialize_partition(graph, avg_community_size=10)

    if starting_communities:
        logger.info("Using starting communities for Hierarchical Leiden clustering (ECCD).")
    else:
        logger.info("No starting communities for Hierarchical Leiden clustering (default).")
    
    # Workaround for index out of bounds error when passing LCC to graspologic-native:
    # Create edge list to ensure proper node handling
    edges = [(u, v, d.get('weight', 1.0)) for u, v, d in graph.edges(data=True)]
    
    community_mapping = hierarchical_leiden(
        edges,  # Pass edge list instead of graph to avoid index out of bounds error
        max_cluster_size=max_cluster_size, 
        random_seed=seed,
        starting_communities=starting_communities
    )
    
    results: dict[int, dict[str, int]] = {}
    hierarchy: dict[int, int] = {}
    for partition in community_mapping:
        level = partition.level
        if level not in results:
            results[level] = {}            
        results[level][partition.node] = partition.cluster

        # Update hierarchy mapping for parent-child cluster relationships
        if partition.cluster not in hierarchy:
            hierarchy[partition.cluster] = (
                partition.parent_cluster if partition.parent_cluster is not None else -1
            )

    return results, hierarchy
