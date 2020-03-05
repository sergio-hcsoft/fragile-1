import numpy as np
import hvplot
import hvplot.pandas
import hvplot.networkx as hvnx
import networkx as nx
import holoviews as hv
from fragile.core.utils import resize_frame
from umap import UMAP
import copy
import warnings
from holoviews.operation.datashader import datashade, bundle_graph

warnings.filterwarnings("ignore")
hv.extension("bokeh")


def get_path_nodes_and_edges(g, leaf_name):
    parent = -100
    nodes = [int(leaf_name)]
    edges = []
    while parent != 0:
        parents = list(g.in_edges([leaf_name]))
        try:
            parent = parents[0][0]
            nodes.append(parent)
            edges.append(tuple([parent, leaf_name]))
            leaf_name = int(parent)
        except:
            print(parent, leaf_name)
            return nodes, edges
    return nodes, edges


def get_best_path(swarm):
    best_ix = swarm.walkers.states.cum_rewards.argmax()
    best = swarm.walkers.states.id_walkers[best_ix]
    leaf_name = swarm.tree.node_names[best]
    nodes, edges = get_path_nodes_and_edges(swarm.tree.data, leaf_name)
    nodes, edges = list(reversed(nodes))[1:], list(reversed(edges))[1:]
    states = [swarm.tree.data.nodes[n]["state"] for n in nodes]
    n_iters = [swarm.tree.data.nodes[n]["n_iter"] for n in nodes]
    actions = [swarm.tree.data.edges[e]["action"] for e in edges]
    return states, actions, n_iters, nodes, edges


def add_image_from_node(swarm, node_id):
    parents = list(swarm.tree.data.in_edges([node_id]))
    if len(parents) > 0:
        parent = parents[0][0]
        action = swarm.tree.data.edges[(parent, node_id)]["action"]
        state = swarm.tree.data.nodes[parent]["state"]
        data = swarm.env._env.step(state=state, action=action)
        obs = swarm.env._env.unwrapped.ale.getScreenRGB()
        obs = resize_frame(obs[:, :, 0][2:170], 60, 60, "L")
        return obs


def create_embedding_layout(swarm):
    nodes = list(swarm.tree.data.nodes())[1:]
    observs = np.array([add_image_from_node(swarm, n) for n in nodes])
    samples = observs.reshape(observs.shape[0], -1)
    embeddings = UMAP(n_components=2, min_dist=0.99, n_neighbors=50).fit_transform(samples)
    return {n: embeddings[i] for i, n in enumerate(nodes)}


def get_plot_graph(swarm):
    plot_g = nx.Graph()
    states, actions, n_iters, nodes, edges = get_best_path(swarm)
    for n in swarm.tree.data.nodes():
        is_best = n in nodes
        node_attrs = copy.deepcopy(swarm.tree.data.nodes[n])
        node_attrs.pop("state")
        plot_g.add_node(
            n,
            final=1 if is_best else 0.3,
            node_alpha=1 if is_best else 0.2,
            line_alpha=1 if is_best else 0.0,
            **node_attrs
        )
    for a, b in swarm.tree.data.edges():
        plot_g.add_edge(
            a,
            b,
            weight=float(swarm.tree.data.edges[(a, b)]["action"]),
            final=1 if (a, b) in edges else 0.3,
        )
    return plot_g


def plot_graph(plot_g, embs):
    graph = hv.Graph.from_networkx(plot_g, embs)

    graph.opts(
        node_color=hv.dim("n_iter"),
        node_cmap="viridis",
        node_size=3,  # hv.dim('final') * 5,
        edge_line_width=hv.dim("final") * 0.2,
        node_line_width=0.5,
        node_alpha=hv.dim("node_alpha"),
        edge_alpha=hv.dim("final"),
        edge_line_color=hv.dim("final"),
        edge_cmap=["white", "red"],
        node_line_color="red",
        node_line_alpha=hv.dim("line_alpha"),
        width=800,
        height=600,
        bgcolor="gray",
        colorbar=True,
    )
    return graph


def create_subgraph(start, end, graph, embs=None, key="n_iter"):
    embs = embs if embs is not None else {}
    g = nx.Graph()
    for n in graph.nodes:
        n_iter = graph.nodes[n][key]
        if start <= n_iter <= end:
            g.add_node(n, **graph.nodes[n])
            g.nodes[n]["last_line_alpha"] = 1 if n_iter == end else 0
            g.nodes[n]["last_size"] = 8 if n_iter == end else 4
    for a, b in graph.edges:
        n_iter_a = graph.nodes[a][key]
        n_iter_b = graph.nodes[b][key]
        if start <= n_iter_a <= end and start <= n_iter_b <= end:
            g.add_edge(a, b, **graph.edges[(a, b)])
    new_embs = {k: v for k, v in embs.items() if k in g.nodes}
    return g, new_embs


def plot_subgraph(plot_g, embs, bundle: bool = False):
    graph = hv.Graph.from_networkx(plot_g, embs)

    graph.opts(
        node_color=hv.dim("cum_reward"),
        node_cmap="viridis",
        node_size=hv.dim("last_size"),
        edge_line_width=hv.dim("final"),
        node_line_width=1.5,
        node_alpha=0.8,
        xaxis=None,
        yaxis=None,
        edge_alpha=hv.dim("final"),
        edge_line_color=hv.dim("final"),
        edge_cmap=["white", "green"],
        node_line_color="red",
        node_line_alpha=hv.dim("last_line_alpha"),
        width=1280,
        height=720,
        bgcolor="gray",
        colorbar=True,
    )
    if bundle:
        bundled = bundle_graph(graph)
        return bundled.opts(norm=dict(framewise=True))
    return graph.opts(norm=dict(framewise=True))


def plot_iteration(
    iteration,
    graph,
    embeddings,
    start=0,
    key="n_iter",
    bundle=False,
    observs=None,
    plot_func=plot_subgraph,
):
    g, new_embs = create_subgraph(
        start=start, end=iteration, graph=graph, embs=embeddings, key=key
    )
    graph = plot_func(g, new_embs, bundle=bundle)
    if observs is not None:
        screen = observs.get(iteration)
        it = iteration
        while screen is None:
            it -= 1
            screen = observs.get(it)
        image = hv.RGB(screen).opts(xaxis=None, yaxis=None, normalize=True, shared_axes=False)
        return image + graph
    return graph


def get_game_observs(swarm):
    states, actions, n_iters, nodes, edges = get_best_path(swarm)
    observs = {}
    for node_id, it in zip(nodes, n_iters):
        parents = list(swarm.tree.data.in_edges([node_id]))
        if len(parents) > 0:
            parent = parents[0][0]
            action = swarm.tree.data.edges[(parent, node_id)]["action"]
            state = swarm.tree.data.nodes[parent]["state"]
            data = swarm.env._env.step(state=state, action=action)
            obs = swarm.env._env.unwrapped.ale.getScreenRGB()
            observs[it] = obs
    return observs
