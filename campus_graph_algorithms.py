"""Campus Navigation and Infrastructure Optimization.

This module builds a synthetic USF campus graph and demonstrates three graph
algorithms that support navigation, infrastructure planning, and security
patrol analysis.

The graph is undirected and weighted. Vertices represent campus buildings and
edge weights represent walking distance between buildings in meters.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from collections import deque
from heapq import heappop, heappush
from statistics import mean
from time import perf_counter
from textwrap import wrap
from typing import Dict, Iterable, List, Tuple


@dataclass
class WeightedGraph:
    """Undirected, weighted graph implemented with an adjacency list."""

    adjacency: Dict[str, Dict[str, int]] = field(default_factory=dict)

    def add_vertex(self, vertex: str) -> None:
        """Add a vertex if it does not already exist."""

        self.adjacency.setdefault(vertex, {})

    def add_edge(self, source: str, target: str, weight: int) -> None:
        """Add an undirected weighted edge between two vertices."""

        if weight <= 0:
            raise ValueError("Edge weights must be positive.")

        self.add_vertex(source)
        self.add_vertex(target)
        self.adjacency[source][target] = weight
        self.adjacency[target][source] = weight

    def vertices(self) -> List[str]:
        """Return all vertices in insertion order."""

        return list(self.adjacency.keys())

    def edges_count(self) -> int:
        """Count each undirected edge once."""

        return sum(len(neighbors) for neighbors in self.adjacency.values()) // 2

    def neighbors(self, vertex: str) -> Dict[str, int]:
        """Return the adjacent vertices and edge weights for a vertex."""

        return self.adjacency.get(vertex, {})

    def has_vertex(self, vertex: str) -> bool:
        return vertex in self.adjacency


USF_BUILDINGS: List[str] = [
    "Student Center",
    "Library",
    "Engineering Building",
    "Science Hall",
    "Business School",
    "Recreation Center",
    "Residence Hall A",
    "Residence Hall B",
    "Dining Commons",
    "Administration Building",
    "Health Services",
    "Art Gallery",
    "Math Tower",
    "Music Complex",
    "Parking Garage North",
    "Parking Garage South",
    "Research Lab",
    "Campus Police",
    "Bookstore",
    "Lecture Hall Complex",
]


FULL_EDGE_LIST: List[Tuple[str, str, int]] = [
    ("Student Center", "Library", 180),
    ("Library", "Engineering Building", 210),
    ("Engineering Building", "Science Hall", 160),
    ("Science Hall", "Business School", 190),
    ("Business School", "Recreation Center", 220),
    ("Recreation Center", "Residence Hall A", 140),
    ("Residence Hall A", "Residence Hall B", 120),
    ("Residence Hall B", "Dining Commons", 130),
    ("Dining Commons", "Administration Building", 170),
    ("Administration Building", "Health Services", 150),
    ("Health Services", "Art Gallery", 200),
    ("Art Gallery", "Math Tower", 175),
    ("Math Tower", "Music Complex", 160),
    ("Music Complex", "Parking Garage North", 240),
    ("Parking Garage North", "Parking Garage South", 260),
    ("Parking Garage South", "Research Lab", 230),
    ("Research Lab", "Campus Police", 190),
    ("Campus Police", "Bookstore", 140),
    ("Bookstore", "Lecture Hall Complex", 160),
    ("Lecture Hall Complex", "Student Center", 250),
    ("Student Center", "Engineering Building", 260),
    ("Student Center", "Business School", 310),
    ("Student Center", "Dining Commons", 280),
    ("Student Center", "Campus Police", 360),
    ("Library", "Science Hall", 150),
    ("Library", "Administration Building", 290),
    ("Library", "Bookstore", 205),
    ("Engineering Building", "Business School", 240),
    ("Engineering Building", "Math Tower", 200),
    ("Science Hall", "Health Services", 235),
    ("Business School", "Administration Building", 180),
    ("Recreation Center", "Dining Commons", 125),
    ("Recreation Center", "Art Gallery", 290),
    ("Residence Hall A", "Health Services", 210),
    ("Residence Hall B", "Campus Police", 185),
    ("Dining Commons", "Bookstore", 165),
    ("Administration Building", "Math Tower", 195),
    ("Health Services", "Music Complex", 250),
    ("Art Gallery", "Parking Garage North", 270),
    ("Math Tower", "Research Lab", 220),
    ("Music Complex", "Bookstore", 215),
    ("Parking Garage North", "Campus Police", 155),
    ("Parking Garage South", "Bookstore", 240),
    ("Research Lab", "Lecture Hall Complex", 175),
    ("Campus Police", "Lecture Hall Complex", 130),
    ("Bookstore", "Student Center", 235),
    ("Lecture Hall Complex", "Engineering Building", 205),
    ("Lecture Hall Complex", "Library", 265),
    ("Science Hall", "Math Tower", 145),
    ("Business School", "Bookstore", 225),
    ("Student Center", "Science Hall", 275),
    ("Student Center", "Residence Hall A", 300),
    ("Library", "Business School", 230),
    ("Library", "Recreation Center", 240),
    ("Engineering Building", "Residence Hall A", 255),
    ("Science Hall", "Recreation Center", 205),
    ("Business School", "Dining Commons", 200),
    ("Recreation Center", "Dining Commons", 125),
    ("Residence Hall A", "Dining Commons", 145),
    ("Residence Hall B", "Campus Police", 185),
    ("Dining Commons", "Bookstore", 165),
    ("Administration Building", "Campus Police", 300),
    ("Health Services", "Bookstore", 260),
    ("Art Gallery", "Bookstore", 275),
    ("Math Tower", "Campus Police", 240),
    ("Administration Building", "Campus Police", 300),
    ("Dining Commons", "Campus Police", 210),
]


SPARSE_EDGE_LIMIT = 28


def build_campus_graph(vertex_count: int = 20, density: str = "dense") -> WeightedGraph:
    """Construct a synthetic USF campus graph.

    The vertex_count argument lets the benchmark test smaller subgraphs, while
    density controls whether the graph uses a sparse or dense subset of the
    hardcoded edge set.
    """

    if vertex_count < 2:
        raise ValueError("vertex_count must be at least 2")

    if density not in {"sparse", "dense"}:
        raise ValueError("density must be either 'sparse' or 'dense'")

    selected_vertices = USF_BUILDINGS[:vertex_count]
    graph = WeightedGraph()
    for building in selected_vertices:
        graph.add_vertex(building)

    edge_limit = SPARSE_EDGE_LIMIT if density == "sparse" else len(FULL_EDGE_LIST)
    selected_edges = FULL_EDGE_LIST[:edge_limit]

    for source, target, weight in selected_edges:
        if source in graph.adjacency and target in graph.adjacency:
            graph.add_edge(source, target, weight)

    return graph


def validate_connected_graph(graph: WeightedGraph) -> None:
    """Ensure the graph is connected enough for the campus use case."""

    vertices = graph.vertices()
    if not vertices:
        raise ValueError("Graph is empty.")

    visited = bfs(graph, vertices[0])[0]
    if len(visited) != len(vertices):
        raise ValueError("The graph must be connected for this project demonstration.")


def dijkstra_shortest_path(graph: WeightedGraph, start: str, goal: str) -> Tuple[List[str], int]:
    # Time complexity: O((V + E) log V) using a binary heap priority queue.
    # Space complexity: O(V) for distance, previous, and priority queue storage.
    # Best case: the goal is reached quickly and only a small portion of the graph is explored.
    # Worst case: all vertices and edges are processed before the shortest path is finalized.
    if not graph.has_vertex(start) or not graph.has_vertex(goal):
        raise ValueError("Both start and goal must exist in the graph.")

    distances = {vertex: float("inf") for vertex in graph.vertices()}
    previous: Dict[str, str | None] = {vertex: None for vertex in graph.vertices()}
    distances[start] = 0

    queue: List[Tuple[int, str]] = [(0, start)]

    while queue:
        current_distance, current_vertex = heappop(queue)
        if current_distance > distances[current_vertex]:
            continue

        if current_vertex == goal:
            break

        for neighbor, weight in graph.neighbors(current_vertex).items():
            new_distance = current_distance + weight
            if new_distance < distances[neighbor]:
                distances[neighbor] = new_distance
                previous[neighbor] = current_vertex
                heappush(queue, (new_distance, neighbor))

    if distances[goal] == float("inf"):
        raise ValueError(f"No path found from {start} to {goal}.")

    path = reconstruct_path(previous, start, goal)
    return path, int(distances[goal])


def prim_minimum_spanning_tree(graph: WeightedGraph, start: str) -> Tuple[List[Tuple[str, str, int]], int]:
    # Time complexity: O(E log V) because each edge may enter the heap and each
    # extraction is logarithmic in the number of queued candidates.
    # Space complexity: O(V + E) for the visited set, heap, and MST edge list.
    # Best case: a graph with few edges or a strongly connected neighborhood around the start.
    # Worst case: dense graphs where many edge candidates are considered before the MST is complete.
    if not graph.has_vertex(start):
        raise ValueError("The start vertex must exist in the graph.")

    visited = {start}
    mst_edges: List[Tuple[str, str, int]] = []
    total_weight = 0
    heap: List[Tuple[int, str, str]] = []

    for neighbor, weight in graph.neighbors(start).items():
        heappush(heap, (weight, start, neighbor))

    while heap and len(visited) < len(graph.vertices()):
        weight, source, target = heappop(heap)
        if target in visited:
            continue

        visited.add(target)
        mst_edges.append((source, target, weight))
        total_weight += weight

        for neighbor, neighbor_weight in graph.neighbors(target).items():
            if neighbor not in visited:
                heappush(heap, (neighbor_weight, target, neighbor))

    if len(visited) != len(graph.vertices()):
        raise ValueError("The graph is disconnected, so a spanning tree cannot be formed.")

    return mst_edges, total_weight


def bfs(graph: WeightedGraph, start: str) -> Tuple[List[str], List[str]]:
    # Time complexity: O(V + E) because each vertex and edge is inspected at most once.
    # Space complexity: O(V) for the queue, visited set, and traversal order.
    # Best case: the starting node is already connected to all relevant buildings.
    # Worst case: the traversal reaches every reachable vertex and inspects every edge.
    if not graph.has_vertex(start):
        raise ValueError("The start vertex must exist in the graph.")

    visited = {start}
    queue = deque([start])
    order: List[str] = []

    while queue:
        current = queue.popleft()
        order.append(current)

        for neighbor in graph.neighbors(current):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

    return order, order.copy()


def reconstruct_path(previous: Dict[str, str | None], start: str, goal: str) -> List[str]:
    """Rebuild the shortest path by backtracking from the goal to the start."""

    path = [goal]
    current = goal

    while current != start:
        parent = previous[current]
        if parent is None:
            return []
        path.append(parent)
        current = parent

    path.reverse()
    return path


def average_runtime(function, *args, repeat: int = 25):
    """Measure average execution time for a callable."""

    durations = []
    result = None
    for _ in range(repeat):
        start_time = perf_counter()
        result = function(*args)
        durations.append(perf_counter() - start_time)
    return result, mean(durations)


def format_table(headers: List[str], rows: List[List[str]]) -> str:
    """Create a left-aligned text table for console output."""

    widths = [len(header) for header in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))

    def format_row(row: Iterable[str]) -> str:
        return " | ".join(cell.ljust(widths[index]) for index, cell in enumerate(row))

    separator = "-+-".join("-" * width for width in widths)
    lines = [format_row(headers), separator]
    lines.extend(format_row(row) for row in rows)
    return "\n".join(lines)


def build_networkx_graph(graph: WeightedGraph):
    """Convert the adjacency-list graph into a NetworkX graph for plotting."""

    import networkx as nx

    nx_graph = nx.Graph()
    seen_edges = set()

    for vertex in graph.vertices():
        nx_graph.add_node(vertex)

    for source, neighbors in graph.adjacency.items():
        for target, weight in neighbors.items():
            edge_key = tuple(sorted((source, target)))
            if edge_key in seen_edges:
                continue
            seen_edges.add(edge_key)
            nx_graph.add_edge(source, target, weight=weight)

    return nx_graph


def visualize_graph(
    graph: WeightedGraph,
    highlight_path: List[str] | None = None,
    save_path: str | None = None,
    show: bool = True,
) -> None:
    """Render the campus graph with optional path highlighting.

    The graph is drawn with node labels and weighted edges. If a shortest path
    is supplied, the route is emphasized to make the navigation result easy to see.
    """

    import matplotlib.pyplot as plt
    import networkx as nx

    nx_graph = build_networkx_graph(graph)
    positions = nx.spring_layout(nx_graph, seed=42, weight="weight")

    path_edges = set()
    path_nodes = set()
    if highlight_path and len(highlight_path) >= 2:
        path_nodes = set(highlight_path)
        path_edges = {
            tuple(sorted((highlight_path[index], highlight_path[index + 1])))
            for index in range(len(highlight_path) - 1)
        }

    plt.figure(figsize=(16, 11))
    plt.title("USF Campus Graph", fontsize=16, weight="bold")
    plt.axis("off")

    base_edges = list(nx_graph.edges())
    highlighted_edge_list = [edge for edge in base_edges if tuple(sorted(edge)) in path_edges]
    normal_edge_list = [edge for edge in base_edges if tuple(sorted(edge)) not in path_edges]

    nx.draw_networkx_edges(
        nx_graph,
        positions,
        edgelist=normal_edge_list,
        width=1.4,
        edge_color="#b8c1cc",
        alpha=0.7,
    )
    if highlighted_edge_list:
        nx.draw_networkx_edges(
            nx_graph,
            positions,
            edgelist=highlighted_edge_list,
            width=3.2,
            edge_color="#d1495b",
        )

    node_colors = [
        "#f4d35e"
        if node in path_nodes and node not in {highlight_path[0], highlight_path[-1]}
        else "#1d7874"
        if highlight_path and node == highlight_path[0]
        else "#ee964b"
        if highlight_path and node == highlight_path[-1]
        else "#4f6d7a"
        for node in nx_graph.nodes()
    ]

    nx.draw_networkx_nodes(
        nx_graph,
        positions,
        node_size=1800,
        node_color=node_colors,
        edgecolors="#1f1f1f",
        linewidths=1.1,
    )
    wrapped_labels = {node: format_node_label(node) for node in nx_graph.nodes()}
    nx.draw_networkx_labels(
        nx_graph,
        positions,
        labels=wrapped_labels,
        font_size=7.2,
        font_color="#101820",
        verticalalignment="center",
        horizontalalignment="center",
    )
    edge_labels = nx.get_edge_attributes(nx_graph, "weight")
    nx.draw_networkx_edge_labels(nx_graph, positions, edge_labels=edge_labels, font_size=7)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=220, bbox_inches="tight")
        print(f"Graph visualization saved to {save_path}")

    if show:
        plt.show()

    plt.close()


def format_node_label(name: str, max_width: int = 12) -> str:
    """Wrap long building names so they fit more cleanly inside nodes."""

    return "\n".join(wrap(name, width=max_width))


def benchmark_algorithms() -> None:
    """Run the algorithms on different graph sizes and densities."""

    graph_sizes = [8, 12, 16, 20]
    densities = ["sparse", "dense"]
    rows: List[List[str]] = []

    for density in densities:
        for size in graph_sizes:
            graph = build_campus_graph(size, density)
            validate_connected_graph(graph)

            shortest_start = graph.vertices()[0]
            shortest_goal = graph.vertices()[-1]

            _, shortest_time = average_runtime(
                dijkstra_shortest_path,
                graph,
                shortest_start,
                shortest_goal,
            )
            _, mst_time = average_runtime(prim_minimum_spanning_tree, graph, shortest_start)
            _, bfs_time = average_runtime(bfs, graph, shortest_start)

            rows.append(
                [
                    density,
                    str(size),
                    str(graph.edges_count()),
                    f"{shortest_time * 1000:.4f}",
                    f"{mst_time * 1000:.4f}",
                    f"{bfs_time * 1000:.4f}",
                ]
            )

    print("\nBenchmark Results (average runtime over 25 runs)")
    print(
        format_table(
            ["Density", "Vertices", "Edges", "Dijkstra ms", "Prim ms", "BFS ms"],
            rows,
        )
    )


def print_algorithm_demo(graph: WeightedGraph) -> None:
    """Show a sample run of all three algorithms on the full campus graph."""

    start = "Student Center"
    goal = "Lecture Hall Complex"
    path, distance = dijkstra_shortest_path(graph, start, goal)
    mst_edges, total_weight = prim_minimum_spanning_tree(graph, start)
    traversal_order, reachable_buildings = bfs(graph, start)

    print("Campus Navigation and Infrastructure Optimization")
    print("-" * 58)
    print(f"Shortest path from {start} to {goal}: {' -> '.join(path)}")
    print(f"Total walking distance: {distance} meters")
    print()
    print(f"MST total cable length: {total_weight} meters")
    print(f"MST edges selected: {len(mst_edges)}")
    print()
    print(f"BFS patrol starting at {start} visits {len(traversal_order)} buildings")
    print(f"Reachable buildings: {', '.join(reachable_buildings)}")


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the demo and visualization."""

    parser = argparse.ArgumentParser(
        description="Campus Navigation and Infrastructure Optimization project"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Display the campus graph in a matplotlib window.",
    )
    parser.add_argument(
        "--save-graph",
        default=None,
        metavar="PATH",
        help="Save the graph visualization to an image file.",
    )
    return parser.parse_args()


def main() -> None:
    """Program entry point."""

    args = parse_args()
    full_graph = build_campus_graph(20, "dense")
    validate_connected_graph(full_graph)
    print_algorithm_demo(full_graph)

    if args.visualize or args.save_graph:
        shortest_path, _ = dijkstra_shortest_path(
            full_graph,
            "Student Center",
            "Lecture Hall Complex",
        )
        visualize_graph(
            full_graph,
            highlight_path=shortest_path,
            save_path=args.save_graph,
            show=args.visualize,
        )

    benchmark_algorithms()


if __name__ == "__main__":
    main()