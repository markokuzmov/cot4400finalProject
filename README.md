# Campus Navigation and Infrastructure Optimization

This project models a university campus as a weighted, undirected graph and demonstrates three core graph algorithms for network optimization and navigation.

## Features

- Synthetic USF campus dataset with 20 buildings
- Adjacency-list graph representation
- Dijkstra shortest path for route planning
- Prim minimum spanning tree for cable/network planning
- BFS traversal for patrol and reachability analysis
- Benchmarking across small and full graph sizes, with sparse and dense edge variants

## Files

- [campus_graph_algorithms.py](campus_graph_algorithms.py): main Python program with the graph, algorithms, demo output, and benchmarks

## Requirements

- Python 3.14+ or a compatible Python 3 environment

## Run

From the project directory, run:

```bash
python campus_graph_algorithms.py
```

If you are using the provided virtual environment, you can also run:

```bash
.venv/bin/python campus_graph_algorithms.py
```

## Output

The program prints:

- A sample shortest path between two campus buildings
- The total weight of the minimum spanning tree
- A BFS traversal order for patrol/reachability
- A formatted benchmark table comparing graph sizes and densities
