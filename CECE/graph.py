import math
from math import inf
import numpy as np
from tqdm.notebook import tqdm
from .refine import *

class Graph:

    def __init__(self, msq):
        self.msq = msq
        self.id_to_node = {id: node for node, id in enumerate(msq.keys())}
        self.node_to_id = {node: id for id, node in self.id_to_node.items()}

    def create_graph(self, filename = None):
        keys = list(self.id_to_node.keys())

        self.matr = np.zeros((len(keys), len(keys)))
        for i in tqdm(range (len(keys))):
            for j in range (i, len(keys)):
                if i != j:
                    cost = refine (self.msq, keys[i], keys[j], False)
                    self.matr[i, j] = cost
                    self.matr[j, i] = cost

        if filename:
            np.save(filename, self.matr)
        return self.matr

    def load_graph(self, filename):
        self.matr = np.load(filename)
        return self.matr


    def array_with_significance(self, dataset, target_labels):
        matr = np.copy(self.matr)

        for i in tqdm(range (len(matr))):
            for j in range (len(matr)):
                matr[i][i] = matr[i][i] / significance(dataset, self.node_to_id[j], self.node_to_id[j], target_labels)
        return matr


def significance(coco, source_id, target_id, target_label):

    source_p = coco.coco[source_id]["pred_class"][1] if coco.coco[source_id]["pred_class"][0] == target_label else 0
    target_p = coco.coco[target_id]["pred_class"][1] if coco.coco[target_id]["pred_class"][0] == target_label else 0

    return sigmoid(target_p - source_p)

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

# def create_graph(msq, filename = None):
#     keys = list(msq.keys())
#     matr = np.zeros((len(keys), len(keys)))
#     for i in tqdm(range (len(keys))):
#         for j in range (i, len(keys)):
#             if i != j:
#                 q1 = msq[keys[i]]
#                 q2 = msq[keys[j]]
#                 cost = refine (msq, keys[i], keys[j], False)
#                 matr[i, j] = cost
#                 matr[j, i] = cost
#
#     if filename:
#         np.save(filename, matr)
#     return matr


# def significance(source_node, target_node, target_label, node_to_id, coco):
#   source_image = node_to_id[source_node]
#   target_image = node_to_id[target_node]
#
#   source_p = coco.coco[source_image]["pred_class"][1] if coco.coco[source_image]["pred_class"][0] == target_label else 0
#   target_p = coco.coco[source_image]["pred_class"][1] if coco.coco[source_image]["pred_class"][0] == target_label else 0
#
#   return sigmoid(target_p - source_p)
#
# def add_significance_to_graph(graph_filename, target_labels, node_to_id, coco):
#   matr = np.load(graph_filename)
#   # add significance on adjecency matrix
#   for i in tqdm(range (len(matr))):
#     for j in range (len(matr)):
#       matr[i][i] = matr[i][i] / significance(i, j, target_labels, node_to_id, coco)
#   return matr




# Credits for Dijkstra implementation
# Author: Cristian Bastidas
# GitHub: https://github.com/crixodia
# Date: 2020-10-7

def find_all(wmat, start, end=-1):
    """
    Returns a tuple with a distances' list and paths' list of
    all remaining vertices with the same indexing.

        (distances, paths)

    For example, distances[x] are the shortest distances from x
    vertex which shortest path is paths[x]. x is an element of
    {0, 1, ..., n-1} where n is the number of vertices

    Args:
    wmat    --  weighted graph's adjacency matrix
    start   --  paths' first vertex
    end     --  (optional) path's end vertex. Return just the 
                distance and its path

    Exceptions:
    Index out of range, Be careful with start and end vertices
    """
    n = len(wmat)

    dist = [inf]*n
    dist[start] = wmat[start][start]  # 0

    spVertex = [False]*n
    parent = [-1]*n

    path = [{}]*n

    for count in range(n-1):
        minix = inf
        u = 0

        for v in range(len(spVertex)):
            if spVertex[v] == False and dist[v] <= minix:
                minix = dist[v]
                u = v

        spVertex[u] = True
        for v in range(n):
            if not(spVertex[v]) and wmat[u][v] != 0 and dist[u] + wmat[u][v] < dist[v]:
                parent[v] = u
                dist[v] = dist[u] + wmat[u][v]

    for i in range(n):
        j = i
        s = []
        while parent[j] != -1:
            s.append(j)
            j = parent[j]
        s.append(start)
        path[i] = s[::-1]

    return (dist[end], path[end]) if end >= 0 else (dist, path)


def find_shortest_path(wmat, start, end=-1):
    """
    Returns paths' list of all remaining vertices.

    Args:
    wmat    --  weigthted graph's adjacency matrix
    start   --  paths' first vertex
    end     --  (optional) path's end vertex. Return just
                the path

    Exceptions:
    Index out of range, Be careful with start and end vertices.
    """
    return find_all(wmat, start, end)[1]


def find_shortest_distance(wmat, start, end=-1):
    """
    Returns distances' list of all remaining vertices.

    Args:
    wmat    --  weigthted graph's adjacency matrix
    start   --  paths' first vertex
    end     --  (optional) path's end vertex. Return just
                the distance

    Exceptions:
    Index out of range, Be careful with start and end vertices.
    """
    return find_all(wmat, start, end)[0]
