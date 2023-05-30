#!/usr/bin/env python

import numpy as np
import time
import networkx as nx


def a_star(start, end, img, slope):
    # f(x) = g(x) + h(x)

    path = []
    neighbours = [(1, 0), (0, 1), (-1, 0), (0, -1), (1, -1), (-1, 1), (1, 1), (-1, -1)]

    closed_set = []
    open_set = {}
    g_score = {}
    f_score = {}
    came_from = {}

    # fill open set with begin values
    g_score[start] = 0
    f_score[start] = np.sqrt((start[0] - end[0]) ** 2 + (start[1] - end[1]) ** 2)
    open_set[start] = start

    while len(open_set) > 0:
        current_point = None
        current_F = None

        for point in open_set.keys():
            if (current_point is None) or (f_score[point] < current_F):
                current_F = f_score[point]
                current_point = point

        if current_point == end:
            path.append(current_point)
            while current_point in came_from.keys():
                current_point = came_from[current_point]
                path.append(current_point)

            path.reverse()

            return path

        open_set.pop(current_point)
        closed_set.append(current_point)

        for neigh in neighbours:
            current_neigh = (current_point[0] + neigh[0], current_point[1] + neigh[1])

            if current_neigh in closed_set:
                continue

            if current_neigh not in open_set.keys():
                if (
                    (current_neigh[0] >= 0)
                    and (current_neigh[0] < img.shape[0])
                    and (current_neigh[1] >= 0)
                    and (current_neigh[1] < img.shape[1])
                ):
                    #                   y                       x
                    current_neigh_value = img[current_neigh[1]][current_neigh[0]]
                    current_point_value = img[current_point[1]][current_point[0]]

                    # change type (because absolute works wrong, must be signed int)
                    current_neigh_value = current_neigh_value.astype(int)
                    current_point_value = current_point_value.astype(int)

                    # difference beetwen two neighboring pixels
                    neigh_score = (
                        abs(current_neigh_value - current_point_value)
                        + 1
                        + g_score[current_point]
                    )

                    open_set[current_neigh] = current_point

                    came_from[current_neigh] = current_point
                    g_score[current_neigh] = neigh_score
                    h = np.sqrt(
                        (current_neigh[0] - end[0]) ** 2
                        + (current_neigh[1] - end[1]) ** 2
                    )

                    f_score[current_neigh] = g_score[current_neigh] + h

            elif neigh_score >= g_score[current_neigh]:
                continue


def add_edge_if_not_exists(G, node1, node2):
    if not G.has_edge(node1, node2) and not G.has_edge(node2, node1):
        G.add_edge(node1, node2)
    return G


def a_star_2(start, end, img, slope):
    G = nx.Graph()
    rows, cols = img.shape

    for i in range(rows):
        for j in range(cols):
            G.add_node((i, j))

    for i in range(rows):
        for j in range(cols):
            neighbors = [
                (i - 1, j),
                (i + 1, j),
                (i, j - 1),
                (i, j + 1),
                (i - 1, j - 1),
                (i - 1, j + 1),
                (i + 1, j - 1),
                (i + 1, j + 1),
            ]
            for neighbor in neighbors:
                if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols:
                    G = add_edge_if_not_exists(G, (i, j), neighbor)

    def weight(u, v, edge):
        node_cost = abs(img[u[1], u[0]] - img[v[1], v[0]])
        return node_cost + 1

    def heuristic(u, v):
        x1, y1 = u
        x2, y2 = v
        return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    path = nx.astar_path(G, start, end, weight=weight, heuristic=heuristic)

    return path
