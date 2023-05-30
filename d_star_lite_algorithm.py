#!/usr/bin/env python
import heapq

import numpy as np

np.set_printoptions(threshold=np.inf)
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
import time

from a_star import a_star
from priority_queue.priority_queue import priority_queue


class d_star_lite:
    def __init__(self, start_point, end_point, graph, img, R, shape_resize, save_path=False):
        self.rhs = {}
        self.g = {}
        self.h = {}

        self.K_m = 0

        self.open_set2 = priority_queue()

        self.cost = {}

        self.neighbours = [
            (1, 0),
            (0, 1),
            (-1, 0),
            (0, -1),
            (1, -1),
            (-1, 1),
            (1, 1),
            (-1, -1),
        ]
        self.neighbours2 = [
            (0, 0),
            (1, 0),
            (0, 1),
            (-1, 0),
            (0, -1),
            (1, -1),
            (-1, 1),
            (1, 1),
            (-1, -1),
        ]

        self.position = start_point
        self.start_point = start_point
        self.end_point = end_point

        self.save_path = save_path
        self.shape_resize = shape_resize
        graph = cv2.resize(
            graph, (self.shape_resize, self.shape_resize), interpolation=cv2.INTER_AREA
        )

        self.oryginal_graph = graph
        self.new_graph = graph
        self.old_graph = graph

        self.orginal_img = img.copy()
        self.img = img.copy()
        self.img_to_go = img.copy()

        self.orginal_img = cv2.resize(
            self.orginal_img,
            (self.shape_resize, self.shape_resize),
            interpolation=cv2.INTER_AREA,
        )
        self.img = cv2.resize(
            self.img,
            (self.shape_resize, self.shape_resize),
            interpolation=cv2.INTER_AREA,
        )
        self.img_to_go = cv2.resize(
            self.img_to_go,
            (self.shape_resize, self.shape_resize),
            interpolation=cv2.INTER_AREA,
        )

        self.path = []

        self.R = R

        self.slope = 300

        self.mask = np.zeros(
            (self.oryginal_graph.shape[0], self.oryginal_graph.shape[1]), np.uint8
        )
        self.mask[:] = 255

        cv2.circle(self.mask, start_point, self.R, (0, 0, 0), -1)
        self.new_graph = cv2.add(self.oryginal_graph, self.mask)
        self.new_graph[self.new_graph == 255] = 100
        self.old_graph = self.new_graph

    def initialize(self):
        for x in range(self.oryginal_graph.shape[1]):
            for y in range(self.oryginal_graph.shape[0]):
                self.rhs[(x, y)] = np.inf
                self.g[(x, y)] = np.inf

        self.update_heuristic(self.position)
        self.rhs[self.end_point] = 0.0
        self.open_set2.insert(self.end_point, (self.h[self.end_point], 0))

    def calc_key(self, point):
        key1 = min(self.g[point], self.rhs[point]) + self.h[point] + self.K_m
        key2 = min(self.g[point], self.rhs[point])

        return key1, key2

    def calc_cost(self, point_a, point_b, graph_):
        cost = abs(
            graph_[point_a[1]][point_a[0]].astype(int)
            - graph_[point_b[1]][point_b[0]].astype(int)
        )
        return cost

    def update_vertex(self, node):
        in_heap = False
        for i in self.open_set2:
            if i == node:
                in_heap = True

        if self.g[node] != self.rhs[node] and in_heap:
            self.open_set2.update(node, self.calc_key(node))

        elif self.g[node] != self.rhs[node] and not in_heap:
            self.open_set2.insert(node, self.calc_key(node))

        elif self.g[node] == self.rhs[node] and in_heap:
            self.open_set2.remove(node)

    def update_heuristic(self, node):
        for x in range(self.oryginal_graph.shape[1]):
            for y in range(self.oryginal_graph.shape[0]):
                dx = abs(x - node[0])
                dy = abs(y - node[1])
                self.h[(x, y)] = max(dx, dy)

    def compute_shortest_path(self):
        while (
            self.open_set2.top_key()[0][1][0] <= self.calc_key(self.position)[0]
            or self.rhs[self.position] > self.g[self.position]
        ):
            node = self.open_set2.top_key()[0][0]

            k_old = self.open_set2.top_key()[0][1][0]

            k_new = self.calc_key(node)[0]

            if k_old < k_new:  # update key
                self.open_set2.update(node, (k_new, min(self.g[node], self.rhs[node])))

            # locally overconsistent
            elif self.g[node] > self.rhs[node]:
                self.g[node] = self.rhs[node]
                # dequeue
                self.open_set2.remove(node)

                for neigh in self.neighbours:
                    neigh_point = (node[0] + neigh[0], node[1] + neigh[1])
                    if (
                        (neigh_point[0] >= 0)
                        and (neigh_point[0] < self.oryginal_graph.shape[0])
                        and (neigh_point[1] >= 0)
                        and (neigh_point[1] < self.oryginal_graph.shape[1])
                    ):
                        if neigh_point != self.end_point:
                            neigh_point_cost = 1 + self.calc_cost(
                                node, neigh_point, self.new_graph
                            )

                            self.rhs[neigh_point] = min(
                                self.rhs[neigh_point], neigh_point_cost + self.g[node]
                            )

                        # update cost for neighbours
                        self.update_vertex(neigh_point)

            # locally underconsistent
            else:
                g_old = self.g[node]
                self.g[node] = np.inf

                # update node rhs
                if node != self.end_point:
                    neigh_values2 = self.min_succ(node)
                    self.rhs[node] = neigh_values2[
                        min(neigh_values2, key=lambda k: neigh_values2[k])
                    ]
                    self.update_vertex(node)

                # update neighbours rhs
                for neigh in self.neighbours:
                    neigh_point = (node[0] + neigh[0], node[1] + neigh[1])
                    if (
                        (neigh_point[0] >= 0)
                        and (neigh_point[0] < self.oryginal_graph.shape[0])
                        and (neigh_point[1] >= 0)
                        and (neigh_point[1] < self.oryginal_graph.shape[1])
                    ):
                        cost1 = 1 + self.calc_cost(node, neigh_point, self.new_graph)

                        if self.rhs[neigh_point] == cost1 + g_old:
                            if neigh_point != self.end_point:
                                neigh_values2 = self.min_succ(neigh_point)

                                self.rhs[neigh_point] = neigh_values2[
                                    min(neigh_values2, key=lambda k: neigh_values2[k])
                                ]

                        self.update_vertex(neigh_point)

    def show_graph(self):
        graph_df2 = pd.DataFrame(
            index=range(0, self.oryginal_graph.shape[1]),
            columns=range(0, self.oryginal_graph.shape[0]),
        )

        for key, val in self.g.items():
            graph_df2.loc[key[1], key[0]] = (self.g[key], self.rhs[key])

            if key == self.position:
                graph_df2.loc[key[1], key[0]] = (self.g[key], self.rhs[key], "X")

    def save_imgs(self):
        filename = str(len(self.path))+'.jpg'
        cv2.imwrite(self.save_path+'/'+filename, self.img_to_go)

    def show_path_to_go(self):
        pos = self.position
        path_to_go = []

        mask_copy = self.mask.copy()
        mask_copy = cv2.cvtColor(mask_copy, cv2.COLOR_GRAY2BGR)
        self.img_to_go = cv2.add(self.img_to_go, mask_copy)
        self.img_to_go[self.img_to_go == 255] = 100

        while pos != self.end_point:
            neigh_values = {}
            for neigh in self.neighbours:
                neigh_point = (pos[0] + neigh[0], pos[1] + neigh[1])
                if (
                    (neigh_point[0] >= 0)
                    and (neigh_point[0] < self.oryginal_graph.shape[0])
                    and (neigh_point[1] >= 0)
                    and (neigh_point[1] < self.oryginal_graph.shape[1])
                ):
                    neigh_values[neigh_point] = self.g[neigh_point]

            pos = min(neigh_values, key=lambda k: neigh_values[k])
            path_to_go.append(pos)

        cv2.circle(self.img_to_go, self.start_point, 2, (235, 23, 19), -1)
        cv2.circle(self.img_to_go, self.end_point, 2, (0, 255, 230), -1)

        cv2.circle(self.img_to_go, self.position, self.R, (207, 203, 4), 1)

        for point in path_to_go:
            if point != self.start_point and point != self.end_point:
                self.img_to_go[point[1]][point[0]] = (0, 0, 255)

        for point in self.path:
            if point != self.start_point and point != self.end_point:
                self.img_to_go[point[1]][point[0]] = (0, 255, 0)

        # Save images
        if self.save_path:
            self.save_imgs()

        cv2.imshow("img to go", self.img_to_go)

    def min_succ(self, point_):
        neigh_values2 = {}
        for neigh2 in self.neighbours:
            neigh_point = (point_[0] + neigh2[0], point_[1] + neigh2[1])
            if (
                (neigh_point[0] >= 0)
                and (neigh_point[0] < self.oryginal_graph.shape[0])
                and (neigh_point[1] >= 0)
                and (neigh_point[1] < self.oryginal_graph.shape[1])
            ):
                if neigh_point != self.end_point:
                    cost2 = 1 + self.calc_cost(point_, neigh_point, self.new_graph)

                    neigh_values2[neigh_point] = self.g[neigh_point] + cost2

        return neigh_values2

    def move_to_end(self):
        last_node = self.position
        self.initialize()

        self.compute_shortest_path()

        self.path.append(self.position)

        sum = 0
        counter = 0

        while self.position != self.end_point:
            cv2.circle(self.mask, self.position, self.R, (0, 0, 0), -1)
            self.new_graph = cv2.add(self.oryginal_graph, self.mask)
            self.new_graph[self.new_graph == 255] = 100

            # ----------------------------
            self.show_path_to_go()
            cv2.waitKey(100)
            # ----------------------------

            self.img_to_go = self.orginal_img.copy()

            if self.g[self.position] == np.inf:
                print("Path not exist")
                return 0

            neigh_values = {}
            for neigh in self.neighbours:
                neigh_point = (self.position[0] + neigh[0], self.position[1] + neigh[1])
                # neigh_point = (self.position[1] + neigh[1], self.position[0] + neigh[0])
                if (
                    (neigh_point[0] >= 0)
                    and (neigh_point[0] < self.oryginal_graph.shape[0])
                    and (neigh_point[1] >= 0)
                    and (neigh_point[1] < self.oryginal_graph.shape[1])
                ):
                    neigh_values[neigh_point] = self.g[neigh_point]

            self.position = min(neigh_values, key=lambda k: neigh_values[k])

            self.update_heuristic(self.position)

            self.path.append(self.position)

            if (self.new_graph - self.old_graph).any():
                self.K_m = self.K_m + self.h[last_node]

                last_node = self.position

                # find changes between maps
                dif = self.new_graph - self.old_graph
                list_with_changes = []
                for x in range(dif.shape[1]):
                    for y in range(dif.shape[0]):
                        if dif[y][x] and (x, y) != self.end_point:
                            list_with_changes.append((x, y))

                # update cost outgoing edges
                # u - point
                # v - neigh point
                for point in list_with_changes:
                    for neigh in self.neighbours:
                        neigh_point = (point[0] + neigh[0], point[1] + neigh[1])

                        if (
                            (neigh_point[0] >= 0)
                            and (neigh_point[0] < self.oryginal_graph.shape[0])
                            and (neigh_point[1] >= 0)
                            and (neigh_point[1] < self.oryginal_graph.shape[1])
                        ):
                            c_old = 1 + self.calc_cost(
                                point, neigh_point, self.old_graph
                            )
                            c_new = 1 + self.calc_cost(
                                point, neigh_point, self.new_graph
                            )

                            if c_old > c_new:
                                if neigh_point != self.end_point:
                                    self.rhs[point] = min(
                                        self.rhs[point], c_new + self.g[neigh_point]
                                    )

                            elif self.rhs[point] == (c_old + self.g[neigh_point]):
                                if neigh_point != self.end_point:
                                    point_succ = self.min_succ(point)

                                    self.rhs[point] = point_succ[
                                        min(point_succ, key=lambda k: point_succ[k])
                                    ]

                            self.update_vertex(point)

                # update incoming edges
                # u - neigh point
                # v - point
                for point in list_with_changes:
                    for neigh in self.neighbours:
                        neigh_point = (point[0] + neigh[0], point[1] + neigh[1])
                        if (
                            (neigh_point[0] >= 0)
                            and (neigh_point[0] < self.oryginal_graph.shape[0])
                            and (neigh_point[1] >= 0)
                            and (neigh_point[1] < self.oryginal_graph.shape[1])
                        ):
                            c_old = 1 + self.calc_cost(
                                point, neigh_point, self.old_graph
                            )
                            c_new = 1 + self.calc_cost(
                                point, neigh_point, self.new_graph
                            )

                            if c_old > c_new:
                                if neigh_point != self.end_point:
                                    self.rhs[neigh_point] = min(
                                        self.rhs[neigh_point], c_new + self.g[point]
                                    )

                            elif self.rhs[neigh_point] == (c_old + self.g[point]):
                                if neigh_point != self.end_point:
                                    neigh_point_succ = self.min_succ(neigh_point)

                                    self.rhs[neigh_point] = neigh_point_succ[
                                        min(
                                            neigh_point_succ,
                                            key=lambda k: neigh_point_succ[k],
                                        )
                                    ]

                            self.update_vertex(neigh_point)

                start = time.time()
                self.compute_shortest_path()
                end = time.time()

                sum = sum + end - start
                counter = counter + 1

            self.old_graph = self.new_graph

        print("Mean time", sum / counter)
        return self.path
