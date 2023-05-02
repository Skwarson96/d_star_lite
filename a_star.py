#!/usr/bin/env python

import numpy as np
import time

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

        # current_point = min(f_score, key=f_score.get)
        for point in open_set.keys():
            if (current_point is None) or (f_score[point] < current_F):
                current_F = f_score[point]
                current_point = point


        if current_point == end:
            path.append(current_point)
            while current_point in came_from.keys():
                current_point = came_from[current_point]
                # print(current_point)
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
                if ((current_neigh[0] >= 0) and (current_neigh[0] < img.shape[0]) and (current_neigh[1] >= 0) and (
                        current_neigh[1] < img.shape[1])):
                    #                   y                       x
                    current_neigh_value = img[current_neigh[1]][current_neigh[0]]
                    current_point_value = img[current_point[1]][current_point[0]]

                    # change type (because absolute works wrong, must be signed int)
                    current_neigh_value = current_neigh_value.astype(int)
                    current_point_value = current_point_value.astype(int)

                    # print(current_point, current_point_value, current_neigh, current_neigh_value, abs(current_neigh_value - current_point_value) )

                    # difference beetwen two neighboring pixels
                    neigh_score = abs(current_neigh_value - current_point_value) + 1 + g_score[current_point]

                    open_set[current_neigh] = current_point

                    came_from[current_neigh] = current_point
                    g_score[current_neigh] = neigh_score
                    h = np.sqrt((current_neigh[0] - end[0]) ** 2 + (current_neigh[1] - end[1]) ** 2)
                    # h = abs(current_neigh[0] - end[0]) + abs(current_neigh[1] - end[1])
                    f_score[current_neigh] = g_score[current_neigh] + h

                    # if abs(current_neigh_value - current_point_value) < slope or current_neigh_value == 0:
                    #     # continue
                    #     open_set[current_neigh] = current_point

            elif neigh_score >= g_score[current_neigh]:
                continue

            # came_from[current_neigh] = current_point
            # g_score[current_neigh] = neigh_score
            # h = np.sqrt((current_neigh[0] - end[0]) ** 2 + (current_neigh[1] - end[1]) ** 2)
            # # h = abs(current_neigh[0] - end[0]) + abs(current_neigh[1] - end[1])
            # f_score[current_neigh] = g_score[current_neigh] + h
