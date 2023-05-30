#!/usr/bin/env python

import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from a_star import a_star, a_star_2

from d_star_lite_algorithm import d_star_lite


def plot_2d_path(path, img2):
    print("Ilosc pikseli w sciezce: ", len(path))

    if path == None:
        print("Path no exist")
        return 0

    for point in path:
        cv2.circle(img2, point, 1, (255, 0, 0), -1)

    cv2.imshow("img2", img2)
    cv2.waitKey()


def surface_3d_plot(path, img):
    print("surface_3d_plot")
    xx, yy = np.mgrid[0 : img.shape[0], 0 : img.shape[1]]

    X = [a_tuple[1] for a_tuple in path]
    Y = [a_tuple[0] for a_tuple in path]
    Z = []
    for point in path:
        Z.append(img[point[1]][point[0]])

    # create the figure
    fig = plt.figure(figsize=(8, 6), dpi=150)
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(xx, yy, img, rstride=1, cstride=1, cmap="gray", edgecolor="none")
    ax.plot(X, Y, Z, c="r", marker=".", alpha=1)

    ax.view_init(azim=145, elev=60)

    plt.figure()
    plt.plot(Z)
    plt.title("Height")

    # show it
    plt.show()


if __name__ == "__main__":
    print("path_planning")

    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, "./maps/map-new6-real.bmp")

    # read image
    img = cv2.imread(filename, 1)
    img2 = img.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    start_point = (10, 10)  # (x, y)
    end_point = (30, 30)
    # start point
    cv2.circle(img2, start_point, 2, (0, 0, 255), -1)
    # end point
    cv2.circle(img2, end_point, 2, (0, 255, 0), -1)

    slope = 0
    # A star
    # path_from_start_to_end = a_star(start_point, end_point, img, slope)
    # path_from_start_to_end_2 = a_star_2(start_point, end_point, img, slope)

    # surface_3d_plot(path_from_start_to_end, img)

    # D star
    save_path = './imgs/test_dir/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    d_star_ = d_star_lite(start_point, end_point, img, img2, R=50, shape_resize=64, save_path=save_path)
    d_star_.move_to_end()

    print("done")
