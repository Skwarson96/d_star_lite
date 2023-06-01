# d_star_lite

This repository contains implementations of the A* and D* Lite algorithms, which were developed as part of a master's thesis. The algorithms are applied to images where the color of each pixel represents the terrain height, and the cost of traversing between pixels (graph nodes) is determined by the difference in pixel values.

# A* Algorithm
The A* algorithm is a popular pathfinding algorithm that efficiently finds the shortest path between two points in a graph. It uses a heuristic function to guide its search and provides an optimal path if the heuristic is admissible and consistent.
D* Lite Algorithm

# The D* Lite
algorithm is an incremental search algorithm that provides efficient path updates in dynamic environments. It maintains a goal-directed search tree and updates the path based on changes in the graph or the goal.

# Examples
In the visualizations, the robot (orange cross) moves from the starting point to the destination. It is equipped with a sensor that gives it knowledge about the world around it in a radius of R. The robot on the way (green) to the destination learns new parts of the map and replans the path (red)

<p align="center">
<table>
  <tr>
    <td align="center">
      <img src="https://github.com/Skwarson96/d_star_lite/blob/main/imgs/gifs/map1_r10.gif" width="200"/>
      <br>R=10
    </td>
    <td align="center">
      <img src="https://github.com/Skwarson96/d_star_lite/blob/main/imgs/gifs/map1_r30.gif" width="200"/>
      <br>R=30
    </td>
    <td align="center">
      <img src="https://github.com/Skwarson96/d_star_lite/blob/main/imgs/gifs/map1_r50.gif" width="200"/>
      <br>R=50
    </td>
  </tr>
</table>
</p>

<p align="center">
<table>
  <tr>
    <td align="center">
      <img src="https://github.com/Skwarson96/d_star_lite/blob/main/imgs/gifs/map6_r10.gif" width="200"/>
      <br>R=10
    </td>
    <td align="center">
      <img src="https://github.com/Skwarson96/d_star_lite/blob/main/imgs/gifs/map6_r30.gif" width="200"/>
      <br>R=30
    </td>
    <td align="center">
      <img src="https://github.com/Skwarson96/d_star_lite/blob/main/imgs/gifs/map6_r50.gif" width="200"/>
      <br>R=50
    </td>
  </tr>
</table>
</p>


  
