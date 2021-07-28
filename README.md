# Heuristic-Search-Visualiser
A simple visualisation of the Heuristic (A*) search algorithm on a grid with randomised obstacles. Uses Pygame.

Cell colours:
Pink - Obstacle cell
Light Blue - Starting cell
Light Green - Destination cell
Black - Empty (free) cell
Blue - A cell that is part of the shortest path between the start cell and destination cell
Purple - Cells explored during the algorithm (not a part of the final path)

1. When running the .py file, Pygame will open a window containing a grid with randomised obstacles (represented by pink cells). You can right click any obstacle cell / a start cell / end cell to change it to an empty cell.

<img width="806" alt="Screenshot 2021-07-28 at 10 37 39 PM" src="https://user-images.githubusercontent.com/67233931/127366277-adf9110b-f296-42cd-944d-d755eebe2b4a.png">

2. With a left click on an empty cell, you can place obstacles / mark a cell as a 'starting cell' / mark a cell as a 'destination cell'. If there is no cell marked as a starting cell or destination cell, a left click will mark the clicked empty cell as a starting cell or destination cell (in that order of preference). Otherwise, an obstacle is placed.

<img width="802" alt="Screenshot 2021-07-28 at 10 38 18 PM" src="https://user-images.githubusercontent.com/67233931/127366295-8329b697-56d7-453c-b2a6-eae5d04cca21.png">

3. Once you have your starting cell and ending cell marked, pressing the space bar will run the algorithm. If a path is found, it will be highlighted in blue. Purple is used to represent cells that were visited by the algorithm.
<img width="805" alt="Screenshot 2021-07-28 at 10 38 48 PM" src="https://user-images.githubusercontent.com/67233931/127366301-c71a2588-4392-4dfa-adaa-d54d6e38c88d.png">


4. Close the window to quit. If you want to generate a fresh random grid, press the 'c' key.
