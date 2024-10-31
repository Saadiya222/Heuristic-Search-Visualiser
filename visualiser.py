import pygame
import time 
import psutil
import os
from queue import PriorityQueue # For A*
from collections import deque # For BFS



# display settings for the pygame window
def create_window(rows, cols):
    CELL_SIZE = 40  # Fixed size for each cell
    width = cols * CELL_SIZE
    height = rows * CELL_SIZE
    return pygame.display.set_mode((width, height))


# mentioning the color for cells of each state
OBSTACLE_COLOR = (0, 0, 0)
EXPLORED_COLOR = (211, 211, 211)
YELLOW = (255, 255, 0) ###
CREAM = (250, 249, 246)
PATH_COLOR = (0, 128, 255)
START_COLOR = (255, 215, 0) # knight color gold 
GREY = (64, 64, 64) ###
END_COLOR = (255,192,203) # princess color pink 


class Cell:
    def __init__(self, row, col, cell_size):
        self.row = row
        self.col = col
        self.x = col * cell_size
        self.y = row * cell_size
        self.color = CREAM
        self.neighbours = []
        self.width = cell_size
        self.total_rows = None
        self.total_cols = None

    def get_pos(self): # Returns the cell’s row and column position
        return self.row, self.col

# the following methods check the cell’s current state

    def is_closed(self): 
        return self.color == EXPLORED_COLOR

    def is_open(self):
        return self.color == CREAM

    def is_barrier(self):
        return self.color == OBSTACLE_COLOR

    def is_start(self):
        return self.color == START_COLOR

    def is_end(self):
        return self.color == END_COLOR

    def is_path(self):
        return self.color == PATH_COLOR

    def reset(self):
        self.color = CREAM

# Update the cell’s color to indicate its new state (making the search dynamic)

    def make_start(self): # choose knight initial location
        self.color = START_COLOR

    def make_closed(self): # grey
        self.color = EXPLORED_COLOR

    def make_open(self): 
        self.color = CREAM

    def make_barrier(self):
        self.color = OBSTACLE_COLOR

    def make_end(self): # choose princess location
        self.color = END_COLOR

    def make_path(self): 
        self.color = PATH_COLOR

    def draw(self, win):
        pygame.draw.rect(win, self.color, (self.x, self.y, self.width, self.width))


    def update_neighbours(self, grid): # Updates the neighbours list for each cell by checking for open cells around it
        self.neighbours = []
        self.total_rows = len(grid)
        self.total_cols = len(grid[0])

        # DOWN
        if self.row < self.total_rows - 1 and not grid[self.row + 1][self.col].is_barrier():
            self.neighbours.append(grid[self.row + 1][self.col])

        # UP
        if self.row > 0 and not grid[self.row - 1][self.col].is_barrier():
            self.neighbours.append(grid[self.row - 1][self.col])

        # RIGHT
        if self.col < self.total_cols - 1 and not grid[self.row][self.col + 1].is_barrier():
            self.neighbours.append(grid[self.row][self.col + 1])

        # LEFT
        if self.col > 0 and not grid[self.row][self.col - 1].is_barrier():
            self.neighbours.append(grid[self.row][self.col - 1])

def heuristic_fn(p1, p2):  # for a* : the heuristic function used in our scenario is manhattan distance (sum of the absolute differences between the x and y coordinates of two points)
    x1, y1 = p1
    x2, y2 = p2
    return abs(x1 - x2) + abs(y1 - y2)


def show_path(came_from, cur, start, draw):  # for a* and bfs without removal Bbacktracks through the cells to display the found path after the end cell is reached and displays the shortest path if found
    while cur in came_from:
        cur = came_from[cur]
        if cur != start:
            cur.make_path()
        draw()
'''''
def show_path_with_removals(ancestor, current_state, start, draw, removed_obstacles): # for a* and bfs without removal
    path = []
    path_cells = set()  # Initialize empty set for path cells
    removed_list = []
    actual_removed = set()
    
    current = current_state
    while current in ancestor:
        cell, removals = current
        path_cells.add(cell)  # Add each cell to the set
        
        if current in removed_obstacles:
            removed_list.append(removed_obstacles[current])
            actual_removed.add(cell)
        
        path.append((cell.row, cell.col))
        
        if cell != start:
            if cell in actual_removed:
                cell.make_barrier()
            else:
                cell.make_path()
        
        current = ancestor[current]
        draw()
    
    path.append((start.row, start.col))
    path_cells.add(start)  # Don't forget to add start cell
    
    print("\n=== Path Details ===")
    print(f"Path found with {len(removed_list)} obstacle(s) removed")
    print(f"Removed obstacles at positions: {removed_list}")
    print(f"Complete path: {path[::-1]}")
    
    return path_cells  # Now we always return a set, even if empty
'''

def reconstruct_removal_path(ancestor, current_state, start, draw, removed_obstacles):
    path = []
    path_cells = set()
    removed_list = []
    actual_removed = set()
    
    current = current_state
    while current in ancestor:
        cell, removals = current
        path_cells.add(cell)
        
        if current in removed_obstacles:
            removed_list.append(removed_obstacles[current])
            actual_removed.add(cell)
        
        path.append((cell.row, cell.col))
        
        if cell != start:
            if cell in actual_removed:
                cell.make_barrier()
            else:
                cell.make_path()
        
        current = ancestor[current]
        draw()
    
    # Add start position
    path.append((start.row, start.col))
    path_cells.add(start)
    

    print(f"Path found with {len(removed_list)} obstacle(s) removed")
    print(f"Removed obstacles at positions: {removed_list}")
    print(f"Shortest path: {path[::-1]}")
    print()
    return path_cells

def get_path_coordinates(ancestor, end, start):
    path = []
    current = end
    while current in ancestor:
        path.append((current.row, current.col))
        current = ancestor[current]
    path.append((start.row, start.col))
    return path[::-1]  # Reverse to get start->end order

def bfsWithoutRemoval(draw, grid, start, end):
    start_time = time.time()
    queue = deque([start])
    visited = {start}
    ancestor = {}
    max_queue_size = 1
    
    while queue:
        max_queue_size = max(max_queue_size, len(queue))
        current = queue.popleft()
        
        if current == end:
            end_time = time.time()
            path = get_path_coordinates(ancestor, end, start)
            print("\n=== Search Results (BFS) ===")
            print(f" *** Path found! *** \nCost: ({len(path) -1})")
            print(f"Shortest path: {path}")   
            print(f"Number of nodes visited: {len(visited)}")
            print(f"Visited nodes: {sorted([(node.row, node.col) for node in visited])}")
            
            calculate_performance_metrics(start_time, end_time, visited, max_queue_size, search_type="BFS")
            show_path(ancestor, end, start, draw)
            end.make_end()
            return True
            
        for neighbor in current.neighbours:
            if neighbor not in visited:
                queue.append(neighbor)
                visited.add(neighbor)
                ancestor[neighbor] = current
                neighbor.make_open()
        
        draw()
        if current != start:
            current.make_closed()
    
    end_time = time.time()
    print("\n=== Search Results (BFS) ===")
    print("No path found!")
    calculate_performance_metrics(start_time, end_time, visited, max_queue_size, search_type="BFS")
    return False

def bfsWithRemoval(draw, grid, start, end, max_removals):
    start_time = time.time()
    initial_state = (start, 0)
    queue = deque([initial_state])
    visited = {initial_state}
    ancestor = {}
    removed_obstacles = {}
    max_queue_size = 1
    explored_cells = set()
    
    while queue:
        max_queue_size = max(max_queue_size, len(queue))
        current_state = queue.popleft()
        current_cell, removals_used = current_state
        
        explored_cells.add(current_cell)
        if current_cell != start and current_cell != end:
            current_cell.make_closed()
        
        if current_cell == end:
            end_time = time.time()
            print("\n=== Search Results (BFS with removal) ===")
            print(f" *** Path found! *** ")
            
            # Get unique visited cells (without duplicates from different removal states)
            unique_visited = set((state[0].row, state[0].col) for state in visited)
            visited_coordinates = sorted(list(unique_visited))
            
            print(f"Number of nodes visited: {len(unique_visited)}")
            print(f"Visited nodes: {visited_coordinates}")
            
            path_cells = reconstruct_removal_path(ancestor, current_state, start, draw, removed_obstacles)
            cost = len(path_cells) - 1
            print(f"Cost : {cost} ")
            
            # Maintain explored cells visualization
            for cell in explored_cells:
                if cell not in path_cells and cell != start and cell != end:
                    cell.make_closed()
            
            end.make_end()
            calculate_performance_metrics(start_time, end_time, visited, max_queue_size, search_type="BFS with removal")
            return True
            
        for neighbor in get_neighbors_with_removals(current_cell, grid, removals_used, max_removals):
            neighbor_cell, new_removals, removed = neighbor
            neighbor_state = (neighbor_cell, new_removals)
            
            if neighbor_state not in visited:
                queue.append(neighbor_state)
                visited.add(neighbor_state)
                ancestor[neighbor_state] = current_state
                if removed:
                    removed_obstacles[neighbor_state] = (neighbor_cell.row, neighbor_cell.col)
                if neighbor_cell not in explored_cells and neighbor_cell != end:
                    neighbor_cell.make_open()
        
        draw()
    
    end_time = time.time()
    print("\n=== Search Results (BFS with removal) ===")
    print("No path found!")
    calculate_performance_metrics(start_time, end_time, visited, max_queue_size, search_type="BFS with removal")
    return False 




def aStarWithoutRemoval(draw, grid, start, end):  # the search algorithm

    # keeping track of the complexities ( Tracks execution time, Measures memory usage, Counts explored nodes, Tracks maximum frontier size, Prints theoretical complexity analysis, Shows actual performance metrics)
    start_time = time.time()
    count = 0
    open_set = PriorityQueue()
    ancestor = {}
    g_score = {cell: float("inf") for row in grid for cell in row}
    g_score[start] = 0
    f_score = {cell: float("inf") for row in grid for cell in row}
    f_score[start] = heuristic_fn(start.get_pos(), end.get_pos()) + g_score[start]

    open_set_hash = {start}
    open_set.put((f_score[start], count, start))
    visited_nodes = set()
    max_frontier_size = 0

    
    count = 0 # so that ties are broken by insertion order in the Pq
    open_set = PriorityQueue() # Frontier | lower-priority (better f scorre) cells being dequeued first
    ancestor = {} # to reconstruct the path once the end cell is reached
    g_score = {cell: float("inf") for row in grid for cell in row} # Initially, all cells are set to infinity except for the g(start)=0, bc wel'll check in line 135
    g_score[start] = 0
    f_score = {cell: float("inf") for row in grid for cell in row} # Initially, all cells are set to infinity
    f_score[start] = heuristic_fn(start.get_pos(), end.get_pos()) + g_score[start]

    open_set_hash = {start} # A set that mirrors open_set, stores the cells currently in the queue to allow for efficient membership checks
    open_set.put((f_score[start], count, start)) # putting the source (start) cell into the Pqueue

    # Track visited nodes
    visited_nodes = set()

    while not open_set.empty():
        max_frontier_size = max(max_frontier_size, len(open_set_hash))
        for event in pygame.event.get():
            if event.type == pygame.QUIT: # if user quits window
                pygame.quit()

        cur = open_set.get()[2] # Retrieves the cell with the lowest f_score from open_set
        open_set_hash.remove(cur) # because we are now exploring it
        visited_nodes.add((cur.row, cur.col))


        if cur == end:
            end_time = time.time() # stop timer 
            calculate_performance_metrics(start_time, end_time, visited_nodes, max_frontier_size)
            path = []
            current = end
            while current in ancestor:
                path.append((current.row, current.col))
                current = ancestor[current]
            path.append((start.row, start.col))
            path.reverse()
            
            print("\n=== Search Results ===")
            print(f" *** Path found! ***  \nCost: {g_score[end]}")
            print(f"Shortest path: {path}")
            print(f"Number of nodes visited: {len(visited_nodes)}")
            print(f"Visited nodes: {sorted(visited_nodes)}")
            
            show_path(ancestor, end, start, draw)
            end.make_end()
            return True
        
        
        for neighbour in cur.neighbours:  # check the neighbours of the current cell 
            if g_score[cur] + 1 < g_score[neighbour]: # is the path to neighbour through cur shorter than the previously recorded path 
                ancestor[neighbour] = cur
                g_score[neighbour] = g_score[cur] + 1
                f_score[neighbour] = g_score[neighbour] + heuristic_fn(neighbour.get_pos(), end.get_pos())
                if neighbour not in open_set_hash:
                    count += 1
                    open_set.put((f_score[neighbour], count, neighbour))
                    open_set_hash.add(neighbour)
                    neighbour.make_open() # Changes the color of neighbour to indicate it has been added to the open set (light grey)
        draw()
        if cur != start: # If cur is not the start cell, its color is changed to indicate that it has been fully explored
            cur.make_closed()
    
    print("No solution found!")
    end_time = time.time()
    calculate_performance_metrics(start_time, end_time, visited_nodes, max_frontier_size) # we get performance metrics for both successful and unsuccessful searches
    return False
    


def aStarWithRemoval(draw, grid, start, end, max_removals): # A* search with and without obstacle removal
    
    # keeping track of the complexities ( Tracks execution time, Measures memory usage, Counts explored nodes, Tracks maximum frontier size, Prints theoretical complexity analysis, Shows actual performance metrics)
    start_time = time.time()

    count = 0 # so that ties are broken by insertion order in the Pq
    open_set = PriorityQueue()
    ancestor = {} # to reconstruct the path once the end cell is reached
    removed_obstacles = {}  # Track which obstacles were removed in the optimal path
    
    # State includes position and removals used
    g_score = {}
    f_score = {}
    
    # Initialize scores for start state
    initial_state = (start, 0)  # (position, removals_used)
    g_score[initial_state] = 0
    f_score[initial_state] = heuristic_fn(start.get_pos(), end.get_pos())
    
    open_set_hash = {initial_state} # A set that mirrors open_set, stores the cells currently in the queue to allow for efficient membership checks
    open_set.put((f_score[initial_state], count, initial_state)) # putting the source (start) cell into the Pqueue
    visited_nodes = set() #####
    max_frontier_size = 0

    while not open_set.empty():
        max_frontier_size = max(max_frontier_size, len(open_set_hash))
        
        current_state = open_set.get()[2]
        current_cell, removals_used = current_state
        open_set_hash.remove(current_state)
        visited_nodes.add((current_cell.row, current_cell.col))

        if current_cell == end:
            end_time = time.time()
            calculate_performance_metrics(start_time, end_time, visited_nodes, max_frontier_size)
            show_path_with_removals(ancestor, current_state, start, draw, removed_obstacles,visited_nodes)
            return True

        # Get neighbors including potential obstacle removals
        for neighbor in get_neighbors_with_removals(current_cell, grid, removals_used, max_removals):
            neighbor_cell, new_removals, removed = neighbor
            neighbor_state = (neighbor_cell, new_removals)
            
            temp_g_score = g_score[current_state] + 1

            if neighbor_state not in g_score or temp_g_score < g_score[neighbor_state]:
                ancestor[neighbor_state] = current_state
                if removed:
                    removed_obstacles[neighbor_state] = (neighbor_cell.row, neighbor_cell.col)
                g_score[neighbor_state] = temp_g_score
                f_score[neighbor_state] = temp_g_score + heuristic_fn(neighbor_cell.get_pos(), end.get_pos())
                
                if neighbor_state not in open_set_hash:
                    count += 1
                    open_set.put((f_score[neighbor_state], count, neighbor_state))
                    open_set_hash.add(neighbor_state)
                    neighbor_cell.make_open() 

        draw()
        if current_cell != start: # If cur is not the start cell, its color is changed to indicate that it has been fully explored
            current_cell.make_closed()

# we get performance metrics for both successful and unsuccessful searches
    end_time = time.time()
    calculate_performance_metrics(start_time, end_time, visited_nodes, max_frontier_size)
    return False

def get_neighbors_with_removals(cell, grid, removals_used, max_removals):
    neighbors = []
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]  # Down, Up, Right, Left
    
    for dx, dy in directions:
        new_row = cell.row + dx
        new_col = cell.col + dy
        
        if 0 <= new_row < len(grid) and 0 <= new_col < len(grid[0]):
            neighbor = grid[new_row][new_col]
            if not neighbor.is_barrier():
                neighbors.append((neighbor, removals_used, False))
            elif removals_used < max_removals:
                neighbors.append((neighbor, removals_used + 1, True))
                
    return neighbors

def show_path_with_removals(ancestor, current_state, start, draw, removed_obstacles,visited_nodes):
    path = []
    removed_list = []
    
    while current_state in ancestor:
        cell, removals = current_state
        if current_state in removed_obstacles:
            removed_list.append(removed_obstacles[current_state])
        path.append((cell.row, cell.col))
        if cell != start:
            cell.make_path()
        current_state = ancestor[current_state]
        draw()
    
    print("\n=== Path Details ===")
    print(f"Path found with {len(removed_list)} obstacle(s) removed")
    print(f"Removed obstacles at positions: {removed_list}")
    print(f"Shortest path: {path[::-1]}")
    print(f"Cost: {len(path)-1}")
    print(f"Number of visited nodes: {len(visited_nodes)}")
    print(f"Visited nodes: {sorted(visited_nodes)}")

def create_grid(rows, cols, cell_size, obstacle_coords):
    grid = []
    for i in range(rows):
        grid.append([])
        for j in range(cols):
            cell = Cell(i, j, cell_size)
            if (i, j) in obstacle_coords:
                cell.make_barrier()
            grid[i].append(cell)
    return grid




def draw_grid(win, rows, cols, cell_size):
    # Draw horizontal lines
    for i in range(rows + 1):
        pygame.draw.line(win, GREY, (0, i * cell_size), (cols * cell_size, i * cell_size))
    # Draw vertical lines
    for j in range(cols + 1):
        pygame.draw.line(win, GREY, (j * cell_size, 0), (j * cell_size, rows * cell_size))



def draw(win, grid, rows, cols, width):
    win.fill(CREAM)
    for row in grid:
        for cell in row:
            cell.draw(win)
    draw_grid(win, rows, cols, width)
    pygame.display.update()
    time.sleep(0.1)  # Adds a 0.1 second delay


def read_input_file(filename):
    with open(filename, 'r') as file:
        # Read grid dimensions
        rows, cols = map(int, file.readline().strip().split())
        
        # Read obstacles
        obstacle_coords = []
        line = file.readline().strip()
        while line and line[0].isdigit():
            row, col = map(int, line.split())
            obstacle_coords.append((row, col))
            line = file.readline().strip()
            
        # Read number of obstacles to remove
        obstacles_to_remove = int(line.split('=')[1].strip())
        
        # Read search type
        search_type = file.readline().strip()
        
        return rows, cols, obstacle_coords, obstacles_to_remove, search_type
    
def calculate_performance_metrics(start_time, end_time, visited_nodes, open_set_size, search_type="A*"):
    # he parameter 'search_type="A*"' is a default value that ensures backward compatibility with existing code. When no search type is specified in the function call, it assumes we're running A*. This default value makes the function flexible - it can handle both old calls (which don't specify a search type) and new calls (which do specify BFS or other search types). It's a good programming practice that allows us to extend functionality while maintaining existing code.
    time_taken = end_time - start_time
    process = psutil.Process(os.getpid())
    memory_used = process.memory_info().rss / 1024 / 1024
    
    print(f"\n=== Performance Metrics ({search_type}) ===")
    print(f"Time taken: {time_taken:.4f} seconds")
    print(f"Memory used: {memory_used:.2f} MB")
    print(f"Space complexity: O(|V|) where |V| = {len(visited_nodes)} nodes explored")
    print(f"Maximum frontier size: {open_set_size} nodes")
    print(f"Time complexity: O(|V| + |E|) where |E| = number of edges explored")
    

def main():
    CELL_SIZE = 40
    
    try:
        rows, cols, obstacle_coords, obstacles_to_remove, search_type = read_input_file('input.txt')
        
        # Validate input
        if rows <= 0 or cols <= 0:
            raise ValueError("Invalid grid dimensions")
        for row, col in obstacle_coords:
            if row < 0 or row >= rows or col < 0 or col >= cols:
                raise ValueError("Invalid obstacle coordinates")
    
        # Create window based on grid size
        win = create_window(rows, cols)
        
        grid = create_grid(rows, cols, CELL_SIZE, obstacle_coords)
        
        # Set fixed start and end positions
        start = grid[0][0]  # Top-left corner
        end = grid[rows-1][cols-1]  # Bottom-right corner
        
        start.make_start()
        end.make_end()
        
        run = True
        
        while run:
            draw(win, grid, rows, cols, CELL_SIZE)
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False
                    
                if event.type == pygame.KEYDOWN:
                    # A* controls
                    if event.key == pygame.K_SPACE:
                        for row in grid:
                            for cell in row:
                                cell.update_neighbours(grid)
                        print("\n=== Running A* Search Without Obstacle Removal ===")
                        aStarWithoutRemoval(lambda: draw(win, grid, rows, cols, CELL_SIZE), grid, start, end)
                    
                    if event.key == pygame.K_r:
                        for row in grid:
                            for cell in row:
                                cell.update_neighbours(grid)
                        print("\n=== Running A* Search With Obstacle Removal ===")
                        aStarWithRemoval(lambda: draw(win, grid, rows, cols, CELL_SIZE), grid, start, end, obstacles_to_remove)
                    
                    # BFS controls
                    if event.key == pygame.K_b:
                        for row in grid:
                            for cell in row:
                                cell.update_neighbours(grid)
                        print("\n=== Running BFS Without Obstacle Removal ===")
                        bfsWithoutRemoval(lambda: draw(win, grid, rows, cols, CELL_SIZE), grid, start, end)
                    
                    if event.key == pygame.K_v:
                        for row in grid:
                            for cell in row:
                                cell.update_neighbours(grid)
                        print("\n=== Running BFS With Obstacle Removal ===")
                        bfsWithRemoval(lambda: draw(win, grid, rows, cols, CELL_SIZE), grid, start, end, obstacles_to_remove)
                    
                    # Reset grid
                    if event.key == pygame.K_c:
                        grid = create_grid(rows, cols, CELL_SIZE, obstacle_coords)
                        start = grid[0][0]
                        end = grid[rows-1][cols-1]
                        start.make_start()
                        end.make_end()
        
        pygame.quit()

    except FileNotFoundError:
        print("Error: Input file not found")
    except ValueError as e:
        print(f"Error: Invalid input format - {str(e)}")
    except Exception as e:
        print(f"Error: Unexpected error occurred - {str(e)}")

pygame.display.set_caption("Saving the Princess")
main()



