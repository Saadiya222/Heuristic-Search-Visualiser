import pygame
import time 
import psutil
import os
from queue import PriorityQueue # For A*
from collections import deque # For BFS



# display settings for the pygame window
def create_window(rows, cols):
    CELL_SIZE = 40  # Fixed size for each cell
    width = (cols * CELL_SIZE) 
    height = rows * CELL_SIZE
    INFO_HEIGHT = 100
    WINDOW_HEIGHT = height + INFO_HEIGHT
    WINDOW = pygame.display.set_mode((width, WINDOW_HEIGHT))
    return WINDOW 


# mentioning the color for cells of each state
current_algorithm = ""
current_removed_obstacles = None
current_path_cost = None
INFO_HEIGHT = 100
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
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

def draw_info_panel(win, rows, cols, cell_size, y_offset=0):
    # Calculate font size based on window width
    font_size = min(20, int(cols * cell_size / 30))
    font = pygame.font.SysFont('arial', font_size)
    
    # Calculate dynamic spacing
    line_height = font_size + 5
    x_margin = int(cols * cell_size * 0.02)  # 2% of window width
    
    # Background
    pygame.draw.rect(win, WHITE, (0, 0, cols * cell_size, y_offset))
    pygame.draw.line(win, GREY, (0, y_offset), (cols * cell_size, y_offset), 2)
    
    # Dynamic text positioning
    if current_algorithm:
        title = font.render(f"Algorithm: {current_algorithm}", True, BLACK)
        win.blit(title, (x_margin, line_height))
    
    if current_removed_obstacles:
        obstacle_text = font.render(f"Removed obstacles at: {current_removed_obstacles}", True, BLACK)
        win.blit(obstacle_text, (x_margin, 2 * line_height))
    
    if current_path_cost is not None:
        path_found = font.render("*** PATH FOUND! ***", True, (0, 128, 0))
        win.blit(path_found, (x_margin, 3 * line_height))
        cost_text = font.render(f"Path cost: {current_path_cost}", True, BLACK)
        win.blit(cost_text, (x_margin, 4 * line_height))




def reconstruct_removal_path(ancestor, current_state, start, draw, removed_obstacles):
    path = []
    path_cells = set()
    removed_list = []
    
    current = current_state
    while current in ancestor:
        cell, removals = current
        path_cells.add(cell)
        
        if current in removed_obstacles:
            removed_list.append(removed_obstacles[current])
        
        path.append((cell.row, cell.col))
        if cell != start:
            cell.make_path()
        
        current = ancestor[current]
        draw()
    
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

def find_removable_obstacles(grid, start, end, max_removals):
    queue = deque([(start, 0, [])])  # (cell, removals_used, removed_obstacles)
    visited = {(start, 0)}
    
    while queue:
        current, removals_used, removed = queue.popleft()
        
        if current == end:
            return removed
            
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            new_row = current.row + dx
            new_col = current.col + dy
            
            if 0 <= new_row < len(grid) and 0 <= new_col < len(grid[0]):
                neighbor = grid[new_row][new_col]
                if neighbor.is_barrier() and removals_used < max_removals:
                    new_state = (neighbor, removals_used + 1)
                    if new_state not in visited:
                        visited.add(new_state)
                        queue.append((neighbor, removals_used + 1, removed + [(new_row, new_col)]))
                elif not neighbor.is_barrier():
                    new_state = (neighbor, removals_used)
                    if new_state not in visited:
                        visited.add(new_state)
                        queue.append((neighbor, removals_used, removed))
    return []


def bfsWithoutRemoval(draw, grid, start, end):
    global current_algorithm, current_removed_obstacles, current_path_cost
    current_algorithm = "BFS"  
    current_removed_obstacles = None
    current_path_cost = None
    start_time = time.time()
    queue = deque([start])
    visited = {start}
    ancestor = {}
    max_queue_size = 1
    
    while queue:
        max_queue_size = max(max_queue_size, len(queue))
        current = queue.popleft()

        if current != start and current != end:
            current.make_start()  # Shows knight's current position in gold
            draw()
            time.sleep(0.2)  # Optional: adds a slight pause to make movement more visible
            current.make_closed()  # Then marks it as explored

        if current == end:
            end_time = time.time()
            path = get_path_coordinates(ancestor, end, start)
            current_path_cost = len(path) - 1
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
    global current_algorithm, current_removed_obstacles, current_path_cost
    current_algorithm = "BFS with removal"
    current_removed_obstacles = None
    current_path_cost = None

    obstacles_to_remove = find_removable_obstacles(grid, start, end, max_removals)
    
    # Pre-process grid and update neighbors
    for row, col in obstacles_to_remove:
        grid[row][col].make_open()
        # Update neighbors for the cell and its adjacent cells
        grid[row][col].update_neighbours(grid)
        # Update neighbors for adjacent cells
        for dx, dy in [(0,1), (0,-1), (1,0), (-1,0)]:
            new_row, new_col = row + dx, col + dy
            if 0 <= new_row < len(grid) and 0 <= new_col < len(grid[0]):
                grid[new_row][new_col].update_neighbours(grid)

    start_time = time.time()
    queue = deque([start])
    visited = {start}
    ancestor = {}
    max_queue_size = 1
    explored_cells = set()
    path_cells = set()  # Initialize path_cells set
    
    while queue:
        max_queue_size = max(max_queue_size, len(queue))
        current = queue.popleft()
        path_cells.add(current)  # Add current cell to path_cells

        if current != start and current != end:
            current.make_start()
            draw()
            time.sleep(0.2)
            current.make_closed()

        explored_cells.add(current)
        if current != start and current != end:
            current.make_closed()
            
        if current == end:
            end_time = time.time()
            path = get_path_coordinates(ancestor, end, start)
            current_path_cost = len(path) - 1            
            current_removed_obstacles = len(obstacles_to_remove)
            current_removed_obstacles = obstacles_to_remove  # Show the actual coordinates
            print("\n=== Search Results (BFS with removal) ===")
            print(f" *** Path found! *** ")
            print(f"Shortest path: {path}")
            print(f"Number of obstacles removed: {len(obstacles_to_remove)}")
            print(f"Obstacles removed at: {obstacles_to_remove}")
            
            unique_visited = set((node.row, node.col) for node in visited)
            print(f"Number of nodes visited: {len(unique_visited)}")
            print(f"Visited nodes: {sorted(list(unique_visited))}")
            
            show_path(ancestor, end, start, draw)
            end.make_end()
            calculate_performance_metrics(start_time, end_time, visited, max_queue_size, search_type="BFS with removal")
            return True
            
        for neighbor in current.neighbours:
            if neighbor not in visited:
                queue.append(neighbor)
                visited.add(neighbor)
                ancestor[neighbor] = current
                if neighbor not in explored_cells and neighbor != end:
                    neighbor.make_open()
        draw()

    return False






def aStarWithoutRemoval(draw, grid, start, end):
    global current_algorithm, current_removed_obstacles, current_path_cost
    current_algorithm = "A*" 
    current_removed_obstacles = None
    current_path_cost = None
    start_time = time.time()
    count = 0
    open_set = PriorityQueue()
    open_set.put((0, count, start))
    ancestor = {}
    g_score = {cell: float("inf") for row in grid for cell in row}
    g_score[start] = 0
    f_score = {cell: float("inf") for row in grid for cell in row}
    f_score[start] = heuristic_fn(start.get_pos(), end.get_pos())
    
    open_set_hash = {start}
    visited_nodes = set()
    max_frontier_size = 0
    
    while not open_set.empty():
        max_frontier_size = max(max_frontier_size, len(open_set_hash))
        current = open_set.get()[2]
        open_set_hash.remove(current)
        visited_nodes.add((current.row, current.col))
        
        # Show knight's movement
        if current != end:
            current.make_start()
            draw()
            time.sleep(0.2)
            if current != start:
                current.make_closed()
        
        if current == end:
            end_time = time.time()
            path = get_path_coordinates(ancestor, end, start)
            current_path_cost = len(path) - 1
            print("\n=== Search Results (A*) ===")
            print(f" *** Path found! *** ")
            print(f"Shortest path: {path}")
            print(f"Number of nodes visited: {len(visited_nodes)}")
            print(f"Visited nodes: {sorted(list(visited_nodes))}")
            
            show_path(ancestor, end, start, draw)
            end.make_end()
            calculate_performance_metrics(start_time, end_time, visited_nodes, max_frontier_size, search_type="A*")
            return True
            
        for neighbor in current.neighbours:
            temp_g_score = g_score[current] + 1
            
            if temp_g_score < g_score[neighbor]:
                ancestor[neighbor] = current
                g_score[neighbor] = temp_g_score
                f_score[neighbor] = temp_g_score + heuristic_fn(neighbor.get_pos(), end.get_pos())
                if neighbor not in open_set_hash:
                    count += 1
                    open_set.put((f_score[neighbor], count, neighbor))
                    open_set_hash.add(neighbor)
                    neighbor.make_open()
        
        draw()
    
    return False

    


def aStarWithRemoval(draw, grid, start, end, max_removals):
    global current_algorithm, current_removed_obstacles, current_path_cost
    current_algorithm = "A* with removal"
    current_removed_obstacles = None
    current_path_cost = None

    obstacles_to_remove = find_removable_obstacles(grid, start, end, max_removals)
    
    # Pre-process grid and update neighbors
    for row, col in obstacles_to_remove:
        grid[row][col].make_open()
        # Update neighbors for the cell and its adjacent cells
        grid[row][col].update_neighbours(grid)
        # Update neighbors for adjacent cells
        for dx, dy in [(0,1), (0,-1), (1,0), (-1,0)]:
            new_row, new_col = row + dx, col + dy
            if 0 <= new_row < len(grid) and 0 <= new_col < len(grid[0]):
                grid[new_row][new_col].update_neighbours(grid)
    
    # Regular A* visualization
    start_time = time.time()
    path_cells = set()  # Add it here
    count = 0
    open_set = PriorityQueue()
    open_set.put((0, count, start))
    ancestor = {}
    g_score = {cell: float("inf") for row in grid for cell in row}
    g_score[start] = 0
    f_score = {cell: float("inf") for row in grid for cell in row}
    f_score[start] = heuristic_fn(start.get_pos(), end.get_pos())
    
    open_set_hash = {start}
    visited_nodes = set()
    max_frontier_size = 0
    
    while not open_set.empty():
        max_frontier_size = max(max_frontier_size, len(open_set_hash))
        current = open_set.get()[2]
        path_cells.add(current)  # Add this line
        open_set_hash.remove(current)
        visited_nodes.add((current.row, current.col))
        
        # Show knight's movement
        if current != end:
            current.make_start()
            draw()
            time.sleep(0.2)
            if current != start:
                current.make_closed()
        
        if current == end:
            end_time = time.time()
            path = get_path_coordinates(ancestor, end, start)
            current_path_cost = len(path) - 1
            current_removed_obstacles = obstacles_to_remove
            print("\n=== Search Results (A* with removal) ===")
            print(f" *** Path found! *** ")
            print(f"Shortest path: {path}")
            print(f"Number of obstacles removed: {len(obstacles_to_remove)}")
            print(f"Obstacles removed at: {obstacles_to_remove}")
            print(f"Number of nodes visited: {len(visited_nodes)}")
            print(f"Visited nodes: {sorted(list(visited_nodes))}")
            
            show_path(ancestor, end, start, draw)
            end.make_end()
            calculate_performance_metrics(start_time, end_time, visited_nodes, max_frontier_size, search_type="A* with removal")
            return True
            
        for neighbor in current.neighbours:
            temp_g_score = g_score[current] + 1
            
            if temp_g_score < g_score[neighbor]:
                ancestor[neighbor] = current
                g_score[neighbor] = temp_g_score
                f_score[neighbor] = temp_g_score + heuristic_fn(neighbor.get_pos(), end.get_pos())
                if neighbor not in open_set_hash:
                    count += 1
                    open_set.put((f_score[neighbor], count, neighbor))
                    open_set_hash.add(neighbor)
                    neighbor.make_open()
        
        draw()
    
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
    
    # Draw info panel
    draw_info_panel(win, rows, cols, width, INFO_HEIGHT)
    
    # Draw grid with offset
    for row in grid:
        for cell in row:
            # Modify cell's y position to account for info panel
            original_y = cell.y
            cell.y += INFO_HEIGHT
            cell.draw(win)
            cell.y = original_y
            
    # Draw grid lines with offset
    for i in range(rows + 1):
        pygame.draw.line(win, GREY, (0, i * width + INFO_HEIGHT), (cols * width, i * width + INFO_HEIGHT))
    for j in range(cols + 1):
        pygame.draw.line(win, GREY, (j * width, INFO_HEIGHT), (j * width, rows * width + INFO_HEIGHT))
    
    pygame.display.update()
    time.sleep(0.1)


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
    time_taken = end_time - start_time
    process = psutil.Process(os.getpid())
    memory_used = process.memory_info().rss / 1024 / 1024
    
    # Determine time complexity based on algorithm type
    if "A*" in search_type:
        time_complexity = "O(|V| log |V|)"
    else:  # BFS
        time_complexity = "O(|V| + |E|)"
    
    print(f"\n=== Performance Metrics ({search_type}) ===")
    print(f"Time taken: {time_taken:.4f} seconds")
    print(f"Memory used: {memory_used:.2f} MB")
    print(f"Space complexity: O(|V|) where |V| = {len(visited_nodes)} nodes explored")
    print(f"Maximum frontier size: {open_set_size} nodes")
    print(f"Time complexity: {time_complexity} where |V| = number of vertices, |E| = number of edges")
    

def main():
    CELL_SIZE = 40
    pygame.init()
    pygame.font.init()
    
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



