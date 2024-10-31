import pygame
import math
import random
from queue import PriorityQueue

# credits - Tech With Tim

# display settings for the pygame window

WIDTH = 800 # sets the window width to 800 pixels
WIN = pygame.display.set_mode((WIDTH, WIDTH)) # creates a Pygame window where the grid and pathfinding visualization will be displayed

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
    def __init__(self, row, col, width, total_rows): # Initializes each cell with attributes like its position, color, and list of neighboring cells
        self.row = row
        self.col = col
        self.x = row * width
        self.y = col * width
        self.color = CREAM
        self.neighbours = []
        self.width = width
        self.total_rows = total_rows

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
        if self.row < self.total_rows - 1 and not grid[self.row + 1][self.col].is_barrier():  # DOWN
            self.neighbours.append(grid[self.row + 1][self.col])
        if self.row > 0 and not grid[self.row - 1][self.col].is_barrier():  # UP
            self.neighbours.append(grid[self.row - 1][self.col])
        if self.col < self.total_rows - 1 and not grid[self.row][self.col + 1].is_barrier():  # RIGHT
            self.neighbours.append(grid[self.row][self.col + 1])
        if self.col > 0 and not grid[self.row][self.col - 1].is_barrier():  # LEFT
            self.neighbours.append(grid[self.row][self.col - 1])

    def __lt__(self, other):
        return False


def heuristic_fn(p1, p2):  # the heuristic function used in our scenario is manhattan distance (sum of the absolute differences between the x and y coordinates of two points)
    x1, y1 = p1
    x2, y2 = p2
    return abs(x1 - x2) + abs(y1 - y2)


def show_path(came_from, cur, start, draw):  # backtracks through the cells to display the found path after the end cell is reached and displays the shortest path if found
    while cur in came_from:
        cur = came_from[cur]
        if cur != start:
            cur.make_path()
        draw()


def search(draw, grid, start, end):  # the search algorithm
    count = 0 # so that ties are broken by insertion order in the Pq
    open_set = PriorityQueue() # Frontier | lower-priority (better f scorre) cells being dequeued first
    ancestor = {} # to reconstruct the path once the end cell is reached
    g_score = {cell: float("inf") for row in grid for cell in row} # Initially, all cells are set to infinity except for the g(start)=0, bc wel'll check in line 135
    g_score[start] = 0
    f_score = {cell: float("inf") for row in grid for cell in row} # Initially, all cells are set to infinity
    f_score[start] = heuristic_fn(start.get_pos(), end.get_pos()) + g_score[start]
    open_set_hash = {start} # A set that mirrors open_set, stores the cells currently in the queue to allow for efficient membership checks
    open_set.put((f_score[start], count, start))  # putting the source (start) cell into the Pqueue
    while not open_set.empty():
        for event in pygame.event.get():
            if event.type == pygame.QUIT: # if user quits window
                pygame.quit()
        cur = open_set.get()[2] # Retrieves the cell with the lowest f_score from open_set
        open_set_hash.remove(cur) # because we are now exploring it
        if cur == end:  # if the destination is reached, display the path
            show_path(ancestor, end, start, draw)
            end.make_end()
            return True # path was found
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
    return False


def create_grid(rows, width):
    grid = []
    gap = width // rows
    for i in range(rows):
        grid.append([])
        for j in range(rows):
            cell = Cell(i, j, gap, rows)
            r = random.random()
            if r >= 0.7:  # each cell has a 30% chance of becoming an obstacle!
                cell.make_barrier()
            grid[i].append(cell)
    return grid


def draw_grid(win, rows, width):
    gap = width // rows
    for i in range(rows):
        pygame.draw.line(win, GREY, (0, i * gap), (width, i * gap))
        for j in range(rows):
            pygame.draw.line(win, GREY, (j * gap, 0), (j * gap, width))


def draw(win, grid, rows, width):
    win.fill(CREAM)
    for row in grid:
        for cell in row:
            cell.draw(win)
    draw_grid(win, rows, width)
    pygame.display.update()


def get_clicked_pos(pos, rows, width):
    gap = width // rows
    y, x = pos
    row = y // gap
    col = x // gap
    return row, col


def main(win, width):
    rows = 20
    #cols 30
    grid = create_grid(rows, width)
    start = None
    end = None
    run = True
    started = False
    while run:
        draw(win, grid, rows, width)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            if pygame.mouse.get_pressed()[0]:  # LEFT
                pos = pygame.mouse.get_pos()
                row, col = get_clicked_pos(pos, rows, width)
                cell = grid[row][col]
                if not start and cell != end:
                    start = cell
                    cell.make_start()
                elif not end and cell != start:
                    end = cell
                    cell.make_end()
                elif cell != end and cell != start:
                    cell.make_barrier()
            elif pygame.mouse.get_pressed()[2]:  # RIGHT
                pos = pygame.mouse.get_pos()
                row, col = get_clicked_pos(pos, rows, width)
                cell = grid[row][col]
                cell.reset()
                if cell == start:
                    start = None
                elif cell == end:
                    end = None
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and start and end:
                    for row in grid:
                        for cell in row:
                            if cell.is_closed() or cell.is_path():
                                cell.make_open()
                            cell.update_neighbours(grid)
                    search(lambda: draw(win, grid, rows, width), grid, start, end)
                if event.key == pygame.K_c:
                    start = None
                    end = None
                    grid = create_grid(rows, width)
                    #grid = create_grid(rows,cols, width)
    pygame.quit()


pygame.display.set_caption("Heuristic Search (A*)")

main(WIN, WIDTH)
