import pygame
import math
import random
from queue import PriorityQueue

# display settings for the pygame window

WIDTH = 800
WIN = pygame.display.set_mode((WIDTH, WIDTH))

# mentioning the color for cells of each state

OBSTACLE_COLOR = (204, 0, 102)
EXPLORED_COLOR = (51, 0, 102)
YELLOW = (255, 255, 0)
BLACK = (0, 0, 0)
PATH_COLOR = (0, 128, 255)
START_COLOR = (155, 255, 255)
GREY = (64, 64, 64)
END_COLOR = (153, 255, 204)


class Cell:
    def __init__(self, row, col, width, total_rows):
        self.row = row
        self.col = col
        self.x = row * width
        self.y = col * width
        self.color = BLACK
        self.neighbours = []
        self.width = width
        self.total_rows = total_rows

    def get_pos(self):
        return self.row, self.col

    def is_closed(self):
        return self.color == EXPLORED_COLOR

    def is_open(self):
        return self.color == BLACK

    def is_barrier(self):
        return self.color == OBSTACLE_COLOR

    def is_start(self):
        return self.color == START_COLOR

    def is_end(self):
        return self.color == END_COLOR

    def is_path(self):
        return self.color == PATH_COLOR

    def reset(self):
        self.color = BLACK

    def make_start(self):
        self.color = START_COLOR

    def make_closed(self):
        self.color = EXPLORED_COLOR

    def make_open(self):
        self.color = BLACK

    def make_barrier(self):
        self.color = OBSTACLE_COLOR

    def make_end(self):
        self.color = END_COLOR

    def make_path(self):
        self.color = PATH_COLOR

    def draw(self, win):
        pygame.draw.rect(win, self.color, (self.x, self.y, self.width, self.width))

    def update_neighbours(self, grid):
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


def heuristic_fn(p1, p2):  # the heuristic function used in this scenario is manhattan distance
    x1, y1 = p1
    x2, y2 = p2
    return abs(x1 - x2) + abs(y1 - y2)


def show_path(came_from, cur, start, draw):  # displays the shortest path if found
    while cur in came_from:
        cur = came_from[cur]
        if cur != start:
            cur.make_path()
        draw()


def search(draw, grid, start, end):  # the search algorithm
    count = 0
    open_set = PriorityQueue()
    ancestor = {}
    g_score = {cell: float("inf") for row in grid for cell in row}
    g_score[start] = 0
    f_score = {cell: float("inf") for row in grid for cell in row}
    f_score[start] = heuristic_fn(start.get_pos(), end.get_pos())
    open_set_hash = {start}
    open_set.put((f_score[start], count, start))  # putting the source cell into the queue
    while not open_set.empty():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
        cur = open_set.get()[2]
        open_set_hash.remove(cur)
        if cur == end:  # if the destination is reached, display the path
            show_path(ancestor, end, start, draw)
            end.make_end()
            return True
        for neighbour in cur.neighbours:  # check the neighbours of the current cell
            if g_score[cur] + 1 < g_score[neighbour]:
                ancestor[neighbour] = cur
                g_score[neighbour] = g_score[cur] + 1
                f_score[neighbour] = g_score[neighbour] + heuristic_fn(neighbour.get_pos(), end.get_pos())
                if neighbour not in open_set_hash:
                    count += 1
                    open_set.put((f_score[neighbour], count, neighbour))
                    open_set_hash.add(neighbour)
                    neighbour.make_open()
        draw()
        if cur != start:
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
    win.fill(BLACK)
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
    rows = 50
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
    pygame.quit()


pygame.display.set_caption("Heuristic Search (A*)")

main(WIN, WIDTH)
