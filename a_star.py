import pygame
import math
import sys
from queue import PriorityQueue

# Initialize pygame
pygame.init()

# Constants
WIDTH = 800
HEIGHT = 800
GRID_SIZE = 50
NODE_WIDTH = WIDTH // GRID_SIZE
NODE_HEIGHT = HEIGHT // GRID_SIZE

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
PURPLE = (128, 0, 128)
ORANGE = (255, 165, 0)
GREY = (128, 128, 128)
TURQUOISE = (64, 224, 208)

# Create display
WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("A* Path Finding Algorithm")

class Node:
    def __init__(self, row, col):
        self.row = row
        self.col = col
        self.x = row * NODE_WIDTH
        self.y = col * NODE_HEIGHT
        self.color = WHITE
        self.neighbors = []
    
    def get_pos(self):
        return self.row, self.col
    
    def is_closed(self):
        return self.color == RED
    
    def is_open(self):
        return self.color == GREEN
    
    def is_barrier(self):
        return self.color == BLACK
    
    def is_start(self):
        return self.color == ORANGE
    
    def is_end(self):
        return self.color == PURPLE
    
    def reset(self):
        self.color = WHITE
    
    def make_closed(self):
        self.color = RED
    
    def make_open(self):
        self.color = GREEN
    
    def make_barrier(self):
        self.color = BLACK
    
    def make_start(self):
        self.color = ORANGE
    
    def make_end(self):
        self.color = PURPLE
    
    def make_path(self):
        self.color = BLUE
    
    def draw(self, win):
        pygame.draw.rect(win, self.color, (self.x, self.y, NODE_WIDTH, NODE_HEIGHT))
    
    def update_neighbors(self, grid):
        self.neighbors = []
        
        # Check all four directions (DOWN, UP, RIGHT, LEFT)
        # DOWN
        if self.row < GRID_SIZE - 1 and not grid[self.row + 1][self.col].is_barrier():
            self.neighbors.append(grid[self.row + 1][self.col])
        
        # UP
        if self.row > 0 and not grid[self.row - 1][self.col].is_barrier():
            self.neighbors.append(grid[self.row - 1][self.col])
        
        # RIGHT
        if self.col < GRID_SIZE - 1 and not grid[self.row][self.col + 1].is_barrier():
            self.neighbors.append(grid[self.row][self.col + 1])
        
        # LEFT
        if self.col > 0 and not grid[self.row][self.col - 1].is_barrier():
            self.neighbors.append(grid[self.row][self.col - 1])
        
        # Optional: Add diagonal movements
        # DOWN-RIGHT
        if self.row < GRID_SIZE - 1 and self.col < GRID_SIZE - 1 and not grid[self.row + 1][self.col + 1].is_barrier():
            if not grid[self.row + 1][self.col].is_barrier() and not grid[self.row][self.col + 1].is_barrier():
                self.neighbors.append(grid[self.row + 1][self.col + 1])
        
        # DOWN-LEFT
        if self.row < GRID_SIZE - 1 and self.col > 0 and not grid[self.row + 1][self.col - 1].is_barrier():
            if not grid[self.row + 1][self.col].is_barrier() and not grid[self.row][self.col - 1].is_barrier():
                self.neighbors.append(grid[self.row + 1][self.col - 1])
        
        # UP-RIGHT
        if self.row > 0 and self.col < GRID_SIZE - 1 and not grid[self.row - 1][self.col + 1].is_barrier():
            if not grid[self.row - 1][self.col].is_barrier() and not grid[self.row][self.col + 1].is_barrier():
                self.neighbors.append(grid[self.row - 1][self.col + 1])
        
        # UP-LEFT
        if self.row > 0 and self.col > 0 and not grid[self.row - 1][self.col - 1].is_barrier():
            if not grid[self.row - 1][self.col].is_barrier() and not grid[self.row][self.col - 1].is_barrier():
                self.neighbors.append(grid[self.row - 1][self.col - 1])
    
    def __lt__(self, other):
        return False

def h(p1, p2):
    """Heuristic function (Manhattan distance)"""
    x1, y1 = p1
    x2, y2 = p2
    return abs(x1 - x2) + abs(y1 - y2)

def reconstruct_path(came_from, current, draw):
    """Reconstruct the path from end to start"""
    while current in came_from:
        current = came_from[current]
        if not current.is_start() and not current.is_end():
            current.make_path()
        draw()

def algorithm(draw, grid, start, end):
    """A* pathfinding algorithm"""
    count = 0
    open_set = PriorityQueue()
    open_set.put((0, count, start))
    came_from = {}
    
    # g_score: cost from start to current node
    g_score = {node: float("inf") for row in grid for node in row}
    g_score[start] = 0
    
    # f_score: g_score + heuristic (estimated cost to end)
    f_score = {node: float("inf") for row in grid for node in row}
    f_score[start] = h(start.get_pos(), end.get_pos())
    
    open_set_hash = {start}
    
    while not open_set.empty():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        
        current = open_set.get()[2]
        open_set_hash.remove(current)
        
        # If we've reached the end
        if current == end:
            reconstruct_path(came_from, end, draw)
            end.make_end()
            start.make_start()
            return True
        
        for neighbor in current.neighbors:
            # Calculate g_score for this neighbor
            # 1.414 (sqrt(2)) for diagonal, 1 for cardinal directions
            if (abs(neighbor.row - current.row) + abs(neighbor.col - current.col)) == 2:
                temp_g_score = g_score[current] + 1.414
            else:
                temp_g_score = g_score[current] + 1
            
            # If we found a better path
            if temp_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = temp_g_score
                f_score[neighbor] = temp_g_score + h(neighbor.get_pos(), end.get_pos())
                
                if neighbor not in open_set_hash:
                    count += 1
                    open_set.put((f_score[neighbor], count, neighbor))
                    open_set_hash.add(neighbor)
                    if not neighbor.is_end():
                        neighbor.make_open()
        
        draw()
        
        if current != start and current != end:
            current.make_closed()
    
    return False

def make_grid():
    """Create the grid of nodes"""
    grid = []
    for i in range(GRID_SIZE):
        grid.append([])
        for j in range(GRID_SIZE):
            node = Node(i, j)
            grid[i].append(node)
    
    return grid

def draw_grid(win):
    """Draw grid lines"""
    for i in range(GRID_SIZE):
        # Horizontal lines
        pygame.draw.line(win, GREY, (0, i * NODE_HEIGHT), (WIDTH, i * NODE_HEIGHT))
        # Vertical lines
        pygame.draw.line(win, GREY, (i * NODE_WIDTH, 0), (i * NODE_WIDTH, HEIGHT))

def draw(win, grid, show_instructions):
    """Draw everything on the screen"""
    win.fill(WHITE)
    
    # Draw all nodes
    for row in grid:
        for node in row:
            node.draw(win)
    
    draw_grid(win)
    
    # Only draw instructions if flag is True
    if show_instructions:
        draw_instructions(win)
    
    pygame.display.update()

def get_clicked_pos(pos):
    """Convert mouse position to grid position"""
    x, y = pos
    row = x // NODE_WIDTH
    col = y // NODE_HEIGHT
    return row, col

def draw_instructions(win):
    """Draw instructions on the screen"""
    font = pygame.font.SysFont('Arial', 20)
    texts = [
        "LEFT CLICK: Draw obstacles/Set start & end points",
        "RIGHT CLICK: Erase",
        "FIRST LEFT CLICK: Set start (orange)",
        "SECOND LEFT CLICK: Set end (purple)",
        "SUBSEQUENT LEFT CLICKS: Draw barriers (black)",
        "SPACE: Start algorithm",
        "C: Clear grid",
        "H: Toggle instructions",
        "ESC: Quit"
    ]
    
    # Create semi-transparent background for instructions
    background = pygame.Surface((WIDTH, 30 * len(texts)))
    background.set_alpha(200)
    background.fill(BLACK)
    win.blit(background, (0, 0))
    
    for i, text in enumerate(texts):
        text_surface = font.render(text, True, WHITE)
        win.blit(text_surface, (10, 5 + 30 * i))

def main():
    """Main function"""
    grid = make_grid()
    
    start = None
    end = None
    
    run = True
    started = False
    show_instructions = True  # Start with instructions visible
    
    clock = pygame.time.Clock()  # Add clock to control frame rate
    
    while run:
        clock.tick(60)  # Limit to 60 frames per second
        
        # Draw the grid (and instructions if flag is True)
        draw(WIN, grid, show_instructions)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            
            # Once algorithm has started, ignore mouse events
            if started:
                continue
            
            # Left mouse button
            if pygame.mouse.get_pressed()[0]:
                pos = pygame.mouse.get_pos()
                row, col = get_clicked_pos(pos)
                try:
                    node = grid[row][col]
                    
                    # First set start, then end, then barriers
                    if not start and node != end:
                        start = node
                        start.make_start()
                    elif not end and node != start:
                        end = node
                        end.make_end()
                    elif node != start and node != end:
                        node.make_barrier()
                except IndexError:
                    pass
            
            # Right mouse button (erase)
            elif pygame.mouse.get_pressed()[2]:
                pos = pygame.mouse.get_pos()
                row, col = get_clicked_pos(pos)
                try:
                    node = grid[row][col]
                    if node == start:
                        start = None
                    elif node == end:
                        end = None
                    node.reset()
                except IndexError:
                    pass
            
            # Key presses
            if event.type == pygame.KEYDOWN:
                # H to toggle instructions
                if event.key == pygame.K_h:
                    show_instructions = not show_instructions
                
                # Space to start algorithm
                if event.key == pygame.K_SPACE and start and end and not started:
                    for row in grid:
                        for node in row:
                            node.update_neighbors(grid)
                    
                    started = True
                    # Create a custom draw function that preserves instruction state
                    algorithm(lambda: draw(WIN, grid, show_instructions), grid, start, end)
                    started = False
                
                # C to clear grid
                if event.key == pygame.K_c:
                    start = None
                    end = None
                    grid = make_grid()
                
                # ESC to quit
                if event.key == pygame.K_ESCAPE:
                    run = False
    
    pygame.quit()

if __name__ == "__main__":
    main()