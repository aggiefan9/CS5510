#import astar_fixed
import astar
import time

def generate_maze():
    maze = [[0 for j in range(36)] for i in range(36)]
    for i in range(35):
        maze[i][0] = 1
        maze[35][i] = 1
    for i in range(36):
        maze[i][35] = 1
        maze[0][i] = 1
    for i in range(0, 25):
        maze[15][i] = 1
    for i in range(20):
        maze[25][35-i] = 1
    return maze

s_time = time.time()
maze1 = generate_maze()
path = astar.astar(maze=maze1, start=(10, 10), end=(30, 30), allow_diagonal_movement=True)
t_time = time.time() - s_time
print("Cost: ", len(path))
print("Time: ", t_time)