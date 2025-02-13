# Credit for this: Nicholas Swift
# as found at https://medium.com/@nicholas.w.swift/easy-a-star-pathfinding-7e6689c7f7b2
from warnings import warn
# import heapq
from heapdict import heapdict
import math

class Node:
    """
    A node class for A* Pathfinding
    """

    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position

        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.position == other.position
    
    def __repr__(self):
      return f"{self.position}"
    
    def __hash__(self) -> int:
       return hash(self.position)

    # defining less than for purposes of heap queue
    def __lt__(self, other):
      return self.f < other.f
    
    # defining greater than for purposes of heap queue
    def __gt__(self, other):
      return self.f > other.f

def return_path(current_node):
    path = []
    current = current_node
    while current is not None:
        path.append(current.position)
        current = current.parent
    return path[::-1]  # Return reversed path


def elevation_star(maze, start, end, allow_diagonal_movement = False):
    """
    Returns a list of tuples as a path from the given start to the given end in the given maze
    :param maze:
    :param start:
    :param end:
    :return:
    """

    # Create start and end node
    start_node = Node(None, start)
    start_node.g = start_node.h = start_node.f = 0
    end_node = Node(None, end)
    end_node.g = end_node.h = end_node.f = 0

    # Initialize both open and closed list
    open_list = heapdict()
    closed_list = set()

    open_list[start_node] = 0 # = start_node.f

    # Adding a stop condition
    outer_iterations = 0
    # max_iterations = (len(maze[0]) * len(maze) // 2)
    max_iterations = (len(maze[0]) * len(maze)) * 4

    # what squares do we search
    adjacent_squares = ((0, -1), (0, 1), (-1, 0), (1, 0),)
    if allow_diagonal_movement:
        adjacent_squares = ((0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1),)

    # Loop until you find the end
    while len(open_list) > 0:
        outer_iterations += 1

        if outer_iterations > max_iterations:
          # if we hit this point return the path such as it is
          # it will not contain the destination
          warn("giving up on pathfinding too many iterations")
          return return_path(current_node), None       
        
        # Get the current node
        current_node = open_list.popitem()[0]
        closed_list.add(current_node)

        # Found the goal
        if current_node == end_node:
            return return_path(current_node), current_node.g

        # Generate children
        children = []
        
        for new_position in adjacent_squares: # Adjacent squares

            # Get node position
            node_position = (current_node.position[0] + new_position[0], current_node.position[1] + new_position[1])

            # Make sure within range
            if node_position[0] > (len(maze) - 1) or node_position[0] < 0 or node_position[1] > (len(maze[len(maze)-1]) -1) or node_position[1] < 0:
                continue

            # Make sure walkable terrain
            if maze[node_position[0]][node_position[1]] < 0:
                continue

            # Create new node
            new_node = Node(current_node, node_position)

            # Append
            children.append(new_node)
        
        
        # Loop through children
        for child in children:
            new_g = current_node.g + 4
            if maze[child.position[0]][child.position[1]] > maze[current_node.position[0]][current_node.position[1]]:
              new_g += 2
            elif maze[child.position[0]][child.position[1]] < maze[current_node.position[0]][current_node.position[1]]:
              new_g -= 1
            if child in open_list:
                continue
            if child in closed_list:
                continue
            
            # Create the f, g, and h values
            child.g = new_g
            child.h = ((child.position[0] - end_node.position[0]) ** 2) + ((child.position[1] - end_node.position[1]) ** 2)
            child.f = child.g*10 + math.sqrt(child.h)

            # Add the child to the open list
            open_list[child] = child.f

    warn("Couldn't get a path to destination")
    return None, None




def print_path(maze, path, print_maze=True):

    if print_maze:
      for step in path:
        maze[step[0]][step[1]] = 256
      
      for row in maze:
        line = []
        for col in row:
          if col <= 0:
            line.append(" \u2588")
          elif col == 256:
            line.append(" .")
          else:
            line.append(" " + str(col))
          
        print("".join(line))

    print(path)