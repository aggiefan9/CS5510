from astar_elevation import elevation_star, print_path

maze = ((4, 3, 2, 2, 2, 2, 1, 1, 1, 1),
        (3, 2, 2, -1, 2, 1, 1, 1, 2, 1),
        (3, 2, 2, -1, 2, 1, 4, 1, 2, 1),
        (2, 2, 1, -1, 4, 2, 4, 4, 3, 1),
        (2, 1, 1, -1, 4, 4, 4, 4, 2, 1),
        (1, 1, 1, -1, 3, 4, 3, 2, 2, 1),
        (1, 2, 2, 2, 2, 3, 1, 1, 1, 1),
        (1, 1, 1, 1, 1, 1, 1, 1, 2, 2))

start = (7, 0)
end = (2, 6)

path, cost = elevation_star(maze, start, end, allow_diagonal_movement=True)
modifiable = [*maze,]
for i in range(len(modifiable)):
    modifiable[i] = [*modifiable[i],]
print_path(modifiable, path)
print("Cost = ", cost)