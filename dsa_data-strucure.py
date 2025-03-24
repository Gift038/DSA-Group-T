# Adjacency list (data structure) representation of the graph
# python dictionary with key value pairs where;
# keys represent the cities and,
# the values are lists of tuples representing (city to which the key(city) traverses, distance between the 2 cities)
graph_adj_list = {
    1: [(2, 12), (3, 10), (7, 12)],  # (2,12) represents (city 2, distance between cities 1 and 2)
    2: [(1, 12), (3, 8), (4, 12)],
    3: [(1, 10), (2, 8), (4, 11), (5, 3), (7, 9)],
    4: [(2, 12), (3, 11), (5, 11), (6, 10)],
    5: [(3, 3), (4, 11), (6, 6), (7, 7)],
    6: [(5, 6), (7, 9), (4, 10)],
    7: [(1, 12), (3, 9), (5, 7), (6, 9)]
}

# iterate through all the cities(nodes) and access each cities' neighbors and their distances
for city, neighbors in graph_adj_list.items():
    print(f"Neighbors of city {city}: {neighbors}")
    for neighbor, distance in neighbors:
        print(f"From CITY {city} to city {neighbor}, the distance is {distance}.")

