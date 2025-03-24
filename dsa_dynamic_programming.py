#Code implementation for the dynamic programming algorithm of the TSP graph

#adjacency list where the TSP graph is stored
graph_adj_list = {
    1: [(2, 12), (3, 10), (7, 12)],
    2: [(1, 12), (3, 8), (4, 12)],
    3: [(1, 10), (2, 8), (4, 11), (5, 3), (7, 9)],
    4: [(2, 12), (3, 11), (5, 11), (6, 10)],
    5: [(3, 3), (4, 11), (6, 6), (7, 7)],
    6: [(5, 6), (7, 9), (4, 10)],
    7: [(1, 12), (3, 9), (5, 7), (6, 9)]
}

def tsp_dynamic_programming(graph_adj_list):
    cities = list(graph_adj_list.keys())
    n = len(cities)
    
    # memoization table to store minimum cost for each subset
    memo = {}
    
    # function to find the shortest path using dynamic programming and bitmasking
    def tsp(mask, pos):
         # pos represents the current city in the TSP
         # mask is a bitmask (binary number) used to track which cities have been visited
        if mask == (1 << n) - 1:  # all cities visited
            return graph_adj_list.get(pos, {}).get(1, float('infinity'))  # return to start
        
        if (mask, pos) in memo:
            return memo[(mask, pos)]
        
        min_cost = float('infinity')
        for neighbor, cost in graph_adj_list.get(pos, {}).items():
            if not (mask & (1 << (neighbor - 1))):  # if city is not visited
                new_cost = cost + tsp(mask | (1 << (neighbor - 1)), neighbor)
                if new_cost < min_cost:
                    min_cost = new_cost
                    memo[(mask, pos)] = min_cost
        
        return min_cost
    
    # reconstrunt the optimal route
    def find_optimal_route():
        mask = 1  # start from city 1
        pos = 1
        route = [1]
        
        while len(route) < n:
            next_city = None
            min_cost = float('infinity')
            for neighbor, cost in graph_adj_list.get(pos, {}).items():
                if not (mask & (1 << (neighbor - 1))):
                    new_cost = cost + tsp(mask | (1 << (neighbor - 1)), neighbor)
                    if new_cost < min_cost:
                        min_cost = new_cost
                        next_city = neighbor
            
            if next_city is None:
                break
            
            route.append(next_city)
            mask |= (1 << (next_city - 1))
            pos = next_city
        
        route.append(1)  # return to start
        return route
    
    optimal_route = find_optimal_route()
    min_cost = tsp(1, 1)
    
    return optimal_route, min_cost

# to convert adjacency list to dictionary format for lookup
graph_dict = {city: {neighbor: cost for neighbor, cost in neighbors} for city, neighbors in graph_adj_list.items()}
optimal_route, min_cost = tsp_dynamic_programming(graph_dict)

print("Minimum Tour Cost:", min_cost)
print("Optimal Tour:", optimal_route)

