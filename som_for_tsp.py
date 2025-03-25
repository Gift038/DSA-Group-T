import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# ------------------------------------------------------------------------------
# Utility functions for Euclidean distance and route evaluation
# ------------------------------------------------------------------------------

def euclidean_distance(a, b):
    """
    Return the Euclidean distances between each point in array a and point(s) in array b.
    """
    return np.linalg.norm(a - b, axis=1)

def select_closest(candidates, origin):
    """
    Given an array of candidate neuron positions and an input city (origin),
    return the index of the candidate that is closest (winner neuron).
    """
    return euclidean_distance(candidates, origin).argmin()

def route_distance(cities, route):
    """
    Compute the total route distance.
    'cities' is a DataFrame with columns 'x' and 'y'.
    'route' is an ordered list of indices (cities) representing the tour.
    The route is assumed to be cyclic (returning to the starting city).
    """
    route_cities = cities.reindex(route)
    # Append starting city to complete the cycle
    route_cities = pd.concat([route_cities, route_cities.iloc[[0]]])
    points = route_cities[['x', 'y']].values
    return np.sum(np.linalg.norm(points[1:] - points[:-1], axis=1))

# ------------------------------------------------------------------------------
# Plotting functions
# ------------------------------------------------------------------------------

def plot_network(cities, neurons, name='diagram.png', ax=None):
    """
    Plot the neuron network and the positions of cities.
    Saves the figure to file 'name' if ax is not provided.
    """
    mpl.rcParams['agg.path.chunksize'] = 10000

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.set_aspect('equal', adjustable='datalim')
        ax.axis('off')
        ax.scatter(cities['x'], cities['y'], color='red', s=40, label='Cities')
        for idx, row in cities.iterrows():  #Labelling city dots 
            ax.text(row['x'], row['y'], str(idx), fontsize=12, color='black', ha='right', va='bottom')
        ax.plot(neurons[:, 0], neurons[:, 1], '-', color='#0063ba', markersize=2, label='Neuron network')
        ax.legend()
        plt.savefig(name, bbox_inches='tight', pad_inches=0, dpi=200)
        plt.close()
    else:
        ax.scatter(cities['x'], cities['y'], color='red', s=40)
        ax.plot(neurons[:, 0], neurons[:, 1], '-', color='#0063ba', markersize=2)
        return ax

def plot_route(cities, route, name='route.png', ax=None):
    """
    Plot the final TSP route obtained by reordering the cities based on the SOM.
    """
    mpl.rcParams['agg.path.chunksize'] = 10000

    # Reorder cities by the route and complete the cycle
    route_cities = cities.reindex(route)
    route_cities = pd.concat([route_cities, route_cities.iloc[[0]]])
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.set_aspect('equal', adjustable='datalim')
        ax.axis('off')
        ax.scatter(cities['x'], cities['y'], color='red', s=40, label='Cities')
        for idx, row in cities.iterrows():  #Labelling city dots 
            ax.text(row['x'], row['y'], str(idx), fontsize=12, color='black', ha='right', va='bottom')

        ax.plot(route_cities['x'], route_cities['y'], color='purple', linewidth=2, label='TSP Route')
        ax.legend()
        plt.savefig(name, bbox_inches='tight', pad_inches=0, dpi=200)
        plt.close()
    else:
        ax.scatter(cities['x'], cities['y'], color='red', s=40)
        ax.plot(route_cities['x'], route_cities['y'], color='purple', linewidth=2)
        return ax

# ------------------------------------------------------------------------------
# SOM (Self-Organizing Map) for TSP
# ------------------------------------------------------------------------------

def generate_network(size):
    """
    Generate a network of neurons.
    Each neuron is represented as a 2D point in [0, 1] x [0, 1].
    """
    return np.random.rand(size, 2)

def get_neighborhood(center, radius, domain_size):
    """
    Compute a Gaussian neighborhood function for the given 'center' neuron index.
    Returns an array of influence values for all neurons in the network.
    """
    if radius < 1:
        radius = 1
    indices = np.arange(domain_size)
    deltas = np.abs(center - indices)
    distances = np.minimum(deltas, domain_size - deltas)
    return np.exp(- (distances ** 2) / (2 * (radius ** 2)))

def get_route(cities, network):
    """
    Assign each city to its closest neuron and then order the cities by the neuron indices.
    This produces an approximate TSP route.
    """
    # Convert each row to a numpy array explicitly
    winners = cities[['x', 'y']].apply(lambda c: select_closest(network, np.array(c)), axis=1)
    cities = cities.copy()
    cities['winner'] = winners
    return cities.sort_values('winner').index

def som(cities, iterations, initial_learning_rate=0.8, initial_radius_factor=0.1):
    """
    Train a Self-Organizing Map (SOM) to solve the TSP.
    Returns the ordered route (list of city indices) corresponding to the TSP tour.
    """
    cities_norm = cities.copy()
    # Determine number of neurons (heuristic: 8 times the number of cities)
    num_neurons = cities.shape[0] * 8
    network = generate_network(num_neurons)
    print(f"Network of {num_neurons} neurons created. Starting iterations:")

    learning_rate = initial_learning_rate
    radius = num_neurons * initial_radius_factor

    for i in range(iterations):
        if i % 100 == 0:
            print(f"Iteration {i}/{iterations}", end="\r")
        city = cities_norm.sample(1)[['x', 'y']].values[0]
        winner_idx = select_closest(network, city)
        gaussian = get_neighborhood(winner_idx, radius, network.shape[0])
        network += gaussian[:, np.newaxis] * learning_rate * (city - network)
        learning_rate *= 0.99997
        radius *= 0.9997

        # Save intermediate network plots for visualization
        if i % 1000 == 0:
            plot_network(cities, network, name=f'diagrams/network_{i:05d}.png')

        if radius < 1 or learning_rate < 0.001:
            print(f"Parameters decayed sufficiently at iteration {i}.")
            break

    print(f"\nCompleted {i+1} iterations.")
    plot_network(cities, network, name='diagrams/network_final.png')
    route = get_route(cities, network)
    plot_route(cities, route, name='diagrams/route_final.png')
    return route

# ------------------------------------------------------------------------------
# Data setup for the TSP (7 cities)
# ------------------------------------------------------------------------------
def load_cities():
    """
    Create a DataFrame for the 7-city TSP instance.
    The coordinates are normalized values based on the graph in DSAPracticalAssignment.pdf.
    """
    data = {
        'x': [0.25, 0.4, 0.4, 0.6, 0.5, 0.7, 0.33],
        'y': [0.35, 0.67, 0.45, 0.58, 0.3, 0.1, 0.21]
    }
    cities = pd.DataFrame(data, index=[1, 2, 3, 4, 5, 6, 7])
    return cities

def actual_route_distance(cities, route, scaling_factor=36.36363636):
    """
    Compute the actual total route distance based on normalized city coordinates
    and a scaling factor that converts normalized distance to actual distance.
    
    I munually calculated the scaling_factor by dividing actual distance on the graph by the 
    the Euclidean distance between city 2 and 3.
"""
    #total normalized distance
    normalized_distance = route_distance(cities, route)
    
    # Convert the normalized distance to actual distance
    actual_distance = normalized_distance * scaling_factor
    return actual_distance
# ------------------------------------------------------------------------------
# Main function to run the SOM-based TSP solver
# ------------------------------------------------------------------------------
def main_som():
    """
    Main function to execute the SOM approach for the TSP.
    Loads the TSP data, ensures necessary directories exist, trains the SOM,
    computes the final route and its total distance, and generates plots.
    """
    # Create the directory for saving diagrams if it doesn't exist
    os.makedirs('diagrams', exist_ok=True)
    
    cities = load_cities()
    route = som(cities, iterations=3000)
    total_distance = route_distance(cities, route)
    actual_total_distance = actual_route_distance(cities, route)
    print("\nFinal TSP route (city order):", list(route))
    print("Approximate normalized total route distance:", total_distance)
    print("Total Distance:", actual_total_distance)
    print("\nCheck the output folder 'DIAGRAMS' for the final image route.")

if __name__ == "__main__":
    main_som()
