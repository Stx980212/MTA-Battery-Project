#!/Users/shentianxiao/opt/anaconda3/bin/python

from googlemaps import Client
from tqdm import tqdm
import pandas as pd
import sys

GOOGLE_MAPS_API_KEY = "AIzaSyA69HVKvhpNowVcaCLcw4kN-kIoiaxDCno"
STOPS_BY_ROUTE_PATH = "../../data/stops_by_route.csv"

def get_stops(route):
    stops_by_route_df = pd.read_csv(STOPS_BY_ROUTE_PATH)
    stops_by_route_df = stops_by_route_df.set_index(["Route", "Direction"])
    desired_stops = stops_by_route_df.loc[route]
    direction1, direction2 = desired_stops.iloc[0].dropna().str.upper(), desired_stops.iloc[1].dropna().str.upper()
    return direction1, direction2

def get_graph_table(direction1, direction2, depot):
    all_stops = list(set(direction1.tolist() + direction2.tolist())) + [depot]
    graph = pd.DataFrame(columns=all_stops, index=all_stops)
    return graph

def get_distance(origin, destination, gmaps):
    distance = gmaps.distance_matrix(origin, destination)["rows"][0]["elements"][0]["distance"]["value"]
    return distance/1000 # Km

def populate_direction(direction, graph, gmaps):
    for i in tqdm(range(direction.size - 1), desc=direction.name): # Iterate from first stop to second-to-last stop
        origin, destination = direction.iloc[i], direction.iloc[i + 1]
        distance = get_distance(origin + ", New York, NY", destination + ", New York, NY", gmaps)
        graph.loc[origin, destination] = str(distance) + " " + direction.name

def complete_route_cycle(direction1, direction2, graph, gmaps):
    origin, destination = direction2.iloc[-1], direction1.iloc[0]
    if origin != destination:
        distance = get_distance(origin + ", New York, NY", destination + ", New York, NY", gmaps)
        graph.loc[origin, destination] = str(distance) + " " + direction1.name
    origin, destination = direction1.iloc[-1], direction2.iloc[0]
    if origin != destination:
        distance = get_distance(origin + ", New York, NY", destination + ", New York, NY", gmaps)
        graph.loc[origin, destination] = str(distance) + " " + direction1.name

def fill_in_depot_connection(direction1, depot, route, graph, gmaps):
    min_dist = float("inf")
    min_total_dist = float("inf")
    stop_name = None
    stops = list(set(direction1.tolist()))
    depot_addresses = pd.read_csv("../../data/depot_data/depots.csv").set_index("Depot")
    depot_address = depot_addresses.loc[' ' + depot]["Address"]
    for i in tqdm(range(len(stops)), desc="calculating optimal stop to depot"):
        origin = stops[i]
        distance = get_distance(origin + ", New York, NY", depot_address, gmaps)
        total_distance = distance + get_distance(depot_address, origin + ", New York, NY", gmaps)
        if total_distance < min_total_dist:
            min_total_dist = total_distance
            min_dist = distance
            stop_name = origin
    graph.loc[stop_name, depot] = str(min_dist) + " to depot of " + route
    graph.loc[depot, stop_name] = str(min_total_dist - min_dist) + " " + direction1.name

def main():
    bus_route, depot = input().split(",") # expects a <route>,<depot_name> pair. Must match spelling in the correct .csv in data/
    print(f"Processing {bus_route} route", file=sys.stderr)
    gmaps = Client(GOOGLE_MAPS_API_KEY)
    direction1, direction2 = get_stops(bus_route)
    graph = get_graph_table(direction1, direction2, depot)
    populate_direction(direction1, graph, gmaps)
    populate_direction(direction2, graph, gmaps)
    complete_route_cycle(direction1, direction2, graph, gmaps)
    fill_in_depot_connection(direction1, depot, bus_route, graph, gmaps)
    graph.to_csv(sys.stdout)

if __name__ == "__main__":
    main()
