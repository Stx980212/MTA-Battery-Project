#!/usr/bin/sh

while IFS="$\n" read -r line
do
    route="${line%%,*}"
    echo "$line" | ./bus_stop_distance_scrapper.py > "../../data/routes/$route""_stop_graph.csv"
done < "../../data/routes/Supplement.txt"

