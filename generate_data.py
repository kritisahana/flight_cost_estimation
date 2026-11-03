"""
generate_data.py
Step 1: build a small made-up airline for one day.

Creates two files in the data/ folder:
  flights.csv     - every flight: which plane, where from/to, scheduled times
  passengers.csv  - every passenger: which flight they're on, and their
                    connecting flight (if they have one)
"""

import os
import random

import pandas as pd

import config


def duration_between(a, b):
    """Look up flight time between two airports (works in either direction)."""
    return config.FLIGHT_DURATION.get((a, b)) or config.FLIGHT_DURATION[(b, a)]


def to_clock(minutes):
    """Turn 'minutes after midnight' into a readable HH:MM string."""
    return f"{int(minutes // 60):02d}:{int(minutes % 60):02d}"


def generate_flights():
    """
    Give each plane a chain of flights for the day.
    Each flight leaves from wherever the previous one landed,
    after a planned turnaround gap.
    """
    flights = []
    flight_no = 101

    for plane_index in range(config.NUM_AIRCRAFT):
        aircraft_id = f"VT-{chr(65 + plane_index)}{chr(65 + plane_index)}A"  # e.g. VT-AAA
        location = random.choice(config.AIRPORTS)
        # stagger start times so planes don't all leave at 06:00
        departure = config.FIRST_DEPARTURE + plane_index * 20

        for leg in range(config.FLIGHTS_PER_AIRCRAFT):
            destination = random.choice([a for a in config.AIRPORTS if a != location])
            arrival = departure + duration_between(location, destination)

            # planned spare time before this plane's next flight
            is_last_leg = leg == config.FLIGHTS_PER_AIRCRAFT - 1
            gap = None if is_last_leg else random.randint(*config.TURNAROUND_GAP_RANGE)

            flights.append({
                "flight_id": f"FL{flight_no}",
                "aircraft_id": aircraft_id,
                "leg_number": leg + 1,
                "origin": location,
                "destination": destination,
                "sched_departure": departure,
                "sched_arrival": arrival,
                "dep_time": to_clock(departure),
                "arr_time": to_clock(arrival),
                "turnaround_gap": gap,
                "passengers": int(config.SEATS_PER_AIRCRAFT
                                  * random.uniform(*config.LOAD_FACTOR_RANGE)),
            })

            flight_no += 1
            location = destination
            if not is_last_leg:
                departure = arrival + gap

    return pd.DataFrame(flights)


def generate_passengers(flights):
    """
    Create every passenger on every flight.
    About 25% get a connecting flight: a later flight that leaves from
    the airport where they land, within a realistic connection window.
    """
    passengers = []
    pax_no = 1
    low, high = config.CONNECTION_GAP_RANGE

    for _, flight in flights.iterrows():
        # flights a passenger landing here could connect to
        possible_connections = flights[
            (flights["origin"] == flight["destination"])
            & (flights["sched_departure"] - flight["sched_arrival"] >= low)
            & (flights["sched_departure"] - flight["sched_arrival"] <= high)
        ]

        for _ in range(flight["passengers"]):
            connecting = (random.random() < config.CONNECTING_SHARE
                          and len(possible_connections) > 0)

            if connecting:
                onward = possible_connections.sample(
                    1, random_state=random.randint(0, 10**9)).iloc[0]
                onward_flight = onward["flight_id"]
                connection_gap = onward["sched_departure"] - flight["sched_arrival"]
            else:
                onward_flight = None
                connection_gap = None

            passengers.append({
                "passenger_id": f"P{pax_no:05d}",
                "flight_id": flight["flight_id"],
                "connecting_flight": onward_flight,
                "connection_gap": connection_gap,
            })
            pax_no += 1

    return pd.DataFrame(passengers)


if __name__ == "__main__":
    random.seed(config.RANDOM_SEED)
    os.makedirs("data", exist_ok=True)

    flights = generate_flights()
    passengers = generate_passengers(flights)

    flights.to_csv("data/flights.csv", index=False)
    passengers.to_csv("data/passengers.csv", index=False)

    connecting = passengers["connecting_flight"].notna().sum()
    print(f"Created {len(flights)} flights on {config.NUM_AIRCRAFT} aircraft")
    print(f"Created {len(passengers)} passengers ({connecting} with connections)")
    print("\nFirst plane's day:")
    first = flights[flights["aircraft_id"] == flights["aircraft_id"].iloc[0]]
    print(first[["flight_id", "origin", "destination", "dep_time",
                 "arr_time", "turnaround_gap", "passengers"]].to_string(index=False))
