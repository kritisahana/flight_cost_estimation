"""
propagation.py
Step 2: follow a delay through the network.

Given one delayed flight, work out:
  1. how late every later flight on the same aircraft becomes
  2. which connecting passengers miss their onward flight

Rule 1 - aircraft (the "knock-on" delay)
    A plane needs MIN_TURNAROUND minutes on the ground (cleaning, fuel,
    boarding). Any planned ground time beyond that is SLACK, which can
    absorb delay:
        slack            = turnaround_gap - MIN_TURNAROUND
        next_flight_late = max(0, this_flight_late - slack)
    The delay passes down the chain until slack soaks it all up.

Rule 2 - passengers (missed connections)
    A passenger needs MIN_CONNECTION_TIME to reach the next gate.
        time_available = planned_gap - inbound_delay + onward_delay
        missed if time_available < MIN_CONNECTION_TIME
    (If the onward flight is late too, the passenger gets extra time.)
    Passengers staying on the same aircraft can't miss it: the plane
    can't leave without arriving first.
"""

import pandas as pd

import config


def propagate_aircraft_delay(flights, delayed_flight_id, initial_delay):
    """
    Return a dict {flight_id: minutes late} for the delayed flight and
    every later flight flown by the same aircraft.
    """
    start = flights[flights["flight_id"] == delayed_flight_id].iloc[0]
    chain = (flights[(flights["aircraft_id"] == start["aircraft_id"])
                     & (flights["leg_number"] >= start["leg_number"])]
             .sort_values("leg_number"))

    delays = {}
    delay = initial_delay
    for _, flight in chain.iterrows():
        if delay <= 0:
            break                      # the chain has recovered
        delays[flight["flight_id"]] = delay

        if pd.isna(flight["turnaround_gap"]):
            break                      # last flight of the day
        slack = flight["turnaround_gap"] - config.MIN_TURNAROUND
        delay = max(0, delay - slack)  # slack absorbs part of the delay

    return delays


def find_missed_connections(flights, passengers, flight_delays):
    """
    Return the connecting passengers who miss their onward flight,
    with how much connection time they ended up with.
    """
    aircraft_of = dict(zip(flights["flight_id"], flights["aircraft_id"]))

    # only passengers arriving on a delayed flight can be affected
    at_risk = passengers[
        passengers["flight_id"].isin(flight_delays.keys())
        & passengers["connecting_flight"].notna()
    ].copy()

    at_risk["inbound_delay"] = at_risk["flight_id"].map(flight_delays)
    at_risk["onward_delay"] = at_risk["connecting_flight"].map(flight_delays).fillna(0)
    at_risk["time_available"] = (at_risk["connection_gap"]
                                 - at_risk["inbound_delay"]
                                 + at_risk["onward_delay"])

    same_aircraft = (at_risk["flight_id"].map(aircraft_of)
                     == at_risk["connecting_flight"].map(aircraft_of))

    missed = at_risk[(at_risk["time_available"] < config.MIN_CONNECTION_TIME)
                     & ~same_aircraft]
    return missed


def simulate(flights, passengers, delayed_flight_id, initial_delay):
    """Run both rules for one delay scenario."""
    flight_delays = propagate_aircraft_delay(flights, delayed_flight_id, initial_delay)
    missed = find_missed_connections(flights, passengers, flight_delays)
    return flight_delays, missed


if __name__ == "__main__":
    flights = pd.read_csv("data/flights.csv")
    passengers = pd.read_csv("data/passengers.csv")

    # Example: FL109 (BOM->BLR, lands 14:16) arrives 60 minutes late
    flight_delays, missed = simulate(flights, passengers, "FL109", 60)

    print("Scenario: FL109 arrives 60 minutes late\n")
    print("Knock-on delay down the aircraft's chain:")
    for fid, mins in flight_delays.items():
        f = flights[flights["flight_id"] == fid].iloc[0]
        print(f"  {fid} {f['origin']}->{f['destination']} "
              f"(dep {f['dep_time']}): {mins:.0f} min late")

    print(f"\nMissed connections: {len(missed)} passengers")
    if len(missed):
        summary = missed.groupby(["flight_id", "connecting_flight"]).size()
        for (inbound, onward), count in summary.items():
            print(f"  {count} passengers on {inbound} missed {onward}")

    # Quick look at how impact grows with delay
    print("\nDelay   Flights delayed   Missed connections")
    for d in range(0, 181, 15):
        fd, m = simulate(flights, passengers, "FL109", d)
        print(f"{d:>4} min  {len(fd):>10}   {len(m):>14}")