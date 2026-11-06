"""
fast_sim.py
A fast version of the same rules in propagation.py + costs.py.

The original files check one scenario at a time, which is easy to read
but slow (1,000 runs x 61 delay levels takes minutes). This file applies
the SAME rules to all 1,000 runs at once using numpy arrays, so the
dashboard can answer in about a second.

check_matches_original() proves the two give identical answers when the
randomness is switched off.
"""

import numpy as np
import pandas as pd

import config


class PlaneChain:
    """Everything needed to simulate a delay starting at one flight."""

    def __init__(self, flights, passengers, flight_id):
        start = flights.loc[flights["flight_id"] == flight_id].iloc[0]
        chain = (flights[(flights["aircraft_id"] == start["aircraft_id"])
                         & (flights["leg_number"] >= start["leg_number"])]
                 .sort_values("leg_number"))

        self.flight_id = flight_id
        self.aircraft_id = start["aircraft_id"]
        self.ids = chain["flight_id"].tolist()
        self.gaps = chain["turnaround_gap"].to_numpy(dtype=float)  # last is NaN
        self.pax = chain["passengers"].to_numpy(dtype=float)
        self.sched_arrival = chain["sched_arrival"].to_numpy(dtype=float)

        # Connecting passengers on these flights who change to a DIFFERENT plane
        plane_of = dict(zip(flights["flight_id"], flights["aircraft_id"]))
        conn = passengers[passengers["flight_id"].isin(self.ids)
                          & passengers["connecting_flight"].notna()].copy()
        conn = conn[conn["connecting_flight"].map(plane_of) != self.aircraft_id]
        position = {f: i for i, f in enumerate(self.ids)}
        self.conn = conn.reset_index(drop=True)
        self.conn_leg = conn["flight_id"].map(position).to_numpy()
        self.conn_gap = conn["connection_gap"].to_numpy(dtype=float)


def propagate(chain, delay, gap_noise):
    """
    Rule 1 for many runs at once.
    gap_noise: array (runs, legs) of minutes added to each turnaround gap.
    Returns delays: array (runs, legs) of minutes late per flight.
    """
    runs, legs = gap_noise.shape
    delays = np.zeros((runs, legs))
    current = np.full(runs, float(delay))
    for leg in range(legs):
        delays[:, leg] = current
        if np.isnan(chain.gaps[leg]):
            break                                   # last flight of the day
        slack = chain.gaps[leg] + gap_noise[:, leg] - config.MIN_TURNAROUND
        current = np.where(current > 0, np.maximum(0, current - slack), 0)
    return delays


def costs_for(chain, delays, conn_noise):
    """
    Rule 2 + all cost rules for many runs at once.
    conn_noise: array (runs, connecting passengers) of minutes added to
    each passenger's connection time.
    Returns a dict of arrays (one value per run) per cost category.
    """
    cancelled = delays >= config.CANCELLATION_THRESHOLD           # (runs, legs)
    cancelled_pax = (cancelled * chain.pax).sum(axis=1)

    inbound = delays[:, chain.conn_leg]                           # (runs, conn)
    time_left = chain.conn_gap + conn_noise - inbound
    missed = (time_left < config.MIN_CONNECTION_TIME) & (inbound > 0)
    flew = ~cancelled[:, chain.conn_leg]
    missed_flew = missed & flew

    arrival = chain.sched_arrival[chain.conn_leg] + inbound
    stranded = (missed_flew & (arrival > config.LAST_FLIGHT_OF_DAY)).sum(axis=1)

    meal_pax = ((delays >= config.MEAL_THRESHOLD) * chain.pax).sum(axis=1)

    out = {
        "aircraft_delay": delays.sum(axis=1) * config.COST_PER_DELAY_MINUTE,
        "rebooking": missed_flew.sum(axis=1) * config.REBOOKING_COST,
        "meals": meal_pax * config.MEAL_COST,
        "hotels": (stranded + cancelled_pax) * config.HOTEL_COST,
        "cancellation": cancelled_pax * (config.CANCELLATION_COMPENSATION
                                         + config.REBOOKING_COST),
    }
    out["total"] = sum(out.values())
    out["missed"] = missed_flew.sum(axis=1)
    out["flights_delayed"] = (delays > 0).sum(axis=1)
    out["flights_cancelled"] = cancelled.sum(axis=1)
    return out


def scenario(chain, delay):
    """One fixed (no randomness) scenario, with full details for display."""
    legs = len(chain.ids)
    delays = propagate(chain, delay, np.zeros((1, legs)))
    result = {k: float(v[0]) for k, v in
              costs_for(chain, delays, np.zeros((1, len(chain.conn_gap)))).items()}

    flight_rows = [{"flight_id": f, "minutes_late": float(delays[0, i]),
                    "cancelled": bool(delays[0, i] >= config.CANCELLATION_THRESHOLD)}
                   for i, f in enumerate(chain.ids) if delays[0, i] > 0]

    inbound = delays[0, chain.conn_leg]
    time_left = chain.conn_gap - inbound
    hit = (time_left < config.MIN_CONNECTION_TIME) & (inbound > 0)
    missed = chain.conn.loc[hit, ["passenger_id", "flight_id",
                                  "connecting_flight", "connection_gap"]].copy()
    missed["time_left"] = time_left[hit]
    return result, pd.DataFrame(flight_rows), missed


def cost_curve(chain, runs, turnaround_var, connection_var, seed=config.RANDOM_SEED):
    """
    Monte Carlo: for every 5-minute delay level, simulate `runs` random days
    and return average and worst-case (average of worst 10%) cost per category.
    """
    rng = np.random.default_rng(seed)
    legs, n_conn = len(chain.ids), len(chain.conn_gap)
    rows = []
    for delay in range(0, config.MAX_DELAY + 1, config.DELAY_STEP):
        gap_noise = rng.normal(0, turnaround_var, (runs, legs))
        conn_noise = rng.uniform(-connection_var, connection_var, (runs, n_conn))
        res = costs_for(chain, propagate(chain, delay, gap_noise), conn_noise)
        worst_n = max(1, runs // 10)
        row = {"delay_min": delay}
        for k, v in res.items():
            row[k] = v.mean()
        row["worst_case"] = np.sort(res["total"])[-worst_n:].mean()
        rows.append(row)
    table = pd.DataFrame(rows)
    table["extra_vs_previous"] = table["total"].diff().fillna(table["total"])
    return table


def check_matches_original(flights, passengers):
    """Confirm the fast version gives the same answer as costs.py (no randomness)."""
    from costs import cost_of_delay
    problems = 0
    for fid in flights["flight_id"]:
        chain = PlaneChain(flights, passengers, fid)
        for d in range(0, config.MAX_DELAY + 1, config.DELAY_STEP):
            fast = scenario(chain, d)[0]["total"]
            slow = cost_of_delay(flights, passengers, fid, d)["total"]
            if abs(fast - slow) > 0.01:
                problems += 1
                print(f"Mismatch {fid} at {d} min: fast {fast} vs original {slow}")
    print("All scenarios match." if problems == 0 else f"{problems} mismatches.")
    return problems == 0


if __name__ == "__main__":
    flights = pd.read_csv("data/flights.csv")
    passengers = pd.read_csv("data/passengers.csv")
    check_matches_original(flights, passengers)
