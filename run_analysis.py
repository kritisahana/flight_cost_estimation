"""
run_analysis.py
Step 4: the full analysis - 5-minute steps, randomness, 1,000 runs, chart.

You type in which flight is late. For every delay level (0, 5, 10 ...
300 minutes) the program runs the whole simulation 1,000 times. Each
run, two uncertain things come out slightly differently:

  1. Turnaround: some days the ground crew is faster, some days slower,
     so the time a plane needs on the ground varies by a few minutes.
  2. Passengers: some passengers run and make a tight connection,
     others are slow, so the time each one needs to change planes varies.

For each delay level it reports:
  - the AVERAGE cost over the 1,000 runs (the expected cost)
  - the WORST-CASE cost (the average of the worst 1 in 10 runs)

Results are saved to results/<flight>_costs.csv and drawn as a chart
in results/<flight>_chart.png.
"""

import os

import matplotlib
matplotlib.use("Agg")          # draw straight to a file, no window needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from costs import calculate_costs, rupees
from propagation import simulate

# How much the uncertain things vary from run to run (minutes)
TURNAROUND_VARIATION = 5    # ground time needed is usually within +/- 5 min of normal
CONNECTION_VARIATION = 10   # a passenger may need 10 min more or less than normal


def ask_for_flight(flights):
    """Keep asking until the user types a real flight ID, e.g. FL109."""
    while True:
        flight_id = input("Which flight is delayed? (e.g. FL109): ").strip().upper()
        if flight_id in flights["flight_id"].values:
            return flight_id
        print(f"  {flight_id} isn't in the schedule. Try FL101 to FL130.")


def ask_for_runs():
    """Ask how many random runs per delay level (Enter = the config value)."""
    answer = input(f"How many runs per delay level? "
                   f"(press Enter for {config.NUM_SIMULATIONS}): ").strip()
    return int(answer) if answer.isdigit() and int(answer) > 0 else config.NUM_SIMULATIONS


def relevant_passengers(flights, passengers, flight_id):
    """
    Only connecting passengers on this plane's flights can be affected.
    Keeping just them makes 1,000 runs much faster.
    """
    plane = flights.loc[flights["flight_id"] == flight_id, "aircraft_id"].iloc[0]
    plane_flights = flights.loc[flights["aircraft_id"] == plane, "flight_id"]
    return passengers[passengers["flight_id"].isin(plane_flights)
                      & passengers["connecting_flight"].notna()]


def one_random_run(flights, passengers, flight_id, delay, rng):
    """Run the simulation once, with today's random 'luck'."""
    # 1. Turnaround luck: change each plane's usable ground time a little
    lucky_flights = flights.copy()
    lucky_flights["turnaround_gap"] += rng.normal(0, TURNAROUND_VARIATION,
                                                  len(flights))

    # 2. Passenger luck: change each passenger's usable connection time a little
    lucky_passengers = passengers.copy()
    lucky_passengers["connection_gap"] += rng.uniform(-CONNECTION_VARIATION,
                                                      CONNECTION_VARIATION,
                                                      len(passengers))

    flight_delays, missed = simulate(lucky_flights, lucky_passengers,
                                     flight_id, delay)
    return calculate_costs(lucky_flights, lucky_passengers,
                           flight_delays, missed)["total"]


def five_minute_table(flights, passengers, flight_id, runs):
    """For each delay level: average and worst-case cost over many runs."""
    rng = np.random.default_rng(config.RANDOM_SEED)
    pax = relevant_passengers(flights, passengers, flight_id)

    rows = []
    previous_average = 0
    for delay in range(0, config.MAX_DELAY + 1, config.DELAY_STEP):
        totals = [one_random_run(flights, pax, flight_id, delay, rng)
                  for _ in range(runs)]
        average = np.mean(totals)
        rows.append({
            "delay_min": delay,
            "average_cost": average,
            # worst case = average of the worst 10% of runs
            "worst_case_cost": np.mean(sorted(totals)[-max(1, runs // 10):]),
            "extra_vs_previous": average - previous_average,
        })
        previous_average = average
        print(f"  simulated {delay:>3} min", end="\r")   # progress
    return pd.DataFrame(rows)


def draw_chart(table, flight_id, runs, path):
    """
    Two charts, one above the other, sharing the same delay axis:
      top    - total cost as the delay grows (average, with the worst-case band)
      bottom - the extra cost of each 5-minute step (the jumps stand out)
    """
    blue, ink, muted, grid = "#2a78d6", "#0b0b0b", "#52514e", "#e6e5e1"
    delay = table["delay_min"]
    average = table["average_cost"] / 100000          # in lakh
    worst = table["worst_case_cost"] / 100000
    extra = table["extra_vs_previous"] / 100000

    fig, (top, bottom) = plt.subplots(2, 1, figsize=(11, 8), sharex=True,
                                      gridspec_kw={"height_ratios": [3, 2]})
    fig.patch.set_facecolor("#fcfcfb")

    # Top: total cost
    top.fill_between(delay, average, worst, color=blue, alpha=0.18,
                     linewidth=0, label="Worst case (worst 10% of runs)")
    top.plot(delay, average, color=blue, linewidth=2, label="Average cost")
    top.set_ylabel("Total cost (Rs lakh)", color=muted)
    top.set_title(f"Cost of {flight_id} being late  "
                  f"({runs} simulated runs per 5-minute step)",
                  loc="left", color=ink, fontsize=13)
    top.legend(frameon=False, loc="upper left", labelcolor=muted)

    # Bottom: extra cost of each 5-minute step
    bottom.bar(delay, extra, width=3.5, color=blue)
    bottom.set_ylabel("Extra cost for\nthese 5 min (Rs lakh)", color=muted)
    bottom.set_xlabel("Delay of the first flight (minutes)", color=muted)

    # Label the biggest jumps (skipping any too close to one already labelled)
    labelled = []
    for i in extra.nlargest(5).index:
        if len(labelled) == 2 or any(abs(delay[i] - delay[j]) < 20 for j in labelled):
            continue
        labelled.append(i)
        bottom.annotate(f"+{extra[i]:.1f} lakh\nat {delay[i]:.0f} min",
                        (delay[i], extra[i]), xytext=(0, 4),
                        textcoords="offset points", ha="center",
                        va="bottom", fontsize=9, color=ink)
    bottom.set_ylim(0, extra.max() * 1.35)

    for ax in (top, bottom):
        ax.set_facecolor("#fcfcfb")
        ax.grid(axis="y", color=grid, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(grid)
        ax.tick_params(colors=muted)
    bottom.set_xticks(range(0, int(delay.max()) + 1, 30))

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    flights = pd.read_csv("data/flights.csv")
    passengers = pd.read_csv("data/passengers.csv")

    flight_id = ask_for_flight(flights)
    runs = ask_for_runs()
    print(f"Running {runs} simulations per delay level...")
    table = five_minute_table(flights, passengers, flight_id, runs)

    print(f"\nCost of {flight_id} being late "
          f"(average of {runs} runs):\n")
    print("Delay     Average cost       Worst case (worst 10%)   Extra for these 5 min")
    for _, row in table.iterrows():
        print(f"{row['delay_min']:>4.0f} min  {rupees(row['average_cost'])}   "
              f"{rupees(row['worst_case_cost'])}       "
              f"+ {rupees(row['extra_vs_previous'])}")

    os.makedirs("results", exist_ok=True)
    out = f"results/{flight_id}_costs.csv"
    table.to_csv(out, index=False)
    chart = f"results/{flight_id}_chart.png"
    draw_chart(table, flight_id, runs, chart)
    print(f"\nSaved table to {out}")
    print(f"Saved chart to {chart}")