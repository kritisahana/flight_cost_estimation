"""
costs.py
Step 3: turn "what broke" into rupees.

Takes the result of propagation.py (which flights are late, which
passengers missed connections) and adds up five kinds of cost:

  1. Aircraft delay   - every minute a flight is late costs money
                        (crew, fuel, maintenance) -> COST_PER_DELAY_MINUTE
  2. Rebooking        - each passenger who misses a connection must be
                        put on another flight -> REBOOKING_COST
  3. Meals            - once a flight is 2+ hours late, every passenger
                        on it gets a meal (DGCA rule) -> MEAL_COST
  4. Hotels           - passengers stranded overnight get a hotel:
                          * missed a connection and landed after 22:00
                            (no flights left today), or
                          * their flight was cancelled
  5. Cancellation     - a flight 4+ hours late is assumed cancelled:
                        every passenger gets DGCA compensation and is
                        rebooked on the next day's flight

Simplification: a passenger on a cancelled flight is counted under
cancellation only (not also as a missed connection), so nobody is
charged twice.
"""

import pandas as pd

import config
from propagation import simulate


def calculate_costs(flights, passengers, flight_delays, missed):
    """Return a dict of cost per category, plus the total, in rupees."""
    info = flights.set_index("flight_id")

    # Which late flights are bad enough to be cancelled?
    cancelled = [f for f, d in flight_delays.items()
                 if d >= config.CANCELLATION_THRESHOLD]
    cancelled_pax = info.loc[cancelled, "passengers"].sum() if cancelled else 0

    # 1. Aircraft delay: minutes late x cost per minute, for every late flight
    aircraft = sum(flight_delays.values()) * config.COST_PER_DELAY_MINUTE

    # 2. Rebooking: missed connections (on flights that still flew)
    still_flew = missed[~missed["flight_id"].isin(cancelled)]
    rebooking = len(still_flew) * config.REBOOKING_COST

    # 3. Meals: everyone on a flight delayed 2+ hours
    long_delays = [f for f, d in flight_delays.items()
                   if d >= config.MEAL_THRESHOLD]
    meal_pax = info.loc[long_delays, "passengers"].sum() if long_delays else 0
    meals = meal_pax * config.MEAL_COST

    # 4. Hotels: stranded after the last flight of the day, or cancelled
    actual_arrival = (still_flew["flight_id"].map(info["sched_arrival"])
                      + still_flew["inbound_delay"])
    stranded = (actual_arrival > config.LAST_FLIGHT_OF_DAY).sum()
    hotels = (stranded + cancelled_pax) * config.HOTEL_COST

    # 5. Cancellation: compensation + rebooking for everyone on a cancelled flight
    cancellation = cancelled_pax * (config.CANCELLATION_COMPENSATION
                                    + config.REBOOKING_COST)

    costs = {
        "aircraft_delay": aircraft,
        "rebooking": rebooking,
        "meals": meals,
        "hotels": hotels,
        "cancellation": cancellation,
    }
    costs["total"] = sum(costs.values())
    return costs


def cost_of_delay(flights, passengers, flight_id, delay):
    """One-line shortcut: 'how much does this delay cost?'"""
    flight_delays, missed = simulate(flights, passengers, flight_id, delay)
    return calculate_costs(flights, passengers, flight_delays, missed)


def rupees(x):
    """Show a number as Indian-style lakhs, e.g. 8.5 lakh."""
    return f"Rs {x / 100000:6.2f} lakh"


if __name__ == "__main__":
    flights = pd.read_csv("data/flights.csv")
    passengers = pd.read_csv("data/passengers.csv")

    # Example: FL109 arrives 60 minutes late
    c = cost_of_delay(flights, passengers, "FL109", 60)
    print("Scenario: FL109 arrives 60 minutes late\n")
    print("Cost breakdown:")
    for name, value in c.items():
        if name == "total":
            print("  " + "-" * 32)
        print(f"  {name:<15} {rupees(value)}")

    # How cost grows as the delay grows
    print("\nFL109 delay   Total cost       Extra cost vs previous row")
    previous = 0
    for d in range(0, config.MAX_DELAY + 1, 30):
        total = cost_of_delay(flights, passengers, "FL109", d)["total"]
        print(f"{d:>6} min   {rupees(total)}   + {rupees(total - previous)}")
        previous = total