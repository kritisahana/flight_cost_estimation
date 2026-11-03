"""
config.py
All the numbers the project depends on, kept in one place.
Change a value here and every other file picks it up.
"""

# ---------------------------------------------------------------
# 1. Settings for the synthetic (made-up) airline
# ---------------------------------------------------------------
RANDOM_SEED = 42          # same seed = same fake data every time you run

AIRPORTS = ["DEL", "BOM", "BLR", "MAA", "HYD"]

NUM_AIRCRAFT = 6          # how many planes the airline has
FLIGHTS_PER_AIRCRAFT = 5  # how many flights each plane flies in a day
FIRST_DEPARTURE = 6 * 60  # first flight of the day at 06:00 (stored as minutes after midnight)

SEATS_PER_AIRCRAFT = 180               # typical narrow-body (A320 / 737)
LOAD_FACTOR_RANGE = (0.75, 0.95)       # planes are 75%-95% full

TURNAROUND_GAP_RANGE = (35, 70)        # planned gap between a plane's flights (minutes)
MIN_TURNAROUND = 30                    # a plane needs at least this long on the ground

CONNECTING_SHARE = 0.25                # 25% of passengers have a connecting flight
CONNECTION_GAP_RANGE = (40, 180)       # their planned gap between flights (minutes)
MIN_CONNECTION_TIME = 30               # time needed to walk to the next gate

# Flight durations between airport pairs (minutes)
FLIGHT_DURATION = {
    ("DEL", "BOM"): 125, ("DEL", "BLR"): 165, ("DEL", "MAA"): 165, ("DEL", "HYD"): 130,
    ("BOM", "BLR"): 95,  ("BOM", "MAA"): 110, ("BOM", "HYD"): 85,
    ("BLR", "MAA"): 60,  ("BLR", "HYD"): 75,
    ("MAA", "HYD"): 80,
}

# ---------------------------------------------------------------
# 2. Cost values (Indian Rupees)
#    Sources:
#    - Aircraft cost per minute: Airlines for America 2024, ~$100/min
#      (crew, fuel, maintenance, ownership), converted at ~Rs 84/$
#    - Passenger care and compensation: DGCA CAR Section 3, Series M, Part IV
#    - Rebooking, meal, hotel amounts: assumptions, stated in the report
# ---------------------------------------------------------------
COST_PER_DELAY_MINUTE = 8400        # Rs per minute a plane is delayed
REBOOKING_COST = 4500               # Rs per passenger who misses a connection
MEAL_COST = 500                     # Rs per passenger
MEAL_THRESHOLD = 120                # meals owed once delay reaches 2 hours
HOTEL_COST = 5500                   # Rs per passenger (room + transport)
CANCELLATION_THRESHOLD = 240        # assume flight is cancelled beyond 4 hours of delay
CANCELLATION_COMPENSATION = 7500    # Rs per passenger (DGCA medium-length flight)
LAST_FLIGHT_OF_DAY = 22 * 60        # connections after 22:00 have no same-day alternative

# ---------------------------------------------------------------
# 3. Analysis settings
# ---------------------------------------------------------------
DELAY_STEP = 5           # analyse in 5-minute increments
MAX_DELAY = 300          # up to 5 hours
NUM_SIMULATIONS = 1000   # Monte Carlo runs per delay level
