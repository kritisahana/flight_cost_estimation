"""
agent.py
The AI analyst: a Claude model that answers questions about delay costs
by CALLING THE SIMULATOR, never by guessing numbers itself.

Principle: the agent decides and explains, the engine calculates.

Tools the agent can use:
  list_flights       - the day's schedule
  simulate_delay     - cost + cascade for one flight at one delay
  cost_curve         - where the cost jumps for a flight (5-minute steps)
  rank_flights       - which flights are most expensive to delay
  evaluate_hold      - recovery option: hold a connecting flight N minutes
                       to save the passengers who would miss it
"""

import json

import numpy as np
import pandas as pd

import config
import fast_sim

DEFAULT_MODEL = "claude-sonnet-5-5"

SYSTEM_PROMPT = """You are an airline operations cost analyst working with a \
delay-cost simulator for a small synthetic airline (6 aircraft, 30 flights, \
one day, 5 Indian airports).

Rules:
- Get every number from a tool. Never calculate or estimate costs yourself.
- Explain in plain language for a non-technical operations manager. Show money \
in rupees lakh (1 lakh = 100,000).
- When explaining a cascade, name the flights and say why (e.g. "FL110 is late \
because the same aircraft arrives late as FL109").
- When asked what to do, compare options with evaluate_hold or simulate_delay \
and recommend the cheapest, stating the saving.
- Be brief. Use short paragraphs or a small table.
- The data is synthetic and the costs are industry reference values; mention \
this only if the user asks how reliable the numbers are."""

TOOLS = [
    {
        "name": "list_flights",
        "description": "List all flights in the schedule: flight ID, aircraft, "
                       "route, departure/arrival time, passengers, connecting passengers.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "simulate_delay",
        "description": "Simulate one flight arriving late by a number of minutes "
                       "(no randomness). Returns cost by category, which flights "
                       "become late or cancelled, and missed connections.",
        "input_schema": {
            "type": "object",
            "properties": {
                "flight_id": {"type": "string", "description": "e.g. FL109"},
                "delay_minutes": {"type": "integer", "minimum": 0, "maximum": 300},
            },
            "required": ["flight_id", "delay_minutes"],
        },
    },
    {
        "name": "cost_curve",
        "description": "Monte Carlo cost curve for one flight from 0 to 300 minutes "
                       "in 5-minute steps. Returns cost at key delays and the "
                       "biggest 5-minute cost jumps with what caused them.",
        "input_schema": {
            "type": "object",
            "properties": {"flight_id": {"type": "string"}},
            "required": ["flight_id"],
        },
    },
    {
        "name": "rank_flights",
        "description": "Rank every flight by how much it costs if it is late by "
                       "the given number of minutes. Use to find the most critical flights.",
        "input_schema": {
            "type": "object",
            "properties": {"delay_minutes": {"type": "integer", "minimum": 0,
                                             "maximum": 300}},
            "required": ["delay_minutes"],
        },
    },
    {
        "name": "evaluate_hold",
        "description": "Recovery option: when a late flight makes passengers miss a "
                       "connecting flight, hold that connecting flight for some "
                       "minutes so they make it. Compares total cost with and "
                       "without the hold (the held flight's own knock-on delay included).",
        "input_schema": {
            "type": "object",
            "properties": {
                "flight_id": {"type": "string", "description": "the late flight"},
                "delay_minutes": {"type": "integer"},
                "hold_flight_id": {"type": "string",
                                   "description": "the connecting flight to hold"},
                "hold_minutes": {"type": "integer", "minimum": 1, "maximum": 120},
            },
            "required": ["flight_id", "delay_minutes", "hold_flight_id", "hold_minutes"],
        },
    },
]


def _lakh(x):
    return round(float(x) / 100000, 2)


class Tools:
    """Runs the tools against the current data and config values."""

    def __init__(self, flights, passengers):
        self.flights = flights
        self.passengers = passengers
        self.ids = set(flights["flight_id"])

    def _check(self, *flight_ids):
        for f in flight_ids:
            if f not in self.ids:
                raise ValueError(f"{f} is not in the schedule (FL101-FL130).")

    def list_flights(self):
        conn = (self.passengers.dropna(subset=["connecting_flight"])
                .groupby("flight_id").size())
        rows = []
        for _, f in self.flights.iterrows():
            rows.append({"flight": f["flight_id"], "aircraft": f["aircraft_id"],
                         "route": f"{f['origin']}-{f['destination']}",
                         "dep": f["dep_time"], "arr": f["arr_time"],
                         "passengers": int(f["passengers"]),
                         "connecting": int(conn.get(f["flight_id"], 0))})
        return rows

    def simulate_delay(self, flight_id, delay_minutes):
        self._check(flight_id)
        chain = fast_sim.PlaneChain(self.flights, self.passengers, flight_id)
        result, late, missed = fast_sim.scenario(chain, delay_minutes)
        return {
            "costs_lakh": {k: _lakh(result[k]) for k in
                           ["aircraft_delay", "rebooking", "meals", "hotels",
                            "cancellation", "total"]},
            "late_flights": late.to_dict("records"),
            "missed_connections": (missed.groupby(["flight_id", "connecting_flight"])
                                   .size().rename("passengers").reset_index()
                                   .to_dict("records")),
        }

    def cost_curve(self, flight_id):
        self._check(flight_id)
        chain = fast_sim.PlaneChain(self.flights, self.passengers, flight_id)
        t = fast_sim.cost_curve(chain, 1000, 5, 10)
        key = t[t["delay_min"].isin([15, 30, 60, 90, 120, 180, 240, 300])]
        jumps = []
        for i in t["extra_vs_previous"].nlargest(4).index:
            before, after = t.loc[i - 1], t.loc[i]
            causes = [c for c in ["rebooking", "meals", "hotels", "cancellation"]
                      if after[c] - before[c] > 1000]
            jumps.append({"at_minutes": int(after["delay_min"]),
                          "extra_lakh": _lakh(after["extra_vs_previous"]),
                          "caused_by": causes or ["aircraft_delay"]})
        return {
            "average_cost_lakh_at": {int(r.delay_min): _lakh(r.total)
                                     for r in key.itertuples()},
            "worst_case_lakh_at": {int(r.delay_min): _lakh(r.worst_case)
                                   for r in key.itertuples()},
            "normal_cost_per_5_min_lakh": _lakh(t["extra_vs_previous"].median()),
            "biggest_jumps": jumps,
        }

    def rank_flights(self, delay_minutes):
        rows = []
        for f in self.flights["flight_id"]:
            chain = fast_sim.PlaneChain(self.flights, self.passengers, f)
            r = fast_sim.scenario(chain, delay_minutes)[0]
            rows.append({"flight": f, "total_lakh": _lakh(r["total"]),
                         "flights_delayed": int(r["flights_delayed"]),
                         "missed_connections": int(r["missed"])})
        return sorted(rows, key=lambda r: -r["total_lakh"])

    def evaluate_hold(self, flight_id, delay_minutes, hold_flight_id, hold_minutes):
        self._check(flight_id, hold_flight_id)
        chain = fast_sim.PlaneChain(self.flights, self.passengers, flight_id)
        without = fast_sim.scenario(chain, delay_minutes)[0]

        # Holding the onward flight gives its passengers extra connection time
        held = chain.conn["connecting_flight"].to_numpy() == hold_flight_id
        original_gap = chain.conn_gap.copy()
        chain.conn_gap = np.where(held, chain.conn_gap + hold_minutes, chain.conn_gap)
        with_hold = fast_sim.scenario(chain, delay_minutes)[0]
        chain.conn_gap = original_gap

        # ...but the held flight is now late, with its own knock-on effects
        held_chain = fast_sim.PlaneChain(self.flights, self.passengers, hold_flight_id)
        hold_cost = fast_sim.scenario(held_chain, hold_minutes)[0]

        total_with = with_hold["total"] + hold_cost["total"]
        return {
            "without_hold": {"total_lakh": _lakh(without["total"]),
                             "missed_connections": int(without["missed"])},
            "with_hold": {"total_lakh": _lakh(total_with),
                          "missed_connections": int(with_hold["missed"]),
                          "cost_of_holding_lakh": _lakh(hold_cost["total"]),
                          "missed_on_held_flights_chain": int(hold_cost["missed"])},
            "saving_lakh": _lakh(without["total"] - total_with),
            "recommendation": "hold" if total_with < without["total"] else "don't hold",
            "note": "Approximation: the two effects are added together.",
        }

    def run(self, name, args):
        try:
            return getattr(self, name)(**args)
        except Exception as e:  # report tool errors back to the model
            return {"error": str(e)}


def ask(question, history, flights, passengers, api_key, model=DEFAULT_MODEL,
        on_tool=None, max_steps=8):
    """
    Send the question to Claude, let it call tools until it has an answer.
    history: list of previous {"role", "content"} messages (text only).
    on_tool: optional callback(name, args) to show progress in the UI.
    Returns the answer text.
    """
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    tools = Tools(flights, passengers)
    messages = list(history) + [{"role": "user", "content": question}]

    for _ in range(max_steps):
        response = client.messages.create(
            model=model, max_tokens=1500, system=SYSTEM_PROMPT,
            tools=TOOLS, messages=messages)

        if response.stop_reason != "tool_use":
            return "".join(b.text for b in response.content if b.type == "text")

        messages.append({"role": "assistant", "content": response.content})
        results = []
        for block in response.content:
            if block.type == "tool_use":
                if on_tool:
                    on_tool(block.name, block.input)
                output = tools.run(block.name, block.input)
                results.append({"type": "tool_result", "tool_use_id": block.id,
                                "content": json.dumps(output, default=float)})
        messages.append({"role": "user", "content": results})

    return "I needed more steps than allowed to answer that. Try a narrower question."


if __name__ == "__main__":
    # Quick offline check of the tools (no API key needed)
    f = pd.read_csv("data/flights.csv")
    p = pd.read_csv("data/passengers.csv")
    t = Tools(f, p)
    print(json.dumps(t.simulate_delay("FL109", 60), indent=1, default=float))
    print(json.dumps(t.cost_curve("FL109"), indent=1))
    print(t.rank_flights(60)[:5])
    print(json.dumps(t.evaluate_hold("FL109", 60, "FL130", 10), indent=1))
