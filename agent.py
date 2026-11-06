"""
agent.py
The AI analyst, built with LangChain. It answers questions about delay
costs by CALLING THE SIMULATOR, never by guessing numbers itself.

Principle: the agent decides and explains, the engine calculates.

Works with three AI providers (pick one in the dashboard):
  Gemini  - Google, free tier, best default        (needs GOOGLE_API_KEY)
  Groq    - very fast open models, free tier       (needs GROQ_API_KEY)
  Ollama  - runs on your own computer, no key      (needs Ollama installed)

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
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool

import config
import fast_sim

# Provider name -> (default model, environment variable holding the key)
PROVIDERS = {
    "Gemini (Google, free)": ("gemini-3.1-flash-lite", "GOOGLE_API_KEY"),
    "Groq (free)": ("llama-3.3-70b-versatile", "GROQ_API_KEY"),
    "Ollama (local, no key)": ("qwen3:8b", None),
}


def system_prompt():
    """Instructions + background knowledge, using the CURRENT config values."""
    return f"""You are an airline operations cost analyst working with a \
delay-cost simulator for a small synthetic airline (one day, 5 Indian airports: \
DEL, BOM, BLR, MAA, HYD).

Rules:
- Get every number from a tool. Never calculate or estimate costs yourself.
- Explain in plain language for a non-technical operations manager. Show money \
in rupees lakh (1 lakh = 100,000).
- When explaining a cascade, name the flights and say why (e.g. "FL110 is late \
because the same aircraft arrives late as FL109").
- When asked what to do, compare options with evaluate_hold or simulate_delay \
and recommend the cheapest, stating the saving.
- Be brief. Use short paragraphs or a small table.

Background knowledge (use it to explain WHY, not to calculate):
- Delay propagation: a delay passes to the same aircraft's next flight. Ground \
time beyond the {config.MIN_TURNAROUND}-minute minimum turnaround is slack that \
absorbs part of the delay.
- Missed connection: a passenger needs at least {config.MIN_CONNECTION_TIME} \
minutes to change planes. Passengers staying on the same aircraft can't miss it.
- Cost rules: aircraft delay Rs {config.COST_PER_DELAY_MINUTE:,} per minute per \
late flight (crew, fuel, maintenance, ownership; based on Airlines for America \
2024 average of about $100 per block minute). Rebooking Rs {config.REBOOKING_COST:,} \
per missed connection. Meals Rs {config.MEAL_COST:,} per passenger once a flight \
is {config.MEAL_THRESHOLD}+ minutes late. Hotel Rs {config.HOTEL_COST:,} per \
passenger stranded overnight. A flight {config.CANCELLATION_THRESHOLD}+ minutes \
late is assumed cancelled: compensation Rs {config.CANCELLATION_COMPENSATION:,} \
plus rebooking plus hotel per passenger.
- Indian rules (DGCA CAR Section 3, Series M, Part IV): airlines must provide \
meals during long delays, hotel accommodation when an overnight stay is needed, \
and on cancellation an alternate flight or refund plus compensation of roughly \
Rs 5,000-10,000 depending on flight length.
- Monte Carlo: the cost curve simulates each 5-minute step many times with \
random turnaround and connection times; expected cost = average, worst case = \
average of the worst 10% of runs.
- Limitations: synthetic data; crew schedules, seat availability, aircraft \
types, refunds and lost revenue are not modelled. Mention these only if asked \
how reliable the numbers are."""


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


def make_tools(flights, passengers):
    """Wrap the simulator functions as LangChain tools."""
    t = Tools(flights, passengers)

    def list_flights() -> str:
        """List all flights: flight ID, aircraft, route, departure/arrival time,
        passengers and number of connecting passengers."""
        return json.dumps(t.run("list_flights", {}), default=float)

    def simulate_delay(flight_id: str, delay_minutes: int) -> str:
        """Simulate one flight (e.g. FL109) arriving late by delay_minutes (0-300),
        with no randomness. Returns cost by category in Rs lakh, which flights
        become late or cancelled, and missed connections."""
        return json.dumps(t.run("simulate_delay", {"flight_id": flight_id,
                                                   "delay_minutes": delay_minutes}),
                          default=float)

    def cost_curve(flight_id: str) -> str:
        """Monte Carlo cost curve for one flight from 0 to 300 minutes in 5-minute
        steps. Returns cost at key delays, the normal cost per 5 minutes, and the
        biggest 5-minute cost jumps with what caused them."""
        return json.dumps(t.run("cost_curve", {"flight_id": flight_id}), default=float)

    def rank_flights(delay_minutes: int) -> str:
        """Rank every flight by how much it costs if it is late by delay_minutes.
        Use to find the most critical flights."""
        return json.dumps(t.run("rank_flights", {"delay_minutes": delay_minutes}),
                          default=float)

    def evaluate_hold(flight_id: str, delay_minutes: int,
                      hold_flight_id: str, hold_minutes: int) -> str:
        """Recovery option. flight_id is late by delay_minutes and some of its
        passengers miss hold_flight_id. Test holding hold_flight_id for
        hold_minutes (1-120) so they make it. Compares total cost with and
        without the hold, including the held flight's own knock-on delay."""
        return json.dumps(t.run("evaluate_hold", {
            "flight_id": flight_id, "delay_minutes": delay_minutes,
            "hold_flight_id": hold_flight_id, "hold_minutes": hold_minutes}),
            default=float)

    return [StructuredTool.from_function(f) for f in
            (list_flights, simulate_delay, cost_curve, rank_flights, evaluate_hold)]


def make_model(provider, model, api_key=None):
    """Create the chat model for the chosen provider.
    max_retries=1 and timeout=60: fail fast with a clear error instead of
    silently retrying for minutes when a free-tier limit is hit."""
    model = (model or "").strip()          # ignore stray spaces in the name
    api_key = (api_key or "").strip() or None
    if provider.startswith("Gemini"):
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=model, google_api_key=api_key,
                                      temperature=0, max_retries=1, timeout=60)
    if provider.startswith("Groq"):
        from langchain_groq import ChatGroq
        return ChatGroq(model=model, api_key=api_key, temperature=0,
                        max_retries=1, timeout=60)
    from langchain_ollama import ChatOllama
    return ChatOllama(model=model, temperature=0)


def _text(message):
    """Get plain text from a model reply (some providers return a list of parts)."""
    content = message.content
    if isinstance(content, list):
        content = "".join(p.get("text", "") if isinstance(p, dict) else str(p)
                          for p in content)
    return content.strip()


def ask(question, history, flights, passengers, provider, model, api_key=None,
        on_tool=None, max_steps=8, llm=None):
    """
    Send the question to the model and let it call tools until it can answer.
    history: earlier {"role": "user"/"assistant", "content": text} messages.
    on_tool: optional callback(name, args) to show progress in the dashboard.
    llm: optional ready-made model (used for testing).
    Returns the answer text.
    """
    tools = make_tools(flights, passengers)
    by_name = {tool.name: tool for tool in tools}
    llm = (llm or make_model(provider, model, api_key)).bind_tools(tools)

    messages = [SystemMessage(system_prompt())]
    for m in history:
        cls = HumanMessage if m["role"] == "user" else AIMessage
        messages.append(cls(m["content"]))
    messages.append(HumanMessage(question))

    for _ in range(max_steps):
        reply = llm.invoke(messages)
        messages.append(reply)
        if not reply.tool_calls:
            return _text(reply) or "(The model returned an empty answer.)"
        for call in reply.tool_calls:
            if on_tool:
                on_tool(call["name"], call["args"])
            tool = by_name.get(call["name"])
            output = (tool.invoke(call["args"]) if tool else
                      json.dumps({"error": f"Unknown tool {call['name']}"}))
            messages.append(ToolMessage(output, tool_call_id=call["id"]))

    return "I needed more steps than allowed to answer that. Try a narrower question."


if __name__ == "__main__":
    # Quick offline check of the tools (no API key needed)
    f = pd.read_csv("data/flights.csv")
    p = pd.read_csv("data/passengers.csv")
    for tool in make_tools(f, p):
        print(f"{tool.name}: {list(tool.args)}")
    print(make_tools(f, p)[1].invoke({"flight_id": "FL109", "delay_minutes": 60}))