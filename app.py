"""
app.py
Interactive dashboard for the flight delay cost model.

Run with:   streamlit run app.py
"""

import os
import random

import altair as alt
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

import config
import fast_sim

# Read API keys from the .env file (if there is one) into the environment
load_dotenv()
# ---------------------------------------------------------------
# Page setup and colours
# ---------------------------------------------------------------
st.set_page_config(page_title="Flight Delay Cost Estimator", page_icon="✈️",
                   layout="wide")

BLUE, ORANGE, AQUA, YELLOW, MAGENTA, VIOLET = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6")
GREY, INK, MUTED, GRID = "#c9c8c2", "#0b0b0b", "#52514e", "#e6e5e1"

COST_NAMES = {
    "aircraft_delay": "Aircraft delay",
    "rebooking": "Rebooking",
    "meals": "Meals",
    "hotels": "Hotels",
    "cancellation": "Cancellation",
}
COST_COLOURS = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA]

# Default values of every editable assumption, read once from config.py
EDITABLE = {
    "COST_PER_DELAY_MINUTE": ("Aircraft cost per minute late (Rs)", 100),
    "REBOOKING_COST": ("Rebooking per passenger (Rs)", 100),
    "MEAL_COST": ("Meal per passenger (Rs)", 50),
    "HOTEL_COST": ("Hotel per passenger (Rs)", 100),
    "CANCELLATION_COMPENSATION": ("Cancellation compensation (Rs)", 500),
    "MEAL_THRESHOLD": ("Meals owed after (min)", 15),
    "CANCELLATION_THRESHOLD": ("Cancel flight after (min)", 15),
    "MIN_TURNAROUND": ("Minimum turnaround (min)", 5),
    "MIN_CONNECTION_TIME": ("Minimum connection time (min)", 5),
}
if "defaults" not in st.session_state:
    st.session_state.defaults = {k: getattr(config, k) for k in EDITABLE}


def lakh(x):
    return f"Rs {x / 100000:,.2f} lakh"


def clock(minutes):
    minutes = int(round(minutes))
    return f"{minutes // 60 % 24:02d}:{minutes % 60:02d}"


@st.cache_data
def load_data(version):
    """Read the network files (version changes when the network is rebuilt)."""
    return pd.read_csv("data/flights.csv"), pd.read_csv("data/passengers.csv")


@st.cache_data(show_spinner="Running Monte Carlo simulation...")
def monte_carlo(version, flight_id, runs, t_var, c_var, assumptions):
    """Cached cost curve. `assumptions` is part of the cache key."""
    for k, v in assumptions:
        setattr(config, k, v)
    flights, passengers = load_data(version)
    chain = fast_sim.PlaneChain(flights, passengers, flight_id)
    return fast_sim.cost_curve(chain, runs, t_var, c_var)


def rebuild_network(num_aircraft, seed):
    import generate_data
    config.NUM_AIRCRAFT = num_aircraft
    # Give the generator its own private random source, so nothing else
    # running inside Streamlit can disturb it (same result as the script)
    generate_data.random = random.Random(seed)
    flights = generate_data.generate_flights()
    passengers = generate_data.generate_passengers(flights)
    generate_data.random = random
    os.makedirs("data", exist_ok=True)
    flights.to_csv("data/flights.csv", index=False)
    passengers.to_csv("data/passengers.csv", index=False)
    st.session_state.data_version += 1
    st.session_state.pop("flight", None)


if "data_version" not in st.session_state:
    st.session_state.data_version = 0
if not os.path.exists("data/flights.csv"):
    rebuild_network(config.NUM_AIRCRAFT, config.RANDOM_SEED)

flights, passengers = load_data(st.session_state.data_version)

# ---------------------------------------------------------------
# Sidebar: every control
# ---------------------------------------------------------------
with st.sidebar:
    st.header("Scenario")
    labels = {r.flight_id: f"{r.flight_id}  ·  {r.origin}→{r.destination}  ·  "
                           f"{r.dep_time}  ·  {r.aircraft_id}"
              for r in flights.itertuples()}
    ids = list(labels)
    flight_id = st.selectbox("Delayed flight", ids, key="flight",
                             index=ids.index("FL109") if "FL109" in ids else 0,
                             format_func=labels.get)
    delay = st.slider("Arrival delay (minutes)", 0, config.MAX_DELAY, 60,
                      step=config.DELAY_STEP)

    st.header("Simulation")
    runs = st.select_slider("Monte Carlo runs per 5-minute step",
                            [100, 500, 1000, 5000], value=1000)
    t_var = st.slider("Turnaround uncertainty (± min, std dev)", 0, 15, 5)
    c_var = st.slider("Passenger connection uncertainty (± min)", 0, 20, 10)

    with st.expander("Cost assumptions"):
        if st.button("Reset to defaults", use_container_width=True):
            for k, v in st.session_state.defaults.items():
                st.session_state[f"cfg_{k}"] = v
        for k, (label, step) in EDITABLE.items():
            st.number_input(label, min_value=0, step=step, key=f"cfg_{k}",
                            value=st.session_state.defaults[k])

    with st.expander("Airline network"):
        n_planes = st.slider("Number of aircraft", 3, 12, len(flights["aircraft_id"].unique()))
        seed = st.number_input("Random seed", value=config.RANDOM_SEED, step=1)
        c1, c2 = st.columns(2)
        if c1.button("Rebuild", use_container_width=True):
            rebuild_network(n_planes, int(seed))
            st.rerun()
        if c2.button("Default", use_container_width=True):
            rebuild_network(6, 42)
            st.rerun()
        st.caption(f"{len(flights)} flights · {len(passengers):,} passengers · "
                   f"{passengers['connecting_flight'].notna().sum()} connecting")

# Apply the edited assumptions to the model
assumptions = tuple((k, st.session_state[f"cfg_{k}"]) for k in EDITABLE)
for k, v in assumptions:
    setattr(config, k, v)

curve = monte_carlo(st.session_state.data_version, flight_id, runs,
                    t_var, c_var, assumptions)
chain = fast_sim.PlaneChain(flights, passengers, flight_id)
fixed, late_flights, missed = fast_sim.scenario(chain, delay)
row = curve.loc[curve["delay_min"] == delay].iloc[0]
info = flights.set_index("flight_id")
sel = info.loc[flight_id]

# ---------------------------------------------------------------
# Header and headline numbers
# ---------------------------------------------------------------
st.title("Flight Delay Cost Estimator")
st.caption(f"**{flight_id}** {sel['origin']} → {sel['destination']}, scheduled "
           f"{sel['dep_time']}–{sel['arr_time']}, aircraft {sel['aircraft_id']}, "
           f"{int(sel['passengers'])} passengers · arriving **{delay} min late** · "
           f"{runs:,} simulated runs per step")

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Expected cost (Rs lakh)", f"{row['total'] / 1e5:,.2f}",
          f"+{row['extra_vs_previous'] / 1e5:,.2f} in last 5 min",
          delta_color="inverse")
k2.metric("Worst case (Rs lakh)", f"{row['worst_case'] / 1e5:,.2f}",
          help="Average of the worst 10% of simulated runs.")
k3.metric("Flights delayed", f"{row['flights_delayed']:.1f}",
          help="Average over the simulated runs, including the delayed flight.")
k4.metric("Missed connections", f"{row['missed']:.0f}",
          help="Average number of passengers who miss a connection.")
k5.metric("Flights cancelled", f"{row['flights_cancelled']:.1f}",
          help=f"A flight more than {config.CANCELLATION_THRESHOLD} min late "
               "is assumed cancelled.")

tabs = st.tabs(["📈 Cost curve", "🔗 What breaks", "🧾 Cost breakdown",
                "🏆 Compare flights", "🗺️ Network", "🤖 AI analyst", "ℹ️ Method"])

# ---------------------------------------------------------------
# Tab 1: cost curve
# ---------------------------------------------------------------
with tabs[0]:
    plot = curve.assign(
        avg_lakh=curve["total"] / 1e5, worst_lakh=curve["worst_case"] / 1e5,
        extra_lakh=curve["extra_vs_previous"] / 1e5,
        selected=curve["delay_min"] == delay)
    base = alt.Chart(plot).encode(
        x=alt.X("delay_min:Q", title="Arrival delay (minutes)",
                scale=alt.Scale(domain=[0, config.MAX_DELAY]),
                axis=alt.Axis(values=list(range(0, config.MAX_DELAY + 1, 30)))))
    tip = [alt.Tooltip("delay_min:Q", title="Delay (min)"),
           alt.Tooltip("avg_lakh:Q", title="Expected (lakh)", format=".2f"),
           alt.Tooltip("worst_lakh:Q", title="Worst 10% (lakh)", format=".2f"),
           alt.Tooltip("extra_lakh:Q", title="Extra this 5 min (lakh)", format=".2f")]
    hover = alt.selection_point(fields=["delay_min"], nearest=True,
                                on="pointerover", empty=False)

    band = base.mark_area(color=BLUE, opacity=0.18).encode(
        y=alt.Y("avg_lakh:Q", title="Total cost (Rs lakh)"), y2="worst_lakh:Q")
    line = base.mark_line(color=BLUE, strokeWidth=2).encode(y="avg_lakh:Q")
    points = base.mark_point(color=BLUE, size=60, filled=True).encode(
        y="avg_lakh:Q", opacity=alt.condition(hover, alt.value(1), alt.value(0)),
        tooltip=tip).add_params(hover)
    rule = alt.Chart(pd.DataFrame({"d": [delay]})).mark_rule(
        color=MUTED, strokeDash=[4, 4]).encode(x="d:Q")
    top = (band + line + points + rule).properties(
        height=320, title="Expected total cost (line) and worst case (shaded band)")

    bars = base.mark_bar(cornerRadiusTopLeft=2, cornerRadiusTopRight=2).encode(
        y=alt.Y("extra_lakh:Q", title="Extra cost (Rs lakh)"),
        color=alt.condition("datum.selected", alt.value(ORANGE), alt.value(BLUE)),
        tooltip=tip).properties(
        height=200, title="Extra cost of each 5-minute step (orange = selected delay)")

    st.altair_chart(alt.vconcat(top, bars).configure_view(stroke=None)
                    .configure_axis(gridColor=GRID, labelColor=MUTED,
                                    titleColor=MUTED, domainColor=GRID)
                    .configure_title(color=INK, anchor="start", fontSize=14),
                    use_container_width=True)

    big = curve.nlargest(3, "extra_vs_previous")
    st.markdown("**Biggest jumps:** " + " · ".join(
        f"+{lakh(r.extra_vs_previous)} at {r.delay_min} min"
        for r in big.sort_values("delay_min").itertuples()))

    table = curve.rename(columns={**COST_NAMES, "total": "Expected total",
                                  "worst_case": "Worst case (10%)",
                                  "extra_vs_previous": "Extra vs previous 5 min",
                                  "delay_min": "Delay (min)",
                                  "missed": "Missed connections",
                                  "flights_delayed": "Flights delayed",
                                  "flights_cancelled": "Flights cancelled"})
    c1, c2 = st.columns([1, 4])
    c1.download_button("⬇ Download table (CSV)", table.to_csv(index=False),
                       f"{flight_id}_cost_curve.csv", "text/csv",
                       use_container_width=True)
    with c2.expander("Show full 5-minute table"):
        st.dataframe(table.round(0), hide_index=True, use_container_width=True)

# ---------------------------------------------------------------
# Tab 2: what breaks (fixed rules, no randomness)
# ---------------------------------------------------------------
with tabs[1]:
    st.caption("This view uses the fixed rules (no randomness) so you can follow "
               "the cascade step by step.")
    if delay == 0:
        st.info("Move the delay slider above 0 to see the cascade.")
    else:
        story = []
        for i, f in enumerate(late_flights.itertuples()):
            r = info.loc[f.flight_id]
            status = "**cancelled**" if f.cancelled else f"{f.minutes_late:.0f} min late"
            if i == 0:
                story.append(f"1. **{f.flight_id}** ({r['origin']}→{r['destination']}) "
                             f"arrives {status}.")
            else:
                prev = late_flights.iloc[i - 1]
                slack = info.loc[prev["flight_id"], "turnaround_gap"] - config.MIN_TURNAROUND
                story.append(f"{i + 1}. The same aircraft then flies **{f.flight_id}** "
                             f"({r['origin']}→{r['destination']}). "
                             f"{slack:.0f} min of spare ground time absorbs part of it, "
                             f"so it leaves {status}.")
        st.markdown("\n".join(story))

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Late flights")
            show = late_flights.copy()
            show["Route"] = show["flight_id"].map(
                lambda f: f"{info.loc[f, 'origin']}→{info.loc[f, 'destination']}")
            show["Scheduled dep"] = show["flight_id"].map(info["dep_time"])
            show["New dep"] = [clock(info.loc[f, "sched_departure"] + m)
                               for f, m in zip(show["flight_id"], show["minutes_late"])]
            show["Status"] = show["cancelled"].map({True: "Cancelled", False: "Late"})
            st.dataframe(show[["flight_id", "Route", "Scheduled dep", "New dep",
                               "minutes_late", "Status"]].rename(
                columns={"flight_id": "Flight", "minutes_late": "Min late"}),
                hide_index=True, use_container_width=True)
        with c2:
            st.subheader("Missed connections")
            if missed.empty:
                st.success("No passenger misses a connection at this delay.")
            else:
                grp = (missed.groupby(["flight_id", "connecting_flight"])
                       .agg(passengers=("passenger_id", "size"),
                            planned_gap=("connection_gap", "first"),
                            time_left=("time_left", "first")).reset_index())
                st.dataframe(grp.rename(columns={
                    "flight_id": "Arriving on", "connecting_flight": "Missed",
                    "passengers": "Passengers", "planned_gap": "Planned gap (min)",
                    "time_left": "Time left (min)"}),
                    hide_index=True, use_container_width=True)
                st.caption(f"A passenger needs at least {config.MIN_CONNECTION_TIME} "
                           "min to change planes.")

# ---------------------------------------------------------------
# Tab 3: cost breakdown
# ---------------------------------------------------------------
with tabs[2]:
    c1, c2 = st.columns([2, 3])
    with c1:
        st.subheader(f"At {delay} min")
        parts = pd.DataFrame({"Category": list(COST_NAMES.values()),
                              "lakh": [row[k] / 1e5 for k in COST_NAMES]})
        st.altair_chart(alt.Chart(parts).mark_bar(
            cornerRadiusTopRight=4, cornerRadiusBottomRight=4).encode(
            y=alt.Y("Category:N", sort=None, title=None),
            x=alt.X("lakh:Q", title="Expected cost (Rs lakh)"),
            color=alt.Color("Category:N", scale=alt.Scale(
                domain=list(COST_NAMES.values()), range=COST_COLOURS), legend=None),
            tooltip=["Category", alt.Tooltip("lakh:Q", format=".2f", title="Rs lakh")])
            .properties(height=260).configure_view(stroke=None)
            .configure_axis(gridColor=GRID, labelColor=MUTED, titleColor=MUTED),
            use_container_width=True)
        st.dataframe(parts.assign(**{"Rs lakh": parts["lakh"].round(2)})
                     [["Category", "Rs lakh"]], hide_index=True,
                     use_container_width=True)
    with c2:
        st.subheader("How the mix changes with delay")
        long = curve.melt("delay_min", list(COST_NAMES), "cat", "cost")
        long["Category"] = long["cat"].map(COST_NAMES)
        long["lakh"] = long["cost"] / 1e5
        st.altair_chart(alt.Chart(long).mark_area().encode(
            x=alt.X("delay_min:Q", title="Arrival delay (minutes)"),
            y=alt.Y("lakh:Q", stack=True, title="Expected cost (Rs lakh)"),
            color=alt.Color("Category:N", scale=alt.Scale(
                domain=list(COST_NAMES.values()), range=COST_COLOURS),
                legend=alt.Legend(orient="top", title=None)),
            order=alt.Order("order:Q"),
            tooltip=["delay_min", "Category",
                     alt.Tooltip("lakh:Q", format=".2f", title="Rs lakh")])
            .transform_calculate(order="indexof(['Aircraft delay','Rebooking',"
                                       "'Meals','Hotels','Cancellation'], datum.Category)")
            .properties(height=360).configure_view(stroke=None)
            .configure_axis(gridColor=GRID, labelColor=MUTED, titleColor=MUTED),
            use_container_width=True)

# ---------------------------------------------------------------
# Tab 4: compare all flights at the selected delay
# ---------------------------------------------------------------
with tabs[3]:
    st.caption(f"Cost if **each** flight were {delay} min late (fixed rules). "
               "Flights early in an aircraft's day, or with many connecting "
               "passengers, are the most expensive to delay.")
    rank = []
    for f in flights["flight_id"]:
        r = fast_sim.scenario(fast_sim.PlaneChain(flights, passengers, f), delay)[0]
        rank.append({"Flight": f, "Route": f"{info.loc[f, 'origin']}→"
                                           f"{info.loc[f, 'destination']}",
                     "Departs": info.loc[f, "dep_time"],
                     "Rs lakh": r["total"] / 1e5,
                     "Flights delayed": int(r["flights_delayed"]),
                     "Missed connections": int(r["missed"])})
    rank = pd.DataFrame(rank).sort_values("Rs lakh", ascending=False)
    rank["selected"] = rank["Flight"] == flight_id
    st.altair_chart(alt.Chart(rank).mark_bar(
        cornerRadiusTopRight=4, cornerRadiusBottomRight=4).encode(
        y=alt.Y("Flight:N", sort="-x", title=None,
                axis=alt.Axis(labelOverlap=False)),
        x=alt.X("Rs lakh:Q", title=f"Cost of a {delay}-min delay (Rs lakh)"),
        color=alt.condition("datum.selected", alt.value(ORANGE), alt.value(BLUE)),
        tooltip=["Flight", "Route", "Departs",
                 alt.Tooltip("Rs lakh:Q", format=".2f"),
                 "Flights delayed", "Missed connections"])
        .properties(height=max(300, 26 * len(rank))).configure_view(stroke=None)
        .configure_axis(gridColor=GRID, labelColor=MUTED, titleColor=MUTED),
        use_container_width=True)
    st.download_button("⬇ Download ranking (CSV)",
                       rank.drop(columns="selected").to_csv(index=False),
                       f"flight_ranking_{delay}min.csv", "text/csv")

# ---------------------------------------------------------------
# Tab 5: network timeline
# ---------------------------------------------------------------
with tabs[4]:
    st.caption("Each row is one aircraft's day. Bars are flights at their "
               "scheduled times, coloured by what happens in this scenario.")
    late_map = dict(zip(late_flights.get("flight_id", []),
                        late_flights.get("minutes_late", [])))
    cancelled = set(late_flights.loc[late_flights["cancelled"], "flight_id"]) \
        if not late_flights.empty else set()
    hit_by_missed = set(missed["connecting_flight"]) if not missed.empty else set()

    def status(f):
        if f in cancelled:
            return "Cancelled"
        if f == flight_id and delay > 0:
            return "Delayed flight"
        if f in late_map:
            return "Knock-on delay"
        if f in hit_by_missed:
            return "Passengers missed it"
        return "Not affected"

    g = flights.assign(start=flights["sched_departure"] / 60,
                       end=flights["sched_arrival"] / 60,
                       Status=flights["flight_id"].map(status),
                       late=flights["flight_id"].map(late_map).fillna(0),
                       route=flights["origin"] + "→" + flights["destination"])
    domain = ["Delayed flight", "Knock-on delay", "Cancelled",
              "Passengers missed it", "Not affected"]
    bars = alt.Chart(g).mark_bar(height=22, cornerRadius=4).encode(
        x=alt.X("start:Q", title="Time of day (hour)",
                scale=alt.Scale(domain=[5, 25])),
        x2="end:Q",
        y=alt.Y("aircraft_id:N", title=None),
        color=alt.Color("Status:N", scale=alt.Scale(
            domain=domain, range=[BLUE, ORANGE, VIOLET, YELLOW, GREY]),
            legend=alt.Legend(orient="top", title=None)),
        tooltip=[alt.Tooltip("flight_id", title="Flight"), "route",
                 alt.Tooltip("dep_time", title="Departs"),
                 alt.Tooltip("arr_time", title="Arrives"),
                 alt.Tooltip("passengers", title="Passengers"),
                 alt.Tooltip("late:Q", title="Min late", format=".0f"),
                 "Status"])
    text = alt.Chart(g).mark_text(fontSize=10).encode(
        x=alt.X("mid:Q"), y="aircraft_id:N", text="flight_id",
        color=alt.condition("datum.Status == 'Not affected'",
                            alt.value(INK), alt.value("white"))).transform_calculate(
        mid="(datum.start + datum.end) / 2")
    st.altair_chart((bars + text).properties(height=60 * g["aircraft_id"].nunique())
                    .configure_view(stroke=None)
                    .configure_axis(gridColor=GRID, labelColor=MUTED, titleColor=MUTED),
                    use_container_width=True)
    with st.expander("Full schedule"):
        st.dataframe(flights.drop(columns=["sched_departure", "sched_arrival"]),
                     hide_index=True, use_container_width=True)

# ---------------------------------------------------------------
# Tab 6: AI analyst
# ---------------------------------------------------------------
with tabs[5]:
    import agent

    c1, c2, c3, c4 = st.columns([2, 3, 2, 1])
    provider = c1.selectbox("AI provider", list(agent.PROVIDERS))
    default_model, env_var = agent.PROVIDERS[provider]

    key = ""
    if env_var:
        key = os.environ.get(env_var, "")
        try:
            key = key or st.secrets.get(env_var, "")
        except Exception:
            pass
        typed = c2.text_input(
            f"{provider.split()[0]} API key", type="password",
            placeholder=("already set from " + env_var) if key
            else f"paste key here (or set {env_var})",
            help="Used only while the dashboard is open; never saved to a file.")
        key = typed or key
    else:
        c2.text_input("API key", "Not needed: runs on this computer",
                      disabled=True,
                      help="Install Ollama from ollama.com, then run: "
                           f"ollama pull {default_model}")
    model = c3.text_input("Model", default_model, key=f"model_{provider}")
    c4.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
    if c4.button("Clear chat", use_container_width=True):
        st.session_state.chat = []

    st.caption("The analyst answers by running the simulator (it never makes up "
               "numbers) and uses the cost assumptions set in the sidebar.")
    if "chat" not in st.session_state:
        st.session_state.chat = []

    suggestions = [
        f"Explain what happens if {flight_id} is {delay} minutes late.",
        f"At what delay does {flight_id} become really expensive, and why?",
        "Which 3 flights are most costly to delay by 60 minutes?",
        f"Should we hold a connecting flight to save passengers when "
        f"{flight_id} is {delay} min late?",
    ]
    st.markdown("**Try asking:**")
    cols = st.columns(2)
    clicked = None
    for i, s in enumerate(suggestions):
        if cols[i % 2].button(s, use_container_width=True, key=f"suggest_{i}"):
            clicked = s

    for m in st.session_state.chat:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    question = st.chat_input("Ask about delay costs...") or clicked
    if question:
        if env_var and not key:
            st.warning(f"Add your {provider.split()[0]} API key above to use the analyst.")
        else:
            with st.chat_message("user"):
                st.markdown(question)
            with st.chat_message("assistant"):
                status_box = st.status("Thinking...", expanded=False)

                def show_tool(name, args):
                    status_box.write(f"🔧 `{name}` {args}")

                try:
                    answer = agent.ask(question, st.session_state.chat, flights,
                                       passengers, provider, model, key,
                                       on_tool=show_tool)
                    status_box.update(label="Done", state="complete")
                except Exception as e:
                    status_box.update(label="Error", state="error")
                    msg = str(e).lower()
                    if any(w in msg for w in ("api key", "api_key", "401",
                                              "permission", "invalid")):
                        hint = "The API key looks wrong. Check you pasted all of it."
                    elif any(w in msg for w in ("429", "quota", "rate", "exhausted")):
                        hint = ("The free-tier limit was reached. Wait a minute, "
                                "or switch to another provider.")
                    elif "not found" in msg or "404" in msg:
                        hint = "That model name wasn't found. Check the Model box."
                    elif any(w in msg for w in ("connect", "refused", "11434")):
                        hint = ("Couldn't reach the AI service. Check your internet, "
                                "or for Ollama make sure the Ollama app is running.")
                    else:
                        hint = "Something went wrong talking to the AI service."
                    answer = f"Sorry, the request failed. {hint}\n\nDetails: `{e}`"
                st.markdown(answer)
            st.session_state.chat += [{"role": "user", "content": question},
                                      {"role": "assistant", "content": answer}]

# ---------------------------------------------------------------
# Tab 7: method
# ---------------------------------------------------------------
with tabs[6]:
    st.markdown(f"""
**What the model does**

1. **Delay propagation.** A delay passes down the same aircraft's later flights.
   At each stop, ground time beyond the {config.MIN_TURNAROUND}-minute minimum
   turnaround absorbs part of it.
2. **Missed connections.** A passenger misses their connection if their time to
   change planes falls below {config.MIN_CONNECTION_TIME} minutes. Passengers
   staying on the same aircraft can't miss it.
3. **Costs.** Aircraft delay per minute, rebooking per missed connection, meals
   after {config.MEAL_THRESHOLD} min, hotels for passengers stranded overnight,
   and cancellation (compensation + rebooking + hotel) after
   {config.CANCELLATION_THRESHOLD} min.
4. **Monte Carlo.** Each 5-minute step is simulated {runs:,} times with random
   turnaround times (±{t_var} min) and connection times (±{c_var} min). The
   expected cost is the average; the worst case is the average of the worst 10%.

**Sources for cost values**
- Aircraft cost per minute: Airlines for America 2024 (~$100/min), converted to rupees.
- Meals, hotels, cancellation compensation: DGCA CAR Section 3, Series M, Part IV.
- Rebooking, meal and hotel amounts: stated assumptions.

**Limitations:** synthetic network and passengers; crew schedules, seat
availability, aircraft types, refunds and lost revenue are not modelled.
""")
