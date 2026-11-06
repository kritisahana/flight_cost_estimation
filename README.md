# Flight Delay Cost Estimator

Estimates the financial cost of a flight delay in 5-minute increments,
including knock-on effects on the same aircraft's later flights and on
connecting passengers. Uses a rule-based network simulation with Monte Carlo
randomness, an interactive dashboard, and an optional AI analyst.

## Quick start

```bash
pip3 install -r requirements.txt
python3 generate_data.py          # build the synthetic airline (once)
python3 -m streamlit run app.py    # open the dashboard
```

The dashboard opens in your browser at http://localhost:8501.

## Files

| File | What it does |
|---|---|
| `config.py` | All settings and cost assumptions, with sources |
| `generate_data.py` | Builds a synthetic one-day airline network into `data/` |
| `propagation.py` | Rules for how a delay spreads and who misses connections |
| `costs.py` | Turns what broke into rupees (5 cost categories) |
| `run_analysis.py` | Command-line version: 5-minute steps, 1,000 random runs, chart |
| `fast_sim.py` | Fast version of the same rules for the dashboard (checked identical) |
| `agent.py` | AI analyst that answers questions by calling the simulator |
| `app.py` | The Streamlit dashboard |

## Dashboard

- **Sidebar:** choose the delayed flight and delay, set Monte Carlo runs and
  uncertainty, edit any cost assumption, rebuild the airline network.
- **Cost curve:** expected cost and worst case at every 5 minutes, plus the
  extra cost of each 5-minute step. Download as CSV.
- **What breaks:** the cascade step by step: late flights and missed connections.
- **Cost breakdown:** cost by category at the chosen delay, and how the mix changes.
- **Compare flights:** which flights are most expensive to delay.
- **Network:** each aircraft's day, coloured by what the delay affects.
- **AI analyst:** ask questions in plain English (free Gemini or Groq key, or Ollama with no key).
- **Method:** how the model works, sources and limitations.

## AI analyst

The analyst is built with LangChain and has five tools: simulate a delay, cost
curve, rank flights, evaluate holding a connecting flight, and list flights. It
gets every number from the simulator rather than calculating itself.

Choose the AI provider in the dashboard:

| Provider | Default model | Key |
|---|---|---|
| Gemini (recommended) | `gemini-2.5-flash` | Free from aistudio.google.com |
| Groq | `llama-3.3-70b-versatile` | Free from console.groq.com |
| Ollama | `qwen3:8b` | None: runs on your computer (install from ollama.com) |

Paste the key into the dashboard, or set it before starting:

```bash
export GOOGLE_API_KEY="your-key"      # for Gemini
export GROQ_API_KEY="your-key"        # for Groq
python3 -m streamlit run app.py
```

Never commit your key to GitHub.

## Check the fast engine

```bash
python3 fast_sim.py   # prints "All scenarios match."
```
