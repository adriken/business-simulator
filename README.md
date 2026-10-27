# Business Simulator

A general-purpose Monte Carlo business simulator. Describe any business as building blocks, give every uncertain number as a range, and the engine plays out thousands of possible futures to show the odds: typical profit, how bad a bad year gets, the chance of a loss, when cash could run out, and which assumptions matter most.

**Live app:** _link added after deployment_

## Why

A spreadsheet forecast gives one number ("you'll make 1.7M this year") built on guesses about guests, prices and costs. This simulator treats those guesses honestly. You enter "between 70 and 200 room-nights a month, probably 130", and it tells you, for example, that the median year makes 1.76M KES, one year in ten makes less than 123k, and 8% of years lose money.

## What it shows

For any business:

- Profit and cash distributions with P10, median and P90
- Probability of making a loss
- Break-even point (share of sales volume needed to cover overheads)
- When cash runs out, and how often
- A tornado chart of which inputs move profit most
- A side-by-side comparison of two plans, e.g. raise prices vs add capacity

## How it works

1. **Describe the business** in a YAML file: revenue streams (units × price), capacity limits, losses, fixed / per-unit / one-off costs, and a time period (days, weeks or months).
2. **Every number can be a range.** `4500` is fixed, `[3500, 5500]` is uniform, and `[3500, 4500, 5500]` is triangular (low, most likely, high).
3. **Sample once per scenario.** Each scenario draws one value for every input, so a scenario is one consistent version of the business. Variation within a year comes from seasonality.
4. **Run the period-by-period arithmetic** for all scenarios at once, as NumPy arrays of shape (scenarios × periods):

   ```
   demand   = base demand × seasonality      (or active customers, for subscriptions)
   produced = demand, scaled down if a shared capacity pool is full
   sold     = produced × (1 − loss)
   revenue  = sold × price
   profit   = revenue − fixed − per-unit − one-off costs
   cash     = starting cash + running total of profit
   ```

5. **Summarise across scenarios** into percentiles and probabilities.

Plans are compared with **common random numbers**: each named input has its own random stream seeded from `(seed, input name)`, so both plans face identical futures and any difference comes from the plan, not from luck.

## A business is a YAML file

No code is needed for a new business. This is the shape of `templates/guesthouse.yaml`:

```yaml
name: Small guesthouse
period: month
horizon: 12
currency: KES
starting_cash: 250000

capacity:
  room_nights: 240

streams:
  - name: rooms
    demand: [70, 130, 200]
    price: [3500, 4500, 5500]
    capacity: room_nights
    loss: [0.02, 0.04, 0.08]
    seasonality: [1.2, 0.8, 0.8, 0.9, 1.0, 0.9, 1.1, 1.3, 1.0, 0.9, 1.0, 1.4]

costs:
  fixed:
    - {name: staff, amount: [120000, 135000, 150000]}
  per_unit:
    - {name: breakfast_and_amenities, stream: rooms, basis: sold, amount: [600, 750, 900]}
  one_off:
    - {name: repainting, period: 3, amount: [40000, 50000, 60000]}

plans:
  - name: Raise price 10%
    changes:
      - {path: streams.rooms.price, op: multiply, value: 1.10}
      - {path: streams.rooms.demand, op: multiply, value: [0.85, 0.92, 0.98]}
```

Other building blocks: several streams sharing one capacity pool (`capacity_per_unit`), per-unit costs charged on `produced` or `sold`, and recurring streams with `churn` for subscriptions. Files are validated with pydantic, so a typo or an impossible range is reported with its exact location.

## Run it locally

Requires Python 3.12.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows (macOS/Linux: source .venv/bin/activate)
python -m pip install -r requirements.txt

python -m streamlit run app.py                     # the dashboard
python -m simulator templates/guesthouse.yaml      # a text report
python -m pytest -v                                # the tests
```

## Project structure

```
app.py                     Streamlit dashboard
templates/                 business definitions (YAML)
simulator/
    distributions.py       ranges -> samples (inverse CDF), seeded Sampler
    model.py               YAML schema and validation (pydantic)
    engine.py              the vectorised simulation
    metrics.py             percentiles, loss probability, break-even, runway
    sensitivity.py         tornado analysis
    compare.py             plan comparison with common random numbers
    editing.py             the dashboard's editable inputs table
    formatting.py          display helpers
    __main__.py            command-line report
tests/                     pytest suite
```

## Design decisions

- **Inverse-CDF sampling.** Uniform draws are generated separately from the distributions, which is what makes common random numbers possible.
- **Vectorised.** No Python loop over scenarios: 20,000 scenarios × 12 months run in about 0.1 s. Only subscriptions loop over time, because each period depends on the last.
- **Validated input.** YAML is read with `yaml.safe_load` and checked by pydantic; unknown keys, wrong order (`low > high`) and broken references are errors, never silent fixes.
- **One-at-a-time sensitivity.** The tornado chart swings one input while holding the rest at their likely values. It is fast and easy to read, but it can't show interactions between inputs (for example, extra capacity only matters when demand is high).

## Stack

Python 3.12 · NumPy · pandas · pydantic · PyYAML · Streamlit · Plotly · pytest

Template figures are illustrative examples, not real business data.
