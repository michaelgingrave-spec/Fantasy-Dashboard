# Fantasy-Dashboard

MJ's fantasy football / best-ball / DFS tool. Streamlit app, auto-deployed from `main`.

**Live app:** https://fantasy-dashboard-mrc9ttshbbb8kagam4em8b.streamlit.app/

## Run locally

```
pip install -r requirements.txt
streamlit run dashboard.py
```

## Screens

Sidebar nav (`dashboard.py` → `_SCREENS` / `dfs/screens.py`):

- **Best-ball / rankings / projections / roster optimizer** — the original tool.
- **🏈 DFS Optimizer** — DraftKings NFL Classic MILP lineup builder (PuLP): lock/exclude,
  stacking, multi-lineup + exposure caps, variance control. Optional "Apply regression lean".
- **🎯 DFS Matchup Edge**, **💡 DFS Suggestions**, **🔍 DFS Data Check**.
- **🔎 DFS Player Lookup** — per-player research: FantasyPoints projection vs regression lean,
  recent usage, FP-vs-xFP regression flag, historical coverage splits, opponent scheme
  tendencies. Descriptive; does not feed the optimizer.

## Weekly data

- **DK salaries** auto-pull from DraftKings' public JSON on a residential IP; the deployed app
  (datacenter IP, gets 403) falls back to `data/dfs/dk_snapshot/`, refreshed by
  `python -m dfs.refresh_dk --push`.
- **Projections**: drop the FantasyPoints weekly export at `data/projections.week{N}.csv`.
- **FantasyPoints Data Suite** tables live in `data/dfs/matchup/`. The scheduled task
  `fantasypoints-weekly-pull` (every Wednesday) grabs the previous NFL week via
  `matchup_model/weekly_pull.py` and pushes.

## Tests

```
pytest
```
