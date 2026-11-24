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
- **🎮 DFS Game Projections** — pick a game and see our projection for every player in it: FantasyPoints'
  number next to ours (DK points), the projected stat line by position (pass/rush/receiving), DK salary,
  the injury tag and the game's spread / implied totals. By default only players FantasyPoints also
  projects (above 0) are shown, so depth players stay out; a min-FP box, "hide players ruled Out" and a
  position picker narrow it further.
- **🏆 DFS Millionaire Maker** — a lineup builder for DraftKings' NFL Millionaire Maker (the Sunday main slate),
  built from a study of 138 past main slates: the best possible lineup of each week (exact MILP over the real
  salaries and scores), what those lineups share, and a forward test of which of those traits pay when lineups are
  built before kickoff. The screen shows the evidence tables (stacks, spending by position, game totals and
  opposing defenses) next to the controls. Code: `dfs/history.py` (salaries + scores 2014-21 from the Rotoguru
  archive, 2026 from DraftKings), `dfs/hindsight.py`, `dfs/team_form.py`, `dfs/mm_study.py`, `dfs/mm_spend.py`,
  `dfs/millionaire.py`, `dfs/mm_view.py`; write-up in `dfs/mm_report.md`. Rebuild the study with
  `python -m dfs.hindsight`, then `python -m dfs.mm_study --grid main --n 40` and `--grid env --n 40`, the same two
  with `--source fp --n 60` for the 2026 weeks, and `python -m dfs.mm_study --evidence` to write the file the screen
  reads (about an hour and a half on 6 cores in all; the steps are listed at the top of `dfs/mm_study.py`).
- **🎯 DFS Matchup Edge**, **💡 DFS Suggestions**, **🔍 DFS Data Check**.
- **🔎 DFS Player Lookup** — per-player research: FantasyPoints projection vs regression lean,
  recent usage, FP-vs-xFP regression flag, historical coverage splits, opponent scheme
  tendencies. Descriptive; does not feed the optimizer.

## Weekly data

- **DK salaries** auto-pull from DraftKings' public JSON on a residential IP; the deployed app
  (datacenter IP, gets 403) falls back to `data/dfs/dk_snapshot/`, refreshed by
  `python -m dfs.refresh_dk --push`.
- **Projections**: drop the FantasyPoints weekly export at `data/projections.week{N}.csv`
  (or let `weekly` below install the newest `projections.weekly*.csv` from Downloads).
- **The whole weekly job in one command** (`dfs/weekly.py`): `weekly` (= `weekly.cmd`, i.e.
  `python -m dfs.weekly --execute --push`) refreshes the box scores, buys the finished week's DraftKings /
  FanDuel prop lines from the Odds API (5 markets, 50 credits a game, closing snapshot, never below
  `--floor`, nothing bought twice), logs them to `data/dfs/props/line_history.csv`, grades every ungraded
  line, prints the week's results, installs the newest FantasyPoints export, refreshes the version-exam and
  forward-test reports (`--fast` skips them) and commits + pushes exactly the files it touched.
  `python -m dfs.weekly` alone is a dry run that shows the plan and the credit cost and changes nothing.
  The one manual step is the FantasyPoints download (signed-in browser: fantasypoints.com/nfl/projections/weekly
  -> Download CSV; the next week's file is posted Tuesday night). Run it after Monday night's game.
- **FantasyPoints Data Suite** tables live in `data/dfs/matchup/`. The scheduled task
  `fantasypoints-weekly-pull` (every Wednesday) grabs the previous NFL week via
  `matchup_model/weekly_pull.py` and pushes.

## Tests

```
pytest
```
