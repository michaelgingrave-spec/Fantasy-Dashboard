# Millionaire Maker study

*Built 2026-11-23. Code: `dfs/history.py` (salaries and scores), `dfs/lineup_milp.py` (the exact lineup solver), `dfs/hindsight.py` (best lineups and their structure), `dfs/team_form.py` (opponent strength as of kickoff),
`dfs/mm_study.py` (the forward test and the evidence file), `dfs/mm_spend.py` (spending, price tiers, game environment), `dfs/millionaire.py` (the builder), screen `dfs/mm_view.py` ("🏆 DFS Millionaire Maker").
How to rebuild every table is at the top of `dfs/mm_study.py`.*

## The question

DraftKings' NFL Millionaire Maker runs on the Sunday main slate (the 1 PM and 4 PM ET games) and pays its top prize to roughly one lineup in a hundred thousand. You asked for a
tool built from history: take every past week's salaries and results, find the lineup that scored the most, look at what those lineups have in common, and build lineups to match.

The first half is mechanical. The second half has a trap, so this study has two parts:

1. **Hindsight.** The best possible lineup of each past week, found exactly (an integer program over the real salaries and the real scores, all of DraftKings' roster rules), and
   what the best lineup and the next 24 best of every week have in common, against what an ordinary optimizer builds.
2. **Forward test.** A trait can be common among the best lineups only because of which players happened to score that week (that is selection on the outcome). So each candidate
   rule is also tested the way it would be used: build lineups *before* kickoff with and without the rule, from salary-implied projections, and see how they then scored.

The builder's defaults come from what survived part 2, not from part 1.

## Data

| Period | Source | Weeks |
|---|---|---|
| 2026 weeks 1-11 | DraftKings' own player CSV for each week's main-slate draft group (the www endpoint still serves expired groups) + nflverse box scores through DraftKings scoring | 11 |
| 2014-2021 | The Rotoguru archive of DraftKings salaries and DraftKings points (the 2022-2025 pages are empty) | 127 |

138 main slates in all (a Sunday window of at least six games; 2016 week 16 had one). DraftKings points recomputed from box scores match the archive's recorded points exactly for 98.9% of
2021 player-weeks, so the two sources agree on scoring. **There is no free source of 2022-2025 salaries**, and none of past ownership or contest results, which limits what can be tested
(see the end).

The archive pages are third-party compiled data, so they and the full-lineup player lists stay on this machine (git-ignored); only the structural tables derived from them are in the repo.

## Method

**Best lineup.** For each week: maximise actual DraftKings points over the week's main-slate players subject to the $50,000 cap, QB 1, RB 2-3, WR 3-4, TE 1-2, one FLEX (RB/WR/TE) and DST 1, nine players in all
(`dfs/lineup_milp.py`, HiGHS through SciPy when installed, CBC through PuLP otherwise; checked against brute force and against the Optimizer's own solver). The next 24 lineups
each differ from every earlier one by at least three players.

**Structure.** Each lineup is described by what the study looks for: how many of the QB's own WR/TE and RB it holds, whether it has a bring-back from the QB's opponent, players per team and per
game, the number of teams and games, whether the defense faces its own offense, the salary shape (QB price, the $8,000+ players, the cheapest skill players, the FLEX position), and the game
environment (the QB's game total rank, favorite or not).

**Forward test.** For every week the pre-game expected points of a player is the points that salary has averaged *in the other seasons* (a quadratic in salary per position, so no week's outcome
leaks into its own projection). Under that projection every lineup that spends the cap has about the same expected score, so a rule can only help by changing how scores are distributed (a
stack's correlation fattens the upper tail), not by picking better players. Each lineup maximises its projection times a random jitter of up to ±35%, so a week's lineups differ the way a field's
do, and every rule set sees the same jitter. Each rule set builds 40 lineups in each of the 138 weeks.

*The measure.* A lineup is a top-5% (top-1%) lineup when it scores at least the 95th (99th) percentile of **every lineup built that week by every rule set pooled**, a stand-in for a field
with many kinds of lineups. A rule's *lift* is its rate over the unconstrained optimizer's rate (so 1.0 = no better than ordinary), and the interval comes from resampling whole weeks, since a
week's lineups all share one slate. The 2026 weeks are repeated with the FantasyPoints projections, the ones the builder really uses.

*A flaw in the first version of this test, and why it matters.* The first version measured each rule against the 95th percentile of the unconstrained optimizer's own 40 lineups. That made 25
of 27 rule sets look "better", opposite rules included (a QB at $7,000+ *and* a QB at $6,500 or less). The cause is the yardstick: the unconstrained lineups are all alike, so the spread of their
scores within a week is narrow, and any rule set whose players merely differ from theirs lands above or below that narrow band by the luck of which players scored that week. A bounded tail rate
is convex in that luck, so it averages above its nominal 5% for every rule set, helpful or not. Against the pooled field of all rule sets, which carries that between-composition spread inside it,
the rules that only change which players are used fall to 1.0, and only the rules that change how the scores correlate keep their lift. `test_a_rule_that_only_changes_which_players_are_used_shows_no_lift`
(`dfs/tests/test_mm_study.py`) reproduces it on synthetic data so it cannot come back.

## What "the game around the player" means here, and how the Data Suite fits

Two kinds of pre-game information are tested, both as of kickoff (nothing from the game itself):

* **Game totals.** The closing game total, the spread and each team's implied total (the nflverse schedule file's closing lines). A lineup can be built around the highest totals, around favorites
  or underdogs, or around teams with a high or low implied total.
* **Opponent defensive efficiency.** For each player, the quality of the unit he faces, measured four ways: DraftKings points the defense has allowed to the player's position per game, points
  allowed per game, EPA per dropback / per rush allowed (play-by-play), and, for a defense (DST), how weak the opposing offense is. Each is built only from games before the week and shrunk
  toward last season's finish (four games' worth of last season, pulled halfway to the league average), so week 1 is not a blank.

The FantasyPoints Data Suite's weekly team-defense table (EPA per dropback and per rush, pass rate and so on) covers 2022-2025. DraftKings salaries exist here for 2014-2021 and 2026, so the Data
Suite's own weeks and the salary weeks do not overlap, and a lineup test cannot be run on its table directly. What was done instead: the same measure was rebuilt from the nflverse play-by-play and checked
against the Data Suite table for 2022-2025 (correlation .998-.999 for EPA per dropback and .975-.983 for EPA per rush, team by team and week by week), and the rebuilt measure is what the 138-week
test uses. The Data Suite's season-level points-allowed exports cannot be used as a pre-game measure at all: a season total includes the week being predicted and every week after it.
EPA needs play-by-play, which is cached locally for 2021 onward only, so the EPA-based environment tests cover 2021 and 2026 (the points-allowed measure covers all 138 weeks).

## Findings (descriptive: no lineup involved yet)

### What each price tier has returned

Every main-slate player of the 138 study weeks (45,930 player-weeks), cut by position and price. `5x rate` is the share who scored at least five points per $1,000 of salary; `top finish` the share who
ranked among the week's top 3 QBs / 6 RBs / 9 WRs / 3 TEs / 3 DSTs. Intervals resample weeks.

| Position and price | Points | Points per $1k | 5x rate | Top finish | Share of pool | Share of top finishes |
|---|---|---|---|---|---|---|
| QB under $5,500 | 7.9 | 1.59 | 5.0% | 3.0% | 54% | 18% |
| QB $5,500-6,499 | 18.2 | 3.08 | 9.0% (7.3-10.8) | 13.0% | 26% | 36% |
| QB $6,500-7,499 | 20.3 | 2.97 | 6.5% (4.7-8.6) | 19.6% | 14% | 31% |
| QB $7,500 and up | 22.1 | 2.77 | 1.7% (0.4-3.3) | 26.0% | 5% | 15% |
| RB $4,500-5,999 | 10.6 | 2.08 | 5.9% | 11.6% | 19% | 32% |
| RB $6,000-7,499 | 15.1 | 2.30 | 3.6% | 23.3% | 8% | 26% |
| RB $7,500 and up | 21.2 | 2.54 | 4.5% | 45.1% | 4% | 24% |
| WR $4,500-5,999 | 11.2 | 2.20 | 6.1% | 12.8% | 16% | 26% |
| WR $6,000-7,499 | 15.4 | 2.34 | 6.0% | 26.4% | 9% | 31% |
| WR $7,500-8,499 | 18.4 | 2.35 | 4.0% | 36.1% | 3% | 13% |
| WR $8,500 and up | 20.3 | 2.29 | 1.6% (0.0-3.6) | 48.9% | 1% | 7% |
| TE $3,500-4,499 | 8.8 | 2.28 | 7.7% (5.8-9.9) | 11.8% | 9% | 25% |
| TE $4,500-5,499 | 10.7 | 2.21 | 5.6% | 16.0% | 5% | 19% |
| TE $5,500 and up | 14.0 | 2.19 | 3.6% (1.7-6.2) | 33.0% | 3% | 26% |
| DST under $2,600 | 5.4 | 2.35 | 13.7% | 7.6% | 33% | 19% |
| DST $2,600-2,999 | 6.9 | 2.52 | 14.2% | 12.8% | 29% | 29% |
| DST $3,000-3,399 | 8.2 | 2.61 | 12.4% | 15.7% | 21% | 25% |
| DST $3,400 and up | 9.0 | 2.46 | 9.9% (7.5-12.5) | 20.0% | 17% | 27% |

(The cheapest RB / WR / TE tiers, the players below $4,500 / $4,500 / $3,500, are mostly backups and return under 1.3 points per $1,000; they are in the evidence file, not here.)

* **QB.** Points per dollar peak between $5,500 and $7,500, and a QB at $7,500+ almost never returns five times his price (1.7%), but he finishes as a top-3 QB 26% of the time: the pay-up QB buys the
  top finish, not the value.
* **RB and WR.** Points per dollar rise a little with price at RB and are flat from $4,500 up at WR. The WR at $8,500+ is the worst tail value on the board (1.6% reach 5x).
* **TE.** The $3,500-4,499 tier is the best value at any position outside DST (7.7% reach 5x); the premium TE at $5,500+ is a top-3 TE a third of the time but reaches 5x 3.6%.
* **DST.** The cheap tiers reach 5x most often (14% vs 10% at $3,400+), and points per dollar are flat to slightly rising up to $3,400.

### Is the game already in the price?

For every starter-priced player, the points he scored beyond what his salary averages in other seasons, regressed on the game around him (per one standard deviation of the factor; t-statistics
clustered by week; 138 weeks; a positive number means the price did not fully include the factor).

| Position | Implied team total | Weaker defense for the position | Better own offense | Opponent allows more points |
|---|---|---|---|---|
| QB | -0.09 (t -0.6) | +0.33 (t +2.2) | **-0.68 (t -4.0)** | +0.15 (t +1.0) |
| RB | +0.12 (t +1.3) | +0.32 (t +3.5) | +0.05 (t +0.5) | +0.19 (t +2.2) |
| WR | +0.23 (t +2.2) | +0.25 (t +2.1) | +0.07 (t +0.6) | +0.08 (t +0.8) |
| TE | +0.38 (t +3.3) | +0.28 (t +2.4) | +0.13 (t +1.1) | +0.34 (t +2.8) |
| DST | +0.17 (t +1.5) | +0.25 (t +2.3) | +0.03 (t +0.3) | +0.29 (t +2.6) |

* A weaker opposing defense is worth about a quarter to a third of a point per standard deviation at every position: the same sign everywhere, which is more convincing than any single t.
* The implied total adds points for WR and TE, nothing for QB or RB.
* A QB on a *good* offense is **over**-priced by about two thirds of a point per standard deviation (t -4.0): the salary pays for the offense, and the QB's points do not follow it. Underdog and low-offense QBs beat their price.
* All of these are small. A lineup built entirely from opponents one standard deviation weaker would gain about 2-3 points on roughly 140, against a week-to-week lineup spread of about 26. They are real
  but cannot be seen through a lineup-level test unless they are larger than they look here; the forward test below measures what they are worth as rules.
* The EPA-based opponent measure (2021 and 2026 only, 565-2,777 player-weeks per position) is inconsistent in sign and mostly insignificant; with a fifth of the weeks it cannot confirm the points-allowed result.

## What the best lineups had in common (hindsight)

The best lineup of each of the 138 weeks scored 277.9 on average (median 276.5, from 229 to 373 across weeks); each week's 25 best average 267.7. The table gives the share of weeks the single best
lineup had each trait, with what an ordinary salary-only optimizer builds beside it. That comparison is fair for *structure* (the optimizer's stacks and game choices are accidents), and is left out
for how the cap is split, where that optimizer has habits of its own (it puts a TE in the FLEX 56% of the time). Intervals resample weeks.

| Trait | Best lineup of the week | Ordinary lineups |
|---|---|---|
| QB + at least one of his WR/TE | 60% (51-68) | 22% |
| QB + at least two of his WR/TE | 10% (6-15) | 2% |
| No QB teammate at all (WR, TE or RB) | 31% (24-39) | 69% |
| A bring-back (an opposing WR/TE) | 29% (22-36) | 20% |
| Stack (1+) with a bring-back | 14% (9-20) | 4% |
| Stack (2+) with a bring-back | 2% (0-4) | 0% |
| 3+ players from the QB's game | 41% (33-49) | 16% |
| At most 3 offensive players from one team | 99% (98-100) | 100% |
| 6 to 8 different teams | 91% (86-95) | 81% |
| Nobody in the lineup faces the lineup's own defense | 91% (86-95) | 70% |

* **Stacking is the one structural trait that stands out.** The QB's own pass catcher was in the best lineup 2.7 times as often as in an ordinary one, a stack plus a bring-back 3.8 times as often. The counts for
  the larger stacks are small (the best lineup was a QB + 2 + bring-back in 2% of weeks), which is why the hindsight table cannot say how big a stack to build; the forward test can.
* "At most 3 from one team" carries no information: ordinary lineups obey it too. 4+ from one team was the best lineup in 1% of weeks.
* **The game around the players barely separates the winners from ordinary lineups.** Two or more players from a top-3-total game: 80% of the best lineups, 79% of ordinary ones. The QB's team a favorite: 60% vs 67%.
  Nobody facing a top-8 defense for his position: 15% vs 12%. A QB facing one of the 10 weakest defenses: 34% vs 33%. The largest gap is four or more players from 25+ implied totals (46% vs 35%).
  This is what the regression above predicts when the price already carries most of the game.

### How the best lineups split the $50,000

Mean spend at each slot of the single best lineup of every week (an interval from resampling weeks), the mean over each week's 25 best, the middle 80% of weeks, and the mean by era. No ordinary-lineup column
(see above): whether a split *pays* is the forward test's question, not this table's.

| Slot | Best lineup (95% interval) | Best 25 | Middle 80% of weeks | 2014-17 | 2018-21 | 2026 (11 weeks) |
|---|---|---|---|---|---|---|
| QB | $6,369 (6,210-6,530) | $6,339 | 5,200-7,700 | $6,506 | $6,276 | $6,082 |
| RBs (all) | $14,267 (13,581-14,917) | $14,302 | 9,670-19,860 | $13,360 | $14,855 | $16,318 |
| WRs (all) | $20,031 (19,364-20,664) | $20,127 | 15,800-25,030 | $20,382 | $19,774 | $19,409 |
| TEs (all) | $5,102 (4,720-5,506) | $4,868 | 2,600-8,230 | $5,009 | $5,252 | $4,809 |
| DST | $2,988 (2,910-3,066) | $2,999 | 2,370-3,700 | $3,042 | $2,942 | $2,936 |
| Left unspent | $1,242 (968-1,559) | $1,365 | 0-3,620 | $1,702 | $902 | $445 |

The shares: a QB at about 13% of the cap, RBs 29%, WRs 40%, the TE 10%, the DST 6%. The weeks differ a great deal (the QB from $5,200 to $7,700 in the middle 80%), so no single price is "the" winner's. Two drifts
across eras: the RB share has risen (about $13,400, then $14,900, then $16,300) and the money left unspent has fallen ($1,700, then $900, then $400), which fits DraftKings' pricing having tightened (the 2026
figures are 11 weeks).

## What survives the forward test

Every rule set below builds 40 lineups in each of 138 weeks from the salary-implied projection, before kickoff, and is scored on what happened. `Lift` is the share of its lineups in a week's top 5% (top 1%) of
the pooled field, over the unconstrained optimizer's share (4.8% and 0.83% over the 138 weeks; 4.5% and 0.45% in the 11 FantasyPoints weeks). An interval that stays above 1.0 is a rule that helped. The right-hand
columns repeat the 2026 weeks with the FantasyPoints projections (the ones the builder uses); 11 weeks is a short sample (the unconstrained optimizer lands only 3 of 655 lineups in the top 1% there), so those columns say
which way the effect points, not how big it is.

### Stacking and team structure

| Rule set | Top 5%: 138 weeks | Top 1%: 138 weeks | Top 5%: 2026, FantasyPoints | Top 1%: 2026, FantasyPoints |
|---|---|---|---|---|
| QB + 2 + bring-back, nobody facing the defense | **1.36x** (1.19-1.58) | **2.3x** (1.7-3.4) | 1.73x (1.11-2.69) | 4.7x (1.3-19.0) |
| QB + 2 + bring-back, at most 3 from a team | 1.33x (1.15-1.54) | 2.3x (1.7-3.3) | 1.70x (1.11-2.56) | 4.7x (1.2-19.6) |
| QB + 2 + bring-back, 5+ games, 6+ teams | 1.30x (1.14-1.50) | 2.3x (1.7-3.3) | 1.53x (0.94-2.39) | 3.3x (0.8-14.0) |
| QB + 2 pass catchers + a bring-back | 1.30x (1.13-1.50) | 2.3x (1.7-3.3) | 1.67x (1.11-2.52) | 4.7x (1.2-20.0) |
| QB + 2 pass catchers | 1.17x (1.03-1.34) | 1.5x (1.0-2.2) | 1.50x (0.97-2.27) | 5.0x (1.7-18.0) |
| QB + 1 pass catcher | 1.13x (1.01-1.25) | 1.4x (1.0-1.9) | 1.00x (0.80-1.26) | 3.7x (1.3-13.0) |
| QB + 1 pass catcher + a bring-back | 1.13x (0.99-1.28) | 1.4x (1.0-2.1) | 1.13x (0.74-1.67) | 3.7x (0.8-15.0) |
| At most 3 from one team (no stack) | 1.00x (1.00-1.01) | 1.0x (1.0-1.1) | 1.00x (1.00-1.00) | 1.0x (1.0-1.0) |

* **This is the result.** Lift rises with how much the lineup's scores are tied to one game: QB + 1 pass catcher 1.13x, QB + 2 1.17x, adding a bring-back 1.30x, and a defense that does not face any of the lineup's
  offense 1.36x. At the extreme tail (the top 1%) it is larger, 2.3x, which is what correlation should do to a tail, and the Millionaire Maker's top prize is far deeper in the tail than 1%. That is an
  argument from the shape of the curve; this sample cannot measure the 1-in-100,000 level directly.
* The bring-back helps **with a two-man stack, not without one**: with QB + 1 it adds nothing (1.13x either way), with QB + 2 it takes the top-1% lift from 1.5x to 2.3x.
* The cap on players per team (at most 3), a minimum number of games and teams, and the defense rule are all within noise of each other (1.30-1.36x): they are safe, not additive.
* "At most 3 from one team" by itself is 1.00x: the optimizer never wanted more.

### Spending by position

| Rule set | Top 5%: 138 weeks | Top 1%: 138 weeks | Top 5%: 2026, FantasyPoints | Top 1%: 2026, FantasyPoints |
|---|---|---|---|---|
| Two or more $8,000+ players | 1.14x (0.91-1.38) | 1.7x (1.2-2.5) | 0.86x (0.48-1.29) | 3.3x (1.0-13.0) |
| FLEX is a second TE | 1.04x (0.94-1.13) | 1.1x (0.8-1.4) | 1.03x (0.74-1.35) | 1.7x (1.0-3.0) |
| QB at $7,000 or more | 1.03x (0.85-1.22) | 1.3x (0.9-1.8) | 1.12x (0.43-2.21) | 0.7x (0.0-4.0) |
| QB at $6,500 or less | 1.00x (0.90-1.10) | 1.1x (0.8-1.4) | 1.00x (0.77-1.20) | 0.7x (0.0-3.0) |
| TE at $4,500 or less | 1.00x (0.90-1.10) | 1.4x (1.0-1.8) | 1.00x (0.78-1.31) | 2.0x (1.0-6.0) |
| $17,000 or less on WRs (WR-light) | 0.99x (0.87-1.12) | 1.2x (0.8-1.6) | 1.17x (0.70-1.78) | 2.0x (1.1-6.0) |
| $12,000 or less on RBs (RB-light) | 0.99x (0.85-1.14) | 1.2x (0.8-1.8) | 0.97x (0.52-1.46) | 3.3x (0.7-12.0) |
| At most one $8,000+ player | 0.97x (0.93-1.00) | 0.9x (0.8-1.0) | 1.00x (0.69-1.27) | 1.0x (0.0-4.0) |
| FLEX is a third RB | 0.96x (0.83-1.11) | 1.2x (0.8-1.6) | 0.97x (0.70-1.26) | 1.3x (0.1-6.0) |
| No $8,000+ player | 0.96x (0.84-1.09) | 1.0x (0.8-1.3) | 0.90x (0.45-1.42) | 1.0x (0.0-5.0) |
| TE at $5,500 or more | 0.95x (0.75-1.22) | 1.2x (0.8-1.8) | 0.53x (0.23-0.92) | 1.0x (0.0-5.0) |
| $21,000+ on WRs (WR-heavy) | 0.95x (0.86-1.05) | 1.2x (0.9-1.6) | 1.17x (0.94-1.44) | 1.3x (0.2-5.0) |
| FLEX is a fourth WR | 0.92x (0.83-1.02) | 1.2x (0.9-1.6) | 1.03x (0.73-1.37) | 1.7x (0.5-6.0) |
| $16,000+ on RBs (RB-heavy) | 0.92x (0.81-1.04) | 1.1x (0.8-1.6) | 1.00x (0.78-1.24) | 1.3x (0.1-6.0) |
| One TE (the FLEX is an RB or WR) | 0.92x (0.84-1.02) | 1.2x (0.9-1.6) | 0.93x (0.74-1.11) | 1.7x (0.7-6.0) |

* **No way of splitting the cap helped at the top 5%.** A cheap or dear QB, a cheap or dear TE, RB-heavy or RB-light, WR-heavy or WR-light, no / one / two or more $8,000+ players, and the FLEX as an RB, WR or a
  second TE all sit at about 1.0, with intervals that straddle it.
* **The hindsight pattern does not pay when it is forced.** The best lineups look cheap at QB and TE and light on stars, and the "shape" rule built from exactly that (a cheaper QB, at most one $8,000+ player, one
  TE) is 0.98x (0.87-1.09). Added to the best stack it *lowers* the lift (QB + 2 + bring-back 1.30x, with the shape 1.22x; at the top 1%, 2.3x and 1.8x). A week's best lineup is the one in which some cheap players
  happened to score; that is known only afterwards, and a rule that forces cheap players beforehand gets the cheap players without the scoring.
* Three top-1% flags (two or more $8,000+ players 1.7x, a TE at $4,500 or less 1.4x, at most one $8,000+ player 0.9x, the first and last being the same finding from two sides) are about what chance gives across 15
  rules and two thresholds. The 2026 estimates point the same way (3.3x and 2.0x) with intervals (1.0-13 and 1.0-6.0) too wide to confirm. A hint that two stars thicken the extreme tail, not a result.
* One 2026 rule is flagged *worse* (a TE at $5,500+, 0.53x on 11 weeks); it is 0.95x on the 138 weeks, so it is not repeated.
* The descriptive price tiers above still describe where points per dollar are best (a QB at $5,500-7,499, a TE at $3,500-4,499, DSTs under $3,400, no WR at $8,500+) as a way to *choose between* players of
  a position, but a rule that forces those tiers shows no lift.

### Game totals and opponent defenses

| Rule set | Top 5%: 138 weeks | Top 1%: 138 weeks | Top 5%: 2026, FantasyPoints | Top 1%: 2026, FantasyPoints |
|---|---|---|---|---|
| At most 1 player faces one of the 8 toughest defenses for his position | 1.12x (1.01-1.23) | 1.4x (1.1-1.9) | 1.13x (0.92-1.36) | 1.3x (1.0-3.0) |
| 5+ players face one of the 12 weakest defenses for their position | 1.12x (0.98-1.27) | 1.4x (0.9-1.9) | 1.50x (1.03-2.15) | 2.7x (0.8-9.0) |
| 5+ players from teams with an implied total of 24+ | 1.08x (0.93-1.22) | 1.3x (1.0-1.8) | 0.80x (0.48-1.10) | 0.7x (0.0-2.0) |
| At most 1 player from a team with an implied total of 20 or less | 1.03x (0.94-1.12) | 1.2x (0.9-1.6) | 1.07x (0.81-1.32) | 1.7x (0.5-5.0) |
| QB's team is an underdog | 0.99x (0.89-1.09) | 1.2x (0.9-1.7) | 1.00x (0.77-1.28) | 1.7x (0.4-7.0) |
| DST's team is a favorite by a point or more | 0.98x (0.89-1.08) | 1.0x (0.7-1.3) | 0.93x (0.76-1.16) | 1.0x (0.5-3.0) |
| QB's game among the 5 highest totals | 0.96x (0.86-1.06) | 1.0x (0.8-1.2) | 1.17x (0.86-1.55) | 1.3x (0.0-4.0) |
| QB faces one of the 10 weakest defenses against QBs | 0.95x (0.84-1.06) | 1.1x (0.8-1.5) | 0.90x (0.68-1.17) | 2.0x (0.7-6.0) |
| DST faces one of the 10 weakest offenses | 0.94x (0.83-1.06) | 0.9x (0.6-1.2) | 0.93x (0.78-1.13) | 1.7x (0.6-6.0) |

* **Game totals add nothing as a lineup rule.** Building around the five highest totals, around underdogs or favorites, around implied totals of 24+ or away from 20 and below: 0.96-1.08x, all within noise. This matches
  the hindsight table (the best lineups are no more concentrated in the high-total games than ordinary lineups) and the regression (only WR and TE get a small, +0.2 to +0.4 points per standard deviation, bonus from a higher total).
* **The opposing defense is a mild, consistent tilt, not a structure rule.** At most one player against one of the eight toughest defenses for his position: 1.12x (1.01-1.23), 1.4x in the top 1%, and 1.13x on the 2026 weeks;
  five or more players against the twelve weakest: 1.12x on the 138 weeks and 1.50x (1.03-2.15) on 2026. The regression finds the same sign at every position (+0.25 to +0.33 points per standard deviation). But one of
  nine rule sets clearing 1.0 by a hair (the lower end of its interval is 1.01) is a result chance gives in about one set of nine tests in five, and a rule on QBs facing weak pass defenses (0.95x) or a DST facing
  weak offenses (0.94x) shows nothing.
* **Neither adds to the stack.** QB + 2 + bring-back with five or more players from implied totals of 24+: 1.33x; with five or more against weak defenses: 1.33x; the stack alone: 1.30x. On 2026 the three are
  1.73x, 1.90x and 1.67x. The matchup conditions ride on the stack and do not improve it measurably.

### Combinations

| Rule set | Top 5%: 138 weeks | Top 1%: 138 weeks | Top 5%: 2026, FantasyPoints | Top 1%: 2026, FantasyPoints |
|---|---|---|---|---|
| QB + 2 + bring-back, 5+ players from implied totals of 24+ | 1.33x (1.14-1.56) | 2.4x (1.7-3.3) | 1.73x (1.11-2.64) | 5.3x (1.2-23.0) |
| QB + 2 + bring-back, 5+ players against weak defenses | 1.33x (1.15-1.52) | 2.2x (1.5-3.1) | 1.90x (1.13-3.08) | 5.3x (0.9-24.0) |
| QB + 2 + bring-back + the salary shape | 1.22x (1.04-1.45) | 1.8x (1.3-2.6) | 1.40x (0.86-2.24) | 4.0x (1.6-14.0) |
| QB + 1 + bring-back + the salary shape | 1.11x (0.94-1.31) | 1.4x (0.9-2.0) | 1.07x (0.65-1.64) | 4.0x (1.4-13.0) |
| Cheaper QB + at most one $8,000+ + one TE | 0.98x (0.87-1.09) | 1.3x (0.9-1.8) | 0.90x (0.53-1.21) | 1.3x (0.2-6.0) |

### Does it hold in every era?

The lift of the main stack rules by era (the 138 weeks split; 2026 with the salary-implied projection like the rest, so the eras are comparable). Intervals resample weeks.

| Rule set | 2014-17 (65 weeks) | 2018-21 (62 weeks) | 2026 (11 weeks) | All 138 |
|---|---|---|---|---|
| QB + 2 + bring-back, nobody facing the defense | 1.40x (1.17-1.65) | 1.27x (0.99-1.61) | 1.76x (1.24-2.75) | 1.36x (1.19-1.58) |
| QB + 2 + bring-back | 1.35x (1.11-1.61) | 1.19x (0.93-1.50) | 1.76x (1.18-2.73) | 1.30x (1.13-1.50) |
| QB + 2 pass catchers | 1.18x (0.96-1.42) | 1.16x (0.94-1.42) | 1.18x (0.71-1.92) | 1.17x (1.03-1.34) |
| QB + 1 pass catcher | 1.08x (0.90-1.28) | 1.15x (0.99-1.33) | 1.41x (0.95-2.15) | 1.13x (1.01-1.25) |
| At most 1 player against a top-8 defense | 1.14x (1.00-1.30) | 1.08x (0.92-1.25) | 1.29x (1.00-1.64) | 1.12x (1.01-1.23) |

The main stack is above 1.0 in each era (the bring-back version's interval clears 1.0 in 2014-17 and 2026, and 2018-21 is the weakest at 1.19-1.27x with an interval reaching just below 1.0). The estimates for 2026,
whether with salary-implied or FantasyPoints projections, are the highest, but they rest on 11 weeks and their intervals contain every other era's estimate.

### Does the lift depend on how alike a week's lineups are?

The jitter (a random nudge of up to ±35% to every player's value, different for every lineup) sets how much a week's 40 lineups differ from each other, so it could in principle manufacture a lift. The four
main rule sets were rerun at ±15% (lineups alike) and ±60% (lineups very different), 40 lineups per rule set in each of the 138 weeks, each in a pooled field of those four rule sets. (At ±35% the same four-set field
reproduces the 39-set field's number for the main stack, 1.30x, so the choice of field is not what moves the numbers.)

| Jitter | QB + 1 + bring-back | QB + 2 pass catchers | QB + 2 + bring-back | QB + 2 + bring-back, top 1% |
|---|---|---|---|---|
| ±15% | 1.23x (1.05-1.42) | 1.20x (1.03-1.42) | 1.38x (1.19-1.60) | 1.7x (1.2-2.3) |
| ±35% (the main study) | 1.14x (0.99-1.33) | 1.19x (1.04-1.38) | 1.30x (1.12-1.51) | 2.2x (1.6-3.1) |
| ±60% | 1.22x (1.06-1.41) | 1.16x (1.01-1.35) | 1.41x (1.23-1.62) | 1.9x (1.4-2.7) |

The main stack's top-5% lift is 1.30-1.41x at every setting with an interval above 1.0 each time, and its top-1% lift 1.7-2.2x. The result does not depend on how alike the week's lineups are.

## How the builder uses this

The builder (`dfs/millionaire.py`, the screen `dfs/mm_view.py`) builds lineups from the FantasyPoints projections (a blend of median and ceiling, a "ceiling lean" of 0.5 by default) with these defaults, each of
which the table above supports or is neutral to:

| Setting | Default | Why |
|---|---|---|
| QB stack | QB + 2 of his WR/TE | 1.17x alone, 1.30x with the bring-back: the best-supported structure |
| Bring-back | on (1 opposing WR/TE) | only pays on top of the two-man stack (1.17x to 1.30x; top 1% 1.5x to 2.3x) |
| Defense | nobody in the lineup faces it | 1.36x, the best of the stack variants (within noise of 1.30x); 91% of the best lineups did this |
| Most from one team | 3 | 99% of the best lineups held 3 or fewer; measured effect none, harmless |
| Min salary | $49,000 | the forward test's own condition; a floor, not a target (the best lineups left $1,242 on average) |
| Lineups / exposure / uniqueness | 20 / 40% / 3 players | portfolio hygiene, not tested |

Not forced, on purpose: how the cap is split, the game total, and the opposing defense. None of them beat 1.0 by more than noise, so the builder leaves them to the projections. The ownership rules are an option
only: no historical ownership exists to test them (the screen's ownership figure is a heuristic).

## Limits

* **The contest is not the test.** The Millionaire Maker's top prize goes to about one entry in a hundred thousand; this sample has 138 weeks. The top 5% and top 1% of a pooled field of optimizer lineups is the
  closest proxy it allows, and the 2.3x at the top 1% is an argument for what happens further out, not a measurement of it.
* **The salary-implied projection has no skill by construction**, so the forward test isolates structure. It does not say whether the FantasyPoints projections beat it. The 2026 rerun with them (11 weeks) is the
  only check on the real inputs and is short.
* **There is no field data.** The lift is against an ordinary optimizer, which stacks 22% of the time with a teammate. If, as is often said, a large share of a real field already stacks, the stack's edge *over the field* is
  smaller than the lift shown here. Ownership, contest payouts and the field's own lineups are not available, so this study cannot say.
* **Salaries exist for 2014-21 and 2026 only** (138 main slates; nine archive weeks with an incomplete player list were skipped). DraftKings' pricing and the league's scoring have changed since; the stack lift is
  above 1.0 in each era, which is the best available answer.
* **Multiple testing.** About forty rule sets at two thresholds were tested (some 60 tests among the thirty that are not stacks), so a few "better" and "worse" flags are expected from chance alone (about three at a
  95% interval). A flag that is not a stack, and does not repeat on 2026, is not evidence. That is the reason this report treats the star-count and opponent-defense results as hints.
* **Opponent EPA exists for 2021 and 2026 only** (play-by-play is cached from 2021); the points-allowed measure covers all 138 weeks. The Data Suite's own team-defense tables (2022-2025) do not overlap the salary
  weeks, so the measure was rebuilt and validated against them (correlation .975-.999) rather than used directly.
* **The Rotoguru $0 salaries** (270 skill players, 2014-20, which DraftKings did not price) were free roster spots in the first version of the test and are now dropped; every result above is from after that fix.
  The first version of the forward-test yardstick was also biased (see Method); the results above use the pooled field.
