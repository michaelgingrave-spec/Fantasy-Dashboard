@echo off
REM The whole weekly job in one command (see dfs/weekly.py): box scores, the finished week's prop lines
REM (bought, logged, graded), the newest FantasyPoints projections from Downloads, the forward-test
REM reports, then commit + push. Extra options are passed on, e.g.  weekly 7   weekly --fast
REM A dry run that spends nothing and changes nothing:  python -m dfs.weekly
cd /d "%~dp0"
python -m dfs.weekly --execute --push %*
