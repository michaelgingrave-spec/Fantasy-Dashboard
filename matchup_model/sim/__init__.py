"""Game simulator for same-game parlays.

`engine.simulate` plays one game many times from the model's own projections (means) and a small set of calibrated noise parameters; `sgp` prices a parlay from those
paths; `inputs` builds the two `TeamSpec`s from the app's tables; `history`, `calibrate` and `validate` fit the noise on past seasons and test it on lines it has not seen.
"""
