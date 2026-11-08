# Research index

Model implementations live in ../src: my_model (XGBoost), combo_model (local/hierarchical/neural ensemble), hgb_upgrade (history variants), win_model (win/amount decomposition), lag_model (changing lag effects), pdf_model (history geometry/peer changes), calibrated_model (episode-cross-fitted calibration).

Use ../lab.py from the repository root. Tools in tools/ compare predictions or reproduce historical team-blend experiments. Scratch predictions belong in ../runs. Never fit and evaluate calibration on the same episodes when claiming held-out performance. Fold priors must use training episodes only.
