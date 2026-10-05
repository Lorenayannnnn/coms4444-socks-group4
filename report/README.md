# Tournament report analysis

The report includes `report/analysis/results_section.tex` and `report/analysis/appendix_section.tex` from `report/main.tex`.
All figures use plain Matplotlib, with PDF files for LaTeX and PNG copies for inspection.

Reproduce the figures and numerical summaries from the repository root:

```sh
python report/analyze_tournament.py
latexmk -pdf -outdir=/private/tmp/socks-report-build report/main.tex
cp /private/tmp/socks-report-build/main.pdf report/group4_report.pdf
```

The Python environment needs numpy, pandas, and matplotlib. The only input is
`results/7_raw_runs.csv`; the supplied overall standings are used to check the
roommate-weighted scores. Raw inputs are preserved.

## Aggregation

- A simulation is identified by `sim_id`. A roommate is identified by `(sim_id, seat)`.
- Scores first average members of the same group within a simulation, then average
  simulations. Identical group members are not treated as independent repetitions.
- Household spending and first failed replacement purchase are counted once per simulation.
- Mismatch embarrassment subtracts `65536 * sockless_days` from total embarrassment.
- Daily rates divide by the run duration. Mismatch per dressed day pools the group
  simulation totals and dressed days before dividing, weighting by dressed days.
- A group sockless run means at least one member of that group had a sockless day.
- Error bars in the overall figure span the three seed-specific aggregates; they
  are descriptive ranges, not confidence intervals.
- Four-versus-five pairs match roster, household type, group role, budget, duration,
  drawer multiplier and seed. Actual capacity differs, so this is not a fixed-capacity
  estimate of the extra draw's effect.
- Five-person roster comparisons retain the same four groups and replace the fifth.
  Overlapping comparisons are not independent experiments.

## Outputs

1. `01_overall`: total scores and mismatch per dressed day.
2. `02_budget`: Group 4 homogeneous households by budget, duration and size.
3. `03_exhaustion`: positive finite-budget homogeneous households by replacement pressure.
4. `04_neighbors`: Group 4 with each other group in two-person and nine-person households.
5. `05_capacity`: capacity effects within fixed budget-duration settings.
6. `06_unit`: paired four-versus-five changes by household size.
7. `07_externalities`: changes for the retained four groups when Group 4 enters a roster.
8. `08_households`: Group 4's mean rank by household type and each type's share of the gap to the leader.
9. `09_discard_ablation`: the replay without voluntary discards (needs `analysis/discard_ablation.csv`).

## Discard ablation

`discard_ablation.py` replays every finite-budget simulation in which Group 4 held most of the
seats, plus the nine-person households with a single Group 4 player, with Group 4's discard
allowance fixed at zero. Everything else (roster, seat order, drawer, budget, duration, seed)
matches the tournament. Run it from a checkout of the course repository's final `main`, so the
simulator and the other groups' players are the tournament versions:

```sh
python /path/to/report/discard_ablation.py /path/to/results/7_raw_runs.csv \
    /path/to/report/analysis/discard_ablation.csv --workers 8
```

It also replays 40 random simulations with the unmodified player and stops unless every seat
reproduces its tournament score. `analyze_tournament.py` checks those rows again before using
the file.

The CSV summaries and `metrics.json` contain the calculations behind report claims.
The script checks identifiers, repetition counts, repeated household fields,
nonnegative mismatch penalties, pair counts, and reproduction of supplied overall means.
