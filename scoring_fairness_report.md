# Safe-cracking scoring fairness analysis

## Recommendation

Treat `0.5` as missing evidence, not half a crack. Adjust peer scores for rater harshness, and adjust cracking and resistance for opponent difficulty, while retaining the published 45% / 20% / 25% / 10% weights.

## Data checks

- 18 teams, 162 recorded matchups.
- 123 valid binary outcomes and 39 invalid/unreliable `0.5` records.
- Overall valid crack rate: 55.3%.
- Cross-validated regularisation strength: 1.0.
- Cross-validated log loss: 0.610; intercept-only baseline: 0.688.
- Peer regularisation strength: 4.0; cross-validated RMSE: 0.144; mean-only baseline: 0.164.

## Adjusted ranking

| Rank | Code | Team / safe | Raw total | Judges /45 | Peers /20 | Crack /25 | Resist /10 | Fair score | Model range | Status |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 3A | 3A Tonbridge: Cooke's Arcade | 78.19 | 37.80 | 14.68 | 18.47 | 6.98 | 77.93 | 70.4 - 85.4 | Leader - moderate |
| 2 | 1E | 1E Simon Langton Grammar: Dead or Alive | 79.54 | 36.60 | 16.00 | 19.55 | 4.70 | 76.85 | 69.5 - 84.2 | Winner review |
| 3 | 1D | 1D Dulwich: Rice to the Occasion | 75.76 | 34.20 | 13.47 | 20.47 | 5.15 | 73.29 | 66.5 - 80.1 | Winner review |
| 4 | 3F | 3F Eltham: FZN | 73.33 | 42.00 | 16.19 | 7.37 | 7.08 | 72.65 | 65.4 - 79.9 | Winner review |
| 5 | 1B | 1B Westminster: Operation Ovenbreak | 69.17 | 30.00 | 13.72 | 18.21 | 5.59 | 67.53 | 59.6 - 75.5 | Outside review group |
| 6 | 2B | 2B Sevenoaks: The Oaks Heist | 69.27 | 30.60 | 14.45 | 15.21 | 6.59 | 66.84 | 58.8 - 74.9 | Outside review group |
| 7 | 2A | 2A St Paul's: Lableak | 64.30 | 37.80 | 13.01 | 13.55 | 1.90 | 66.26 | 57.4 - 75.2 | Outside review group |
| 8 | 3C | 3C HABS: The Maxwell Demon | 64.22 | 27.00 | 12.62 | 17.96 | 5.27 | 62.84 | 55.0 - 70.7 | Outside review group |
| 9 | 3E | 3E Eltham: MAXimum Security | 60.01 | 29.40 | 13.22 | 15.13 | 2.77 | 60.52 | 51.6 - 69.4 | Outside review group |
| 10 | 2E | 2E Emanuel: BlackHole | 55.76 | 28.20 | 12.34 | 10.23 | 6.43 | 57.20 | 47.9 - 66.5 | Outside review group |
| 11 | 1A | 1A Westminster:  Da Vinici Heist | 55.92 | 25.20 | 11.86 | 16.17 | 3.60 | 56.83 | 48.2 - 65.5 | Outside review group |
| 12 | 1C | 1C Dulwich: Crack or be Kraken | 56.38 | 27.60 | 13.15 | 11.84 | 3.95 | 56.54 | 48.2 - 64.9 | Outside review group |
| 13 | 2F | 2F Emanuel: Dixon's Safe | 50.72 | 30.00 | 12.52 | 10.01 | 2.36 | 54.90 | 46.6 - 63.2 | Outside review group |
| 14 | 3D | 3D Ibstock Place: Simon | 53.36 | 25.80 | 10.76 | 14.05 | 3.72 | 54.33 | 45.5 - 63.1 | Outside review group |
| 15 | 1F | 1F Simon Langton Grammar: WeHeartPhysics (underwater) | 51.48 | 25.20 | 12.09 | 13.05 | 2.37 | 52.71 | 43.5 - 61.9 | Outside review group |
| 16 | 2D | 2D Sutton:  ZipZap | 48.24 | 22.80 | 12.66 | 9.35 | 4.37 | 49.18 | 40.0 - 58.3 | Outside review group |
| 17 | 3B | 3B Tonbridge: Circuit Circus | 38.87 | 16.20 | 10.80 | 10.59 | 4.30 | 41.89 | 33.5 - 50.2 | Outside review group |
| 18 | 2C | 2C LEH: Fast & Curious | 39.01 | 14.40 | 13.35 | 12.08 | 1.78 | 41.61 | 32.6 - 50.6 | Outside review group |

## Schedule and uncertainty

The estimated easiest schedule was 3C (63.9% expected crack rate for an average attacker); the hardest was 2F (48.5%).

Largest ranking changes after adjustment:

- 1C: 10 to 12 (-2 places).
- 2E: 12 to 10 (+2 places).
- 2F: 15 to 13 (+2 places).
- 1B: 6 to 5 (+1 places).
- 1E: 1 to 2 (-1 places).

Winner-review group: **3A, 1E, 1D, 3F**. Their modelled differences from the provisional leader include zero at the approximate 95% level.

Intervals use conditional approximations at the selected regularisation strengths. They include model uncertainty in adjusted peer, cracking, and resistance scores; judge marks are treated as fixed. They do not account for model-selection uncertainty or the fact that the leader was selected from the same data.

Because each team has few valid attempts, use the fair ordering as evidence and moderate the winner-review group with the published judging rubric. Do not use extra decimal places as a tie-break.

## Competition rule for future rounds

1. Record outcomes as `1 = cracked`, `0 = valid failure`, and blank or a separate status for broken/unsolvable attempts.
2. Publish before competition that peer-rater and opponent-adjusted estimates will be used.
3. Balance schedules using prior/pilot safe difficulty where possible; statistical adjustment is a fallback, not a substitute for good allocation.
4. Require a minimum number of valid attempts or reduce the crack-component weight when data loss is high.
5. Resolve near ties with a declared moderation rule, not extra decimal places from the model.
