# Safe-cracking scoring analysis

This project adjusts peer scores for rater harshness and adjusts cracking and resistance scores for the difficulty of the safes and attackers each team faced. It keeps the competition's published weights:

- Judges: 45%
- Peers: 20%
- Cracking: 25%
- Resistance: 10%

## Run the analysis

### Streamlit app

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open the displayed local URL, upload an `.xlsx` scoring workbook, and use the tabs for:

- **Decision**: raw total, weighted fair components, final score, model range, status, and downloads.
- **Audit**: raw/adjusted components and attempt-level evidence.
- **Peer fairness**: peer adjustments and harsh/generous rater tendencies.
- **Cracking fairness**: schedule difficulty, crack uncertainty, and invalid-attempt burden.
- **Data & method**: model checks, assumptions, and limitations.

The app defaults to the workspace spreadsheet for demonstration. Uncheck **Use workspace spreadsheet** to require an upload.

The Decision tab has three component display modes:

- **Scores out of 100**: comparable component scores before weighting.
- **Weighted contributions**: points out of 45, 20, 25 and 10; these add directly to the fair score.
- **Component ranks**: rank within each component, where 1 is best and tied scores share a rank.

In Peer fairness, **Adjusted peer score** describes the team's own safe after correcting for who rated it. **How this team marked others** describes the marks that team gave to other safes. The latter is used to correct ratings received from harsh or generous scorers; it does not directly reward or penalise the scorer's own safe.

Peer charts always use the same row order so teams align horizontally. The shared ordering can be changed between **Team code**, **Raw peer score**, **Adjusted peer score**, and **Scoring behaviour (harshest first)**.

### Notebook

1. Save and close `scoring_spreadsheet_2026.xlsx` in Excel so formula values are current.
2. Open `analyse_scoring.ipynb` in VS Code.
3. Select the **Python (.venv safe-cracking)** kernel.
4. Choose **Run All**.

The notebook's **Decision table for judges** is the recommended judging view; the following **Audit table** explains every adjustment. The run also regenerates:

- `scoring_fairness_analysis.xlsx`: compact `Decision table`, detailed `Audit detail`, attempt and peer-rater audits, and method notes.
- `scoring_fairness_report.md`: concise text report.

The command-line equivalent is:

```powershell
.\.venv\Scripts\python.exe .\analyse_scoring.py
```

## Method

Recorded outcomes are interpreted as:

- `1`: successful crack.
- `0`: valid unsuccessful crack.
- `0.5`: invalid or unreliable attempt, excluded from model fitting.
- Blank: no attempt.

A ridge-regularised Rasch logistic model estimates attacker ability and safe difficulty simultaneously. The regularisation strength is selected by deterministic five-fold cross-validation. Adjusted cracking is performance against an average-difficulty safe; adjusted resistance is performance against an average-strength attacker.

A separate ridge-regularised additive model estimates each safe's peer-rated quality and each scoring team's tendency to be harsh or generous. Its regularisation strength is selected by five-fold held-out-rating RMSE. Adjusted peer score means the predicted score from an average rater. Positive rater generosity means a team tends to award higher marks after accounting for the safes it scored.

The approximate 95% model ranges are conditional on the selected regularisation strengths. They cover model uncertainty in adjusted peer, cracking, and resistance scores. They do not include variation in judge marking, regularisation-selection uncertainty, or all possible model misspecification.

The **winner-review** flag means a team's modelled difference from the provisional leader includes zero at the approximate 95% level. Judges should moderate that group together using the published rubric rather than treating decimal places as decisive.

For future competitions, publish both the peer-rater adjustment and opponent-difficulty adjustment before scoring begins.

## Recreate the environment

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m ipykernel install --user --name safe-cracking --display-name "Python (.venv safe-cracking)"
```

## Checks

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The script validates required sheets, weights, team identifiers, crack-result values, self-matches, usable evidence, cached formula values, and model convergence before producing results.