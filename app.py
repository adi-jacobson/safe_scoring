from __future__ import annotations

import tempfile
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from analyse_scoring import SOURCE, analyse, write_outputs


st.set_page_config(
    page_title="Safe scoring review",
    page_icon=":material/lock:",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root { --ink: #17242b; --muted: #52636b; --teal: #007a78; --amber: #d97941; --line: #dbe2e5; --panel: #ffffff; --sidebar: #edf2f1; --canvas: #f4f7f6; --note: #fff8ef; --method: #eef8f7; }
    .stApp { background: linear-gradient(180deg, var(--canvas) 0, var(--panel) 280px); color: var(--ink); }
    [data-testid="stHeader"] { background: color-mix(in srgb, var(--canvas) 92%, transparent); }
    [data-testid="stSidebar"] { background: var(--sidebar); border-right: 1px solid var(--line); }
    h1, h2, h3 { font-family: Georgia, 'Times New Roman', serif; letter-spacing: 0; }
    h1 { font-size: 2.2rem !important; }
    div[data-testid="stMetric"] { background: var(--panel); border: 1px solid var(--line); padding: 14px; }
    div[data-testid="stMetric"] label { color: var(--muted); }
    .status-note { border-left: 4px solid var(--amber); background: var(--note); padding: 12px 16px; }
    .method-note { border-left: 4px solid var(--teal); background: var(--method); padding: 12px 16px; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def run_analysis(file_bytes: bytes, file_name: str) -> tuple[list[dict], list[dict], dict, bytes, str]:
    """Analyse uploaded bytes and return model results plus downloadable outputs."""
    safe_name = Path(file_name).name
    directory = Path(tempfile.mkdtemp(prefix="safe-scoring-"))
    try:
        source_path = directory / safe_name
        workbook_path = directory / "scoring_fairness_analysis.xlsx"
        report_path = directory / "scoring_fairness_report.md"
        source_path.write_bytes(file_bytes)
        rows, attempts, diagnostics = analyse(source_path)
        write_outputs(rows, attempts, diagnostics, workbook_path, report_path)
        return rows, attempts, diagnostics, workbook_path.read_bytes(), report_path.read_text(encoding="utf-8")
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def decision_frame(
    ranking: pd.DataFrame,
    weights: dict[str, float],
    display_mode: str,
) -> pd.DataFrame:
    decision = ranking.copy()
    if display_mode == "Scores out of 100":
        component_columns = ["Judges /100", "Peers /100", "Cracking /100", "Resistance /100"]
        decision["Judges /100"] = 100 * decision["judge"]
        decision["Peers /100"] = 100 * decision["adjusted_peer"]
        decision["Cracking /100"] = 100 * decision["adjusted_attack"]
        decision["Resistance /100"] = 100 * decision["adjusted_resistance"]
    elif display_mode == "Weighted contributions":
        component_columns = ["Judges /45", "Peers /20", "Cracking /25", "Resistance /10"]
        decision["Judges /45"] = 100 * decision["judge"] * weights["judge"]
        decision["Peers /20"] = 100 * decision["adjusted_peer"] * weights["peer"]
        decision["Cracking /25"] = 100 * decision["adjusted_attack"] * weights["attack"]
        decision["Resistance /10"] = 100 * decision["adjusted_resistance"] * weights["resistance"]
    else:
        component_columns = ["Judges rank", "Peers rank", "Cracking rank", "Resistance rank"]
        rank_sources = {
            "Judges rank": "judge",
            "Peers rank": "adjusted_peer",
            "Cracking rank": "adjusted_attack",
            "Resistance rank": "adjusted_resistance",
        }
        for column, source in rank_sources.items():
            decision[column] = decision[source].rank(method="min", ascending=False).astype(int)
    decision["Model range"] = decision.apply(
        lambda row: f"{row['adjusted_total_low']:.1f} - {row['adjusted_total_high']:.1f}", axis=1
    )
    decision["Status"] = np.select(
        [decision["adjusted_rank"].eq(1), decision["winner_review"]],
        ["Provisional leader - moderate", "Winner review"],
        default="Outside winner-review group",
    )
    return decision[
        ["adjusted_rank", "code", "team", "current_total", *component_columns, "adjusted_total", "Model range", "Status"]
    ].rename(
        columns={
            "adjusted_rank": "Rank",
            "code": "Code",
            "team": "Team / safe",
            "current_total": "Raw total",
            "adjusted_total": "Fair score",
        }
    )


def audit_frame(ranking: pd.DataFrame) -> pd.DataFrame:
    audit = ranking.copy()
    percentage_columns = [
        "judge", "peer", "adjusted_peer", "peer_rater_effect", "current_attack",
        "adjusted_attack", "current_resistance", "adjusted_resistance",
    ]
    audit[percentage_columns] = 100 * audit[percentage_columns]
    audit["Peer range"] = audit.apply(
        lambda row: f"{100 * row['peer_low']:.1f} - {100 * row['peer_high']:.1f}", axis=1
    )
    audit["Crack range"] = audit.apply(
        lambda row: f"{100 * row['attack_low']:.1f} - {100 * row['attack_high']:.1f}", axis=1
    )
    audit["Resistance range"] = audit.apply(
        lambda row: f"{100 * row['resistance_low']:.1f} - {100 * row['resistance_high']:.1f}", axis=1
    )
    audit["Crack evidence"] = audit.apply(
        lambda row: f"{int(row['valid_attacks'])} attack / {int(row['valid_defences'])} defence", axis=1
    )
    return audit[
        [
            "code", "team", "current_rank", "adjusted_rank", "rank_change", "judge", "peer",
            "adjusted_peer", "Peer range", "peer_rater_effect", "current_attack", "adjusted_attack",
            "Crack range", "current_resistance", "adjusted_resistance", "Resistance range",
            "Crack evidence", "current_total", "adjusted_total",
        ]
    ].rename(
        columns={
            "code": "Code", "team": "Team / safe", "current_rank": "Raw rank",
            "adjusted_rank": "Fair rank", "rank_change": "Places gained", "judge": "Judge (%)",
            "peer": "Raw peer (%)", "adjusted_peer": "Adjusted peer (%)",
            "peer_rater_effect": "How this team marked others (pp)", "current_attack": "Raw crack (%)",
            "adjusted_attack": "Adjusted crack (%)", "current_resistance": "Raw resistance (%)",
            "adjusted_resistance": "Adjusted resistance (%)", "current_total": "Raw total",
            "adjusted_total": "Fair score",
        }
    )


def style_axes(axis: plt.Axes) -> None:
    axis.figure.patch.set_facecolor(colours["panel"])
    axis.set_facecolor(colours["panel"])
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.grid(axis="x", color=colours["grid"], linewidth=0.8)
    axis.set_axisbelow(True)
    axis.tick_params(colors=colours["ink"])
    axis.xaxis.label.set_color(colours["ink"])
    axis.yaxis.label.set_color(colours["ink"])
    axis.title.set_color(colours["ink"])
    if axis.legend_ is not None:
        for text in axis.legend_.get_texts():
            text.set_color(colours["ink"])


with st.sidebar:
    st.header("Workbook")
    uploaded_file = st.file_uploader("Upload scoring spreadsheet", type=["xlsx"])
    use_example = st.checkbox("Use workspace spreadsheet", value=uploaded_file is None)
    theme = st.selectbox("Appearance", ["Light", "Dark"], help="Choose the app colours for this session.")
    st.caption("Save the workbook in Excel first so cached formula values are current.")
    st.divider()
    st.markdown("**Outcome coding**")
    st.caption("1 = cracked · 0 = valid failure · 0.5 = invalid/unreliable · blank = no attempt")

if theme == "Dark":
    colours = {
        "ink": "#edf4f2", "muted": "#b9c8c6", "teal": "#55c7bd", "amber": "#f0a36f",
        "line": "#40504f", "panel": "#1d292b", "sidebar": "#162123", "canvas": "#11191b",
        "note": "#3a2d24", "method": "#173735", "grid": "#40504f", "raw": "#8fa4ad",
        "blue": "#63a9d1", "zero": "#b9c8c6",
    }
else:
    colours = {
        "ink": "#17242b", "muted": "#52636b", "teal": "#007a78", "amber": "#d97941",
        "line": "#dbe2e5", "panel": "#ffffff", "sidebar": "#edf2f1", "canvas": "#f4f7f6",
        "note": "#fff8ef", "method": "#eef8f7", "grid": "#dbe2e5", "raw": "#a7b6c2",
        "blue": "#2878a5", "zero": "#333333",
    }
st.markdown(
    f"""
    <style>
    :root {{ --ink: {colours['ink']}; --muted: {colours['muted']}; --teal: {colours['teal']}; --amber: {colours['amber']}; --line: {colours['line']}; --panel: {colours['panel']}; --sidebar: {colours['sidebar']}; --canvas: {colours['canvas']}; --note: {colours['note']}; --method: {colours['method']}; }}
    </style>
    """,
    unsafe_allow_html=True,
)

if uploaded_file is not None:
    source_bytes = uploaded_file.getvalue()
    source_name = uploaded_file.name
elif use_example and SOURCE.exists():
    source_bytes = SOURCE.read_bytes()
    source_name = SOURCE.name
else:
    st.title("Safe scoring review")
    st.info("Upload an `.xlsx` scoring workbook in the sidebar to begin.")
    st.stop()

st.title("Safe scoring review")
st.caption(f"Source: {source_name} · peer scores adjusted for rater tendency · cracking adjusted for schedule difficulty")

try:
    with st.spinner("Validating workbook and fitting fairness models..."):
        rows, attempts, diagnostics, workbook_bytes, report_text = run_analysis(source_bytes, source_name)
except Exception as error:
    st.error("The workbook could not be analysed.")
    st.exception(error)
    st.stop()

ranking = pd.DataFrame(rows).sort_values("adjusted_rank").reset_index(drop=True)
attempt_log = pd.DataFrame(attempts)
audit = audit_frame(ranking)
review_codes = ranking.loc[ranking["winner_review"], "code"].tolist()

metric_columns = st.columns(4)
metric_columns[0].metric("Teams", diagnostics["team_count"])
metric_columns[1].metric("Valid cracks", diagnostics["valid_count"])
metric_columns[2].metric("Invalid attempts", diagnostics["invalid_count"])
metric_columns[3].metric("Winner review", len(review_codes))

decision_tab, audit_tab, peer_tab, cracking_tab, method_tab = st.tabs(
    ["Decision", "Audit", "Peer fairness", "Cracking fairness", "Data & method"]
)

with decision_tab:
    st.subheader("Decision table")
    st.markdown(
        '<div class="status-note"><strong>Moderate together:</strong> '
        + ", ".join(review_codes)
        + ". Their modelled differences from the provisional leader include zero at the approximate 95% level.</div>",
        unsafe_allow_html=True,
    )
    display_mode = st.selectbox(
        "Component display",
        ["Scores out of 100", "Weighted contributions", "Component ranks"],
        help=(
            "Scores out of 100 compare components on one scale. Weighted contributions add to the fair score. "
            "Component ranks show 1 as best for each component; tied values share a rank."
        ),
    )
    decision = decision_frame(ranking, diagnostics["weights"], display_mode)
    if display_mode == "Scores out of 100":
        component_columns = ["Judges /100", "Peers /100", "Cracking /100", "Resistance /100"]
        st.caption(
            "Comparable component scores before weighting. The fair score still uses weights of 45%, 20%, 25% and 10%."
        )
    elif display_mode == "Weighted contributions":
        component_columns = ["Judges /45", "Peers /20", "Cracking /25", "Resistance /10"]
        st.caption("Weighted point contributions. These four columns add directly to the fair score.")
    else:
        component_columns = ["Judges rank", "Peers rank", "Cracking rank", "Resistance rank"]
        st.caption(
            "Rank within each component after adjustment where applicable. 1 is best; tied component scores share a rank. "
            "Ranks do not add to the fair score."
        )
    table_config = {
        "Rank": st.column_config.NumberColumn(format="%d"),
        "Raw total": st.column_config.NumberColumn(format="%.2f"),
        "Fair score": st.column_config.ProgressColumn(format="%.2f", min_value=0, max_value=100),
    }
    component_format = "%d" if display_mode == "Component ranks" else "%.1f"
    table_config.update({column: st.column_config.NumberColumn(format=component_format) for column in component_columns})
    st.dataframe(
        decision,
        width="stretch",
        hide_index=True,
        column_config=table_config,
    )

    plot_data = ranking.sort_values("adjusted_total")
    figure, axis = plt.subplots(figsize=(11, 7))
    positions = np.arange(len(plot_data))
    axis.barh(positions - 0.18, plot_data["current_total"], height=0.34, color=colours["raw"], label="Raw total")
    axis.barh(positions + 0.18, plot_data["adjusted_total"], height=0.34, color=colours["teal"], label="Fair score")
    axis.set(yticks=positions, yticklabels=plot_data["code"], xlabel="Points", title="Raw and fair scores")
    axis.legend(frameon=False, ncol=2)
    style_axes(axis)
    st.pyplot(figure, width="stretch")
    plt.close(figure)

    download_columns = st.columns([1, 1, 3])
    download_columns[0].download_button(
        "Download analysis workbook", workbook_bytes, "scoring_fairness_analysis.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", width="stretch",
    )
    download_columns[1].download_button(
        "Download report", report_text, "scoring_fairness_report.md", "text/markdown", width="stretch",
    )

with audit_tab:
    st.subheader("Adjustment audit")
    st.caption("Raw and adjusted components, uncertainty ranges, evidence counts, and rank movement.")
    st.dataframe(audit, width="stretch", hide_index=True, height=610)

    with st.expander("Attempt-level crack audit"):
        attempt_display = attempt_log[["attacker", "defender", "outcome", "status"]].rename(
            columns={"attacker": "Attacker", "defender": "Defending safe", "outcome": "Recorded value", "status": "Treatment"}
        )
        st.dataframe(attempt_display, width="stretch", hide_index=True, height=420)

with peer_tab:
    peer_overview_tab, peer_math_tab = st.tabs(["Overview", "Maths behind peer scores"])
    with peer_overview_tab:
        st.subheader("Peer scoring fairness")
        st.markdown(
            '<div class="method-note"><strong>Two roles are shown here.</strong> '
            '<strong>Adjusted peer score</strong> is about this team\'s own safe and estimates the mark it would receive from an average rater. '
            '<strong>How this team marked others</strong> is based only on marks this team gave to other safes: negative means harsher than average; positive means more generous. '
            'A team is not rewarded or penalised for its own marking style. Instead, the model corrects the marks received from each harsh or generous rater.</div>',
            unsafe_allow_html=True,
        )
        peer_sort = st.selectbox(
            "Order teams in peer charts",
            [
                "Team code",
                "Raw peer score (highest first)",
                "Adjusted peer score (highest first)",
                "Scoring behaviour (harshest first)",
            ],
            help=(
                "The selected order is applied to both charts, so each horizontal row always refers to the same team."
            ),
        )
        if peer_sort == "Team code":
            peer_view = ranking.sort_values("code", ascending=False)
        elif peer_sort == "Raw peer score (highest first)":
            peer_view = ranking.sort_values("peer")
        elif peer_sort == "Adjusted peer score (highest first)":
            peer_view = ranking.sort_values("adjusted_peer")
        else:
            peer_view = ranking.sort_values("peer_rater_effect", ascending=False)
        rater_view = peer_view
        st.caption("Both charts use the selected order, so each horizontal row refers to the same team.")
        left, right = st.columns(2)
        with left:
            figure, axis = plt.subplots(figsize=(7, 7))
            positions = np.arange(len(peer_view))
            raw_peer = 100 * peer_view["peer"].to_numpy()
            adjusted_peer = 100 * peer_view["adjusted_peer"].to_numpy()
            for position, raw, adjusted in zip(positions, raw_peer, adjusted_peer):
                axis.plot([raw, adjusted], [position, position], color=colours["raw"], linewidth=1.5)
            axis.scatter(raw_peer, positions, color=colours["raw"], label="Raw", zorder=3)
            axis.scatter(adjusted_peer, positions, color=colours["teal"], label="Adjusted", zorder=3)
            axis.set(yticks=positions, yticklabels=peer_view["code"], xlabel="Peer score (%)", title="Raw and adjusted peer scores")
            axis.legend(frameon=False)
            style_axes(axis)
            st.pyplot(figure, width="stretch")
            plt.close(figure)
        with right:
            rater_points = 100 * rater_view["peer_rater_effect"]
            figure, axis = plt.subplots(figsize=(7, 7))
            bar_colours = np.where(rater_points < 0, colours["blue"], colours["amber"])
            axis.barh(rater_view["code"], rater_points, color=bar_colours)
            axis.axvline(0, color=colours["zero"], linewidth=1)
            axis.set(
                xlabel="Difference from an average rater (percentage points)",
                title="How each team marked other teams",
            )
            style_axes(axis)
            st.pyplot(figure, width="stretch")
            plt.close(figure)

        behaviour = ranking[["code", "peer_rater_effect"]].copy()
        behaviour["Typical marking behaviour"] = behaviour["peer_rater_effect"].apply(
            lambda value: (
                f"{abs(100 * value):.1f} points harsher than average"
                if value < -0.0005
                else f"{100 * value:.1f} points more generous than average"
                if value > 0.0005
                else "About average"
            )
        )
        behaviour = behaviour.rename(columns={"code": "Team acting as scorer"})[
            ["Team acting as scorer", "Typical marking behaviour"]
        ]
        with st.expander("Read the scoring-behaviour estimates in plain language"):
            st.dataframe(behaviour, width="stretch", hide_index=True)

        st.caption(
            f"Five-fold prediction RMSE: {diagnostics['peer_cv_rmse']:.3f}; "
            f"mean-only baseline: {diagnostics['peer_baseline_rmse']:.3f}. Lower is better."
        )
    with peer_math_tab:
        st.markdown('<a id="peer-regularisation-maths"></a>', unsafe_allow_html=True)
        st.subheader("How peer scores are regularised")
        st.markdown(
            "Each completed peer-rating cell contributes one observation. The row identifies the team being rated; "
            "the column identifies the team giving the rating. Scores are first divided by the maximum possible peer score, "
            "so the model works between 0 and 1."
        )
        st.latex(r"y_{ij} = \mu + q_i + g_j + \varepsilon_{ij}")
        st.markdown(
            "Here, $y_{ij}$ is the score given by rater $j$ to team $i$. $\\mu$ is the overall level, "
            "$q_i$ is team $i$'s underlying peer quality, and $g_j$ is rater $j$'s tendency to be generous or harsh. "
            "The adjusted peer score is $\\operatorname{clip}(\\mu + q_i, 0, 1)$: the prediction for an average rater."
        )
        st.markdown("The fitted values minimise ordinary squared error plus a ridge penalty:")
        st.latex(r"\sum_{i,j}(y_{ij}-\mu-q_i-g_j)^2 + \lambda\left(\sum_i q_i^2 + \sum_j g_j^2\right)")
        st.markdown(
            "The first term rewards matching the observed marks. The second term is regularisation: it gently pulls "
            "team-quality and rater-tendency estimates towards zero unless the data provide strong evidence. This keeps "
            "a small number of ratings from producing extreme estimates. The overall level $\\mu$ is not penalised."
        )
        st.info(
            f"For this workbook, five-fold cross-validation selected $\\lambda = {diagnostics['peer_regularisation'] if 'peer_regularisation' in diagnostics else 'the fitted value'}$. "
            f"The held-out RMSE is {diagnostics['peer_cv_rmse']:.3f}, compared with {diagnostics['peer_baseline_rmse']:.3f} for predicting every rating with the training mean."
        )
        st.markdown(
            "In five-fold cross-validation, the ratings are split into five groups. The model is fitted on four groups "
            "and tested on the fifth, repeating until every rating has been held out once. The chosen $\\lambda$ is the "
            "candidate with the lowest average held-out error."
        )

with cracking_tab:
    cracking_overview_tab, cracking_math_tab = st.tabs(["Overview", "Maths behind cracking scores"])
    with cracking_overview_tab:
        st.subheader("Cracking and resistance fairness")
        schedule_column, uncertainty_column = st.columns(2)
        with schedule_column:
            schedule = ranking.sort_values("schedule_easiness")
            figure, axis = plt.subplots(figsize=(7, 7))
            bar_colours = np.where(schedule["schedule_easiness"] > schedule["schedule_easiness"].mean(), colours["amber"], colours["blue"])
            axis.barh(schedule["code"], 100 * schedule["schedule_easiness"], color=bar_colours)
            axis.axvline(100 * schedule["schedule_easiness"].mean(), color=colours["zero"], linestyle="--", label="Mean")
            axis.set(xlabel="Expected crack rate for an average team (%)", title="Difficulty of each team's schedule")
            axis.legend(frameon=False)
            style_axes(axis)
            st.pyplot(figure, width="stretch")
            plt.close(figure)
        with uncertainty_column:
            uncertainty = ranking.sort_values("adjusted_attack")
            estimate = 100 * uncertainty["adjusted_attack"].to_numpy()
            lower = 100 * uncertainty["attack_low"].to_numpy()
            upper = 100 * uncertainty["attack_high"].to_numpy()
            figure, axis = plt.subplots(figsize=(7, 7))
            axis.errorbar(estimate, uncertainty["code"], xerr=[estimate - lower, upper - estimate], fmt="o", color=colours["teal"], ecolor=colours["raw"], capsize=3)
            axis.set(xlabel="Adjusted crack probability (%)", title="Cracking estimate and approximate 95% range", xlim=(0, 100))
            style_axes(axis)
            st.pyplot(figure, width="stretch")
            plt.close(figure)

        evidence = ranking.assign(
            usable_evidence=lambda frame: frame["valid_attacks"] + frame["valid_defences"]
        ).sort_values("usable_evidence")
        figure, axis = plt.subplots(figsize=(11, 6))
        positions = np.arange(len(evidence))
        axis.barh(positions - 0.18, evidence["valid_attacks"], height=0.34, color=colours["blue"], label="Usable attacks")
        axis.barh(positions + 0.18, evidence["valid_defences"], height=0.34, color=colours["teal"], label="Usable defences")
        axis.set(
            yticks=positions,
            yticklabels=evidence["code"],
            xlabel="Usable attempts",
            title="Usable cracking evidence: attacks and defences",
        )
        axis.legend(frameon=False, ncol=2)
        style_axes(axis)
        st.pyplot(figure, width="stretch")
        plt.close(figure)
        st.caption("Usable evidence includes only valid 0/1 outcomes. Invalid or unreliable attempts are excluded from both bars.")
    with cracking_math_tab:
        st.markdown('<a id="cracking-regularisation-maths"></a>', unsafe_allow_html=True)
        st.subheader("How cracking scores are regularised")
        st.markdown(
            "Every usable attempt is treated as a contest between an attacker and a defending safe. A recorded 1 means "
            "the crack succeeded, a 0 means it failed, and a 0.5 invalid or unreliable attempt is left out of this model."
        )
        st.latex(r"P(\text{crack}) = \sigma(\alpha + a_i - d_j), \qquad \sigma(x)=\frac{1}{1+e^{-x}}")
        st.markdown(
            "For an attempt by team $i$ against safe $j$, $a_i$ is the attacker's ability, $d_j$ is the safe's difficulty, "
            "$\\alpha$ is the overall baseline, and $\\sigma$ converts the result into a probability between 0 and 1. "
            "A strong attacker increases the probability; a difficult safe decreases it."
        )
        st.markdown("The model chooses these quantities by maximising the fit to the observed successes and failures, while adding a ridge penalty:")
        st.latex(r"-\log L(\alpha,a,d) + \lambda\left(\sum_i a_i^2 + \sum_j d_j^2\right)")
        st.markdown(
            "The log-likelihood term rewards probabilities that match what happened. The penalty pulls attack and difficulty "
            "estimates towards the average unless the evidence is strong. This matters because a team may have faced only a "
            "few safes, so an extreme result should not automatically become an extreme ability estimate. The baseline $\\alpha$ "
            "is left unpenalised."
        )
        st.info(
            f"For this workbook, five-fold cross-validation selected $\\lambda = {diagnostics['regularisation']:.2f}$. "
            f"Its held-out log loss is {diagnostics['cv_losses'][diagnostics['regularisation']]:.3f}; "
            f"the baseline log loss is {diagnostics['baseline_log_loss']:.3f}."
        )
        st.markdown(
            "The adjusted cracking score is $\\sigma(\\alpha+a_i)$, meaning performance against an average-difficulty safe. "
            "The adjusted resistance score is $\\sigma(d_i-\\alpha)$, meaning the chance that safe $i$ resists an average-strength attacker. "
            "The model therefore compares teams on a common schedule rather than simply averaging the safes they happened to face."
        )

with method_tab:
    st.markdown(
        "**Regularisation maths:** [peer scores](#peer-regularisation-maths) · "
        "[cracking and resistance scores](#cracking-regularisation-maths)"
    )
    st.subheader("Data quality")
    invalid_share = diagnostics["invalid_count"] / diagnostics["attempt_count"]
    quality = pd.DataFrame(
        {
            "Measure": [
                "Teams", "Recorded crack matchups", "Valid crack outcomes", "Invalid / unreliable",
                "Invalid share", "Crack model CV log loss", "Crack baseline log loss",
                "Peer model CV RMSE", "Peer mean-only RMSE",
            ],
            "Value": [
                str(diagnostics["team_count"]), str(diagnostics["attempt_count"]), str(diagnostics["valid_count"]),
                str(diagnostics["invalid_count"]), f"{invalid_share:.1%}",
                f"{diagnostics['cv_losses'][diagnostics['regularisation']]:.3f}",
                f"{diagnostics['baseline_log_loss']:.3f}", f"{diagnostics['peer_cv_rmse']:.3f}",
                f"{diagnostics['peer_baseline_rmse']:.3f}",
            ],
        }
    )
    st.dataframe(quality, width="stretch", hide_index=True)

    st.subheader("How the fair score is built")
    st.markdown(
        """
        - **Judges (45%)**: retained as recorded.
        - **Peers (20%)**: regularised additive model separates safe quality from rater generosity.
        - **Cracking (25%)**: regularised Rasch model estimates performance against an average-difficulty safe.
        - **Resistance (10%)**: the same model estimates resistance against an average-strength attacker.
        - **0.5 outcomes**: excluded because broken or unsolvable attempts are not evidence of success or failure.
        """
    )
    st.warning(
        "Model ranges are conditional approximations. They include peer, cracking and resistance model uncertainty, "
        "but treat judge marks as fixed and do not include model-selection uncertainty."
    )