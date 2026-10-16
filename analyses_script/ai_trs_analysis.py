"""
Analysis code for:
    Fang, Cheng, Besada, & Luo. "How Large Language Models Handle Ethical
    Dilemmas When Providing Mental Health and Emotional Support."

Reproduces every statistical result, table, and figure reported in the
manuscript from the AI Therapeutic Response Scale (AI-TRS) rating file.

    1. Inter-rater reliability of the six core dimensions (ICC 0.61-0.88)
    2. Table 1: mean (SD) by model, one-way ANOVA for overall performance,
       each core dimension, and each topic scenario, plus eta squared
    3. Supplementary Table 1: Tukey HSD pairwise comparisons (with Cohen's d)
    4. Two-way ANOVA: model x scenario
    5. Figure 1: (a) radar chart of dimensions, (b) scenario barplot (SE bars)
    6. Supplementary Table 2: Ethics / Cultural ratings by model
       (one-way ANOVA and Kruskal-Wallis)
    7. Appendix 3: two-stage reliability of the Ethics / Cultural dimensions
       (applicability kappa, then ICC and quadratic-weighted kappa)

Usage
-----
    1. Download the rating data (ratedData_cate_update.csv) from the
       Supplementary Materials: <SUPPLEMENTARY_MATERIALS_URL>
    2. Set DATA_PATH and OUTPUT_DIR below (or pass --data / --out).
    3. pip install pandas numpy scipy statsmodels matplotlib
       python ai_trs_analysis.py

Input data
----------
One row per AI response (4 models x 20 prompts = 80 rows) with columns:
    Model       gpt4 | claude3 | llama3 | deepSeek
    Category    one of the five topic scenarios
    <dim>_<rater> ratings on a 1-9 scale (one column per rater), e.g. empa_CC
    Ethics_1, Ethics_2, Cultural_1, Cultural_2
                ratings on a 0-2 scale; -9 = not applicable (N/A)

Scoring
-------
Each core dimension score is the mean of the available raters for that
response. Overall performance is the equal-weighted mean of the six core
dimension scores. An Ethics / Cultural score is the mean of the non-N/A
ratings; a response counts as applicable if at least one rater rated it.
"""

import argparse
from decimal import ROUND_HALF_UP, Decimal
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.inter_rater import cohens_kappa
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# ----------------------------------------------------------------------------
# Paths: replace the placeholders with locations on your computer
# ----------------------------------------------------------------------------

DATA_PATH = "<PATH_TO_DOWNLOADED_DATA>/ratedData_cate_update.csv"
OUTPUT_DIR = "<PATH_TO_OUTPUT_FOLDER>"

# ----------------------------------------------------------------------------
# Study design
# ----------------------------------------------------------------------------

MODEL_ORDER = ["gpt4", "claude3", "llama3", "deepSeek"]
MODEL_LABELS = {
    "gpt4": "GPT-4o",
    "claude3": "Claude 3.5 Sonnet",
    "llama3": "Llama 3.3 70B",
    "deepSeek": "DeepSeek R1",
}
MODEL_COLORS = ["#E74C3C", "#3498DB", "#F39C12", "#27AE60"]

CORE_DIMENSIONS = {
    "Empathy and Warmth": ["empa_CC", "empa_JW", "empa_UB", "empa_huzi"],
    "Insightfulness": ["insi_PB", "insi_SB", "insi_MI", "insi_jun"],
    "Natural Tone": ["natu_YX", "natu_JW", "natu_SB", "natu_JX"],
    "Clarity and Conciseness": ["cc_YX", "cc_UB", "cc_huzi", "cc_JX"],
    "Open Exploration": ["open_CC", "open_MI", "open_jun"],
    "Free from Assumptions": ["free_PB", "free_MI", "free_JX", "free_SB", "free_JW"],
}
CATEGORICAL_DIMENSIONS = {
    "Ethical concerns": ["Ethics_1", "Ethics_2"],
    "Cultural awareness": ["Cultural_1", "Cultural_2"],
}
NA_CODE = -9

SCENARIOS = [
    "Child & Adolescent Concerns",
    "Severe Psychopathology",
    "Safety Concerns",
    "Therapeutic Relationship & Boundaries",
    "Sociocultural Stressors",
]
SCENARIO_SHORT_LABELS = [
    "Child & Adolescent",
    "Severe Psychopathology",
    "Safety Concerns",
    "Therapeutic Relationship",
    "Sociocultural Stressors",
]
RADAR_LABELS = {
    "Empathy and Warmth": "Empathy and\nWarmth",
    "Insightfulness": "Insightfulness",
    "Natural Tone": "Natural Tone",
    "Clarity and Conciseness": "Clarity and\nConciseness",
    "Open Exploration": "Open Exploration",
    "Free from Assumptions": "Free from Assumptions",
}
OVERALL = "Overall Performance"


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def load_data(path):
    df = pd.read_csv(path, encoding="latin-1")
    rating_columns = [c for cols in CORE_DIMENSIONS.values() for c in cols]
    rating_columns += [c for cols in CATEGORICAL_DIMENSIONS.values() for c in cols]
    df[rating_columns] = df[rating_columns].apply(pd.to_numeric, errors="coerce")

    for dimension, columns in CORE_DIMENSIONS.items():
        df[dimension] = df[columns].mean(axis=1, skipna=True)
    df[OVERALL] = df[list(CORE_DIMENSIONS)].mean(axis=1)

    for dimension, columns in CATEGORICAL_DIMENSIONS.items():
        df[dimension] = df[columns].replace(NA_CODE, np.nan).mean(axis=1, skipna=True)

    df["ModelLabel"] = df["Model"].map(MODEL_LABELS)
    return df


def round_half_up(value, digits=2):
    return Decimal(str(float(value))).quantize(Decimal(1).scaleb(-digits), rounding=ROUND_HALF_UP)


def mean_sd(values):
    return f"{round_half_up(np.mean(values))} ({round_half_up(np.std(values, ddof=1))})"


def format_p(p):
    if p < 0.001:
        return "p < .001"
    return f"p = {p:.3f}".replace("0.", ".", 1)


def stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


def model_groups(data, column):
    return [data.loc[data["Model"] == m, column].dropna().to_numpy() for m in MODEL_ORDER]


def one_way_anova(groups):
    """One-way ANOVA with eta squared (SS_between / SS_total)."""
    f_value, p_value = stats.f_oneway(*groups)
    pooled = np.concatenate(groups)
    ss_between = sum(len(g) * (g.mean() - pooled.mean()) ** 2 for g in groups)
    ss_total = ((pooled - pooled.mean()) ** 2).sum()
    return {
        "F": f_value,
        "p": p_value,
        "df1": len(groups) - 1,
        "df2": len(pooled) - len(groups),
        "eta_sq": ss_between / ss_total,
    }


def cohens_d(a, b):
    """Cohen's d with pooled SD (b minus a, matching the Tukey direction)."""
    na, nb = len(a), len(b)
    pooled_sd = np.sqrt(((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1)) / (na + nb - 2))
    return (np.mean(b) - np.mean(a)) / pooled_sd


def icc_all_forms(ratings):
    """Shrout & Fleiss ICCs for a complete targets x raters matrix."""
    x = np.asarray(ratings, dtype=float)
    n, k = x.shape
    grand = x.mean()
    ss_rows = k * ((x.mean(axis=1) - grand) ** 2).sum()
    ss_cols = n * ((x.mean(axis=0) - grand) ** 2).sum()
    ss_error = ((x - grand) ** 2).sum() - ss_rows - ss_cols
    ms_rows = ss_rows / (n - 1)
    ms_cols = ss_cols / (k - 1)
    ms_error = ss_error / ((n - 1) * (k - 1))

    f_value = ms_rows / ms_error
    df1, df2 = n - 1, (n - 1) * (k - 1)
    f_low = f_value / stats.f.ppf(0.975, df1, df2)
    f_high = f_value * stats.f.ppf(0.975, df2, df1)

    icc21 = (ms_rows - ms_error) / (ms_rows + (k - 1) * ms_error + k * (ms_cols - ms_error) / n)
    return {
        "n": n,
        "k": k,
        "ICC(2,1)": icc21,
        "ICC(2,1) 95% CI": icc21_confidence_interval(ms_rows, ms_cols, ms_error, n, k, icc21),
        "ICC(2,k)": (ms_rows - ms_error) / (ms_rows + (ms_cols - ms_error) / n),
        "ICC(3,1)": (ms_rows - ms_error) / (ms_rows + (k - 1) * ms_error),
        "ICC(3,k)": (ms_rows - ms_error) / ms_rows,
        "ICC(3,k) 95% CI": (1 - 1 / f_low, 1 - 1 / f_high),
    }


def icc21_confidence_interval(ms_rows, ms_cols, ms_error, n, k, icc, alpha=0.05):
    """McGraw & Wong (1996) confidence interval for ICC(2,1)."""
    fc = ms_cols / ms_error
    common = k * icc * fc + n * (1 + (k - 1) * icc) - k * icc
    v = ((n - 1) * (k - 1) * common ** 2) / (
        (n - 1) * k ** 2 * icc ** 2 * fc ** 2 + (n * (1 + (k - 1) * icc) - k * icc) ** 2
    )
    f_upper = stats.f.ppf(1 - alpha / 2, n - 1, v)
    f_lower = stats.f.ppf(1 - alpha / 2, v, n - 1)
    low = n * (ms_rows - f_upper * ms_error) / (
        f_upper * (k * ms_cols + (k * n - k - n) * ms_error) + n * ms_rows
    )
    high = n * (f_lower * ms_rows - ms_error) / (
        k * ms_cols + (k * n - k - n) * ms_error + n * f_lower * ms_rows
    )
    return low, high


def format_ci(ci):
    return f"[{ci[0]:.2f}, {ci[1]:.2f}]"


def section(title):
    print("\n" + "=" * 90 + f"\n{title}\n" + "=" * 90)


# ----------------------------------------------------------------------------
# 1. Inter-rater reliability: six core dimensions
# ----------------------------------------------------------------------------

def core_reliability(df):
    """ICC per core dimension, using responses rated by every rater of that
    dimension. The manuscript reports ICC(3,k) (two-way mixed, consistency,
    average of k raters); absolute-agreement ICC(2,k) is shown for reference.
    """
    section("1. INTER-RATER RELIABILITY - SIX CORE DIMENSIONS")
    rows = []
    for dimension, columns in CORE_DIMENSIONS.items():
        result = icc_all_forms(df[columns].dropna())
        rows.append({
            "Dimension": dimension,
            "Responses": result["n"],
            "Raters": result["k"],
            "ICC(3,k)": round(result["ICC(3,k)"], 3),
            "ICC(3,k) 95% CI": format_ci(result["ICC(3,k) 95% CI"]),
            "ICC(2,k)": round(result["ICC(2,k)"], 3),
            "ICC(2,1)": round(result["ICC(2,1)"], 3),
            "ICC(3,1)": round(result["ICC(3,1)"], 3),
        })
    table = pd.DataFrame(rows)
    print(table.to_string(index=False))
    print(f"\nICC(3,k) range: {table['ICC(3,k)'].min():.2f}-{table['ICC(3,k)'].max():.2f}")
    return table


# ----------------------------------------------------------------------------
# 2. Table 1: descriptive statistics and one-way ANOVA by model
# ----------------------------------------------------------------------------

def table1_model_anova(df):
    section("2. TABLE 1 - MEAN (SD) BY MODEL AND ONE-WAY ANOVA")
    outcomes = [("Overall", OVERALL, df)]
    outcomes += [("Skill dimension", d, df) for d in CORE_DIMENSIONS]
    outcomes += [("Topic scenario", s, df[df["Category"] == s]) for s in SCENARIOS]

    rows = []
    for block, outcome, data in outcomes:
        column = OVERALL if block == "Topic scenario" else outcome
        groups = model_groups(data, column)
        result = one_way_anova(groups)
        row = {"Section": block, "Outcome": outcome}
        row.update({MODEL_LABELS[m]: mean_sd(g) for m, g in zip(MODEL_ORDER, groups)})
        row["Average"] = mean_sd(data[column])
        row["ANOVA"] = (
            f"F({result['df1']}, {result['df2']}) = {result['F']:.2f}, "
            f"{format_p(result['p'])}{stars(result['p'])}"
        )
        row["eta_sq"] = round(result["eta_sq"], 3)
        rows.append(row)

    table = pd.DataFrame(rows)
    print(table.drop(columns="Section").to_string(index=False))
    sizes = ", ".join(f"{s}: {(df['Category'] == s).sum()}" for s in SCENARIOS)
    print(f"\nValues are mean (SD). Responses per scenario: {sizes}.")
    return table


# ----------------------------------------------------------------------------
# 3. Supplementary Table 1: Tukey HSD (+ Cohen's d)
# ----------------------------------------------------------------------------

def tukey_posthoc(df):
    """Tukey HSD for every outcome whose omnibus ANOVA is significant."""
    section("3. SUPPLEMENTARY TABLE 1 - TUKEY HSD PAIRWISE COMPARISONS")
    candidates = [(OVERALL, df, OVERALL)]
    candidates += [(d, df, d) for d in CORE_DIMENSIONS]
    candidates += [(s, df[df["Category"] == s], OVERALL) for s in SCENARIOS]

    rows = []
    for outcome, data, column in candidates:
        if one_way_anova(model_groups(data, column))["p"] >= 0.05:
            continue
        result = pairwise_tukeyhsd(data[column], data["ModelLabel"], alpha=0.05)
        pairs = combinations(result.groupsunique, 2)
        for (g1, g2), diff, p, ci, reject in zip(
            pairs, result.meandiffs, result.pvalues, result.confint, result.reject
        ):
            a = data.loc[data["ModelLabel"] == g1, column].to_numpy()
            b = data.loc[data["ModelLabel"] == g2, column].to_numpy()
            rows.append({
                "Outcome": outcome,
                "Comparison": f"{g2} - {g1}",
                "Mean difference": round(diff, 2),
                "95% CI": format_ci(ci),
                "Adjusted p": round(p, 3),
                "Cohen's d": round(cohens_d(a, b), 2),
                "Significant": bool(reject),
            })

    table = pd.DataFrame(rows)
    print(table.to_string(index=False))
    print(f"\n{table['Significant'].sum()} of {len(table)} comparisons significant (FWER = .05 per outcome).")
    return table


# ----------------------------------------------------------------------------
# 4. Two-way ANOVA: model x scenario
# ----------------------------------------------------------------------------

def two_way_anova(df):
    section("4. TWO-WAY ANOVA - MODEL x SCENARIO (OVERALL PERFORMANCE)")
    data = df.rename(columns={OVERALL: "overall"})
    fit = ols("overall ~ C(Model) * C(Category)", data=data).fit()
    table = anova_lm(fit, typ=2)
    table["partial_eta_sq"] = table["sum_sq"] / (table["sum_sq"] + table.loc["Residual", "sum_sq"])
    table.loc["Residual", "partial_eta_sq"] = np.nan
    print(table.round(4).to_string())
    return table


# ----------------------------------------------------------------------------
# 5. Figure 1: radar chart (a) + scenario barplot (b)
# ----------------------------------------------------------------------------

def figure1(df, out_dir):
    section("5. FIGURE 1 - RADAR CHART AND SCENARIO BARPLOT")
    dimensions = list(CORE_DIMENSIONS)
    angles = np.linspace(0, 2 * np.pi, len(dimensions), endpoint=False).tolist()
    angles += angles[:1]

    cells = (
        df.groupby(["Category", "Model"])[OVERALL]
        .agg(mean="mean", sd="std", n="count")
        .reset_index()
    )
    cells["se"] = cells["sd"] / np.sqrt(cells["n"])

    fig = plt.figure(figsize=(16.5, 7.2), facecolor="white")
    grid = fig.add_gridspec(1, 2, width_ratios=[1.12, 1.0], wspace=0.28)
    ax_radar = fig.add_subplot(grid[0], projection="polar")
    ax_bar = fig.add_subplot(grid[1])

    for color, model in zip(MODEL_COLORS, MODEL_ORDER):
        values = df.loc[df["Model"] == model, dimensions].mean().tolist()
        values += values[:1]
        ax_radar.plot(angles, values, "o-", linewidth=2.5, color=color, markersize=7)
        ax_radar.fill(angles, values, alpha=0.2, color=color)
    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels([RADAR_LABELS[d] for d in dimensions], size=11)
    ax_radar.set_ylim(1, 9)
    ax_radar.set_yticks(range(1, 10))
    ax_radar.set_yticklabels([str(v) for v in range(1, 10)], size=9)
    ax_radar.grid(True, alpha=0.3, linestyle="--")
    ax_radar.set_rlabel_position(45)
    ax_radar.set_facecolor("#f8f9fa")

    x = np.arange(len(SCENARIOS))
    width = 0.16
    offsets = (np.arange(len(MODEL_ORDER)) - (len(MODEL_ORDER) - 1) / 2) * width
    for offset, color, model in zip(offsets, MODEL_COLORS, MODEL_ORDER):
        model_cells = cells[cells["Model"] == model].set_index("Category").loc[SCENARIOS]
        ax_bar.bar(
            x + offset, model_cells["mean"], width, yerr=model_cells["se"], capsize=3,
            color=color, label=MODEL_LABELS[model], edgecolor="white", linewidth=0.6,
            error_kw={"elinewidth": 1.1, "capthick": 1.1, "ecolor": "#333333"},
        )
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(SCENARIO_SHORT_LABELS, fontsize=9, rotation=30, ha="right")
    ax_bar.set_xlim(-0.55, len(SCENARIOS) - 0.45)
    ax_bar.set_ylabel("Overall therapeutic quality (1-9)", fontsize=11)
    ax_bar.set_xlabel("Clinical scenario", fontsize=11)
    ax_bar.set_ylim(0, 9)
    ax_bar.set_yticks(range(0, 10))
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)
    ax_bar.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax_bar.set_axisbelow(True)

    handles, labels = ax_bar.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=True, fontsize=11,
               bbox_to_anchor=(0.5, 1.04))
    fig.text(0.04, 0.96, "A", fontsize=18, fontweight="bold")
    fig.text(0.53, 0.96, "B", fontsize=18, fontweight="bold")

    path = out_dir / "figure1_radar_and_scenario_barplot.png"
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white", pad_inches=0.25)
    plt.close(fig)
    print(f"Saved {path}")
    return cells


# ----------------------------------------------------------------------------
# 6. Supplementary Table 2: Ethics / Cultural by model
# ----------------------------------------------------------------------------

def categorical_model_comparison(df):
    section("6. SUPPLEMENTARY TABLE 2 - ETHICS AND CULTURAL RATINGS BY MODEL")
    rows = []
    for dimension in CATEGORICAL_DIMENSIONS:
        applicable = df.dropna(subset=[dimension])
        groups = model_groups(applicable, dimension)
        anova = one_way_anova(groups)
        kruskal = stats.kruskal(*groups)
        row = {"Dimension (0-2 scale)": f"{dimension} (n = {len(applicable)})"}
        row.update({
            MODEL_LABELS[m]: f"{mean_sd(g)} n={len(g)}" for m, g in zip(MODEL_ORDER, groups)
        })
        row["ANOVA"] = f"F({anova['df1']}, {anova['df2']}) = {anova['F']:.2f}, {format_p(anova['p'])}"
        row["Kruskal-Wallis"] = f"H(3) = {kruskal.statistic:.2f}, {format_p(kruskal.pvalue)}"
        rows.append(row)
    table = pd.DataFrame(rows)
    print(table.to_string(index=False))
    print("\nA response is applicable if at least one rater gave a non-N/A rating.")
    return table


# ----------------------------------------------------------------------------
# 7. Appendix 3: two-stage reliability of Ethics / Cultural
# ----------------------------------------------------------------------------

def categorical_reliability(df):
    section("7. APPENDIX 3 - TWO-STAGE RELIABILITY OF ETHICS AND CULTURAL")
    stage1, stage2 = [], []
    for dimension, columns in CATEGORICAL_DIMENSIONS.items():
        raw = df[columns]
        applicable = raw.notna() & raw.ne(NA_CODE)

        # Stage 1: agreement on applicable vs. not applicable (all 80 responses).
        status = pd.crosstab(applicable.iloc[:, 0].astype(int), applicable.iloc[:, 1].astype(int))
        status = status.reindex(index=[0, 1], columns=[0, 1], fill_value=0)
        kappa = cohens_kappa(status.to_numpy())
        stage1.append({
            "Dimension": dimension,
            "N": len(raw),
            "Both applicable": int(applicable.all(axis=1).sum()),
            "Both N/A": int((~applicable).all(axis=1).sum()),
            "Disagree": int(applicable.iloc[:, 0].ne(applicable.iloc[:, 1]).sum()),
            "Agreement": f"{applicable.iloc[:, 0].eq(applicable.iloc[:, 1]).mean():.1%}",
            "Cohen's kappa": round(kappa.kappa, 3),
            "95% CI": f"[{kappa.kappa_low:.2f}, {kappa.kappa_upp:.2f}]",
        })

        # Stage 2: agreement on the 0-2 rating when both raters judged it applicable.
        paired = raw.replace(NA_CODE, np.nan).dropna()
        crosstab = pd.crosstab(paired.iloc[:, 0], paired.iloc[:, 1])
        crosstab = crosstab.reindex(index=[0, 1, 2], columns=[0, 1, 2], fill_value=0)
        icc = icc_all_forms(paired)
        stage2.append({
            "Dimension": dimension,
            "Jointly applicable n": len(paired),
            "Exact agreement": f"{(paired.iloc[:, 0] == paired.iloc[:, 1]).mean():.1%}",
            "Quadratic-weighted kappa": round(cohens_kappa(crosstab.to_numpy(), wt="quadratic").kappa, 3),
            "ICC(2,1)": round(icc["ICC(2,1)"], 3),
            "ICC(2,1) 95% CI": format_ci(icc["ICC(2,1) 95% CI"]),
            "ICC(2,2)": round(icc["ICC(2,k)"], 3),
        })

    stage1, stage2 = pd.DataFrame(stage1), pd.DataFrame(stage2)
    print("Stage 1 - agreement on applicability (applicable vs. N/A)")
    print(stage1.to_string(index=False))
    print("\nStage 2 - agreement on 0-2 ratings among jointly applicable responses")
    print(stage2.to_string(index=False))
    return stage1, stage2


# ----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data", default=DATA_PATH, type=Path)
    parser.add_argument("--out", default=OUTPUT_DIR, type=Path)
    args = parser.parse_args()
    if "<" in str(args.data) or "<" in str(args.out):
        parser.error("set DATA_PATH and OUTPUT_DIR at the top of the script, or pass --data and --out")
    if not args.data.exists():
        parser.error(f"data file not found: {args.data}")
    args.out.mkdir(parents=True, exist_ok=True)

    df = load_data(args.data)
    print(f"Loaded {len(df)} responses: {df['Model'].value_counts().to_dict()}")

    core_reliability(df).to_csv(args.out / "core_dimension_icc.csv", index=False)
    table1_model_anova(df).to_csv(args.out / "table1_model_anova.csv", index=False)
    tukey_posthoc(df).to_csv(args.out / "supp_table1_tukey_hsd.csv", index=False)
    two_way_anova(df).to_csv(args.out / "two_way_anova_model_by_scenario.csv")
    figure1(df, args.out).to_csv(args.out / "figure1b_scenario_cell_means.csv", index=False)
    categorical_model_comparison(df).to_csv(args.out / "supp_table2_ethics_cultural.csv", index=False)
    stage1, stage2 = categorical_reliability(df)
    stage1.to_csv(args.out / "appendix3_stage1_applicability.csv", index=False)
    stage2.to_csv(args.out / "appendix3_stage2_rating_agreement.csv", index=False)

    print(f"\nAll outputs written to {args.out}")


if __name__ == "__main__":
    main()
