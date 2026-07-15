"""
Grievance Archetype Discovery via Gaussian Mixture Modeling (Latent Profile Analysis)
=======================================================================================

Pipeline:
  1. Load CSV, select moral foundations + curated LIWC-22 features
  2. Standardize features
  3. Fit GMMs for k = 2..15, select best k via BIC + entropy + min class size
     + bootstrap stability (now computed across the full k range)
  4. Profile each archetype (mean feature z-scores)
  5. External validation: archetype vs subcategory, archetype vs context (chi-square)
  6. Export top-loading comments per archetype for qualitative review

USAGE:
  1. Edit the CONFIG section below to match your actual CSV column names.
  2. Run: python grievance_archetypes.py --input your_file.csv --stability
  3. Check outputs/ for: model_selection.csv, archetype_profiles.csv,
     archetype_assignments.csv, validation_crosstabs.csv, top_comments_per_archetype.csv,
     and profile_heatmap.png / bic_curve.png
"""
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score
from sklearn.utils import resample
from joblib import Parallel, delayed
import argparse
import ast
import sys
import itertools
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from scipy.stats import chi2_contingency
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================== CONFIG ==================================

TEXT_COL = "text"
CONTEXT_COL = "context"
SUBCATEGORY_COL = "subcategory_list"
HAS_GRIEVANCE_COL = "has_grievance"
LANGUAGE_COL = "language"
FILTER_TO_ENGLISH = None

FILTER_HAS_GRIEVANCE = True

MFT_COLS = ["care", "fairness", "loyalty", "authority", "sanctity"]

LIWC_COLS = [
    "emo_anger", "emo_sad", "emo_anx", "tone_neg", "tone_pos",
    "cogproc", "cause", "certitude", "discrep", "insight",
    "Social", "we", "they", "i", "family",
    "moral", "conflict", "reward", "risk",
    "Clout", "Authentic", "Analytic",
]

MIN_K, MAX_K = 2, 15
MIN_CLASS_FRACTION = 0.005
RANDOM_STATE = 42
TOP_N_COMMENTS = 15
OUTPUT_DIR = "."
BOOTSTRAP_N_INIT = 4
BOOTSTRAP_MAX_ITER = 200
BOOTSTRAP_N_JOBS = -1
MAX_ARI_COMPARISONS = 300
STABILITY_K_GRID_STEP = 2
# =========================================================================


def parse_subcategory_list(val):
    if isinstance(val, list):
        return val
    if pd.isna(val):
        return []
    try:
        parsed = ast.literal_eval(val)
        return parsed if isinstance(parsed, list) else []
    except (ValueError, SyntaxError):
        return []


def load_and_prepare(path):
    df = pd.read_csv(path)
    before = len(df)
    df = df.drop_duplicates(subset=[TEXT_COL]).reset_index(drop=True)
    print(f"Removed {before - len(df)} duplicate comments.")

    required = [TEXT_COL, CONTEXT_COL, SUBCATEGORY_COL] + MFT_COLS + LIWC_COLS
    missing = [c for c in required if c not in df.columns]
    if missing:
        print("ERROR: the following configured columns were not found in the CSV:")
        for c in missing:
            print(f"  - {c}")
        print("\nActual columns available in your file:")
        print(list(df.columns))
        sys.exit(1)

    df[SUBCATEGORY_COL] = df[SUBCATEGORY_COL].apply(parse_subcategory_list)

    if FILTER_HAS_GRIEVANCE and HAS_GRIEVANCE_COL in df.columns:
        before = len(df)
        df = df[df[HAS_GRIEVANCE_COL] == 1].reset_index(drop=True)
        print(f"Filtered to has_grievance==1: {len(df)}/{before} rows retained "
              f"({len(df)/before:.1%}).")

    if FILTER_TO_ENGLISH and LANGUAGE_COL in df.columns:
        before = len(df)
        df = df[df[LANGUAGE_COL] == FILTER_TO_ENGLISH].reset_index(drop=True)
        print(f"Filtered to language=={FILTER_TO_ENGLISH}: {len(df)}/{before} rows retained.")

    feature_cols = MFT_COLS + LIWC_COLS
    df_clean = df.dropna(subset=feature_cols).reset_index(drop=True)
    dropped = len(df) - len(df_clean)
    if dropped:
        print(f"Dropped {dropped} rows with missing values in feature columns "
              f"({dropped/len(df):.1%} of remaining data).")
    return df_clean, feature_cols


def fit_models(X_scaled, k_min=MIN_K, k_max=MAX_K):
    results = {}
    for k in range(k_min, k_max + 1):
        print(f"\nFitting k={k}")
        gmm = GaussianMixture(
            n_components=k, covariance_type="diag",
            n_init=30, max_iter=500, random_state=RANDOM_STATE
        )
        gmm.fit(X_scaled)
        if not gmm.converged_:
            print(f"WARNING: k={k} did not converge")

        labels = gmm.predict(X_scaled)
        probs = gmm.predict_proba(X_scaled)
        entropy = -np.sum(probs * np.log(probs + 1e-12), axis=1) / np.log(k)
        class_sizes = pd.Series(labels).value_counts(normalize=True)

        results[k] = {
            "model": gmm, "labels": labels, "probs": probs,
            "bic": gmm.bic(X_scaled), "aic": gmm.aic(X_scaled),
            "entropy": entropy.mean(), "min_class_fraction": class_sizes.min()
        }
        print(f"BIC={gmm.bic(X_scaled):.1f} | entropy={entropy.mean():.3f} | "
              f"smallest={class_sizes.min():.3f}")
    return results


def select_best_k(results, stability, min_stability_ARI=0.60):
    candidates = []
    for k, r in results.items():
        ari = stability[k]["mean_ARI"]

        if r["min_class_fraction"] < MIN_CLASS_FRACTION:
            print(f"k={k}: rejected (class too small)")
            continue
        if not np.isnan(ari) and ari < min_stability_ARI:
            print(f"k={k}: rejected (stability ARI={ari:.3f})")
            continue

        candidates.append({
            "k": k, "bic": r["bic"], "entropy": r["entropy"], "stability_ARI": ari
        })

    if len(candidates) == 0:
        raise ValueError(
            "No models passed the size/stability filters. "
            "Lower min_stability_ARI, widen the stability k grid, or inspect "
            "model_selection.csv directly."
        )

    candidates = pd.DataFrame(candidates)
    print("\nStable candidate solutions:")
    print(candidates.sort_values("bic"))

    best_k = int(candidates.sort_values("bic").iloc[0]["k"])
    return best_k


def align_cluster_labels(reference, predicted):
    labels_ref = np.unique(reference)
    labels_pred = np.unique(predicted)
    contingency = np.zeros((len(labels_ref), len(labels_pred)))
    for i, r in enumerate(labels_ref):
        for j, p in enumerate(labels_pred):
            contingency[i, j] = np.sum((reference == r) & (predicted == p))
    row_ind, col_ind = linear_sum_assignment(-contingency)
    mapping = {labels_pred[c]: labels_ref[r] for r, c in zip(row_ind, col_ind)}
    return np.array([mapping[label] for label in predicted])


def _fit_one_bootstrap(X_scaled, k, n, fraction, seed):
    idx = resample(np.arange(n), replace=False, n_samples=int(n * fraction), random_state=seed)
    X_boot = X_scaled[idx]
    gmm = GaussianMixture(
        n_components=k, covariance_type="diag",
        n_init=BOOTSTRAP_N_INIT, max_iter=BOOTSTRAP_MAX_ITER, random_state=seed
    )
    gmm.fit(X_boot)
    return idx, gmm.predict(X_boot)


def bootstrap_stability(X_scaled, candidate_ks, repeats=50, fraction=0.8,
                         n_jobs=BOOTSTRAP_N_JOBS, max_comparisons=MAX_ARI_COMPARISONS):
    stability = {}
    n = len(X_scaled)

    for k in candidate_ks:
        print(f"\nBootstrap stability for k={k} (parallel, n_jobs={n_jobs})")

        bootstrap_labels = Parallel(n_jobs=n_jobs, verbose=5)(
            delayed(_fit_one_bootstrap)(X_scaled, k, n, fraction, RANDOM_STATE + i)
            for i in range(repeats)
        )

        all_pairs = list(itertools.combinations(range(len(bootstrap_labels)), 2))
        if len(all_pairs) > max_comparisons:
            rng = np.random.RandomState(RANDOM_STATE)
            sel = rng.choice(len(all_pairs), size=max_comparisons, replace=False)
            pairs = [all_pairs[s] for s in sel]
        else:
            pairs = all_pairs

        def _pair_ari(i, j):
            idx_i, labels_i = bootstrap_labels[i]
            idx_j, labels_j = bootstrap_labels[j]
            common, pos_i, pos_j = np.intersect1d(idx_i, idx_j, return_indices=True)
            if len(common) < 20:
                return None
            return adjusted_rand_score(labels_i[pos_i], labels_j[pos_j])

        ari_scores = [s for s in (_pair_ari(i, j) for i, j in pairs) if s is not None]

        stability[k] = {
            "mean_ARI": np.mean(ari_scores) if ari_scores else np.nan,
            "sd_ARI": np.std(ari_scores) if ari_scores else np.nan,
            "n_comparisons": len(ari_scores)
        }
        print(f"k={k}: ARI={stability[k]['mean_ARI']:.3f} (SD={stability[k]['sd_ARI']:.3f}), "
              f"n_comparisons={stability[k]['n_comparisons']}")

    return stability


def build_stability_k_grid(results, step=STABILITY_K_GRID_STEP, refine_top=3):
    all_ks = sorted(results.keys())
    valid = [k for k in all_ks if results[k]["min_class_fraction"] >= MIN_CLASS_FRACTION]
    grid = valid[::step]
    if valid[-1] not in grid:
        grid.append(valid[-1])
    return sorted(set(grid))


def profile_archetypes(df, feature_cols, X_scaled, labels, probs, best_k):
    df = df.copy()
    df["archetype"] = labels
    df["archetype_confidence"] = probs.max(axis=1)

    scaled_df = pd.DataFrame(X_scaled, columns=feature_cols)
    scaled_df["archetype"] = labels
    profile = scaled_df.groupby("archetype")[feature_cols].mean()
    profile["n_comments"] = pd.Series(labels).value_counts().sort_index()
    profile["pct_of_corpus"] = (profile["n_comments"] / len(df) * 100).round(1)
    return df, profile


def validate_against_labels(df):
    results = {}

    ct = pd.crosstab(df["archetype"], df[CONTEXT_COL])
    chi2, p, dof, _ = chi2_contingency(ct)
    results[CONTEXT_COL] = {
        "crosstab_normalized": pd.crosstab(df["archetype"], df[CONTEXT_COL], normalize="index"),
        "crosstab_counts": ct,
        "chi2": chi2, "p_value": p, "dof": dof,
    }
    print(f"\nArchetype vs {CONTEXT_COL}: chi2={chi2:.1f}, p={p:.2e}, dof={dof}")

    exploded = df[["archetype", SUBCATEGORY_COL]].explode(SUBCATEGORY_COL)
    exploded = exploded.dropna(subset=[SUBCATEGORY_COL]).reset_index(drop=True)
    if len(exploded):
        ct_sub = pd.crosstab(exploded["archetype"], exploded[SUBCATEGORY_COL])
        chi2s, ps, dofs, _ = chi2_contingency(ct_sub)
        results[SUBCATEGORY_COL] = {
            "crosstab_normalized": pd.crosstab(exploded["archetype"], exploded[SUBCATEGORY_COL],
                                                normalize="index"),
            "crosstab_counts": ct_sub,
            "chi2": chi2s, "p_value": ps, "dof": dofs,
        }
        print(f"\nArchetype vs {SUBCATEGORY_COL} (exploded multi-label, "
              f"p-value caveat applies): chi2={chi2s:.1f}, p={ps:.2e}, dof={dofs}")
    else:
        print(f"\nNo non-empty {SUBCATEGORY_COL} values found to validate against.")

    return results


def export_top_comments(df, best_k):
    rows = []
    for a in sorted(df["archetype"].unique()):
        top = (df[df.archetype == a]
               .sort_values("archetype_confidence", ascending=False)
               .head(TOP_N_COMMENTS))
        for _, r in top.iterrows():
            rows.append({
                "archetype": a,
                "confidence": round(r["archetype_confidence"], 3),
                "context": r[CONTEXT_COL],
                "subcategory": ", ".join(r[SUBCATEGORY_COL]) if r[SUBCATEGORY_COL] else "(none)",
                "text": r[TEXT_COL],
            })
    out = pd.DataFrame(rows)
    out.to_csv(f"{OUTPUT_DIR}/top_comments_per_archetype.csv", index=False)
    print(f"\nWrote {OUTPUT_DIR}/top_comments_per_archetype.csv "
          f"— READ THESE before naming your archetypes.")


def plot_bic_curve(results, stability=None):
    ks = sorted(results.keys())
    bics = [results[k]["bic"] for k in ks]
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(ks, bics, marker="o", color="tab:blue")
    ax1.set_xlabel("Number of archetypes (k)")
    ax1.set_ylabel("BIC (lower = better fit)", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    if stability is not None:
        aris = [stability.get(k, {}).get("mean_ARI", np.nan) for k in ks]
        if not all(np.isnan(aris)):
            ax2 = ax1.twinx()
            ax2.plot(ks, aris, marker="s", color="tab:red", linestyle="--")
            ax2.axhline(0.60, color="tab:red", linestyle=":", linewidth=1)
            ax2.set_ylabel("Bootstrap stability (mean ARI)", color="tab:red")
            ax2.tick_params(axis="y", labelcolor="tab:red")
    plt.title("Model selection: BIC and stability by number of archetypes")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/bic_curve.png", dpi=150)
    plt.close()


def plot_profile_heatmap(profile, feature_cols):
    plt.figure(figsize=(max(8, len(feature_cols) * 0.5), max(4, len(profile) * 0.6)))
    plt.imshow(profile[feature_cols].values, cmap="RdBu_r", aspect="auto", vmin=-1.5, vmax=1.5)
    plt.colorbar(label="mean z-score")
    plt.yticks(range(len(profile)), [f"Archetype {i}" for i in profile.index])
    plt.xticks(range(len(feature_cols)), feature_cols, rotation=90)
    plt.title("Archetype feature profiles (standardized)")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/profile_heatmap.png", dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--k", type=int, default=None,
                         help="Force this exact number of archetypes, bypassing automatic selection.")
    parser.add_argument("--k_min", type=int, default=MIN_K,
                         help="Lowest k to fit/test (default: module MIN_K = %d)." % MIN_K)
    parser.add_argument("--k_max", type=int, default=MAX_K,
                         help="Highest k to fit/test (default: module MAX_K = %d)." % MAX_K)
    parser.add_argument("--stability", action="store_true",
                         help="Run bootstrap stability analysis across the k grid.")
    parser.add_argument("--stability_repeats", type=int, default=30,
                         help="Bootstrap repeats per k (default lowered from 50 -> 30; "
                              "300 capped pairwise comparisons still gives a stable ARI estimate).")
    parser.add_argument("--stability_fraction", type=float, default=0.8)
    parser.add_argument("--stability_step", type=int, default=STABILITY_K_GRID_STEP,
                         help="Grid spacing for the stability pass across [k_min, k_max]. "
                              "Use 1 for a full, non-skipped refine pass over a narrow range "
                              "(e.g. --k_min 4 --k_max 8 --stability_step 1).")
    parser.add_argument("--n_jobs", type=int, default=BOOTSTRAP_N_JOBS,
                         help="Cores for parallel bootstrap fitting (-1 = all).")
    args = parser.parse_args()

    print("Loading data...")
    df, feature_cols = load_and_prepare(args.input)
    print(f"Loaded {len(df)} comments with {len(feature_cols)} features.\n")

    X = df[feature_cols].values
    X_scaled = StandardScaler().fit_transform(X)

    print(f"Fitting GMMs for k = {args.k_min}..{args.k_max}...")
    results = fit_models(X_scaled, k_min=args.k_min, k_max=args.k_max)

    # FIX: full-range, evenly-spaced grid instead of "5 lowest-BIC k" (which,
    # given monotonic BIC, always meant the 5 largest k).
    candidate_ks = build_stability_k_grid(results, step=args.stability_step)
    print(f"\nRunning stability for k grid: {candidate_ks}")

    stability = {k: {"mean_ARI": np.nan, "sd_ARI": np.nan, "n_comparisons": 0} for k in results}

    if args.stability:
        stability.update(
            bootstrap_stability(
                X_scaled, candidate_ks,
                repeats=args.stability_repeats,
                fraction=args.stability_fraction,
                n_jobs=args.n_jobs,
            )
        )

    model_selection_df = pd.DataFrame({
        k: {
            "bic": r["bic"], "aic": r["aic"], "entropy": r["entropy"],
            "min_class_fraction": r["min_class_fraction"],
            "stability_ARI": stability[k]["mean_ARI"],
            "stability_sd": stability[k]["sd_ARI"],
            "stability_comparisons": stability[k]["n_comparisons"],
            "stability_pass": (stability[k]["mean_ARI"] >= 0.60
                                if not np.isnan(stability[k]["mean_ARI"]) else np.nan)
        }
        for k, r in results.items()
    }).T
    model_selection_df.to_csv(f"{OUTPUT_DIR}/model_selection.csv")
    plot_bic_curve(results, stability)

    if args.k is not None:
        best_k = args.k
        print(f"\nUsing manually specified k={best_k}")
    else:
        best_k = select_best_k(results, stability, min_stability_ARI=0.60)
        print(f"\nAutomatically selected k={best_k}")

    print(f"\nSelected k={best_k}. Inspect model_selection.csv before trusting this — "
          "k values NOT in the stability grid above have NaN stability and were only "
          "screened on class size, not on stability.")

    best = results[best_k]
    df, profile = profile_archetypes(df, feature_cols, X_scaled, best["labels"], best["probs"], best_k)

    confidence_summary = (
        df.groupby("archetype")["archetype_confidence"]
        .agg(mean="mean", median="median", min="min", max="max", count="count")
    )
    confidence_summary.to_csv(f"{OUTPUT_DIR}/confidence_summary.csv")

    profile.to_csv(f"{OUTPUT_DIR}/archetype_profiles.csv")
    plot_profile_heatmap(profile, feature_cols)

    df[[TEXT_COL, CONTEXT_COL, SUBCATEGORY_COL, "archetype", "archetype_confidence"]].to_csv(
        f"{OUTPUT_DIR}/archetype_assignments.csv", index=False)

    validation = validate_against_labels(df)
    for col, v in validation.items():
        v["crosstab_normalized"].to_csv(f"{OUTPUT_DIR}/validation_crosstab_{col}.csv")
        v["crosstab_counts"].to_csv(f"{OUTPUT_DIR}/validation_crosstab_{col}_counts.csv")

    export_top_comments(df, best_k)

    print("\nDone. Review, in this order:")
    print("  1. model_selection.csv + bic_curve.png  -> confirm k choice (check stability_pass!)")
    print("  2. archetype_profiles.csv + profile_heatmap.png -> what defines each archetype")
    print("  3. top_comments_per_archetype.csv -> read these BEFORE naming archetypes")
    print("  4. validation_crosstab_*.csv -> is archetype membership tied to subcategory/context?")


if __name__ == "__main__":
    main()
