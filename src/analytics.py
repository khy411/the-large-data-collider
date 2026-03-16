import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import seaborn as sns
from pathlib import Path
from sklearn.metrics import roc_auc_score
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import OUTPUT_DIR

# colors
CLR_BG = 'dodgerblue'
CLR_SIG = 'darkviolet'

FEATURE_NAMES = [
    "lepton_pT", "lepton_eta", "lepton_phi",
    "missing_energy_magnitude", "missing_energy_phi",
    "jet1_pt", "jet1_eta", "jet1_phi", "jet1_b_tag",
    "jet2_pt", "jet2_eta", "jet2_phi", "jet2_b_tag",
    "jet3_pt", "jet3_eta", "jet3_phi", "jet3_b_tag",
    "jet4_pt", "jet4_eta", "jet4_phi", "jet4_b_tag",
    "m_jj", "m_jjj", "m_lv", "m_jlv",
    "m_bb", "m_wbb", "m_wwbb"
]

MASS_FEATURES = ["m_bb", "m_wwbb", "m_wbb", "m_jj"]

JET_PT_FEATURES = ["jet1_pt", "jet2_pt", "jet3_pt", "jet4_pt"]


# plot 1: feature distributions
def plot_feature_distributions(df: pd.DataFrame):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    signal = df[df["label"] == 1]
    background = df[df["label"] == 0]

    fig, axes = plt.subplots(nrows=7, ncols=4, figsize=(22, 26))
    fig.suptitle("Higgs Boson Dataset - Feature Distributions\nSignal (violet) vs Background (blue)",
                 fontsize=16, fontweight='bold', y=0.98)
    axes = axes.flatten()

    for i, feat in enumerate(FEATURE_NAMES):
        ax = axes[i]
        ax.hist(background[feat], bins=50, alpha=0.6, color=CLR_BG,
                label='Background', density=True)
        ax.hist(signal[feat], bins=50, alpha=0.6, color=CLR_SIG,
                label='Signal', density=True)
        ax.set_title(feat, fontsize=9, fontweight='bold')
        ax.set_xlabel("Value", fontsize=7)
        ax.set_ylabel("Density", fontsize=7)
        ax.tick_params(labelsize=6)
        if i == 0:
            ax.legend(fontsize=7)

    for j in range(len(FEATURE_NAMES), len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out_path = OUTPUT_DIR / "feature_distributions.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.close()


# plot 2: correlation heatmap 
def plot_correlation_heatmap(df: pd.DataFrame):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    corr = df[FEATURE_NAMES + ["label"]].corr()

    fig, ax = plt.subplots(figsize=(18, 15))
    sns.heatmap(
        corr,
        ax=ax,
        cmap=sns.diverging_palette(220, 280, as_cmap=True),
        center=0,
        vmin=-1, vmax=1,
        annot=False,
        linewidths=0.4,
        linecolor='white',
        square=True,
        cbar_kws={"shrink": 0.8, "label": "Pearson Correlation"}
    )
    ax.set_title("Feature Correlation Heatmap", fontsize=16, fontweight='bold', pad=20)
    ax.tick_params(axis='x', rotation=45, labelsize=8)
    ax.tick_params(axis='y', rotation=0, labelsize=8)

    plt.tight_layout()
    out_path = OUTPUT_DIR / "correlation_heatmap.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.close()


# plot 3: AUC separation score per feature
def plot_auc_separation(df: pd.DataFrame):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    scores = {}
    for feat in FEATURE_NAMES:
        auc = roc_auc_score(df["label"], df[feat])
        # flip scores below 0.5 (negative correlation is still separation)
        scores[feat] = max(auc, 1 - auc)

    scores_sorted = dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))

    fig, ax = plt.subplots(figsize=(12, 9))
    bars = ax.barh(
        list(scores_sorted.keys()),
        list(scores_sorted.values()),
        color=[CLR_SIG if v >= 0.55 else CLR_BG for v in scores_sorted.values()],
        alpha=0.85,
        edgecolor='white'
    )
    ax.axvline(x=0.5, color='gray', linestyle='--', linewidth=1.2, label='No separation (AUC=0.5)')
    ax.set_xlabel("AUC Score (higher = better separation)", fontsize=11)
    ax.set_title("Feature Separation Power\nSignal vs Background (AUC per Feature)",
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=9)
    ax.set_xlim(0.45, 0.75)
    ax.tick_params(labelsize=9)
    ax.invert_yaxis()

    # annotate bars
    for bar, val in zip(bars, scores_sorted.values()):
        ax.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
                f'{val:.3f}', va='center', fontsize=7)

    plt.tight_layout()
    out_path = OUTPUT_DIR / "auc_separation.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.close()


# plot 4: missing jet rate
def plot_missing_jet_rate(df: pd.DataFrame):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    zero_rates = {feat: (df[feat] == 0).mean() * 100 for feat in JET_PT_FEATURES}

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(
        zero_rates.keys(),
        zero_rates.values(),
        color=[CLR_BG, CLR_BG, CLR_SIG, CLR_SIG],
        alpha=0.85,
        edgecolor='white',
        width=0.5
    )
    ax.set_ylabel("Missing Jet Rate (%)", fontsize=11)
    ax.set_title("Missing Jet Rate per Jet\n(% of events where jet pT = 0 after sentinel fix)",
                 fontsize=13, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.tick_params(labelsize=10)

    for bar, val in zip(bars, zero_rates.values()):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 1,
                f'{val:.1f}%', ha='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    out_path = OUTPUT_DIR / "missing_jet_rate.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.close()


# plot 5: invariant mass distributions
def plot_invariant_mass(df: pd.DataFrame):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    signal = df[df["label"] == 1]
    background = df[df["label"] == 0]

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 10))
    fig.suptitle("Invariant Mass Feature Distributions\nSignal (violet) vs Background (blue)",
                 fontsize=15, fontweight='bold', y=1.01)
    axes = axes.flatten()

    for i, feat in enumerate(MASS_FEATURES):
        ax = axes[i]
        ax.hist(background[feat], bins=60, alpha=0.6, color=CLR_BG,
                label='Background', density=True)
        ax.hist(signal[feat], bins=60, alpha=0.6, color=CLR_SIG,
                label='Signal', density=True)
        ax.set_title(feat, fontsize=13, fontweight='bold')
        ax.set_xlabel("Value", fontsize=10)
        ax.set_ylabel("Density", fontsize=10)
        ax.legend(fontsize=9)
        ax.tick_params(labelsize=9)

    plt.tight_layout()
    out_path = OUTPUT_DIR / "invariant_mass_distributions.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.close()


# summary stats
def print_summary_stats(df: pd.DataFrame):
    print("\nDataset Summary")
    print(f"Total records : {len(df):,}")
    print(f"Signal (1)    : {(df['label']==1).sum():,} ({(df['label']==1).mean()*100:.1f}%)")
    print(f"Background (0): {(df['label']==0).sum():,} ({(df['label']==0).mean()*100:.1f}%)")
    print(f"\nFeature Stats")
    print(df[FEATURE_NAMES].describe().round(4).to_string())


# main 
if __name__ == "__main__":
    from writer import load_parquet

    print("Loading Parquet...")
    df = load_parquet("higgs_train_sample.parquet")
    print_summary_stats(df)

    print("\nGenerating plots...")
    plot_feature_distributions(df)
    plot_correlation_heatmap(df)
    plot_auc_separation(df)
    plot_missing_jet_rate(df)
    plot_invariant_mass(df)

    print("\nDone")