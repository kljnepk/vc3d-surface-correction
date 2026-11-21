"""Summarize the paired scores. Use --figure to regenerate the README figure."""
import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import fmean

ROOT = Path(__file__).resolve().parent
METHODS = ("VC3D", "R094", "R103")


def read_results():
    starts = defaultdict(dict)
    with (ROOT / "paired_scores.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            key, method = row["fixed_start"], row["method"]
            if method in starts[key]:
                raise ValueError(f"Duplicate result for {key} and {method}")
            starts[key][method] = row
    for key, rows in starts.items():
        if set(rows) != set(METHODS):
            raise ValueError(f"Incomplete scores for {key}")
    return starts


def make_figure(regions, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    green, rust, slate = "#197868", "#aa4a23", "#626e74"
    fig, ax = plt.subplots(figsize=(9.2, 5.6))
    fig.subplots_adjust(left=.24, right=.97, bottom=.20, top=.82)
    fig.text(.055, .947, "Change in target-sheet area", weight="bold", fontsize=19, color="#16313a")
    fig.text(.055, .893, "R103 compared with VC3D on 28 starts in seven regions of PHerc. Paris. 4",
             fontsize=10.5, color="#465761")
    ax.axvline(0, color="#4e626a", linewidth=1.2, zorder=2)
    for index, deltas in enumerate(regions.values()):
        if index % 2 == 0:
            ax.axhspan(index - .5, index + .5, color="#f2f5f5", zorder=0)
        for delta, displacement in zip(deltas, (-.27, -.09, .09, .27)):
            color = green if delta > 1e-6 else rust if delta < -1e-6 else slate
            ax.plot(delta, index + displacement, "o", color=color, markersize=6, zorder=3)
    ax.set_yticks(range(len(regions)), list(regions), fontfamily="monospace", fontsize=9.5)
    ax.set_ylim(len(regions) - .5, -.5)
    ax.set_xlim(-68, 337)
    ax.set_xticks([-50, 0, 50, 100, 150, 200, 250, 300])
    ax.set_xlabel("Paired difference in reference area (voxel²)", labelpad=9)
    ax.grid(axis="x", color="#d9e0e0", linewidth=.65)
    for edge in ax.spines.values():
        edge.set_visible(False)
    ax.tick_params(length=0, pad=6)
    keys = [Line2D([], [], linestyle="none", marker="o", markersize=6, color=c, label=s)
            for c, s in [(green, "R103 higher"), (rust, "VC3D higher"), (slate, "Tie")]]
    fig.legend(handles=keys, loc="lower center", bbox_to_anchor=(.53, .012),
               ncol=3, frameon=False, fontsize=9)
    fig.savefig(path, dpi=155, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure", action="store_true", help="Regenerate paired_changes.png using matplotlib")
    args = parser.parse_args()
    starts = read_results()
    print(f"{len(starts)} matched starts")
    for method in METHODS:
        rows = [arms[method] for arms in starts.values()]
        target = fmean(float(r["target_area_vox2"]) for r in rows)
        other = fmean(float(r["other_sheet_area_vox2"]) for r in rows)
        print(f"{method:5}  target {target:.4f}  other {other:.4f}")
    regions = defaultdict(list)
    for case, methods in sorted(starts.items()):
        delta = float(methods["R103"]["target_area_vox2"]) - float(methods["VC3D"]["target_area_vox2"])
        regions[case.rsplit("_seed_", 1)[0]].append(delta)
    deltas = [v for values in regions.values() for v in values]
    gains = sum(v > 1e-6 for v in deltas)
    losses = sum(v < -1e-6 for v in deltas)
    baseline = fmean(float(s["VC3D"]["target_area_vox2"]) for s in starts.values())
    print(f"Mean paired change: {fmean(deltas):+.4f} voxel² ({100 * fmean(deltas) / baseline:+.2f}%)")
    print(f"Higher: {gains}. Lower: {losses}. Tied: {len(deltas) - gains - losses}.")
    print("Mean paired change by region (voxel²)")
    for region, values in regions.items():
        print(f"  {region}  {fmean(values):+.4f}")
    if args.figure:
        make_figure(regions, ROOT / "paired_changes.png")


if __name__ == "__main__":
    main()
