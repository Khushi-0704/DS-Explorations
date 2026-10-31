from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save plots only, never open windows
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.ticker import MaxNLocator

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


# ---------------------------------------------------------------- helpers
def has_col(df, col, what):
    """True if col is set and exists in df, otherwise warn and skip."""
    if not col:
        return False
    if col not in df.columns:
        print(f"  ! skipping {what}: column '{col}' not found")
        return False
    return True


def tidy(ax, count_axis="y"):
    """Whole-number ticks on the count axis + room for the value labels."""
    if count_axis == "y":
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.margins(y=0.12)
    else:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.margins(x=0.15)


def natural_order(index):
    """Sort labels like 5-9 before 10-19 (by first number); else alphabetical."""
    s = pd.Series(index.astype(str))
    start = s.str.extract(r"(\d+)")[0].astype(float)
    if start.notna().all():
        return [index[i] for i in start.sort_values(kind="stable").index]
    return sorted(index, key=str)


def save_fig(path, dpi):
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()


def safe_name(col):
    return col.replace(".", "_").replace(" ", "_")


# ---------------------------------------------------------------- analyses
def missing_values(df, name, out_dir, dpi):
    count = df.isnull().sum()
    pct = count / len(df) * 100
    summary = pd.DataFrame({"missing_count": count, "missing_pct": pct})
    summary.to_csv(out_dir / "missing_values.csv")
    print(summary)

    pct = pct.sort_values()
    plt.figure(figsize=(10, max(4, 0.35 * len(pct))))
    bars = plt.barh(pct.index, pct.values)
    for bar, v in zip(bars, pct.values):
        plt.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                 f"{v:.1f}%", va="center")
    plt.xlim(0, 100)
    plt.xlabel("Missing Values (%)")
    plt.ylabel("Column")
    plt.title(f"Missing Values by Variable - {name}")
    save_fig(out_dir / "missing_values.png", dpi)


def count_barh(df, col, label, title, fname, out_dir, dpi):
    counts = df[col].dropna().value_counts()
    pct = counts / counts.sum() * 100
    data = counts.sort_values(ascending=True)

    plt.figure(figsize=(10, max(4, 0.35 * len(data))))
    bars = plt.barh(data.index.astype(str), data.values)
    for bar, key in zip(bars, data.index):
        plt.text(bar.get_width(), bar.get_y() + bar.get_height() / 2,
                 f"  {int(bar.get_width())} ({pct[key]:.1f}%)", va="center")
    plt.xlabel("Number of Cases")
    plt.ylabel(label)
    plt.title(title)
    tidy(plt.gca(), "x")
    save_fig(out_dir / fname, dpi)
    counts.to_csv(out_dir / fname.replace(".png", ".csv"))


def count_bar(df, col, title, fname, out_dir, dpi):
    counts = df[col].value_counts()
    counts = counts.reindex(natural_order(counts.index))

    plt.figure(figsize=(10, 6))
    bars = plt.bar(counts.index.astype(str), counts.values)
    for bar, c in zip(bars, counts.values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 str(c), ha="center", va="bottom")
    plt.xlabel(col)
    plt.ylabel("Number of Cases")
    plt.title(title)
    plt.xticks(rotation=45)
    tidy(plt.gca(), "y")
    save_fig(out_dir / fname, dpi)
    counts.to_csv(out_dir / fname.replace(".png", ".csv"))


def monthly_cases(df, title, out_dir, dpi):
    monthly = df.groupby("Month").size().reindex(range(1, 13), fill_value=0)
    monthly.index = MONTHS
    plt.figure(figsize=(12, 6))
    bars = plt.bar(monthly.index, monthly.values)
    for bar, c in zip(bars, monthly.values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 str(c), ha="center", va="bottom")
    plt.xlabel("Month")
    plt.ylabel("Number of Cases")
    plt.title(title)
    tidy(plt.gca(), "y")
    save_fig(out_dir / "monthly_cases.png", dpi)
    monthly.to_csv(out_dir / "monthly_cases.csv")


def weekly_cases(df, date_col, title, out_dir, dpi):
    d = df.dropna(subset=[date_col]).copy()
    # 7-day blocks counted from Jan 1, so weeks line up with calendar months
    d["Week"] = (d[date_col].dt.dayofyear - 1) // 7 + 1
    weekly = d.groupby("Week").size()
    month_ranges = d.groupby("Month")["Week"].agg(["min", "max"])

    fig, ax = plt.subplots(figsize=(16, 7))
    ax.plot(weekly.index, weekly.values, marker="o")
    for week, c in weekly.items():
        ax.annotate(str(c), (week, c), xytext=(0, 6),
                    textcoords="offset points", ha="center", fontsize=8)
    for month, row in month_ranges.iterrows():
        ax.text((row["min"] + row["max"]) / 2, -0.14, MONTHS[month - 1],
                transform=ax.get_xaxis_transform(), ha="center",
                fontsize=11, fontweight="bold")
        ax.axvline(row["max"] + 0.5, linestyle="--", alpha=0.4)

    ax.set_xticks(range(int(weekly.index.min()), int(weekly.index.max()) + 1))
    ax.set_xlabel("Week of Year")
    ax.set_ylabel("Number of Cases")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    tidy(ax, "y")
    plt.subplots_adjust(bottom=0.22)
    plt.savefig(out_dir / "weekly_cases_monthly_grouping.png",
                dpi=dpi, bbox_inches="tight")
    plt.close()
    weekly.to_csv(out_dir / "weekly_cases.csv")


def top_subregions(df, region, subregion, labels, top_n, title, out_dir, dpi):
    t = (df.groupby([region, subregion]).size().reset_index(name="Cases")
           .sort_values("Cases", ascending=False).head(top_n)
           .sort_values("Cases", ascending=True))
    t["label"] = (t[subregion].astype(str) + " ("
                  + t[region].astype(str) + ")")
    plt.figure(figsize=(11, 7))
    bars = plt.barh(t["label"], t["Cases"])
    for bar, v in zip(bars, t["Cases"]):
        plt.text(bar.get_width(), bar.get_y() + bar.get_height() / 2,
                 f" {v}", va="center")
    plt.xlabel("Number of Cases")
    plt.ylabel(f"{labels['subregion']} ({labels['region']})")
    plt.title(title)
    tidy(plt.gca(), "x")
    save_fig(out_dir / "top_subregions.png", dpi)
    t.to_csv(out_dir / "top_subregions.csv", index=False)


def heatmap(table, xlabel, ylabel, title, fname, max_rows, out_dir, dpi):
    # keep only the busiest rows so big datasets stay readable
    table = table.loc[table.sum(axis=1).sort_values(ascending=False)
                      .head(max_rows).index]
    plt.figure(figsize=(14, max(6, 0.35 * len(table))))
    sns.heatmap(table, annot=True, fmt="d", linewidths=0.5)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    save_fig(out_dir / fname, dpi)
    table.to_csv(out_dir / fname.replace(".png", ".csv"))


def numeric_hists(df, cols, out_dir, dpi):
    for col in cols:
        plt.figure(figsize=(8, 5))
        df[col].dropna().plot(kind="hist", bins=30)
        plt.xlabel(col)
        plt.title(f"Distribution of {col}")
        tidy(plt.gca(), "y")
        save_fig(out_dir / f"hist_{safe_name(col)}.png", dpi)


# ---------------------------------------------------------------- one dataset
def run_eda(df, name, cfg, out_dir):
    c, lab, p = cfg["columns"], cfg["labels"], cfg["params"]
    dpi = p["dpi"]
    date_col, region, subregion = c.get("date"), c.get("region"), c.get("subregion")
    case = lab["case_label"]
    summary = {"dataset": name, "rows": len(df), "columns": df.shape[1]}

    print("=" * 60)
    print(f"EDA FOR {name}  ({len(df)} rows, {df.shape[1]} columns)")
    print("=" * 60)

    # missing values + duplicates (work on any dataset)
    missing_values(df, name, out_dir, dpi)
    summary["duplicate_rows"] = int(df.duplicated().sum())
    print("Duplicate records:", summary["duplicate_rows"])

    # region counts
    if has_col(df, region, "region counts"):
        count_barh(df, region, lab["region"],
                   f"Reported {case} by {lab['region']} - {name}",
                   "region_cases.png", out_dir, dpi)

    # date based plots
    if has_col(df, date_col, "date plots"):
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df["Month"] = df[date_col].dt.month
        summary["unparsed_dates"] = int(df[date_col].isna().sum())

        monthly_cases(df, f"Monthly {case} - {name}", out_dir, dpi)
        weekly_cases(df, date_col,
                     f"Weekly {case} with Monthly Grouping - {name}",
                     out_dir, dpi)

        if has_col(df, region, "region x month heatmap"):
            table = (df.groupby([region, "Month"]).size()
                       .unstack(fill_value=0)
                       .reindex(columns=range(1, 13), fill_value=0))
            table.columns = MONTHS
            heatmap(table, "Month", lab["region"],
                    f"{lab['region']}-wise Monthly {case} - {name}",
                    "region_month_heatmap.png", p["heatmap_max_rows"],
                    out_dir, dpi)

        for m in p.get("compare_months") or []:
            summary[f"cases_month_{m}"] = int((df["Month"] == m).sum())
        print({k: v for k, v in summary.items() if k.startswith("cases_month")})

    # top subregions
    if has_col(df, region, "top subregions") and has_col(df, subregion, "top subregions"):
        top_subregions(df, region, subregion, lab, p["top_n"],
                       f"Top {p['top_n']} {lab['subregion']}s by {case} - {name}",
                       out_dir, dpi)

    # other categorical columns (age range etc.)
    for col in c.get("breakdowns") or []:
        if not has_col(df, col, f"breakdown {col}"):
            continue
        count_bar(df, col, f"{case} by {col} - {name}",
                  f"{safe_name(col)}_cases.png", out_dir, dpi)
        if has_col(df, region, f"{col} heatmap"):
            table = df.groupby([region, col]).size().unstack(fill_value=0)
            table = table[natural_order(table.columns)]
            heatmap(table, col, lab["region"],
                    f"{lab['region']}-wise {case} by {col} - {name}",
                    f"region_{safe_name(col)}_heatmap.png",
                    p["heatmap_max_rows"], out_dir, dpi)

    # numeric columns
    num = c.get("numeric", [])
    if num == "auto":
        num = list(df.select_dtypes("number").columns)[: p["max_numeric_cols"]]
    num = [x for x in num if has_col(df, x, f"histogram {x}")]
    numeric_hists(df, num, out_dir, dpi)

    pd.Series(summary).to_csv(out_dir / "summary.csv", header=False)


# ---------------------------------------------------------------- entry point
def run(cfg: dict, out_dir: Path) -> None:
    files = sorted(Path(cfg["data"]["dir"]).glob(cfg["data"]["pattern"]))
    if not files:
        raise FileNotFoundError(
            f"No files matching {cfg['data']['pattern']} in {cfg['data']['dir']}"
        )

    date_col = cfg["columns"].get("date")

    for f in files:
        name = f.stem
        print(f"\nProcessing {name}...")

        df = pd.read_csv(f)
        df = df.rename(columns=cfg.get("rename_columns", {}).get(name, {}))

        if cfg["data"].get("split_by_year") and has_col(df, date_col, "year split"):
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            for year, part in df.groupby(df[date_col].dt.year):
                label = f"{name}_{int(year)}"
                ds_out = out_dir / label
                ds_out.mkdir(parents=True, exist_ok=True)
                run_eda(part.copy(), label, cfg, ds_out)
                print(f"Finished {label}")
        else:
            ds_out = out_dir / name
            ds_out.mkdir(parents=True, exist_ok=True)
            run_eda(df, name, cfg, ds_out)
            print(f"Finished {name}")