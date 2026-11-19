"""Compare the model's results with your hand labels.

Usage:
    python evaluate.py data/labeling_set.csv

Uses the prompt_version set in config.yaml. Saves mismatches to
errors_<version>.csv so you can study where the model and you disagree.
"""
import sqlite3
import sys

import pandas as pd
import yaml
from sklearn.metrics import cohen_kappa_score

cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))
version = cfg["prompt_version"]
labels_path = sys.argv[1] if len(sys.argv) > 1 else "data/labeling_set.csv"

labels = pd.read_csv(labels_path)
con = sqlite3.connect(cfg["database"])
results = pd.read_sql(
    "SELECT id, primary_theme, secondary_theme, severity, reason "
    "FROM results WHERE prompt_version = ? AND status = 'ok'",
    con,
    params=(version,),
)

m = labels.merge(results, on="id")
if m.empty:
    sys.exit(f"No results for prompt_version {version}. Run classify.py first.")

exact = m["my_theme"] == m["primary_theme"]
lenient = exact | (m["my_theme"] == m["secondary_theme"])
sev_diff = (m["my_severity"] - m["severity"]).abs()

print(f"Prompt {version}: {len(m)} reviews compared (of {len(labels)} labeled)")
print(f"Theme accuracy (exact):    {exact.mean():.0%}")
print(f"Theme accuracy (lenient):  {lenient.mean():.0%}  (counts the model's secondary theme)")
print(f"Cohen's kappa (themes):    {cohen_kappa_score(m['my_theme'], m['primary_theme']):.2f}")
print(f"Severity exact:            {(sev_diff == 0).mean():.0%}")
print(f"Severity within 1 point:   {(sev_diff <= 1).mean():.0%}")
print(f"Mean severity error:       {sev_diff.mean():.2f}")

errors = m.loc[~exact, ["id", "Review", "my_theme", "primary_theme",
                        "secondary_theme", "my_severity", "severity", "reason"]]
errors.to_csv(f"errors_{version}.csv", index=False)
print(f"\n{len(errors)} theme mismatches saved to errors_{version}.csv")
print("\nMost common confusions (your label -> model's label):")
print(errors.groupby(["my_theme", "primary_theme"]).size().sort_values(ascending=False).head(8))