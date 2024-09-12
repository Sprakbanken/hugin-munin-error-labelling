import pandas as pd
from pathlib import Path


dfs = []
failed = []
data_dir = Path("__file__").parent.parent / "data"
for data_file in sorted((data_dir / "test_set").glob("*.csv")):
    try:
        dfs.append(
            pd.read_csv(data_file, delimiter="|", encoding="utf-8")
            .reset_index(names="line")
            .assign(csv_name=data_file.stem)
        )
    except Exception as e:
        failed.append(data_file.name)
        print(e)

num_failed = len(failed)
num_total = num_failed + len(dfs)
print(
    f"Failed at reading {num_failed}/{num_total} ({num_failed/num_total:.0%}) files:",
    failed,
)

df = pd.concat(dfs).eval(
    "highlight_url=arkindex_page_url.str.cat(arkindex_line_id, sep='?highlight=')"
)
df["callico_id"] = df["callico_task_url"].str.extract(r"annotation/([\w-]+)/annotate/")
df["arkindex_id"] = df["arkindex_page_url"].str.extract(r"/element/([\w-]+)$")
print(df["callico_id"])


# Plasser highlight_url og arkindex_line_id et praktisk sted
cols = [col for col in df.columns if col not in {"highlight_url", "arkindex_line_id"}]
cols.insert(3, "highlight_url")
cols.append("arkindex_line_id")
df = df[cols]

out_dir = data_dir / "behandlet"
out_dir.mkdir(parents=True, exist_ok=True)
df.to_csv(out_dir / "all_lines.csv", index=False)
df.query("ground_truth != pylaia_prediction").to_csv(
    out_dir / "mistaken_lines.csv", index=False
)
with open(out_dir / "failed.txt", "w") as f:
    f.write(str(failed))


error_types = [
    "Strike-though",
    "Punctuation",
    "Accents",
    "Casing",
    "Ambiguity",
    "Segmentation issue",
    "Rotation",
    "Printed text",
    "Hallucinations",
    "Reading order",
]
df[error_types] = 0
df[error_types] = pd.NA

df.query("ground_truth != pylaia_prediction")[
    [
        "callico_id",
        "arkindex_id",
        "arkindex_line_id",
        "ground_truth",
        "pylaia_prediction",
        "highlight_url",
        *error_types,
    ]
].to_csv(out_dir / "labelling_sheet.csv", index=False)
