import pandas as pd

df = pd.read_csv("data/reviews.csv")
print("Rows before cleaning:", len(df))

df = df.drop_duplicates(subset="Review")
df = df[df["Review"].str.len() >= 20]
print("Rows after cleaning:", len(df))

df.sample(500, random_state=42).to_csv("data/sample.csv", index=False)
print("Saved 500 rows to data/sample.csv")