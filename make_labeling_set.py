import pandas as pd

df = pd.read_csv("data/sample.csv")
df.insert(0, "id", range(len(df)))
df.to_csv("data/sample.csv", index=False)  # adds an id column

labeling = df.sample(50, random_state=7)[["id", "Review", "Rating"]]
labeling["my_theme"] = ""
labeling["my_severity"] = ""
labeling.to_csv("data/labeling_set.csv", index=False)
print("Saved 50 reviews to data/labeling_set.csv")