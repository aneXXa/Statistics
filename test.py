# %% Load the survey
import importlib
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display

import upSetPlotFunc

importlib.reload(upSetPlotFunc)
from upSetPlotFunc import upset_plot

csv_path = (
    Path(__file__).with_name("test.csv")
    if "__file__" in globals()
    else Path("test.csv")
)
df = pd.read_csv(csv_path)

print(f"{len(df):,} rows, {df.shape[1]} columns")
display(df.head())

# %% Encode each row as set membership
# UpSet needs boolean columns (one set per feature). The CSV is a survey, so
# each "set" is a flag: the people who reported that condition.
membership = pd.DataFrame(
    {
        "Student": df["Working Professional or Student"].eq("Student"),
        "Suicidal thoughts": df["Have you ever had suicidal thoughts ?"].eq("Yes"),
        "Family history": df["Family History of Mental Illness"].eq("Yes"),
        "Unhealthy diet": df["Dietary Habits"].eq("Unhealthy"),
        "Short sleep (<5h)": df["Sleep Duration"].eq("Less than 5 hours"),
        "High financial stress": df["Financial Stress"].ge(4),
        "Long hours (>=10)": df["Work/Study Hours"].ge(10),
        "High pressure": df["Work Pressure"].ge(4) | df["Academic Pressure"].ge(4),
    }
)

print("People in each set:")
display(membership.sum().rename("n").to_frame())
print(
    f"{membership.any(axis=1).sum():,} rows belong to at least one set; "
    f"{(~membership.any(axis=1)).sum():,} belong to none (dropped by the plot)."
)

# %% Draw the UpSet plot
# upset_plot closes the figure in pyplot so Jupyter does not auto-double it;
# display once here (do not also call plt.show()).
result = upset_plot(
    membership,
    title="Survey flag co-occurrence",
    max_intersections=50,
    sort_sets="size",
    show_set_counts=True,
    intersection_color_mode="exclusive-set",
    set_color=[
        "#4C78A8",
        "#E45756",
        "#F58518",
        "#54A24B",
        "#B279A2",
        "#72B7B2",
        "#ECAE3E",
        "#9D755D",
    ],
    color_dots_by_set=True,
)
display(result.fig)

# %% Same membership, vertical layout
vertical = upset_plot(
    membership,
    orientation="vertical",
    title="Survey flag co-occurrence (vertical)",
    max_intersections=None,
    sort_sets="size",
    show_set_counts=True,
    intersection_color_mode="exclusive-set",
    set_color=[
        "#4C78A8",
        "#E45756",
        "#F58518",
        "#54A24B",
        "#B279A2",
        "#72B7B2",
        "#ECAE3E",
        "#9D755D",
    ],
    color_dots_by_set=True,
)
display(vertical.fig)

# %%
