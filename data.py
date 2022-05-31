import pandas as pd
import numpy as np
from difflib import get_close_matches
from datetime import datetime


def opp_pitcher(x):
    # If player is pitcher, return nothing
    if x["Position"] == "P":
        return np.nan

    series = slate.loc[
        (slate["Team"] == x["Opponent"]) & (slate["Position"] == "P"), "Player"
    ]
    if len(series) > 1:
        raise ValueError("Multiple Opposing Pitchers identified. Data Issues.")
    else:
        return series.iloc[0]


def close_matches(x, possible):
    matches = get_close_matches(x, possible)
    if matches:
        return matches[0]
    else:
        return np.nan


slate = pd.read_csv("./data/slate.csv")

proj = pd.concat([pd.read_csv("./data/proj_1.csv"), pd.read_csv("./data/proj_2.csv")])
# Find closest name matches from slate
proj["Player"] = proj["Player"].apply(lambda x: close_matches(x, slate["Nickname"]))
# Sometimes salary contains commas
proj["Salary"] = proj["Salary"].str.replace(",", "")
proj["Salary"] = proj["Salary"].astype(int)
# Sometimes data is duplicated
proj = proj.drop_duplicates(subset=["Player", "Pos", "Salary"])
proj = proj.dropna()

# Merge projections with slated players
slate = slate.merge(
    proj,
    left_on=["Nickname", "Position", "Salary"],
    right_on=["Player", "Pos", "Salary"],
    how="left",
)

# Drop all pitchers that are not starting
slate = slate.drop(
    slate.loc[(slate["Position"] == "P") & (slate["Probable Pitcher"].isna()), :].index
)
# BIG ASSUMPTION: assume player fills only first position listed.
# Because of the UTIL slot, I assume this has only minimal impact
# upon optimality
slate["Position"] = slate["Position"].str.split("/", expand=True)[0]
# C and 1B players can fill the C/1B slot
slate["Position"] = slate["Position"].replace({"C": "C/1B", "1B": "C/1B"})
# Pitchers have batting order 0
# Non-starting players also have batting order 0
slate["Batting Order"] = slate["Batting Order"].replace(np.nan, 0)
slate["Batting Order"] = slate["Batting Order"].astype(int)
# Opposing Pitcher for each player
slate["Opp_Pitcher"] = slate.apply(opp_pitcher, axis=1)
# Drop players with 0 fantasy points projected
slate = slate[slate["Proj"] > 0]

# Select relevant columns and rename
slate = slate[
    [
        "Player",
        "Position",
        "Salary",
        "Game",
        "Team",
        "Opponent",
        "Batting Order",
        "Opp_Pitcher",
        "Proj",
    ]
]
slate.columns = [
    "Name",
    "Position",
    "Salary",
    "Game",
    "Team",
    "Opponent",
    "Order",
    "Opp_Pitcher",
    "Projection",
]

# Write to csv with todays date
slate.to_csv(f"./data/slate_{datetime.today().strftime('%Y-%m-%d')}.csv", index=False)
