import pandas as pd
import numpy as np
from difflib import get_close_matches
from datetime import datetime


def opp_pitcher(x):
    # If player is pitcher, return nothing
    if x["Position"] == "P":
        return np.nan

    series = slate.loc[
        (slate["Team"] == x["Opponent"]) & (slate["Position"] == "P"), "Id"
    ]
    # Error if no opposing pitcher is found
    if len(series) == 0:
        raise ValueError(f"{x} doesn't have an opposing pitcher")
    # Error if more than one opposing pitcher is found
    if len(series) > 1:
        raise ValueError("Multiple Opposing Pitchers identified. Data Issues.")
    else:
        return series.iloc[0]


def close_matches(x, possible):
    matches = get_close_matches(x, possible, cutoff=0.80)
    if matches:
        return matches[0]
    else:
        return np.nan


# FantasyData projection slate
slate = pd.read_csv("./data/slate.csv")
# Drop any players that have any kind of injury indicator
slate = slate[slate["Injury Indicator"].isna()]
# Drop non-starting pitchers
slate = slate.drop(
    slate[(slate["Position"] == "P") & (slate["Probable Pitcher"].isna())].index
)
# For players that play multiple positions, assume they only play the first listed
# Because of the UTIL slot, I assume this has little impact on optimality
# TODO: Refactor optimization to account for multiple position players
slate["Position"] = slate["Position"].str.split("/", expand=True)[0]
# Convert C and 1B position players to C/1B position
slate["Position"] = slate["Position"].replace({"C": "C/1B", "1B": "C/1B"})
# Get opposing pitchers for each player
slate["Opp_Pitcher"] = slate.apply(opp_pitcher, axis=1)
slate = slate[
    ["Nickname", "Id", "Position", "Salary", "Game", "Team", "Opponent", "Opp_Pitcher"]
]


proj = pd.concat(
    [pd.read_csv("./data/pitchers.csv"), pd.read_csv("./data/batters.csv")]
)
# Match names in projection data to slate data
proj["Name"] = proj["Name"].apply(lambda x: close_matches(x, slate["Nickname"]))
# Drop players with non-matching names
proj = proj.dropna(subset="Name")
proj = proj[["Name", "Team", "BattingOrder", "FantasyPointsFanDuel"]]


slate = slate.merge(
    proj, left_on=["Nickname", "Team"], right_on=["Name", "Team"], how="left"
)
# Only consider players with >0 projections
slate = slate[slate["FantasyPointsFanDuel"] > 0]
# Drop non-pitchers with no batting order
slate = slate.drop(
    slate[(slate["Position"] != "P") & (slate["BattingOrder"].isna())].index
)
# Set pitchers to have 0 batting order
slate.loc[slate["Position"] == "P", "BattingOrder"] = 0
# Change batting order and salary to integer
slate["BattingOrder"] = slate["BattingOrder"].astype(int)
slate["Salary"] = slate["Salary"].astype(int)
slate = slate[
    [
        "Name",
        "Id",
        "Position",
        "Salary",
        "Game",
        "Team",
        "Opponent",
        "BattingOrder",
        "Opp_Pitcher",
        "FantasyPointsFanDuel",
    ]
]
slate.columns = [
    "Name",
    "ID",
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
slate.to_csv(
    f"./data/slates/slate_{datetime.today().strftime('%Y-%m-%d')}.csv", index=False
)
