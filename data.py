import pandas as pd
import numpy as np
from difflib import get_close_matches
from datetime import datetime
import os


def opp_pitcher(x):
    # If player is pitcher, return nothing
    if x["Position"] == "P":
        return np.nan

    series = slate.loc[
        (slate["Team"] == x["Opponent"]) & (slate["Position"] == "P"), "ID"
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


def make_game(row):
    if row["HomeOrAway"] == "HOME":
        return f"{row['Opponent']}@{row['Team']}"
    elif row["HomeOrAway"] == "AWAY":
        return f"{row['Team']}@{row['Opponent']}"


slate = pd.read_csv("./data/fanduel.csv")
players = pd.concat(
    [pd.read_csv("./data/pitchers.csv"), pd.read_csv("./data/batters.csv")]
)
players["Game"] = players.apply(make_game, axis=1)

slate = slate.merge(players, on=["Name", "Team", "Opponent"], how="left")
# FantasyData updates any players that aren't starting pitchers or batters to 0 projection
# so this removes any players we never want to pick
slate = slate[slate["FantasyPointsFanDuel_x"] > 0]
slate = slate[
    [
        "Name",
        "Position_x",
        "OperatorSalary",
        "Game",
        "Team",
        "Opponent",
        "BattingOrder",
        "FantasyPointsFanDuel_x",
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
    "Projection",
]

fd_slate = pd.read_csv("./data/slate.csv")
fd_slate["Nickname"] = fd_slate["Nickname"].apply(
    lambda x: close_matches(x, slate["Name"])
)
fd_slate = fd_slate.rename(columns={"Nickname": "Name"})

# Possible Issue here: two players having the same name and same salary
# Can't join on Team because FantasyData uses different team abbreviations
# We need the FanDuel slate information only for the IDs for CSV import
slate = slate.merge(fd_slate[["Name", "Id", "Salary"]], how="left")
slate = slate.rename(columns={"Id": "ID"})
slate["Opp_Pitcher"] = slate.apply(opp_pitcher, axis=1)
slate.loc[slate["Order"].isna(), "Order"] = 0
slate["Order"] = slate["Order"].astype(int)
# C and 1B players can fill the C/1B slot
slate["Position"] = slate["Position"].replace({"C": "C/1B", "1B": "C/1B"})

slate = slate[
    [
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
]

# Write to csv with todays date
slate.to_csv(
    f"./data/slates/slate_{datetime.today().strftime('%Y-%m-%d')}.csv", index=False
)

# Remove files after writing slate so errors are raised if they aren't updated
os.remove("./data/slate.csv")
os.remove("./data/batters.csv")
os.remove("./data/pitchers.csv")
os.remove("./data/fanduel.csv")
