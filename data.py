import pandas as pd
import numpy as np
from difflib import get_close_matches
from datetime import datetime
from bs4 import BeautifulSoup
import quopri
import re

name_pattern = re.compile(r"\([^()]*\)")
team_pattern = re.compile(r"[\$,]")


def linestar_proj():
    html = open("./data/proj.mhtml", "r")
    html = quopri.decodestring(html.read())
    soup = BeautifulSoup(html, features="html.parser")

    table = soup.find_all("table")[0]

    # Find column index numbers for columns we want
    # This is to avoid issues when columns change between before games and after games
    header_idx = {
        "Player": None,
        "Salary": None,
        "Consensus": None,
        "Order": None,
        "pOwn": None,
    }
    # Ignore first columns that have filters etcetera
    for num, header in enumerate(table.find_all("th")[2:]):
        if header.text in header_idx.keys():
            # Position column always in front, so add 1 to index value
            header_idx[header.text] = num + 1

    data = []
    rows = table.find_all("tr", class_="playerCardRow")
    for row in rows:
        # Ignore first 4 columns that have checkboxes and other things
        cells = row.find_all("td")[4:]
        player_data = {
            "Position": cells[0].text,
            "Player": name_pattern.sub(
                "", cells[header_idx["Player"]].find(class_="playername").text
            ).rstrip(),
            "Team": cells[header_idx["Player"]].find(class_="playerTeam").text[2:],
            "Salary": int(team_pattern.sub("", cells[header_idx["Salary"]].text)),
            "Consensus": float(cells[header_idx["Consensus"]].text),
            "Order": int(cells[header_idx["Order"]].text.replace("-", "0")),
            "pOwn": float(cells[header_idx["pOwn"]].text.replace("%", "")) / 100,
        }
        data.append(player_data)
    return pd.DataFrame(data)


def opp_pitcher(x):
    # If player is pitcher, return nothing
    if x["Position"] == "P":
        return np.nan

    series = slate.loc[
        (slate["Team"] == x["Opponent"]) & (slate["Position"] == "P"), "Id"
    ]
    if len(series) == 0:
        return np.nan
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


# Linestar Slate
slate = pd.read_csv("./data/slate.csv")

proj = linestar_proj()
proj["Player"] = proj["Player"].apply(lambda x: close_matches(x, slate["Nickname"]))

slate = slate.merge(
    proj,
    left_on=["Nickname", "Team", "Salary", "Position"],
    right_on=["Player", "Team", "Salary", "Position"],
    how="left",
)

# Get historical points scored standard deviation for players
stds = pd.read_csv("./data/stds.csv")
stds["Name"] = stds["Name"].apply(lambda x: close_matches(x, slate["Nickname"]))
# Merge stds to slate
slate = slate.merge(stds, left_on="Nickname", right_on="Name", how="left")
# Set any unfilled player standard deviations to the mean value
slate["Scored_Std"] = slate["Scored_Std"].replace(np.nan, slate["Scored_Std"].mean())

# Drop duplicate rows before adjusting position column
slate = slate.drop_duplicates(subset=["Nickname", "Position", "Team"])
slate = slate.dropna(subset=["Player", "Consensus", "Order"])
# Drop all pitchers that are not starting
slate = slate.drop(
    slate[(slate["Position"] == "P") & (slate["Probable Pitcher"].isna())].index
)
# BIG ASSUMPTION: assume player fills only first position listed.
# Because of the UTIL slot, I assume this has only minimal impact
# upon optimality
slate["Position"] = slate["Position"].str.split("/", expand=True)[0]
# C and 1B players can fill the C/1B slot
slate["Position"] = slate["Position"].replace({"C": "C/1B", "1B": "C/1B"})
slate["Order"] = slate["Order"].astype(int)
# Drop batters who aren't starting
slate = slate.drop(slate[(slate["Order"] == 0) & (slate["Position"] != "P")].index)
# Drop batters with injuries
slate = slate[slate["Injury Indicator"].isna()]
# Opposing Pitcher for each player
slate["Opp_Pitcher"] = slate.apply(opp_pitcher, axis=1)
# Sometimes teams dont have a probable pitcher listed, so drop when teams don't
# have an opposing pitcher
slate = slate.drop(
    slate[(slate["Position"] != "P") & (slate["Opp_Pitcher"].isna())].index
)
# Only care about players with positive projections
slate = slate[slate["Consensus"] > 0]

slate = slate[
    [
        "Nickname",
        "Id",
        "Position",
        "Salary",
        "Game",
        "Team",
        "Opponent",
        "Order",
        "Opp_Pitcher",
        "Consensus",
        "pOwn",
        "Scored_Std",
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
    "Proj_Ownership",
    "Hist_Std",
]

# Write to csv with todays date
slate.to_csv(f"./data/slate_{datetime.today().strftime('%Y-%m-%d')}.csv", index=False)
