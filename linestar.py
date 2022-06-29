from bs4 import BeautifulSoup
import quopri
import pandas as pd
import numpy as np
from os import listdir
import re


name_pattern = re.compile(r"\([^()]*\)")
team_pattern = re.compile(r"[\$,]")


def get_opponent(row):
    """
    For full linestar historical data, use the opponent column to
    determine opposing team and opposing pitcher, if relevant
    """
    if "," in row["Opponent"]:
        opp_pitcher, opp_team = row["Opponent"].split(",")
        return (opp_team[1:], opp_pitcher)
    elif "@" in row["Opponent"]:
        opp_team = row["Opponent"].split("@")[1]
        return (opp_team, np.nan)
    elif "vs" in row["Opponent"]:
        opp_team = row["Opponent"].split("vs")[1][1:]
        return (opp_team, np.nan)


def linestar_hist_full(file):
    html = open(file, "r")
    html = quopri.decodestring(html.read())
    soup = BeautifulSoup(html, features="html.parser")

    table = soup.find_all("table")[0]

    # Find column index numbers for columns we want
    # This is to avoid issues when columns change between before games and after games
    header_idx = {
        "Player": None,
        "Salary": None,
        "Opponent": None,
        "Consensus": None,
        "Projection": None,
        "Scored": None,
        "Order": None,
        "pOwn": None,
        "actOwn": None,
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
            "Opponent": cells[header_idx["Opponent"]].text,
            "Salary": int(team_pattern.sub("", cells[header_idx["Salary"]].text)),
            "Consensus": float(cells[header_idx["Consensus"]].text),
            "Projection": float(
                cells[header_idx["Projection"]].find("input").get("value")
            ),
            "Scored": float(cells[header_idx["Scored"]].text),
            "Order": int(cells[header_idx["Order"]].text.replace("-", "0")),
            "pOwn": float(cells[header_idx["pOwn"]].text.replace("%", "")) / 100,
            "actOwn": float(cells[header_idx["actOwn"]].text.replace("%", "")) / 100,
        }
        data.append(player_data)

    slate = pd.DataFrame(data)
    # Change players that have multiple listed positions to just the first one
    slate["Position"] = slate["Position"].str.split("/", expand=True)[0]
    # C and 1B players can fill the C/1B slot
    slate["Position"] = slate["Position"].replace({"C": "C/1B", "1B": "C/1B"})
    # Convert Linestar opponent column into opposing team and opposing pitcher columns
    slate[["Opp_Team", "Opp_Pitcher"]] = slate.apply(
        get_opponent, axis=1, result_type="expand"
    )
    # For some historical data pages, consensus values are not available and are 0.0
    # Where this occurs, replace consensus value with projection value
    slate["Consensus"] = slate["Consensus"].where(
        slate["Consensus"] != 0, slate["Projection"]
    )
    return slate[
        [
            "Player",
            "Position",
            "Team",
            "Opp_Team",
            "Opp_Pitcher",
            "Salary",
            "Consensus",
            "Scored",
            "Order",
            "pOwn",
            "actOwn",
        ]
    ]


frames = []
for file in listdir("./data/linestar/"):
    try:
        frame = linestar_hist_full("./data/linestar/" + file)
    except Exception as exception:
        print(f"{file} FAILED")
        print(exception)
        continue
    frame["Date"] = file[:10]
    frames.append(frame)
    print(f"{file} done.")

data = pd.concat(frames)
data.to_csv("./data/linestar_data.csv", index=False)
