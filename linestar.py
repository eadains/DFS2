from bs4 import BeautifulSoup
import quopri
import pandas as pd
import numpy as np
from os import listdir


def extract_row_data(row):
    cells = row.find_all("td")
    return {
        "Name": cells[5].find(class_="playername").text,
        "Position": cells[4].text,
        # Some teams only have two characters, causing an extra space
        # at the start, so strip that
        "Team": cells[5].find(class_="playerTeam").text[-3:].strip(),
        "Salary": cells[9].text,
        "Scored": cells[10].text,
        "Projection": cells[11].find("input").get("value"),
        "Consensus": cells[12].text,
        "Time": cells[13].text,
        "Opponent": cells[14].text,
        "Order": cells[16].text,
        "Bat/Arm": cells[17].text,
        "Consistent": cells[18].text,
        "Floor": cells[19].text,
        "Ceiling": cells[20].text,
        "Avg FP": cells[22].text,
        "Imp Runs": cells[23].text,
        "pOwn": cells[25].text,
        "actOwn": cells[26].text,
        "Leverage": cells[27].text,
        "Safety": cells[28].text,
    }


def extract_linestar_data(filename):
    html = open(filename, "r")
    html = quopri.decodestring(html.read())
    soup = BeautifulSoup(html, features="html.parser")

    table = soup.find_all("table")[0]
    row_data = []
    for row in table.find_all("tr", class_="playerCardRow"):
        row_data.append(extract_row_data(row))

    return pd.DataFrame(row_data)


frames = []
for file in listdir("./data/linestar/"):
    try:
        frame = extract_linestar_data("./data/linestar/" + file)
    except Exception as exception:
        print(f"{file} FAILED")
        print(exception)
        continue
    frame["Date"] = file[:10]
    frames.append(frame)

data = pd.concat(frames)
# Remove (R) and (L) from pither names
data.loc[data["Position"] == "P", "Player"] = data.loc[
    data["Position"] == "P", "Player"
].str[:-4]
data["Salary"] = data["Salary"].replace("[\$,]", "", regex=True).astype(int)
data["Projection"] = data["Projection"].astype(float)
data["Scored"] = data["Scored"].astype(float)
data[["pOwn", "actOwn"]] = (
    data[["pOwn", "actOwn"]].replace("[\%]", "", regex=True).astype(float)
)
data["Position"] = data["Position"].str.split("/", expand=True)[0]
# Replace players with no batting order with NaN
data["Order"] = data["Order"].replace({"-": np.nan})

data.to_csv("linestar_data.csv", index=False)
