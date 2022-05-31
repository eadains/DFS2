import pandas as pd
import numpy as np
from datetime import datetime

data = pd.read_csv("./data/linestar_data.csv")
data["Opposing Pitcher"] = data.loc[data["Position"] != "P", "Opponent"].str.split(
    ",", expand=True
)[0]

slate = pd.read_csv(f"./data/slate_{datetime.today().strftime('%Y-%m-%d')}.csv")

batters = data[data["Position"] != "P"]
order_scored = batters.groupby(["Date", "Team", "Order"]).sum()["Scored"]
batters_corr = order_scored.unstack().corr()

pitchers = data[data["Position"] == "P"]
batters = batters.merge(
    pitchers[["Name", "Date", "Scored"]],
    left_on=["Date", "Opposing Pitcher"],
    right_on=["Date", "Name"],
    how="left",
    suffixes=[None, " Opposing"],
)
pitchers_corr = batters[["Scored", "Scored Opposing"]].corr()
pitchers_corr = pitchers_corr.loc["Scored", "Scored Opposing"]

corr = pd.DataFrame(columns=slate["Name"], index=slate["Name"], dtype=float)

for row in slate.itertuples():
    # Correlation with themselves is 1
    corr.loc[row.Name, row.Name] = 1

    # If pitcher, set correlation to everyone else to 0
    if row.Position == "P":
        corr.loc[row.Name, corr.columns != row.Name] = 0

    else:
        # Setting correlation to other batters on the same team according to
        # batting order
        for teammate in slate.loc[slate["Team"] == row.Team, :].itertuples():
            # If the teammate is the pitcher, then 0 correlation
            if teammate.Position == "P":
                corr.loc[row.Name, teammate.Name] = 0
                corr.loc[teammate.Name, row.Name] = 0
            else:
                order_corr = batters_corr.loc[row.Order, teammate.Order]
                corr.loc[row.Name, teammate.Name] = order_corr
                corr.loc[teammate.Name, row.Name] = order_corr

        # Set correlation to opposing pitcher
        corr.loc[row.Name, row.Opp_Pitcher] = pitchers_corr
        corr.loc[row.Opp_Pitcher, row.Name] = pitchers_corr
        # Correlations with every other Name is 0
        corr.loc[row.Name, corr.loc[row.Name].isna()] = 0

# Check that the correlation matrix is symmetric and all its eigenvalues are >= 0
# These two conditions jointly imply the matrix is positive semi-definite
if not np.array_equal(corr, corr.T) & np.all(np.linalg.eigvals(corr) >= 0):
    raise ValueError("Correlation matrix not positive semi-definite")

hist_std = data.groupby("Name").std()["Scored"]
# Default standard deviations for players with missing values
default_pitcher_std = 15
default_other_std = 10

# Get historical standard deviation of scored points for players
# on the current slate
hist_std = hist_std.loc[slate["Player"]]

for player in hist_std[hist_std.isna()].index:
    player_position = slate.loc[slate["Player"] == player, "Position"].values[0]
    if player_position == "P":
        hist_std.loc[player] = default_pitcher_std
    else:
        hist_std.loc[player] = default_other_std

cov = np.diag(hist_std) @ corr @ np.diag(hist_std)
cov = pd.DataFrame(cov, columns=slate["Name"], index=slate["Name"])
cov.to_csv("./data/slate_cov.csv")
