import pandas as pd
import numpy as np
from datetime import datetime

historical = pd.read_csv("./data/linestar_data.csv")
historical["Opposing Pitcher"] = historical.loc[
    historical["Position"] != "P", "Opponent"
].str.split(",", expand=True)[0]

batters = historical[historical["Position"] != "P"]
order_scored = batters.groupby(["Date", "Team", "Order"]).sum()["Scored"]
batters_corr = order_scored.unstack().corr()

pitchers = historical[historical["Position"] == "P"]
batters = batters.merge(
    pitchers[["Name", "Date", "Scored"]],
    left_on=["Date", "Opposing Pitcher"],
    right_on=["Date", "Name"],
    how="left",
    suffixes=[None, " Opposing"],
)
pitchers_corr = batters[["Scored", "Scored Opposing"]].corr()
pitchers_corr = pitchers_corr.loc["Scored", "Scored Opposing"]

slate = pd.read_csv(f"./data/slate_{datetime.today().strftime('%Y-%m-%d')}.csv")
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

corr.to_csv(
    f"./data/slates/corr_{datetime.today().strftime('%Y-%m-%d')}.csv",
    index=False,
    header=False,
)
