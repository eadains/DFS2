include("./src/solve.jl")

LB = -Inf
UB = Inf
cuts = BitVector[]
incumbent = 0

slate = get_mlb_slate()

model = Model(() -> Xpress.Optimizer(logfile="optimlog.log", HEUREMPHASIS=1, MIPRELSTOP=0.15))

# The 2 different lineups are along axis 1
# Players selected
@variable(model, x[1:2, [player.ID for player in slate.players]], binary = true)
# Games selected
@variable(model, y[1:2, slate.teams], binary = true)
# Teams selected
@variable(model, z[1:2, slate.games], binary = true)

for j in 1:2
    # Total salary must be <= $35,000
    @constraint(model, sum(player.Salary * x[j, player.ID] for player in slate.players) <= 35000)
    # Must select 9 total players
    @constraint(model, sum(x[j, :]) == 9)
    # Constraints for each position, must always select 1 pitcher, but can select 1 additional
    # player for each other position to fill the utility slot
    @constraint(model, sum(x[j, player.ID] for player in slate.players if player.Position == "P") == 1)
    @constraint(model, 1 <= sum(x[j, player.ID] for player in slate.players if player.Position == "C/1B") <= 2)
    @constraint(model, 1 <= sum(x[j, player.ID] for player in slate.players if player.Position == "2B") <= 2)
    @constraint(model, 1 <= sum(x[j, player.ID] for player in slate.players if player.Position == "3B") <= 2)
    @constraint(model, 1 <= sum(x[j, player.ID] for player in slate.players if player.Position == "SS") <= 2)
    @constraint(model, 3 <= sum(x[j, player.ID] for player in slate.players if player.Position == "OF") <= 4)

    for team in slate.teams
        # Excluding the pitcher, we can select a maximum of 4 players per team
        @constraint(model, sum(x[j, player.ID] for player in slate.players if player.Team == team && player.Position != "P") <= 4)
        # If no players are selected from a team, y is set to 0
        @constraint(model, y[j, team] <= sum(x[j, player.ID] for player in slate.players if player.Team == team))
    end
    # Must have players from at least 3 teams
    @constraint(model, sum(y[j, :]) >= 3)

    for game in slate.games
        # If no players are selected from a game z is set to 0
        @constraint(model, z[j, game] <= sum(x[j, player.ID] for player in slate.players if player.Game == game))
    end
    # Must select players from at least 2 games
    @constraint(model, sum(z[j, :]) >= 2)
end

mu_x1 = @expression(model, x.data[1, :]' * slate.μ)
var_x1 = @expression(model, x.data[1, :]' * slate.Σ * x.data[1, :])
var_x2 = @expression(model, x.data[2, :]' * slate.Σ * x.data[2, :])
cov = @expression(model, x.data[1, :]' * slate.Σ * x.data[2, :])

@objective(model, Max, mu_x1 + (1 / sqrt(2 * pi)) * (1 + var_x1 + var_x2 - 2 * cov))