using JuMP
using GLPK
using Xpress
include("types.jl")


"""
    do_optim(data::CashData)

Runs optimization for cash games
"""
function do_optim(data::MLBCashOptimData)
    model = Model(GLPK.Optimizer)

    # Players selected
    @variable(model, x[data.players.ID], binary = true)
    # Games selected
    @variable(model, y[data.slate.games], binary = true)
    # Teams selected
    @variable(model, z[dats.slate.teams], binary = true)

    # Total salary must be <= $35,000
    @constraint(model, sum(player.Salary * x[player.ID] for player in data.slate.players) <= 35000)
    # Must select 9 total players
    @constraint(model, sum(x) == 9)
    # Constraints for each position, must always select 1 pitcher, but can select 1 additional
    # player for each other position to fill the utility slot
    @constraint(model, sum(x[player.ID] for player in data.slate.players if player.Position == "P") == 1)
    @constraint(model, 1 <= sum(x[player.ID] for player in data.slate.players if player.Position == "C/1B") <= 2)
    @constraint(model, 1 <= sum(x[player.ID] for player in data.slate.players if player.Position == "2B") <= 2)
    @constraint(model, 1 <= sum(x[player.ID] for player in data.slate.players if player.Position == "3B") <= 2)
    @constraint(model, 1 <= sum(x[player.ID] for player in data.slate.players if player.Position == "SS") <= 2)
    @constraint(model, 3 <= sum(x[player.ID] for player in data.slate.players if player.Position == "OF") <= 4)

    for team in data.slate.teams
        # Excluding the pitcher, we can select a maximum of 4 players per team
        @constraint(model, sum(x[player.ID] for player in data.slate.players if player.Team == team && player.Position != "P") <= 4)
        # If no players are selected from a team, y is set to 0
        @constraint(model, y[team] <= sum(x[player.ID] for player in data.slate.players if player.Team == team))
    end
    # Must have players from at least 3 teams
    @constraint(model, sum(y) >= 3)

    for game in data.slate.games
        # If no players are selected from a game z is set to 0
        @constraint(model, z[game] <= sum(x[player.ID] for player in data.slate.players if player.Game == game))
    end
    # Must select players from at least 2 games
    @constraint(model, sum(z) >= 2)

    # Maximize projected fantasy points
    @objective(model, Max, sum(player.Projection * x[player.ID] for player in data.slate.players))

    optimize!(model)
    println(termination_status(model))
    return (objective_value(model), value.(x))
end
