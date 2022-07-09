using JuMP
using GLPK
using Xpress
include("types.jl")


"""
    do_optim(data::MLBCashOptimData)

Runs optimization for cash games
"""
function do_optim(data::MLBCashOptimData)
    model = Model(GLPK.Optimizer)

    # Players selected
    @variable(model, x[[player.ID for player in data.slate.players]], binary = true)
    # Games selected
    @variable(model, y[data.slate.teams], binary = true)
    # Teams selected
    @variable(model, z[data.slate.games], binary = true)

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
    return (objective_value(model), round.(Int, value.(x)))
end


"""
    do_optim(data::MLBTournyOptimData)

Runs optimization for tournaments
"""
function do_optim(data::MLBTournyOptimData, λ::Float64)
    model = Model(Xpress.Optimizer)

    # Players selected
    @variable(model, x[[player.ID for player in data.slate.players]], binary = true)
    # Games selected
    @variable(model, y[data.slate.teams], binary = true)
    # Teams selected
    @variable(model, z[data.slate.games], binary = true)

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

    # If there are any past lineups, ensure that the current lineup doesn't overlap too much with any of them
    if length(data.pastlineups) > 0
        for past in data.pastlineups
            @constraint(model, sum(x[player.ID] * past[player.ID] for player in data.slate.players) <= data.overlap)
        end
    end

    mu_x = @expression(model, x.data' * data.slate.μ - data.opp_mu)
    var_x = @expression(model, x.data' * data.slate.Σ * x.data + data.opp_var - 2 * x.data' * data.opp_cov)
    # Maximize projected fantasy points
    @objective(model, Max, mu_x + λ * var_x)

    optimize!(model)
    println(termination_status(model))
    # Return optimization result vector, as well as estimated probability of exceeding opponents score
    return (round.(Int, value.(x)), 1 - cdf(Normal(), -value(mu_x) / sqrt(value(var_x))))
end


"""
    lambda_max(data::MLBTournyOptimData)

Does optimization over range of λ values and returns the lineup with the highest objective function.
"""
function lambda_max(data::MLBTournyOptimData)
    # I've found that lambdas from around 0 to 0.05 are selected, with strong majority being 0.02
    lambdas = 0:0.01:0.05
    w_star = Vector{Tuple{JuMP.Containers.DenseAxisArray,Float64}}(undef, length(lambdas))
    # Perform optimization over array of λ values
    Threads.@threads for i in 1:length(lambdas)
        w_star[i] = do_optim(data, lambdas[i])
    end

    # Find lambda value that leads to highest objective function and return its corresponding lineup vector
    max_index = argmax(x[2] for x in w_star)
    println("λ max: $(lambdas[max_index])")
    return w_star[max_index][1]
end


"""
    get_lineups(N::Integer)

Solves the tournament optimization problem with N entries.
Appends lineups to OptimData pastlineups array
"""
function tourny_lineups!(data::MLBTournyOptimData, N::Int64)
    for n in 1:N
        println(n)
        lineup = lambda_max(data)
        append!(data.pastlineups, Ref(lineup))
    end
end