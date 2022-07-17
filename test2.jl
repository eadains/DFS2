include("./src/solve.jl")
using CPLEX

slate = get_mlb_slate("2022-07-15")

"""
    do_optim(data::MLBTournyOptimData)

Runs optimization for tournaments
"""
function do_optim(slate::MLBSlate, λ::Float64, pastlineups::Vector{JuMP.Containers.DenseAxisArray})
    model = Model(Xpress.Optimizer)

    # Players selected
    @variable(model, x[[player.ID for player in slate.players]], binary = true)
    # Games selected
    @variable(model, y[slate.teams], binary = true)
    # Teams selected
    @variable(model, z[slate.games], binary = true)

    # Total salary must be <= $35,000
    @constraint(model, sum(player.Salary * x[player.ID] for player in slate.players) <= 35000)
    # Must select 9 total players
    @constraint(model, sum(x) == 9)
    # Constraints for each position, must always select 1 pitcher, but can select 1 additional
    # player for each other position to fill the utility slot
    @constraint(model, sum(x[player.ID] for player in slate.players if player.Position == "P") == 1)
    @constraint(model, 1 <= sum(x[player.ID] for player in slate.players if player.Position == "C/1B") <= 2)
    @constraint(model, 1 <= sum(x[player.ID] for player in slate.players if player.Position == "2B") <= 2)
    @constraint(model, 1 <= sum(x[player.ID] for player in slate.players if player.Position == "3B") <= 2)
    @constraint(model, 1 <= sum(x[player.ID] for player in slate.players if player.Position == "SS") <= 2)
    @constraint(model, 3 <= sum(x[player.ID] for player in slate.players if player.Position == "OF") <= 4)

    for team in slate.teams
        # Excluding the pitcher, we can select a maximum of 4 players per team
        @constraint(model, sum(x[player.ID] for player in slate.players if player.Team == team && player.Position != "P") <= 4)
        # If no players are selected from a team, y is set to 0
        @constraint(model, y[team] <= sum(x[player.ID] for player in slate.players if player.Team == team))
    end
    # Must have players from at least 3 teams
    @constraint(model, sum(y) >= 3)

    for game in slate.games
        # If no players are selected from a game z is set to 0
        @constraint(model, z[game] <= sum(x[player.ID] for player in slate.players if player.Game == game))
    end
    # Must select players from at least 2 games
    @constraint(model, sum(z) >= 2)

    # If there are any past lineups, ensure that the current lineup doesn't overlap too much with any of them
    if length(pastlineups) > 0
        for past in pastlineups
            @constraint(model, sum(x[player.ID] * past[player.ID] for player in slate.players) <= overlap)
        end
    end

    mu_x = @expression(model, x.data' * slate.μ)
    var_x = @expression(model, x.data' * slate.Σ * x.data)
    # Maximize projected fantasy points
    @objective(model, Max, mu_x + λ * var_x)

    optimize!(model)
    println(termination_status(model))
    # Return optimization result vector, as well as estimated probability of exceeding opponents score
    return (round.(Int, value.(x)), 1 - cdf(Normal(), (250 - value(mu_x)) / sqrt(value(var_x))))
end


"""
    lambda_max(data::MLBTournyOptimData)

Does optimization over range of λ values and returns the lineup with the highest objective function.
"""
function lambda_max(slate::MLBSlate)
    # I've found that lambdas from around 0 to 0.05 are selected, with strong majority being 0.02
    lambdas = 0:0.01:0.20
    w_star = Vector{Tuple{JuMP.Containers.DenseAxisArray,Float64}}(undef, length(lambdas))
    # Perform optimization over array of λ values
    for i in 1:length(lambdas)
        println("Trying λ $(lambdas[i])")
        w_star[i] = do_optim(slate, lambdas[i], JuMP.Containers.DenseAxisArray[])
        print("λ probability: $(w_star[i][2])")
    end

    # Find lambda value that leads to highest objective function and return its corresponding lineup vector
    max_index = argmax(x[2] for x in w_star)
    println("λ max: $(lambdas[max_index])")
    return w_star[max_index][1]
end