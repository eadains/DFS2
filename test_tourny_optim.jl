using JuMP
using DataFrames
using CSV
using Dates
using LinearAlgebra
using Distributions
using StatsBase
using SCIP


"""
    makeposdef(mat::Matrix)

Transforms a Hermitian matrix into a positive definite matrix
Computes eigendecomposition of matrix, sets negative and zero eigenvalues
to small value (1e-10) and then reconstructs matrix

Returns Hermitian positive definite matrix
"""
function makeposdef(mat::Hermitian{Float64})
    vals = eigvals(mat)
    vecs = eigvecs(mat)
    vals[vals.<=1e-10] .= 1e-10
    return Hermitian(vecs * Diagonal(vals) * vecs')
end


function do_optim(players::DataFrame, past_lineups, μ::Vector{Float64}, Σ::Hermitian{Float64}, λ::Float64)
    games = unique(players.Game)
    teams = unique(players.Team)
    positions = unique(players.Position)
    overlap = 5

    positions_max = Dict(
        "P" => 1,
        "C/1B" => 2,
        "2B" => 2,
        "3B" => 2,
        "SS" => 2,
        "OF" => 4
    )
    positions_min = Dict(
        "P" => 1,
        "C/1B" => 1,
        "2B" => 1,
        "3B" => 1,
        "SS" => 1,
        "OF" => 3
    )

    model = Model(optimizer_with_attributes(SCIP.Optimizer, "display/verblevel" => 1))

    # Players selected
    @variable(model, x[players.ID], binary = true)
    # Games selected
    @variable(model, y[teams], binary = true)
    # Teams selected
    @variable(model, z[games], binary = true)

    # Total salary of selected players must be <= $35,000
    @constraint(model, sum(player.Salary * x[player.ID] for player in eachrow(players)) <= 35000)

    # Must select 9 total players
    @constraint(model, sum(x) == 9)

    for position in positions
        @constraint(model, positions_min[position] <= sum(x[player.ID] for player in eachrow(players) if player.Position == position) <= positions_max[position])
    end

    for team in teams
        # Excluding the pitcher, we can select a maximum of 4 players per team
        @constraint(model, sum(x[player.ID] for player in eachrow(players) if player.Team == team && player.Position != "P") <= 4)
        # If no players are selected from a team, y is set to 0
        @constraint(model, y[team] <= sum(x[player.ID] for player in eachrow(players) if player.Team == team))
    end
    # Must have players from at least 3 teams
    @constraint(model, sum(y) >= 3)

    for game in games
        # If no players are selected from a game z is set to 0
        @constraint(model, z[game] <= sum(x[player.ID] for player in eachrow(players) if player.Game == game))
    end
    # Must select players from at least 2 games
    @constraint(model, sum(z) >= 2)

    # If any past lineups have been selected, make sure the current lineup doesn't overlap
    if length(past_lineups) > 0
        for past in past_lineups
            @constraint(model, sum(x .* past) <= overlap)
        end
    end

    mu_x = @expression(model, 200 - x.data' * μ)
    var_x = @expression(model, x.data' * Σ * x.data)
    @objective(model, Max, mu_x + λ * var_x)

    optimize!(model)
    println(termination_status(model))
    # SCIP only checks for integer values within a tolerance, so round the result to the nearest integer
    return (round.(Int, value.(x)), 1 - cdf(Normal(), value(mu_x) / sqrt(value(var_x))))
end


function lambda_max(players::DataFrame, past_lineups, μ::Vector{Float64}, Σ::Hermitian{Float64})
    lambdas = 0.05:0.05:0.30
    w_star = Vector{Tuple{JuMP.Containers.DenseAxisArray,Float64}}(undef, length(lambdas))
    # Perform optimization over array of λ values
    Threads.@threads for i in 1:length(lambdas)
        w_star[i] = do_optim(players, past_lineups, μ, Σ, lambdas[i])
    end

    # Find lambda value that leads to highest objective function and return its corresponding lineup vector
    max_index = argmax(x[2] for x in w_star)
    println("λ max: $(lambdas[max_index])")
    return w_star[max_index][1]
end


players = DataFrame(CSV.File("./data/slates/slate_$(Dates.today()).csv"))

μ = players[!, :Projection]
σ = players[!, :Hist_Std]
# covariance matrix must be positive definite so that Distributions.jl MvNormal
# can do a cholesky factorization on it
Σ = makeposdef(Hermitian(Diagonal(σ) * Tables.matrix(CSV.File("./data/slates/corr_$(Dates.today()).csv", header=false)) * Diagonal(σ)))

N = 10
past_lineups = []
for n in 1:N
    println(n)
    lineup = lambda_max(players, past_lineups, μ, Σ)
    append!(past_lineups, Ref(lineup))
end

lineups = []
for lineup in past_lineups
    # Roster positions to fill
    positions = Dict{String,Union{String,Missing}}("P" => missing, "C/1B" => missing, "2B" => missing, "3B" => missing, "SS" => missing, "OF1" => missing, "OF2" => missing, "OF3" => missing, "UTIL" => missing)
    for player in eachrow(players)
        # If player is selected
        if value(lineup[player.ID]) == 1
            # If they're OF, fill open OF slot, or if they're full, then UTIL
            if player.Position == "OF"
                if ismissing(positions["OF1"])
                    positions["OF1"] = player.ID
                elseif ismissing(positions["OF2"])
                    positions["OF2"] = player.ID
                elseif ismissing(positions["OF3"])
                    positions["OF3"] = player.ID
                else
                    positions["UTIL"] = player.ID
                end
            else
                # Otherwise, fill players position, and if it's full, then UTIL
                if ismissing(positions[player.Position])
                    positions[player.Position] = player.ID
                else
                    positions["UTIL"] = player.ID
                end
            end
        end
    end
    append!(lineups, Ref(positions))
end

# Print lineups to CSV in FanDuel format
open("./tourny_lineups.csv", "w") do file
    println(file, "P,C/1B,2B,3B,SS,OF,OF,OF,UTIL")
    for lineup in lineups
        println(file, "$(lineup["P"]),$(lineup["C/1B"]),$(lineup["2B"]),$(lineup["3B"]),$(lineup["SS"]),$(lineup["OF1"]),$(lineup["OF2"]),$(lineup["OF3"]),$(lineup["UTIL"])")
    end
end