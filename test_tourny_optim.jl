using JuMP
using DataFrames
using CSV
using Dates
using SCIP
using LinearAlgebra
using Distributions
using StatsBase
using Threads


"""
    makeposdef(mat::Matrix)

Transforms a Hermitian matrix into a positive definite matrix
Computes eigendecomposition of matrix, sets negative and zero eigenvalues
to small value (1e-10) and then reconstructs matrix

Returns Hermitian positive definite matrix
"""
function makeposdef(mat::Matrix)
    vals = eigvals(mat)
    vecs = eigvecs(mat)
    vals[vals.<=1e-10] .= 1e-10
    return Hermitian(vecs * Diagonal(vals) * vecs')
end


"""
    gen_team(players::DataFrame)

Generates a random MLB FanDuel DFS team.
Expects columns :Position, :ID, :Proj_Ownership
"""
function gen_team(players::DataFrame)
    positions = Dict(
        "P" => 1,
        "C/1B" => 1,
        "2B" => 1,
        "3B" => 1,
        "SS" => 1,
        "OF" => 3,
        "UTIL" => 1
    )

    team = String[]
    for position in keys(positions)
        # Select players that can fill roster position
        # Anyone except pitchers and already selected players can fill the UTIL slot
        if position == "UTIL"
            players_subset = subset(players, :Position => p -> p .!= "P", :ID => id -> id .∉ Ref(team))
        else
            players_subset = subset(players, :Position => p -> p .== position)
        end
        # Normalize probabilities so they sum to 1
        probs = players_subset[!, :Proj_Ownership] ./ sum(players_subset[!, :Proj_Ownership])
        # Randomly selected the required number of players for the position without replacement
        selected_player = sample(players_subset[!, :ID], ProbabilityWeights(probs), positions[position], replace=false)
        append!(team, selected_player)
    end
    # Return players that are on selected team
    return subset(players, :ID => id -> in.(id, Ref(team)))
end


"""
    verify_team(team::DataFrame)

Verifies that a FanDuel MLB DFS team is a valid lineup.
Expects columns :Position, :Salary, :Team, and :Game
"""
function verify_team(team::DataFrame)
    teams_count = countmap(team[!, :Team])
    teams_player_count = countmap(subset(team, :Position => p -> p .!= "P")[!, :Team])
    games_count = countmap(team[!, :Game])
    constraints = Bool[
        34000<=sum(team[!, :Salary])<=35000, # Salary must be in reasonable range
        all(values(teams_player_count) .<= 4), # No more than 4 players from 1 team, excluding the pitcher
        length(keys(teams_count))>=3, # Must select players from at least 3 teams, INCLUDING the pitcher
        length(keys(games_count))>=2  # Must select players from at least 2 games
    ]
    return all(constraints)
end


function order_stat(μ, cutoff, entries)
    


#players = DataFrame(CSV.File("./data/slates/slate_$(Dates.today()).csv"))
players = DataFrame(CSV.File("./data/TEST_SLATE.csv"))

μ = players[!, :Projection]
σ = players[!, :Hist_Std]
# covariance matrix must be positive definite so that Distributions.jl MvNormal
# can do a cholesky factorization on it
Σ = makeposdef(Diagonal(σ) * Tables.matrix(CSV.File("./data/TEST_SLATE_corr.csv", header=false)) * Diagonal(σ))
