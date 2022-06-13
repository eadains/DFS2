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
            players_subset = players[(players.Position.!="P").&(players.ID.∉Ref(team)), :]
        else
            players_subset = players[(players.Position.==position), :]
        end
        # Normalize probabilities so they sum to 1
        probs = players_subset[!, :Proj_Ownership] ./ sum(players_subset[!, :Proj_Ownership])
        # Randomly selected the required number of players for the position without replacement
        selected_player = sample(players_subset[!, :ID], ProbabilityWeights(probs), positions[position], replace=false)
        append!(team, selected_player)
    end
    # Return players that are on selected team
    return players[players.ID.∈Ref(team), :]
end


"""
    verify_team(team::DataFrame)

Verifies that a FanDuel MLB DFS team is a valid lineup.
Expects columns :Position, :Salary, :Team, and :Game
"""
function verify_team(team::DataFrame)
    teams_count = countmap(team[!, :Team])
    teams_player_count = countmap(team[team.Position.!="P", :Team])
    games_count = countmap(team[!, :Game])
    constraints = Bool[
        34000<=sum(team[!, :Salary])<=35000, # Salary must be in reasonable range
        all(values(teams_player_count) .<= 4), # No more than 4 players from 1 team, excluding the pitcher
        length(keys(teams_count))>=3, # Must select players from at least 3 teams, INCLUDING the pitcher
        length(keys(games_count))>=2  # Must select players from at least 2 games
    ]
    return all(constraints)
end


"""
    opp_team_score(players::DataFrame, μ::Vector{Float64})

Generates possible opponent entries until a valid team is generated,
and then returns that lineups expected number of fantasy points by using μ
"""
function opp_team_score(players::DataFrame, μ::Vector{Float64})
    while true
        team = gen_team(players)
        if verify_team(team)
            return μ' * (players.ID .∈ Ref(team[!, :ID]))
        end
    end
end


"""
    order_stat(players::DataFrame, cutoff::Int, entries::Int)

Computes order statistic for opponent lineups. Entires is number of opponent lineups
to generate, cutoff is the order statistics to compute.
"""
function compute_order_stat(players::DataFrame, μ::Vector{Float64}, cutoff::Int, entries::Int)
    opp_scores = Vector{Float64}(undef, entries)
    Threads.@threads for i = 1:entries
        opp_scores[i] = opp_team_score(players, μ)
    end
    sort!(opp_scores, rev=true)
    return opp_scores[cutoff]
end


function do_optim(players, past_lineups, μ::Vector{Float64}, Σ::Hermitian{Float64}, opp_mu::Float64, opp_var::Float64, opp_cov::Vector{Float64}, λ::Float64)
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

    mu_x = @expression(model, x.data' * μ - opp_mu)
    var_x = @expression(model, x.data' * Σ * x.data + opp_var - 2 * x.data' * opp_cov)
    @objective(model, Max, mu_x + λ * var_x)

    optimize!(model)
    println(termination_status(model))
    # SCIP only checks for integer values within a tolerance, so round the result to the nearest integer
    return (round.(Int, value.(x)), 1 - cdf(Normal(), -value(mu_x) / value(var_x)))
end


function lambda_max(players, past_lineups, μ::Vector{Float64}, Σ::Hermitian{Float64}, opp_mu::Float64, opp_var::Float64, opp_cov::Vector{Float64})
    # I've found that lambdas from around 0.03 to 0.05 are selected
    lambdas = 0.03:0.01:0.05
    w_star = Vector{Tuple{JuMP.Containers.DenseAxisArray,Float64}}(undef, length(lambdas))
    # Perform optimization over array of λ values
    Threads.@threads for i in 1:length(lambdas)
        w_star[i] = do_optim(players, past_lineups, μ, Σ, opp_mu, opp_var, opp_cov, lambdas[i])
    end

    # Find lambda value that leads to highest objective function and return its corresponding lineup vector
    max_index = argmax(x[2] for x in w_star)
    println("λ max: $(lambdas[max_index])")
    return w_star[max_index][1]
end


players = DataFrame(CSV.File("./data/slate_$(Dates.today()).csv"))

μ = players[!, :Projection]
σ = players[!, :Hist_Std]
# covariance matrix must be positive definite so that Distributions.jl MvNormal
# can do a cholesky factorization on it
Σ = makeposdef(Diagonal(σ) * Tables.matrix(CSV.File("./data/corr_$(Dates.today()).csv", header=false)) * Diagonal(σ))

# Total opponent entries in tournament
total_entries = 1000
# Focus on maximizing the probability that our lineup ranks in the top 1%
cutoff = Int(0.01 * total_entries)

score_draws = Vector{Float64}[]
order_stats = Float64[]
for i = 1:100
    score_draw = rand(MvNormal(μ, Σ))
    order_stat = compute_order_stat(players, score_draw, cutoff, total_entries)
    println("$(i) done.")
    append!(score_draws, Ref(score_draw))
    append!(order_stats, order_stat)
end

opp_mu = mean(order_stats)
opp_var = var(order_stats)
# This is covariance between each individual player's score draws and the whole group of order statistics
opp_cov = [cov([x[i] for x in score_draws], order_stats) for i in 1:nrow(players)]

N = 40
past_lineups = []
for n in 1:N
    println(n)
    lineup = lambda_max(players, past_lineups, μ, Σ, opp_mu, opp_var, opp_cov)
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