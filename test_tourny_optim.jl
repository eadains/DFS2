using JuMP
using DataFrames
using CSV
using Dates
using Juniper
using Ipopt
using LinearAlgebra
using Distributions
using StatsBase


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


function do_optim(players, past_lineups, μ::Vector{Float64}, Σ::Matrix{Float64}, opp_mu::Float64, opp_var::Float64, opp_cov::Vector{Float64})
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

    nl_solver = optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0)
    minlp_solver = optimizer_with_attributes(Juniper.Optimizer, "nl_solver" => nl_solver)
    model = Model(minlp_solver)

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
    @NLobjective(model, Min, -mu_x / var_x)

    optimize!(model)
    println(termination_status(model))
    return value.(x)
end

#players = DataFrame(CSV.File("./data/slates/slate_$(Dates.today()).csv"))
players = DataFrame(CSV.File("./data/TEST_SLATE.csv"))

μ = players[!, :Projection]
σ = players[!, :Hist_Std]
# covariance matrix must be positive definite so that Distributions.jl MvNormal
# can do a cholesky factorization on it
Σ = makeposdef(Diagonal(σ) * Tables.matrix(CSV.File("./data/TEST_SLATE_corr.csv", header=false)) * Diagonal(σ))

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