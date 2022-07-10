using Distributions
using StatsBase
include("types.jl")


"""
    find_players(slate::MLBSlate, position::String)

Given an MLB slate, find players with given position. Handles the Utility position by returning all non-pitcher players.
Returns integer indices of players found
"""
function find_players(slate::MLBSlate, position::String)
    indices = Int[]
    if position == "UTIL"
        for i in 1:length(slate.players)
            if slate.players[i].Position != "P"
                append!(indices, i)
            end
        end
    else
        for i in 1:length(slate.players)
            if slate.players[i].Position == position
                append!(indices, i)
            end
        end
    end
    return indices
end


"""
    make_probs(slate::Slate, indices::AbstractVector{<:Integer})

Given a slate and a set of indices, calculates the normalized probabilities of selecting
those players specified by the indices
"""
function make_probs(slate::Slate, indices::AbstractVector{<:Integer})
    nominal_weights = [x.Proj_Ownership for x in slate.players[indices]]
    return ProbabilityWeights(nominal_weights ./ sum(nominal_weights))
end


"""
    gen_team(slate::Slate)

Given a slate, generates a random opponent team using projected ownership numbers.
Not gauranteed to be a valid team.
"""
function gen_team(slate::Slate)
    team = Int[]
    for position in keys(slate.positions)
        indices = find_players(slate, position)
        probs = make_probs(slate, indices)
        append!(team, sample(indices, probs, slate.positions[position], replace=false))
    end
    return team
end


"""
    verify_team(slate::MLBSlate, indices::AbstractVector{<:Integer})

Given an MLB slate and indices representing players selected for a lineup, determine if those players
form a valid lineup
"""
function verify_team(slate::MLBSlate, indices::AbstractVector{<:Integer})
    team = slate.players[indices]
    teams_count = countmap([x.Team for x in team])
    teams_player_count = countmap([x.Team for x in team if x.Position != "P"])
    games_count = countmap([x.Game for x in team])
    constraints = [
        length(team) == 9,
        34000 <= sum([x.Salary for x in team]) <= 35000,
        all(values(teams_player_count) .<= 4),
        length(keys(teams_count)) >= 3,
        length(keys(games_count)) >= 2
    ]
    return all(constraints)
end


"""
    opp_team_score(slate::Slate, μ::AbstractVector{<:Real})

Computes the expected score of an opponent lineup given a slate and a vector of expected player scores.
Generates random teams until a valid one is selected, and returns the expected points.
"""
function opp_team_score(slate::Slate, μ::AbstractVector{<:Real})
    while true
        indices = gen_team(slate)
        if verify_team(slate, indices)
            return sum(μ[indices])
        end
    end
end


"""
    compute_order_stat(slate::Slate, μ::AbstractVector{<:Real}, cutoff::Integer, entries::Integer)

Computes the expected order statistics of opponent lineups. Entries is total number of opponent lineups
to simulate and cutoff is the order statistic to compute. A cutoff of 10 represents the tenth highest
opponent score.
"""
function compute_order_stat(slate::Slate, μ::AbstractVector{<:Real}, cutoff::Integer, entries::Integer)
    opp_scores = Vector{Float64}(undef, entries)
    for i = 1:entries
        opp_scores[i] = opp_team_score(slate, μ)
    end
    sort!(opp_scores, rev=true)
    return opp_scores[cutoff]
end


"""
    estimate_opp_stats(slate::Slate, entries::Integer, cutoff::Integer, samples::Integer)

Estimates the mean, variance, and covariance of opponent lineup order statistics.
Assumes players scores are multivariate normal.
"""
function estimate_opp_stats(slate::Slate, entries::Integer, cutoff::Integer, samples::Integer)
    score_draws = Vector{Vector{Float64}}(undef, samples)
    order_stats = Vector{Float64}(undef, samples)
    Threads.@threads for i = 1:samples
        # Draw random player score vector
        score_draw = rand(MvNormal(slate.μ, slate.Σ))
        # Compute opponent order statistic with the random score vector
        order_stat = compute_order_stat(slate, score_draw, cutoff, entries)
        println("$(i) done.")
        score_draws[i] = score_draw
        order_stats[i] = order_stat
    end

    opp_mu = mean(order_stats)
    opp_var = var(order_stats)
    # This is covariance between each individual player's score draws and the whole group of order statistics
    opp_cov = [cov([x[i] for x in score_draws], order_stats) for i in 1:length(slate.players)]

    return (opp_mu, opp_var, opp_cov)
end