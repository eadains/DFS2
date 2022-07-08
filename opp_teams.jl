using DataFrames
using Distributions
using StatsBase
using CSV
using Tables


struct MLBTeam
    players::Vector{<:NamedTuple}
    positions::Dict{String,<:Integer}
end

const positions = Dict(
    "P" => 1,
    "C/1B" => 1,
    "2B" => 1,
    "3B" => 1,
    "SS" => 1,
    "OF" => 3,
    "UTIL" => 1
)


function find_players(players::Vector{T}, position::String) where {T<:NamedTuple}
    indices = Int[]
    if position == "UTIL"
        for i in 1:length(players)
            if players[i].Position != "P"
                append!(indices, i)
            end
        end
    else
        for i in 1:length(players)
            if players[i].Position == position
                append!(indices, i)
            end
        end
    end
    return indices
end


function pick_player(indices::Vector{<:Integer}, probs::ProbabilityWeights, n::Integer) where {T<:NamedTuple}
    return sample(indices, probs, n, replace=false)
end


function make_probs(players::Vector{<:NamedTuple})
    nominal_weights = [x.Proj_Ownership for x in players]
    return ProbabilityWeights(nominal_weights ./ sum(nominal_weights))
end


function gen_team(positions::Dict{String,<:Integer}, players::Vector{T}) where {T<:NamedTuple}
    team = Int[]
    for position in keys(positions)
        subset = find_players(players, position)
        probs = make_probs(players[subset])
        append!(team, pick_player(subset, probs, positions[position]))
    end
    return team
end


function verify_team(team::Vector{<:NamedTuple})
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


function opp_team_score(players::Vector{<:NamedTuple}, μ::Vector{<:Real})
    while true
        team = gen_team(positions, players)
        if verify_team(players[team])
            return sum(μ[team])
        end
    end
end


function compute_order_stat(players::Vector{<:NamedTuple}, μ::Vector{<:Real}, cutoff::Integer, entries::Integer)
    opp_scores = Vector{Float64}(undef, entries)
    for i = 1:entries
        opp_scores[i] = opp_team_score(players, μ)
    end
    sort!(opp_scores, rev=true)
    return opp_scores[cutoff]
end