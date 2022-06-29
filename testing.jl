using DataFrames
using CSV
using Dates
using Statistics
using Threads


slate = CSV.read("./data/slates/slate_$(Dates.today()).csv", DataFrame)
hist = CSV.read("./data/linestar_data.csv", DataFrame)


"""
    euclid_dist(x::Number, y::Number)

Computes the euclidean distance between two numbers.
Returns the absolute value of their difference
"""
function euclid_dist(x::Number, y::Number)
    return abs(x - y)
end


"""
    get_similar_std(num::Integer, hist::DataFrame, player::DataFrameRow)

Using a k-nearest-neighbors-like approach, estimates the standard deviation of scored points
for a given player.

Given a player, this finds the 'num' number of players that have the closest historical projection
to the given player and also have the same position and batting order. Returns the standard
deviation of those players actually scored points.
"""
function get_similar_std(num::Integer, hist::DataFrame, player::DataFrameRow)::Float64
    # TODO: fix dataframes type stability problems
    # Find players with same position and batting order
    similar_players = hist[(hist[!, :Position].==player[:Position]).&(hist[!, :Order].==player[:Order]), :]
    # Form list of tuples where first element is historical points actually scored,
    # and second element is distance between the historical projection and the current players projection
    similar_proj = tuple.(
        similar_players.Scored,
        euclid_dist.(player.Projection, similar_player.Consensus)
    )
    # Sort by projection distance
    sort!(similar_proj, by=x -> x[2])
    # Return standard deviation of actually scored points from num number of players with the closest
    # projection to the current player
    return std([similar_proj[x][1] for x in 1:num])
end


"""
    get_sigma(num::Integer, hist::DataFrame, slate::DataFrame)

Computes standard deviation vector for given slate.
'num' parameter controls the 'get_similar_std' function behavior
"""
function get_sigma(num::Integer, hist::DataFrame, slate::DataFrame)
    σ = Vector{Float64}(undef, nrow(slate))
    for (i, player) in enumerate(eachrow(slate))
        σ[i] = get_similar_std(num, hist, player)
    end
    return σ
end


corr = Matrix{Float64}(undef, nrow(slate), nrow(slate))

# Iterate over upper triangular portion of matrix, including the diagonal
Threads.@threads for i in 1:nrow(slate)
    for j in i:nrow(slate)
        if i == j
            # Diagonal is always 1
            corr[i, j] = 1
        else
            # Otherwise set symmetric correlation entries
            pair_corr = get_corr(100, hist, slate[i, :], slate[j, :])
            corr[i, j] = pair_corr
            corr[j, i] = pair_corr
        end
    end
end


function get_corr(num::Integer, hist::DataFrame, p1::DataFrameRow, p2::DataFrameRow)
    if p1.Team == p2.Team
        return same_team_corr(num, hist, p1, p2)
    elseif p1.Opponent == p2.Team
        return 0
    else
        return 0
    end
end


function same_team_corr(num::Integer, hist::DataFrame, p1::DataFrameRow, p2::DataFrameRow)
    results = Tuple{Float64,Float64,Float64}[]
    for date_frame in groupby(hist, :Date)
        for team_frame in groupby(date_frame, :Team)
            p1_sim = team_frame[(team_frame.Position.==p1.Position).&(team_frame.Order.==p1.Order), :]
            p2_sim = team_frame[(team_frame.Position.==p2.Position).&(team_frame.Order.==p2.Order), :]
            if nrow(p1_sim) == 0 || nrow(p2_sim) == 0
                # There may be no matching players, so skip
                continue
            elseif nrow(p1_sim) > 1 || nrow(p2_sim) > 1
                # If there are data issues and more than 1 player is found for the above selections, just skip
                continue
            else
                # p1_sim and p2_sim are DataFrame not DataFrameRow so we need to index to get a value instead of a vector
                # Last element is sum of euclidean distances between the two players
                push!(results, (p1_sim.Scored[1], p2_sim.Scored[1], euclid_dist(p1.Projection, p1_sim.Consensus[1]) + euclid_dist(p2.Projection, p2_sim.Consensus[1])))
            end
        end
    end
    sort!(results, by=x -> x[3])
    # Sometimes there may be limited samples
    if length(results) < num
        num = length(results)
    end
    return cor([results[x][1] for x in 1:num], [results[x][2] for x in 1:num])
end


results = Tuple{Float64,Float64,Float64}[]
for date_frame in groupby(hist, :Date)
    for team_frame in groupby(date_frame, :Team)
        p1_sim = team_frame[(team_frame.Position.==p1.Position).&(team_frame.Order.==p1.Order), :]
        p2_sim = team_frame[(team_frame.Position.==p2.Position).&(team_frame.Order.==p2.Order), :]
        if nrow(p1_sim) == 0 || nrow(p2_sim) == 0
            # There may be no matching players, so skip
            continue
        elseif nrow(p1_sim) > 1 || nrow(p2_sim) > 1
            # If there are data issues and more than 1 player is found for the above selections, just skip
            continue
        else
            # p1_sim and p2_sim are DataFrame not DataFrameRow so we need to index to get a value instead of a vector
            # Last element is sum of euclidean distances between the two players
            push!(results, (p1_sim.Scored[1], p2_sim.Scored[1], euclid_dist(p1.Projection, p1_sim.Consensus[1]) + euclid_dist(p2.Projection, p2_sim.Consensus[1])))
        end
    end
end