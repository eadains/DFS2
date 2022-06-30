using DataFrames
using CSV
using Dates
using Statistics
using LinearAlgebra


"""
    euclid_dist(x::Number, y::Number)

Computes the euclidean distance between two numbers.
Returns the absolute value of their difference
"""
function euclid_dist(x::Number, y::Number)
    return abs(x - y)
end


"""
    euclid_dist(x::Tuple{Number, Number}, y::Tuple{Number, Number})

Computes the Euclidean distance between two points in 2d space
"""
function euclid_dist(x::Tuple{Number,Number}, y::Tuple{Number,Number})
    return sqrt((x[1] - y[1])^2 + (x[2] - y[2])^2)
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
        euclid_dist.(player.Projection, similar_players.Consensus)
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


"""
    same_team_corr(num::Integer, hist::DataFrame, p1::DataFrameRow, p2::DataFrameRow)

Computes the k-nearest-neighbor correlation between two players on the same team.
Finds pairs of players that are on the same team and have the same respective positions and
batting orders as the given players. Finds the 'num' number of pairs of players whose projections
have the closest euclidean distance to the given pair of players and returns their correlation
"""
function same_team_corr(num::Integer, hist::DataFrame, p1::DataFrameRow, p2::DataFrameRow)
    results = Tuple{Float64,Float64,Float64}[]
    for date_frame in groupby(hist, :Date)
        for team_frame in groupby(date_frame, :Team)
            p1_sim = team_frame[(team_frame.Position.==p1.Position).&(team_frame.Order.==p1.Order), :]
            # Non-starting pitchers are included, so select the one with highest Consensus
            # this is normally the starting pitcher
            if nrow(p1_sim) > 1
                p1_sim = p1_sim[findmax(p1_sim.Consensus)[2], :]
            elseif nrow(p1_sim) == 0
                # There may be no matching players, so skip
                continue
            end
            p2_sim = team_frame[(team_frame.Position.==p2.Position).&(team_frame.Order.==p2.Order), :]
            if nrow(p2_sim) > 1
                p2_sim = p2_sim[findmax(p2_sim.Consensus)[2], :]
            elseif nrow(p2_sim) == 0
                continue
            end
            # p1_sim and p2_sim are DataFrame not DataFrameRow so we need to index to get a value instead of a vector
            # Last element is euclidean distance between historical pair of player's projections and current pair of player's projections
            push!(results, (p1_sim.Scored[1], p2_sim.Scored[1], euclid_dist((p1.Projection, p2.Projection), (p1_sim.Consensus[1], p2_sim.Consensus[1]))))
        end
    end
    sort!(results, by=x -> x[3])
    # Sometimes there may be limited samples
    if length(results) < num
        num = length(results)
    elseif length(results) < 10
        # If there aren't enough samples, return 0
        return 0.0
    end
    return cor([results[x][1] for x in 1:num], [results[x][2] for x in 1:num])
end


"""
    opp_team_corr(num::Integer, hist::DataFrame, p1::DataFrameRow, p2::DataFrameRow)

Finds k-nearest-neighbor correlation between players on opposing teams. Follows similar
logic to same-team-correlation but instead finds pairs of opposing players that match the
given players
"""
function opp_team_corr(num::Integer, hist::DataFrame, p1::DataFrameRow, p2::DataFrameRow)
    results = Tuple{Float64,Float64,Float64}[]
    for date_frame in groupby(hist, :Date)
        for team_frame in groupby(date_frame, :Team)
            # Get a similar player 1
            p1_sim = team_frame[(team_frame.Position.==p1.Position).&(team_frame.Order.==p1.Order), :]
            if nrow(p1_sim) > 1
                # Non-starting pitchers are included, so select the one with highest Consensus
                # this is normally the starting pitcher
                p1_sim = p1_sim[findmax(p1_sim.Consensus)[2], :]
            elseif nrow(p1_sim) == 0
                # If there are no matches, skip
                continue
            end
            # Get a similar player 2 from the opposing team to p1_sim
            p2_sim = date_frame[(date_frame.Team.==p1_sim.Opp_Team).&(date_frame.Position.==p2.Position).&(date_frame.Order.==p2.Order), :]
            if nrow(p2_sim) > 1
                p2_sim = p2_sim[findmax(p2_sim.Consensus)[2], :]
            elseif nrow(p2_sim) == 0
                continue
            else
                # p1_sim and p2_sim are DataFrame not DataFrameRow so we need to index to get a value instead of a vector
                # Last element is sum of euclidean distances between the two players
                push!(results, (p1_sim.Scored[1], p2_sim.Scored[1], euclid_dist((p1.Projection, p2.Projection), (p1_sim.Consensus[1], p2_sim.Consensus[1]))))
            end
        end
    end
    sort!(results, by=x -> x[3])
    if length(results) < num
        if length(results) < 10
            return 0.0
        else
            num = length(results)
        end
    end
    return cor([results[x][1] for x in 1:num], [results[x][2] for x in 1:num])
end


"""
    players_corr(num::Integer, hist::DataFrame, p1::DataFrameRow, p2::DataFrameRow)

Given two players, calls the appropriate correlation computation function. If the players aren't
on the same team or on opposing teams, this means they are from different games and have 0 correlation.
"""
function players_corr(num::Integer, hist::DataFrame, p1::DataFrameRow, p2::DataFrameRow)
    if p1.Team == p2.Team
        return same_team_corr(num, hist, p1, p2)
    elseif p1.Opponent == p2.Team
        return opp_team_corr(num, hist, p1, p2)
    else
        return 0.0
    end
end


"""
    get_corr(num::Integer, hist::DataFrame, slate::DataFrame)

Given a slate of players, computes full correlation matrix for them
"""
function get_corr(num::Integer, hist::DataFrame, slate::DataFrame)
    corr = Matrix{Float64}(undef, nrow(slate), nrow(slate))

    # Iterate over upper triangular portion of matrix, including the diagonal
    Threads.@threads for i in 1:nrow(slate)
        for j in i:nrow(slate)
            if i == j
                # Diagonal is always 1
                corr[i, j] = 1
            else
                # Otherwise set symmetric correlation entries
                pair_corr = players_corr(num, hist, slate[i, :], slate[j, :])
                corr[i, j] = pair_corr
                corr[j, i] = pair_corr
            end
        end
    end
    return corr
end


"""
    get_cov(num::Integer, hist::DataFrame, slate::DataFrame)

Computes covariance matrix for given slate
"""
function get_cov(num::Integer, hist::DataFrame, slate::DataFrame)
    corr = get_corr(num, hist, slate)
    σ = get_sigma(num, hist, slate)
    Σ = Diagonal(σ) * corr * Diagonal(σ)
    return Σ
end


"""
    write_cov()

Computes and writes to file the covariance matrix for today's slate
"""
function write_cov()
    slate = CSV.read("./data/slates/slate_$(Dates.today()).csv", DataFrame)
    hist = CSV.read("./data/linestar_data.csv", DataFrame)
    cov = get_cov(100, hist, slate)
    CSV.write("./data/slates/cov_$(Dates.today()).csv", Tables.table(cov), writeheader=false)
end