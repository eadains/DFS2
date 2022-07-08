using DataFrames
using CSV
using Dates
using Statistics
using LinearAlgebra
using JuMP
using Xpress


struct TournyData
    players::DataFrame
    μ::Vector{<:Real}
    Σ::Symmetric{<:Real}
end


function makeposdef(mat::Symmetric{<:Real})
    vals = eigvals(mat)
    vecs = eigvecs(mat)
    vals[vals.<=1e-10] .= 1e-10
    return Symmetric(vecs * Diagonal(vals) * vecs')
end


function do_optim(data::TournyData, past_lineups::Vector{JuMP.Containers.DenseAxisArray}, overlap::Integer)
    games = unique(data.players.Game)
    teams = unique(data.players.Team)
    positions = unique(data.players.Position)

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

    model = Model(Xpress.Optimizer)

    # Players selected
    @variable(model, x[data.players.ID], binary = true)
    # Games selected
    @variable(model, y[teams], binary = true)
    # Teams selected
    @variable(model, z[games], binary = true)

    # Total salary of selected players must be <= $35,000
    @constraint(model, sum(player.Salary * x[player.ID] for player in eachrow(data.players)) <= 35000)

    # Must select 9 total players
    @constraint(model, sum(x) == 9)

    for position in positions
        @constraint(model, positions_min[position] <= sum(x[player.ID] for player in eachrow(data.players) if player.Position == position) <= positions_max[position])
    end

    for team in teams
        # Excluding the pitcher, we can select a maximum of 4 players per team
        @constraint(model, sum(x[player.ID] for player in eachrow(data.players) if player.Team == team && player.Position != "P") <= 4)
        # If no players are selected from a team, y is set to 0
        @constraint(model, y[team] <= sum(x[player.ID] for player in eachrow(data.players) if player.Team == team))
    end
    # Must have players from at least 3 teams
    @constraint(model, sum(y) >= 3)

    for game in games
        # If no players are selected from a game z is set to 0
        @constraint(model, z[game] <= sum(x[player.ID] for player in eachrow(data.players) if player.Game == game))
    end
    # Must select players from at least 2 games
    @constraint(model, sum(z) >= 2)

    # If any past lineups have been selected, make sure the current lineup doesn't overlap
    if length(past_lineups) > 0
        for past in past_lineups
            @constraint(model, sum(x .* past) <= overlap)
        end
    end

    mu_x = @expression(model, x.data' * data.μ)
    var_x = @expression(model, x.data' * data.Σ * x.data)
    @objective(model, Max, mu_x + 0.03 * var_x)

    optimize!(model)
    println(termination_status(model))
    if termination_status(model) == JuMP.INFEASIBLE
        return "INFEASIBLE"
    else
        # SCIP only checks for integer values within a tolerance, so round the result to the nearest integer
        return round.(Int, value.(x))
    end
end


function make_tourny_data(date::String)
    players = CSV.read("./data/realized_slates/slate_$(date).csv", DataFrame)
    μ::Vector{Float64} = players.Projection
    cov_mat::Matrix{Float64} = CSV.read("./data/slates/cov_$(date).csv", header=false, Tables.matrix)
    # covariance matrix must be positive definite so that Distributions.jl MvNormal
    # can do a cholesky factorization on it
    Σ = makeposdef(Symmetric(cov_mat))
    return TournyData(players, μ, Σ)
end


function max_score(data::TournyData, lineups::Vector{JuMP.Containers.DenseAxisArray})
    scores = Vector{Float64}(undef, length(lineups))
    for (i, lineup) in enumerate(lineups)
        scores[i] = lineup ⋅ data.players.Scored
    end
    return maximum(scores)
end


function overlap_optim(data::TournyData, overlap::Integer)
    past_lineups = JuMP.Containers.DenseAxisArray[]
    for n in 1:50
        println("Overlap: $(overlap) n: $(n)")
        lineup = do_optim(data, past_lineups, overlap)
        if lineup == "INFEASIBLE"
            println("Infeasible overlap constraint")
            return 0
        else
            append!(past_lineups, Ref(lineup))
        end
    end
    return max_score(data, past_lineups)
end


function max_scoring_overlap(data::TournyData)
    overlap_scores = Dict{Int64,Float64}()
    for overlap in 0:9
        overlap_scores[overlap] = overlap_optim(data, overlap)
    end
    return overlap_scores
end


function do_test()
    dates = ["2022-06-09", "2022-06-10", "2022-06-13", "2022-06-15", "2022-06-16",
        "2022-06-17", "2022-06-18", "2022-06-20", "2022-06-21", "2022-06-22", "2022-06-24",
        "2022-06-25", "2022-06-26", "2022-06-28", "2022-06-29", "2022-06-40", "2022-07-01", "2022-07-02", "2022-07-03"]
    for date in dates
        println(date)
        data = make_tourny_data(date)
        num_games = length(unique(data.players.Game))
        overlap_scores = max_scoring_overlap(data)
        open("./data/overlap.csv", "a") do file
            println(file, "$(date),$(num_games),$(overlap_scores[0]),$(overlap_scores[1]),$(overlap_scores[2]),$(overlap_scores[3]),$(overlap_scores[4]),$(overlap_scores[5]),$(overlap_scores[6]),$(overlap_scores[7]),$(overlap_scores[8]),$(overlap_scores[9])")
        end
    end
end