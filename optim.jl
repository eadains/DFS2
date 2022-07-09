using JuMP
using DataFrames
using CSV
using Dates
using LinearAlgebra
using Distributions
using StatsBase
using Xpress
using GLPK

abstract type OptimData end

struct TournyData <: OptimData
    players::DataFrame
    μ::Vector{<:Real}
    Σ::Symmetric{<:Real}
    opp_mu::Real
    opp_var::Real
    opp_cov::Vector{<:Real}
end


struct CashData <: OptimData
    players::DataFrame
end


struct BTSData <: OptimData
    players::DataFrame
    μ::Vector{<:Real}
    Σ::Symmetric{<:Real}
end


"""
    makeposdef(mat::Symmetric{<:Real})

Transforms a Symmetric real-valued matrix into a positive definite matrix
Computes eigendecomposition of matrix, sets negative and zero eigenvalues
to small value (1e-10) and then reconstructs matrix
"""
function makeposdef(mat::Symmetric{<:Real})
    vals = eigvals(mat)
    vecs = eigvecs(mat)
    vals[vals.<=1e-10] .= 1e-10
    return Symmetric(vecs * Diagonal(vals) * vecs')
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
function opp_team_score(players::DataFrame, μ::Vector{<:Real})
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
function compute_order_stat(players::DataFrame, μ::Vector{<:Real}, cutoff::Integer, entries::Integer)
    opp_scores = Vector{Float64}(undef, entries)
    Threads.@threads for i = 1:entries
        opp_scores[i] = opp_team_score(players, μ)
    end
    sort!(opp_scores, rev=true)
    return opp_scores[cutoff]
end


"""
    estimate_opp_stats(players::DataFrame, μ::Vector{<:Reak}, Σ::Symmetric{<:Real}, entries::Integer, cutoff::Integer)

Computes expected statistics about opponent lineups. Computes mean, variance, and covariance of the order statistics
of the opponents in the competition.
"""
function estimate_opp_stats(players::DataFrame, μ::Vector{<:Real}, Σ::Symmetric{<:Real}, entries::Integer, cutoff::Integer)
    score_draws = Vector{Float64}[]
    order_stats = Float64[]
    # Use 100 samples for calculating expectations
    for i = 1:50
        score_draw = rand(MvNormal(μ, Σ))
        order_stat = compute_order_stat(players, score_draw, cutoff, entries)
        println("$(i) done.")
        append!(score_draws, Ref(score_draw))
        append!(order_stats, order_stat)
    end

    opp_mu = mean(order_stats)
    opp_var = var(order_stats)
    # This is covariance between each individual player's score draws and the whole group of order statistics
    opp_cov = [cov([x[i] for x in score_draws], order_stats) for i in 1:nrow(players)]

    return (opp_mu, opp_var, opp_cov)
end


"""
    make_tourny_data()

Reads players and their covariances from file and generates opponent lineup statistics
and returns TournyData object.
"""
function make_tourny_data(entries::Integer, cutoff::Integer)
    players = CSV.read("./data/slates/slate_$(Dates.today()).csv", DataFrame)
    μ::Vector{Float64} = players[!, :Projection]
    cov_mat::Matrix{Float64} = CSV.read("./data/slates/cov_$(Dates.today()).csv", header=false, Tables.matrix)
    # covariance matrix must be positive definite so that Distributions.jl MvNormal
    # can do a cholesky factorization on it
    Σ = makeposdef(Symmetric(cov_mat))

    opp_mu, opp_var, opp_cov = estimate_opp_stats(players, μ, Σ, entries, cutoff)

    return TournyData(players, μ, Σ, opp_mu, opp_var, opp_cov)
end


"""
    make_cash_data()

Creates data struct for cash game optimization
"""
function make_cash_data()
    players = DataFrame(CSV.File("./data/slates/slate_$(Dates.today()).csv"))
    return CashData(players)
end


"""
    make_bts_data()

Reads players and their covariances from file and returns a BTSData object
for use in Beat the Score tournaments
"""
function make_bts_data()
    players = CSV.read("./data/slates/slate_$(Dates.today()).csv", DataFrame)
    μ::Vector{Float64} = players.Projection
    cov_mat::Matrix{Float64} = CSV.read("./data/slates/cov_$(Dates.today()).csv", header=false, Tables.matrix)
    # covariance matrix must be positive definite so that Distributions.jl MvNormal
    # can do a cholesky factorization on it
    Σ = makeposdef(Symmetric(cov_mat))
    return BTSData(players, μ, Σ)
end


"""
    do_optim(data::TournyData, past_lineups::Vector{JuMP.Containers.DenseAxisArray}, λ::Real, overlap::Integer)

Constructs and optimizes tournament model using given value of λ and overlap constraint
Returns a tuple where the first element is the lineup vector and the second 
is the objective value which is the estimated probability that the lineup exceeds opp_mu
"""
function do_optim(data::TournyData, past_lineups::Vector{JuMP.Containers.DenseAxisArray}, λ::Real, overlap::Integer)
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

    mu_x = @expression(model, x.data' * data.μ - data.opp_mu)
    var_x = @expression(model, x.data' * data.Σ * x.data + data.opp_var - 2 * x.data' * data.opp_cov)
    @objective(model, Max, mu_x + λ * var_x)

    optimize!(model)
    println(termination_status(model))
    # SCIP only checks for integer values within a tolerance, so round the result to the nearest integer
    return (round.(Int, value.(x)), 1 - cdf(Normal(), -value(mu_x) / sqrt(value(var_x))))
end


"""
    do_optim(data::CashData)

Runs optimization for cash games
"""
function do_optim(data::CashData)
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

    model = Model(GLPK.Optimizer)

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

    # Maximum and minimum number of players we can select for each position
    # Must always have 1 pitcher, who cannot fill the UTIL position
    # We can select up to 1 additional player from each other position because
    # the second can fill the UTIL position
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

    # Maximize projected fantasy points
    @objective(model, Max, sum(player.Projection * x[player.ID] for player in eachrow(data.players)))

    optimize!(model)
    println(termination_status(model))
    return (objective_value(model), value.(x))
end


"""
    do_optim(data::BTSData, score::Integer)

Does optimization for Beat the Score type tournaments. Score is the cutoff score to be in the money.
"""
function do_optim(data::BTSData, score::Integer, λ::Real)
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

    mu_x = @expression(model, x.data' * data.μ)
    var_x = @expression(model, x.data' * data.Σ * x.data)
    @objective(model, Max, mu_x + λ * var_x)

    optimize!(model)
    println(termination_status(model))
    # SCIP only checks for integer values within a tolerance, so round the result to the nearest integer
    return (round.(Int, value.(x)), 1 - cdf(Normal(), (score - value(mu_x)) / sqrt(value(var_x))))
end


"""
    lambda_max(data::TournyData, past_lineups::Vector{JuMP.Containers.DenseAxisArray})

Does optimization over range of λ values and returns the lineup with the highest objective function.
"""
function lambda_max(data::TournyData, past_lineups::Vector{JuMP.Containers.DenseAxisArray}, overlap::Integer)
    # I've found that lambdas from around 0 to 0.05 are selected, with strong majority being 0.02
    lambdas = 0:0.01:0.05
    w_star = Vector{Tuple{JuMP.Containers.DenseAxisArray,Float64}}(undef, length(lambdas))
    # Perform optimization over array of λ values
    Threads.@threads for i in 1:length(lambdas)
        w_star[i] = do_optim(data, past_lineups, lambdas[i], overlap)
    end

    # Find lambda value that leads to highest objective function and return its corresponding lineup vector
    max_index = argmax(x[2] for x in w_star)
    println("λ max: $(lambdas[max_index])")
    return w_star[max_index][1]
end


"""
    lambda_max(data::BTSData, score::Integer)

Does optimization over range of λ values for Beat the Score tournaments
and returns the lineup with the highest probability of exceeding the cutoff score
"""
function lambda_max(data::BTSData, score::Integer)
    # I've found that lambdas from around 0 to 0.05 are selected, with strong majority being 0.02
    lambdas = -0.05:0.01:0.05
    w_star = Vector{Tuple{JuMP.Containers.DenseAxisArray,Float64}}(undef, length(lambdas))
    # Perform optimization over array of λ values
    Threads.@threads for i in 1:length(lambdas)
        w_star[i] = do_optim(data, score, lambdas[i])
    end

    # Find lambda value that leads to highest objective function and return its corresponding lineup vector
    max_index = argmax(x[2] for x in w_star)
    println("λ max: $(lambdas[max_index])")
    return w_star[max_index][1]
end


"""
    get_lineups(N::Integer)

Solves the tournament optimization problem with N entries.
Returns the vector of lineups
"""
function get_lineups(data::TournyData, N::Integer, overlap::Integer)
    # A vector of all the lineups we've made so far
    lineups = JuMP.Containers.DenseAxisArray[]
    for n in 1:N
        println(n)
        lineup = lambda_max(data, lineups, overlap)
        append!(lineups, Ref(lineup))
    end

    return lineups
end


"""
    transform_lineup(lineup::JuMP.Containers.DenseAxisArray)

Transforms lineup vector from optimization to a dict mapping between roster position and player ID
"""
function transform_lineup(data::OptimData, lineup::JuMP.Containers.DenseAxisArray)
    # Roster positions to fill
    positions = Dict{String,Union{String,Missing}}("P" => missing, "C/1B" => missing, "2B" => missing, "3B" => missing, "SS" => missing, "OF1" => missing, "OF2" => missing, "OF3" => missing, "UTIL" => missing)
    for player in eachrow(data.players)
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
    return positions
end


"""
    write_lineups(lineups::Vector{Dict{String, String}})

Writes rosters to CSV in format acceptable by FanDuel
"""
function write_lineups(lineups::Vector{Dict{String,Union{Missing,String}}})
    open("./tourny_lineups.csv", "w") do file
        println(file, "P,C/1B,2B,3B,SS,OF,OF,OF,UTIL")
        for lineup in lineups
            println(file, "$(lineup["P"]),$(lineup["C/1B"]),$(lineup["2B"]),$(lineup["3B"]),$(lineup["SS"]),$(lineup["OF1"]),$(lineup["OF2"]),$(lineup["OF3"]),$(lineup["UTIL"])")
        end
    end
end


"""
    write_lineups(points <: Number, lineup::Dict{String,String})

Writes cash game lineup to file with expected points
"""
function write_lineup(points::Number, lineup::Dict{String,Union{Missing,String}})
    open("./cash_lineup.csv", "w") do file
        println(file, "Projected Points: $(points)")
        println(file, "P,C/1B,2B,3B,SS,OF,OF,OF,UTIL")
        println(file, "$(lineup["P"]),$(lineup["C/1B"]),$(lineup["2B"]),$(lineup["3B"]),$(lineup["SS"]),$(lineup["OF1"]),$(lineup["OF2"]),$(lineup["OF3"]),$(lineup["UTIL"])")
    end
end


"""
    write_lineups(lineup::Dict{String,String})

Writes Beat the Score lineup to file with expected points
"""
function write_lineup(lineup::Dict{String,Union{Missing,String}})
    open("./BTS_lineup.csv", "w") do file
        println(file, "Projected Points: $(points)")
        println(file, "P,C/1B,2B,3B,SS,OF,OF,OF,UTIL")
        println(file, "$(lineup["P"]),$(lineup["C/1B"]),$(lineup["2B"]),$(lineup["3B"]),$(lineup["SS"]),$(lineup["OF1"]),$(lineup["OF2"]),$(lineup["OF3"]),$(lineup["UTIL"])")
    end
end


"""
    solve_tourny()

Solves tournament optimization problem. Constructs input data, gets lineups, and write them to file.
"""
function solve_tourny()
    # Assume competition has 25000 entries, and we want to maximize the probability
    # of coming in first
    # This takes about 30 minutes to run
    data = make_tourny_data(25000, 1)
    # Ask for number of lineups to generate
    # This takes ~15 minutes to run
    num = 0
    while true
        print("Enter number of tournament lineups to generate: ")
        num = readline()
        try
            num = parse(Int, num)
            break
        catch
            print("Invalid number entered, try again\n")
        end
    end
    overlap = 0
    while true
        print("Enter overlap parameter: ")
        overlap = readline()
        try
            overlap = parse(Int, overlap)
            break
        catch
            print("Invalid number entered, try again\n")
        end
    end
    lineups = get_lineups(data, num, overlap)
    lineups = [transform_lineup(data, x) for x in lineups]
    write_lineups(lineups)
end


"""
    solve_cash()

Solves cash game optimization and writes results to file
"""
function solve_cash()
    data = make_cash_data()
    points, lineup = do_optim(data)
    lineup = transform_lineup(data, lineup)
    write_lineup(points, lineup)
end


"""
    solve_BTS()

Solves Beat the Score optimization and writes results to file
"""
function solve_BTS()
    data = make_bts_data()
    score = 0
    while true
        print("Enter score to beat: ")
        score = readline()
        try
            score = parse(Int, score)
            break
        catch
            print("Invalid number entered, try again\n")
        end
    end
    points, lineup = lambda_max(data, score)
    lineup = transform_lineup(data, lineup)
    write_lineup(points, lineup)
end