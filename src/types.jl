using LinearAlgebra
using JuMP


abstract type Slate end

struct MLBSlate <: Slate
    positions::Dict{String,Int64}
    # Writing this out explicity helps type stability and ensures that all columns needed are present
    players::Vector{NamedTuple{(:Name, :ID, :Position, :Salary, :Game, :Team, :Opponent, :Order, :Opp_Pitcher, :Projection),Tuple{String,String,String,Int64,String,String,String,Int64,Union{Missing,String},Float64}}}
    games::AbstractVector{String}
    teams::AbstractVector{String}
    μ::AbstractVector{<:Real}
    Σ::Symmetric{<:Real}
end


function MLBSlate(players, games, teams, μ, Σ)
    # These are the positions we must fill for a MLB team on FanDuel
    positions = Dict(
        "P" => 1,
        "C/1B" => 1,
        "2B" => 1,
        "3B" => 1,
        "SS" => 1,
        "OF" => 3,
        "UTIL" => 1
    )
    return MLBSlate(positions, players, games, teams, μ, Σ)
end


abstract type OptimData end

struct MLBCashOptimData <: OptimData
    slate::MLBSlate
end


# struct MLBTournyOptimData <: OptimData
#     slate::MLBSlate
#     pastlineups::Vector{JuMP.Containers.DenseAxisArray}
#     overlap::Int64
#     opp_mu::Float64
#     opp_var::Float64
#     opp_cov::AbstractVector{Float64}
# end

struct MLBTournyOptimData <: OptimData
    slate::MLBSlate
    pastlineups::Vector{JuMP.Containers.DenseAxisArray}
    overlap::Int64
end



function MLBTournyOptimData(slate, overlap)
    return MLBTournyOptimData(slate, JuMP.Containers.DenseAxisArray[], overlap)
end