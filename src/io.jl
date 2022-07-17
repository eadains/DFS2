using Tables
using CSV
include("types.jl")


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
    get_mlb_slate(date::String)

Constructs MLBSlate given a date.
"""
function get_mlb_slate(date::String)
    players = CSV.read("./data/slates/slate_$(date).csv", Tables.rowtable)
    μ = [player.Projection for player in players]
    Σ = makeposdef(Symmetric(CSV.read("./data/slates/cov_$(date).csv", header=false, Tables.matrix)))
    games = unique([player.Game for player in players])
    teams = unique([player.Team for player in players])
    return MLBSlate(players, games, teams, μ, Σ)
end


"""
    transform_lineup(lineup::JuMP.Containers.DenseAxisArray)

Transforms lineup vector from optimization to a dict mapping between roster position and player ID
"""
function transform_lineup(slate::MLBSlate, lineup::JuMP.Containers.DenseAxisArray)
    # Roster positions to fill
    positions = Dict{String,Union{String,Missing}}("P" => missing, "C/1B" => missing, "2B" => missing, "3B" => missing, "SS" => missing, "OF1" => missing, "OF2" => missing, "OF3" => missing, "UTIL" => missing)
    for player in slate.players
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

Writes multiple tournament lineups to toury_lineups.csv
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