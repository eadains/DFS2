using JuMP
using DataFrames
using CSV
using Dates
using GLPK

# Need a struct container to index the variable w specified below
struct Stack
    players::Vector{String}
end

function do_optim(players, past_lineups)
    games = unique(players.Game)
    teams = unique(players.Team)
    positions = unique(players.Position)
    pitchers = players[players.Position.=="P", :ID]
    # Maximum overlap parameter between current lineup and all past lineups
    overlap = 3

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
    @variable(model, x[players.ID], binary = true)
    # Games selected
    @variable(model, y[teams], binary = true)
    # Teams selected
    @variable(model, z[games], binary = true)
    # Team stack constraint
    @variable(model, w[teams], binary = true)

    # Total salary of selected players must be <= $35,000
    @constraint(model, sum(player.Salary * x[player.ID] for player in eachrow(players)) <= 35000)

    # Must select 9 total players
    @constraint(model, sum(x) == 9)

    # Maximum and minimum number of players we can select for each position
    # Must always have 1 pitcher, who cannot fill the UTIL position
    # We can select up to 1 additional player from each other position because
    # the second can fill the UTIL position
    for position in positions
        @constraint(model, positions_min[position] <= sum(x[player.ID] for player in eachrow(players) if player.Position == position) <= positions_max[position])
    end

    for team in teams
        # Excluding the pitcher, we can select a maximum of 4 players per team
        @constraint(model, sum(x[player.ID] for player in eachrow(players) if player.Team == team && player.Position != "P") <= 4)
        # If no players are selected from a team, y is set to 0
        @constraint(model, y[team] <= sum(x[player.ID] for player in eachrow(players) if player.Team == team))
        # If less than 4 players, excluding the pitcher, are selected from a team, w is set to 0
        @constraint(model, 4w[team] <= sum(x[player.ID] for player in eachrow(players) if player.Team == team && player.Position != "P"))
    end
    # Must have players from at least 3 teams
    @constraint(model, sum(y) >= 3)
    # Must select at least 2 team stacks
    @constraint(model, sum(w) >= 2)

    for game in games
        # If no players are selected from a game z is set to 0
        @constraint(model, z[game] <= sum(x[player.ID] for player in eachrow(players) if player.Game == game))
    end
    # Must select players from at least 2 games
    @constraint(model, sum(z) >= 2)

    # Ensure that batters are not opposing the selected pitcher
    for pitcher in pitchers
        # This forces the pitcher variable to be 0 if any opposing batters are selected
        # Ignore rows where opposing pitcher is missing (this row is usually the pitcher of the team)
        @constraint(model, sum(x[player.ID] for player in eachrow(players) if !ismissing(player.Opp_Pitcher) && (player.Opp_Pitcher == pitcher)) <= 9(1 - x[pitcher]))
    end

    # If any past lineups have been selected, make sure the current lineup doesn't overlap
    if length(past_lineups) > 0
        for past in past_lineups
            @constraint(model, sum(x .* past) <= overlap)
        end
    end

    # Maximize projected fantasy points
    @objective(model, Max, sum(player.Projection * x[player.ID] for player in eachrow(players)))

    optimize!(model)
    println(termination_status(model))
    return value.(x)
end

players = DataFrame(CSV.File("./data/slate_$(Dates.today()).csv"))
# Number of lineups to generate
N = 60
past_lineups = []

for n in 1:N
    println(n)
    lineup = do_optim(players, past_lineups)
    append!(past_lineups, Ref(value.(lineup)))
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