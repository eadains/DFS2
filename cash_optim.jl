using JuMP
using DataFrames
using CSV
using Dates
using GLPK

players = DataFrame(CSV.File("./data/slate_$(Dates.today()).csv"))
games = unique(players.Game)
teams = unique(players.Team)
positions = unique(players.Position)

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
@variable(model, x[players.Name], binary = true)
# Games selected
@variable(model, y[teams], binary = true)
# Teams selected
@variable(model, z[games], binary = true)

# Total salary of selected players must be <= $35,000
@constraint(model, sum(player.Salary * x[player.Name] for player in eachrow(players)) <= 35000)

# Must select 9 total players
@constraint(model, sum(x) == 9)

# Maximum and minimum number of players we can select for each position
# Must always have 1 pitcher, who cannot fill the UTIL position
# We can select up to 1 additional player from each other position because
# the second can fill the UTIL position
for position in positions
    @constraint(model, positions_min[position] <= sum(x[player.Name] for player in eachrow(players) if player.Position == position) <= positions_max[position])
end

for team in teams
    # Excluding the pitcher, we can select a maximum of 4 players per team
    @constraint(model, sum(x[player.Name] for player in eachrow(players) if player.Team == team && player.Position != "P") <= 4)
    # If no players are selected from a team, y is set to 0
    @constraint(model, y[team] <= sum(x[player.Name] for player in eachrow(players) if player.Team == team))
end
# Must have players from at least 3 teams
@constraint(model, sum(y) >= 3)

for game in games
    # If no players are selected from a game z is set to 0
    @constraint(model, z[game] <= sum(x[player.Name] for player in eachrow(players) if player.Game == game))
end
# Must select players from at least 2 games
@constraint(model, sum(z) >= 2)

# Maximize projected fantasy points
@objective(model, Max, sum(player.Projection * x[player.Name] for player in eachrow(players)))

optimize!(model)
println(termination_status(model))

println("Projected Points: $(objective_value(model))")
println("Roster:")
for player in players.Name
    if value(x[player]) == 1
        println(player)
    end
end