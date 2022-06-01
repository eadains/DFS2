using JuMP
using DataFrames
using CSV
using Dates
using SCIP

players = DataFrame(CSV.File("./data/slate_$(Dates.today()).csv"))
games = unique(players.Game)
teams = unique(players.Team)
positions = unique(players.Position)
pitchers = players[players.Position.=="P", :Name]

# Generate batting order stacks.
# Need a struct container to index the variable w specified below
struct Stack
    players::Vector{String}
end
stack_orders = [[1, 2, 3, 4], [2, 3, 4, 5]]
stacks = Stack[]
# For each team, find players with consectuvive batting orders 1,2,3,4 and 2,3,4,5
for team in teams
    for order in stack_orders
        stack_players = players[(players.Team.==team).*(in.(players.Order, Ref(order))), :Name]
        # Sometimes batting orders are 0's
        if length(stack_players) > 0
            push!(stacks, Stack(stack_players))
        end
    end
end

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

model = Model(SCIP.Optimizer)

# Players selected
@variable(model, x[players.Name], binary = true)
# Games selected
@variable(model, y[teams], binary = true)
# Teams selected
@variable(model, z[games], binary = true)
# Stacks selected
@variable(model, w[stacks], binary = true)

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

# Ensure that batters are not opposing the selected pitcher
for pitcher in pitchers
    # This forces the pitcher variable to be 0 if any opposing batters are selected
    # Ignore rows where opposing pitcher is missing (this row is usually the pitcher of the team)
    @constraint(model, sum(x[player.Name] for player in eachrow(players) if !ismissing(player.Opp_Pitcher) && (player.Opp_Pitcher == pitcher)) <= 9(1 - x[pitcher]))
end

for stack in stacks
    @constraint(model, 4w[stack] <= sum(x[player.Name] for player in eachrow(players) if player.Name in stack.players))
end
# Must select 2 4-man stacks
@constraint(model, sum(w) >= 2)

# Maximize projected fantasy points
@objective(model, Max, sum(player.Projection * x[player.Name] for player in eachrow(players)))

optimize!(model)

println("Projected Points: $(objective_value(model))")
println("Roster:")
for player in players.Name
    if value(x[player]) == 1
        println(player)
    end
end

# overlap = 5
# past_lineups = []
# append!(past_lineups, Ref(value.(x)))

# for past in past_lineups
#     @constraint(model, sum(x .* past) <= overlap)
# end