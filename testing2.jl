using CPLEX
include("./src/solve.jl")


model = Model(CPLEX.Optimizer)
set_optimizer_attribute(model, "CPXPARAM_MIP_Display", 4)
set_optimizer_attribute(model, "CPXPARAM_ScreenOutput", 1)
set_optimizer_attribute(model, "CPXPARAM_Emphasis_MIP", 3)

num_players = length(slate.players)
num_teams = length(slate.teams)
num_games = length(slate.games)
@variable(model, x[1:2, 1:num_players], binary = true)
# Games selected
@variable(model, t[1:2, 1:num_teams], binary = true)
# Teams selected
@variable(model, g[1:2, 1:num_games], binary = true)
# Linearization variables
@variable(model, v[1:2, 1:num_players, 1:num_players], binary = true)
@variable(model, r[1:num_players, 1:num_players], binary = true)

for j in 1:2
    # Total salary must be <= $35,000
    @constraint(model, sum(slate.players[i].Salary * x[j, i] for i in 1:p) <= 35000)
    # Must select 9 total players
    @constraint(model, sum(x[j, i] for i in 1:p) == 9)
    # Constraints for each position, must always select 1 pitcher, but can select 1 additional
    # player for each other position to fill the utility slot
    @constraint(model, sum(x[j, i] for i in 1:p if slate.players[i].Position == "P") == 1)
    @constraint(model, 1 <= sum(x[j, i] for i in 1:p if slate.players[i].Position == "C/1B") <= 2)
    @constraint(model, 1 <= sum(x[j, i] for i in 1:p if slate.players[i].Position == "2B") <= 2)
    @constraint(model, 1 <= sum(x[j, i] for i in 1:p if slate.players[i].Position == "3B") <= 2)
    @constraint(model, 1 <= sum(x[j, i] for i in 1:p if slate.players[i].Position == "SS") <= 2)
    @constraint(model, 3 <= sum(x[j, i] for i in 1:p if slate.players[i].Position == "OF") <= 4)

    for k in 1:num_teams
        # Excluding the pitcher, we can select a maximum of 4 players per team
        @constraint(model, sum(x[j, i] for i in 1:p if (slate.players[i].Team == slate.teams[k]) && (slate.players[i].Position != "P")) <= 4)
        # If no players are selected from a team, t is set to 0
        @constraint(model, t[j, k] <= sum(x[j, i] for i in 1:p if slate.players[i].Team == slate.teams[k]))
    end
    # Must have players from at least 3 teams
    @constraint(model, sum(t[j, i] for i in 1:num_teams) >= 3)

    for k in 1:num_games
        # If no players are selected from a game z is set to 0
        @constraint(model, g[j, k] <= sum(x[j, i] for i in 1:p if slate.players[i].Game == slate.games[k]))
    end
    # Must select players from at least 2 games
    @constraint(model, sum(g[j, i] for i in 1:num_games) >= 2)
end

@objective(model, Max, sum(slate.μ[i] * x[1, i] for i = 1:p))

# Expectation of team 1 and 2
u_1 = @expression(model, sum(slate.μ[i] * x[1, i] for i in 1:p))
u_2 = @expression(model, sum(slate.μ[i] * x[2, i] for i in 1:p))
# Symmetry breaking condition
@constraint(model, u_1 >= u_2)

s = @expression(model, sum(sum(slate.Σ[j, j] * x[i, j] for j in 1:p) + 2 * sum(slate.Σ[j_1, j_2] * v[i, j_1, j_2] for j_1 = 1:p, j_2 = 1:p if j_1 < j_2) for i in 1:2) - 2 * sum(sum(slate.Σ[j_1, j_2] * r[j_1, j_2] for j_2 in 1:p) for j_1 in 1:p))
@constraint(model, [i = 1:2, j_1 = 1:p, j_2 = 1:p], v[i, j_1, j_2] <= x[i, j_1])
@constraint(model, [i = 1:2, j_1 = 1:p, j_2 = 1:p], v[i, j_1, j_2] <= x[i, j_2])
@constraint(model, [i = 1:2, j_1 = 1:p, j_2 = 1:p], v[i, j_1, j_2] >= x[i, j_1] + x[i, j_2] - 1)
@constraint(model, [j_1 = 1:p, j_2 = 1:p], r[j_1, j_2] <= x[1, j_1])
@constraint(model, [j_1 = 1:p, j_2 = 1:p], r[j_1, j_2] <= x[2, j_2])
@constraint(model, [j_1 = 1:p, j_2 = 1:p], r[j_1, j_2] >= x[1, j_1] + x[2, j_2] - 1)

@objective(model, Max, u_1 + 1 / sqrt(2pi) * (1 + s))