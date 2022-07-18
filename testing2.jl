using CPLEX
include("./src/solve.jl")


function find_thetasq_upper_bound(slate)
    model = Model(CPLEX.Optimizer)
    set_optimizer_attribute(model, "CPXPARAM_MIP_Display", 4)
    set_optimizer_attribute(model, "CPXPARAM_ScreenOutput", 1)
    set_optimizer_attribute(model, "CPXPARAM_Emphasis_MIP", 5)
    set_optimizer_attribute(model, "CPXPARAM_TimeLimit", 180)

    p = length(slate.players)
    num_teams = length(slate.teams)
    num_games = length(slate.games)
    @variable(model, x[1:2, 1:p], binary = true)
    # Games selected
    @variable(model, t[1:2, 1:num_teams], binary = true)
    # Teams selected
    @variable(model, g[1:2, 1:num_games], binary = true)
    # Linearization variables
    @variable(model, v[1:2, 1:p, 1:p], binary = true)
    @variable(model, r[1:p, 1:p], binary = true)

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

    @objective(model, Max, s)
    optimize!(model)
    return value(s)
end


function make_thetasq_intervals(upper_bound, n)
    intervals = Vector{Float64}(undef, n + 1)
    intervals[1] = 0
    intervals[2] = 1
    for i in 3:n+1
        intervals[i] = intervals[i-1] + (upper_bound - 1) / (n - 1)
    end
    return intervals
end


function find_delta_upper_bound(slate)
    model = Model(CPLEX.Optimizer)
    set_optimizer_attribute(model, "CPXPARAM_MIP_Display", 4)
    set_optimizer_attribute(model, "CPXPARAM_ScreenOutput", 1)
    set_optimizer_attribute(model, "CPXPARAM_TimeLimit", 180)

    p = length(slate.players)
    num_teams = length(slate.teams)
    num_games = length(slate.games)
    @variable(model, x[1:2, 1:p], binary = true)
    # Games selected
    @variable(model, t[1:2, 1:num_teams], binary = true)
    # Teams selected
    @variable(model, g[1:2, 1:num_games], binary = true)
    # Linearization variables
    @variable(model, v[1:2, 1:p, 1:p], binary = true)
    @variable(model, r[1:p, 1:p], binary = true)

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

    delta = @expression(model, u_1 - u_2)

    @objective(model, Max, delta)
    optimize!(model)
    return value(delta)
end


function make_delta_intervals(upper_bound, n)
    intervals = Vector{Float64}(undef, n + 1)
    for i in 1:n+1
        intervals[i] = ((i - 1) / n) * upper_bound
    end
    return intervals
end


function make_theta_upper_intervals(thetasq_intervals)
    theta_upper_intervals = Vector{Float64}(undef, length(thetasq_intervals) - 1)
    for i in 1:length(theta_upper_intervals)
        if i == 1
            theta_upper_intervals[i] = 1
        else
            theta_upper_intervals[i] = sqrt(thetasq_intervals[i+1])
        end
    end
    return theta_upper_intervals
end


function make_theta_lower_intervals(thetasq_intervals)
    theta_lower_intervals = Vector{Float64}(undef, length(thetasq_intervals) - 1)
    for i in 1:length(theta_lower_intervals)
        if i == 1
            theta_lower_intervals[i] = 0
        else
            theta_lower_intervals[i] = sqrt(thetasq_intervals[i])
        end
    end
    return theta_lower_intervals
end


function make_cdf_constants(theta_lower_intervals, delta_intervals)
    d = length(theta_lower_intervals)
    l = length(delta_intervals) - 1
    cdf_constants = Matrix{Float64}(undef, d, l)
    for q in 1:d
        for k in 1:l
            cdf_constants[q, k] = cdf(Normal(), delta_intervals[k+1] / theta_lower_intervals[q])
        end
    end
    return cdf_constants
end


function find_u_max(slate)
    model = Model(CPLEX.Optimizer)
    set_optimizer_attribute(model, "CPXPARAM_MIP_Display", 4)
    set_optimizer_attribute(model, "CPXPARAM_ScreenOutput", 1)

    p = length(slate.players)
    num_teams = length(slate.teams)
    num_games = length(slate.games)
    @variable(model, x[1:p], binary = true)
    # Games selected
    @variable(model, t[1:num_teams], binary = true)
    # Teams selected
    @variable(model, g[1:num_games], binary = true)

    # Total salary must be <= $35,000
    @constraint(model, sum(slate.players[i].Salary * x[i] for i in 1:p) <= 35000)
    # Must select 9 total players
    @constraint(model, sum(x[i] for i in 1:p) == 9)
    # Constraints for each position, must always select 1 pitcher, but can select 1 additional
    # player for each other position to fill the utility slot
    @constraint(model, sum(x[i] for i in 1:p if slate.players[i].Position == "P") == 1)
    @constraint(model, 1 <= sum(x[i] for i in 1:p if slate.players[i].Position == "C/1B") <= 2)
    @constraint(model, 1 <= sum(x[i] for i in 1:p if slate.players[i].Position == "2B") <= 2)
    @constraint(model, 1 <= sum(x[i] for i in 1:p if slate.players[i].Position == "3B") <= 2)
    @constraint(model, 1 <= sum(x[i] for i in 1:p if slate.players[i].Position == "SS") <= 2)
    @constraint(model, 3 <= sum(x[i] for i in 1:p if slate.players[i].Position == "OF") <= 4)

    for k in 1:num_teams
        # Excluding the pitcher, we can select a maximum of 4 players per team
        @constraint(model, sum(x[i] for i in 1:p if (slate.players[i].Team == slate.teams[k]) && (slate.players[i].Position != "P")) <= 4)
        # If no players are selected from a team, t is set to 0
        @constraint(model, t[k] <= sum(x[i] for i in 1:p if slate.players[i].Team == slate.teams[k]))
    end
    # Must have players from at least 3 teams
    @constraint(model, sum(t[i] for i in 1:num_teams) >= 3)

    for k in 1:num_games
        # If no players are selected from a game z is set to 0
        @constraint(model, g[k] <= sum(x[i] for i in 1:p if slate.players[i].Game == slate.games[k]))
    end
    # Must select players from at least 2 games
    @constraint(model, sum(g[i] for i in 1:num_games) >= 2)

    obj = @expression(model, sum(x[i] * slate.μ[i] for i = 1:p))
    @objective(model, Max, obj)
    optimize!(model)
    return value(obj)
end


function find_z(slate)
    model = Model(CPLEX.Optimizer)
    set_optimizer_attribute(model, "CPXPARAM_MIP_Display", 4)
    set_optimizer_attribute(model, "CPXPARAM_ScreenOutput", 1)

    p = length(slate.players)
    num_teams = length(slate.teams)
    num_games = length(slate.games)
    @variable(model, x[1:p], binary = true)
    # Games selected
    @variable(model, t[1:num_teams], binary = true)
    # Teams selected
    @variable(model, g[1:num_games], binary = true)

    # Total salary must be <= $35,000
    @constraint(model, sum(slate.players[i].Salary * x[i] for i in 1:p) <= 35000)
    # Must select 9 total players
    @constraint(model, sum(x[i] for i in 1:p) == 9)
    # Constraints for each position, must always select 1 pitcher, but can select 1 additional
    # player for each other position to fill the utility slot
    @constraint(model, sum(x[i] for i in 1:p if slate.players[i].Position == "P") == 1)
    @constraint(model, 1 <= sum(x[i] for i in 1:p if slate.players[i].Position == "C/1B") <= 2)
    @constraint(model, 1 <= sum(x[i] for i in 1:p if slate.players[i].Position == "2B") <= 2)
    @constraint(model, 1 <= sum(x[i] for i in 1:p if slate.players[i].Position == "3B") <= 2)
    @constraint(model, 1 <= sum(x[i] for i in 1:p if slate.players[i].Position == "SS") <= 2)
    @constraint(model, 3 <= sum(x[i] for i in 1:p if slate.players[i].Position == "OF") <= 4)

    for k in 1:num_teams
        # Excluding the pitcher, we can select a maximum of 4 players per team
        @constraint(model, sum(x[i] for i in 1:p if (slate.players[i].Team == slate.teams[k]) && (slate.players[i].Position != "P")) <= 4)
        # If no players are selected from a team, t is set to 0
        @constraint(model, t[k] <= sum(x[i] for i in 1:p if slate.players[i].Team == slate.teams[k]))
    end
    # Must have players from at least 3 teams
    @constraint(model, sum(t[i] for i in 1:num_teams) >= 3)

    for k in 1:num_games
        # If no players are selected from a game z is set to 0
        @constraint(model, g[k] <= sum(x[i] for i in 1:p if slate.players[i].Game == slate.games[k]))
    end
    # Must select players from at least 2 games
    @constraint(model, sum(g[i] for i in 1:num_games) >= 2)

    mu = @expression(model, sum(x[i] * slate.μ[i] for i = 1:p))
    var = @expression(model, x' * slate.Σ * x)

    @objective(model, Min, mu + var)
    optimize!(model)
    return value(mu)
end


function LBP(delta, u_max, z)
    model = Model(SCIP.Optimizer)
    @variable(model, theta)

    @constraint(model, theta >= 0)
    @NLconstraint(model, u_max + theta * ((1 / sqrt(2pi)) * exp((delta / theta)^2 / 2)) >= z)

    @objective(model, Min, theta)
    optimize!(model)
    return value(theta)
end


function make_SVIs(delta_intervals, u_max, z)
    l = length(delta_intervals) - 1
    SVIs = Vector{Float64}(undef, l)
    for i in 1:l
        SVIs[i] = LBP(delta_intervals[i], u_max, z)
    end
    return SVIs
end


thetasq_upper_bound = find_thetasq_upper_bound(slate)
thetasq_intervals = make_thetasq_intervals(thetasq_upper_bound, 100)
theta_upper_intervals = make_theta_upper_intervals(thetasq_intervals)
theta_lower_intervals = make_theta_lower_intervals(thetasq_intervals)
delta_upper_bound = find_delta_upper_bound(slate)
delta_intervals = make_delta_intervals(delta_upper_bound, 100)
cdf_constants = make_cdf_constants(theta_lower_intervals, delta_intervals)
u_max = find_u_max(slate)
z = find_z(slate)
SVIs = make_SVIs(delta_intervals, u_max, z)

model = Model(CPLEX.Optimizer)
set_optimizer_attribute(model, "CPXPARAM_MIP_Display", 4)
set_optimizer_attribute(model, "CPXPARAM_ScreenOutput", 1)
set_optimizer_attribute(model, "CPXPARAM_Emphasis_MIP", 3)
set_optimizer_attribute(model, "CPXPARAM_TimeLimit", 1800)

p = length(slate.players)
num_teams = length(slate.teams)
num_games = length(slate.games)
@variable(model, x[1:2, 1:p], binary = true)
# Games selected
@variable(model, t[1:2, 1:num_teams], binary = true)
# Teams selected
@variable(model, g[1:2, 1:num_games], binary = true)
# Linearization variables
@variable(model, v[1:2, 1:p, 1:p], binary = true)
@variable(model, r[1:p, 1:p], binary = true)
# Interval selection variables
@variable(model, w[1:100], binary = true)
@variable(model, y[1:100], binary = true)
@variable(model, u_prime)

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

@constraint(model, sum(w[i] for i = 1:100) == 1)
@constraint(model, sum(y[i] for i = 1:100) == 1)

@constraint(model, [q = 1:100], thetasq_intervals[q] * w[q] <= s)
@constraint(model, [q = 1:100], s <= thetasq_intervals[q+1] + thetasq_intervals[101] * (1 - w[q]))

@constraint(model, [k = 1:100], delta_intervals[k] * y[k] <= u_1 - u_2)
@constraint(model, [k = 1:100], u_1 - u_2 <= delta_intervals[k+1] + delta_intervals[101] * (1 - y[k]))

@constraint(model, [q = 1:100, k = 1:100], u_prime <= u_1 * cdf_constants[q, k] + u_2 * (1 - cdf_constants[q, k]) + 250 * (2 - w[q] - y[k]))

@constraint(model, [k = 1:100], s >= SVIs[k]^2 * y[k])

s_prime = @expression(model, sum(theta_upper_intervals[q] * w[q] for q = 1:100))

@objective(model, Max, u_prime + (1 / sqrt(2pi)) * s_prime)