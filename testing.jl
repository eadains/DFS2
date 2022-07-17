include("./src/solve.jl")


function emax(mu_x1, mu_x2, var_x1, var_x2, cov)
    Φ = x -> cdf(Normal(), x)
    ϕ = x -> pdf(Normal(), x)
    θ = sqrt(var_x1 + var_x2 - 2 * cov)
    return mu_x1 * Φ((mu_x1 - mu_x2) / θ) + mu_x2 * Φ((mu_x2 - mu_x1) / θ) + θ * ϕ((mu_x1 - mu_x2) / θ)
end


function do_optim(slate::MLBSlate, cuts::Vector{JuMP.Containers.DenseAxisArray})
    model = Model(CPLEX.Optimizer)
    set_optimizer_attribute(model, "CPXPARAM_MIP_Display", 1)
    set_optimizer_attribute(model, "CPXPARAM_ScreenOutput", 1)
    set_optimizer_attribute(model, "CPXPARAM_Emphasis_MIP", 5)
    set_optimizer_attribute(model, "CPXPARAM_MIP_Strategy_Probe", 2)
    set_optimizer_attribute(model, "CPXPARAM_MIP_Tolerances_MIPGap", 0.10)

    # The 2 different lineups are along axis 1
    # Players selected
    @variable(model, x[1:2, [player.ID for player in slate.players]], binary = true)
    # Games selected
    @variable(model, y[1:2, slate.teams], binary = true)
    # Teams selected
    @variable(model, z[1:2, slate.games], binary = true)

    for j in 1:2
        # Total salary must be <= $35,000
        @constraint(model, sum(player.Salary * x[j, player.ID] for player in slate.players) <= 35000)
        # Must select 9 total players
        @constraint(model, sum(x[j, :]) == 9)
        # Constraints for each position, must always select 1 pitcher, but can select 1 additional
        # player for each other position to fill the utility slot
        @constraint(model, sum(x[j, player.ID] for player in slate.players if player.Position == "P") == 1)
        @constraint(model, 1 <= sum(x[j, player.ID] for player in slate.players if player.Position == "C/1B") <= 2)
        @constraint(model, 1 <= sum(x[j, player.ID] for player in slate.players if player.Position == "2B") <= 2)
        @constraint(model, 1 <= sum(x[j, player.ID] for player in slate.players if player.Position == "3B") <= 2)
        @constraint(model, 1 <= sum(x[j, player.ID] for player in slate.players if player.Position == "SS") <= 2)
        @constraint(model, 3 <= sum(x[j, player.ID] for player in slate.players if player.Position == "OF") <= 4)

        for team in slate.teams
            # Excluding the pitcher, we can select a maximum of 4 players per team
            @constraint(model, sum(x[j, player.ID] for player in slate.players if player.Team == team && player.Position != "P") <= 4)
            # If no players are selected from a team, y is set to 0
            @constraint(model, y[j, team] <= sum(x[j, player.ID] for player in slate.players if player.Team == team))
        end
        # Must have players from at least 3 teams
        @constraint(model, sum(y[j, :]) >= 3)

        for game in slate.games
            # If no players are selected from a game z is set to 0
            @constraint(model, z[j, game] <= sum(x[j, player.ID] for player in slate.players if player.Game == game))
        end
        # Must select players from at least 2 games
        @constraint(model, sum(z[j, :]) >= 2)
    end

    for cut in cuts
        @constraint(model, sum(cut .* x) <= 17)
    end

    mu_x1 = @expression(model, x.data[1, :]' * slate.μ)
    mu_x2 = @expression(model, x.data[2, :]' * slate.μ)
    @constraint(model, mu_x1 >= mu_x2)

    var_x1 = @expression(model, x.data[1, :]' * slate.Σ * x.data[1, :])
    var_x2 = @expression(model, x.data[2, :]' * slate.Σ * x.data[2, :])
    cov = @expression(model, x.data[1, :]' * slate.Σ * x.data[2, :])
    theta = @expression(model, mu_x1 + (1 / sqrt(2 * pi)) * (1 + var_x1 + var_x2 - 2 * cov))

    @objective(model, Max, theta)
    optimize!(model)
    if termination_status(model) == MOI.OPTIMAL
        println("OPTIMAL")
        return (round.(Int, value.(x)), emax(value(mu_x1), value(mu_x2), value(var_x1), value(var_x2), value(cov)), value(theta))
    elseif termination_status(model) == MOI.INFEASIBLE
        println("INFEASIBLE")
        return ("INFEASIBLE", "INFEASIBLE", "INFEASIBLE")
    end
end


slate = get_mlb_slate("2022-07-16")

LB = -Inf
UB = Inf
cuts = JuMP.Containers.DenseAxisArray[]
incumbent = 0

while true
    proposal, new_LB, new_UB = do_optim(slate, cuts)

    if proposal == "INFEASIBLE"
        break
    end

    UB = new_UB
    println("UB: $(UB)")
    if new_LB > LB
        println("New LB: $(new_LB)")
        LB = new_LB
        incumbent = proposal
    end

    if LB < UB
        append!(cuts, Ref(proposal))
    else
        break
    end
end






function find_thetasq_upper_bound(slate)
    # Run for maximum three minutes
    model = Model(CPLEX.Optimizer)
    set_optimizer_attribute(model, "CPXPARAM_MIP_Display", 4)
    set_optimizer_attribute(model, "CPXPARAM_ScreenOutput", 1)
    set_optimizer_attribute(model, "CPXPARAM_TimeLimit", 180)

    # The 2 different lineups are along axis 1
    # Players selected
    @variable(model, x[1:2, [player.ID for player in slate.players]], binary = true)
    # Games selected
    @variable(model, y[1:2, slate.teams], binary = true)
    # Teams selected
    @variable(model, z[1:2, slate.games], binary = true)

    for j in 1:2
        # Total salary must be <= $35,000
        @constraint(model, sum(player.Salary * x[j, player.ID] for player in slate.players) <= 35000)
        # Must select 9 total players
        @constraint(model, sum(x[j, :]) == 9)
        # Constraints for each position, must always select 1 pitcher, but can select 1 additional
        # player for each other position to fill the utility slot
        @constraint(model, sum(x[j, player.ID] for player in slate.players if player.Position == "P") == 1)
        @constraint(model, 1 <= sum(x[j, player.ID] for player in slate.players if player.Position == "C/1B") <= 2)
        @constraint(model, 1 <= sum(x[j, player.ID] for player in slate.players if player.Position == "2B") <= 2)
        @constraint(model, 1 <= sum(x[j, player.ID] for player in slate.players if player.Position == "3B") <= 2)
        @constraint(model, 1 <= sum(x[j, player.ID] for player in slate.players if player.Position == "SS") <= 2)
        @constraint(model, 3 <= sum(x[j, player.ID] for player in slate.players if player.Position == "OF") <= 4)

        for team in slate.teams
            # Excluding the pitcher, we can select a maximum of 4 players per team
            @constraint(model, sum(x[j, player.ID] for player in slate.players if player.Team == team && player.Position != "P") <= 4)
            # If no players are selected from a team, y is set to 0
            @constraint(model, y[j, team] <= sum(x[j, player.ID] for player in slate.players if player.Team == team))
        end
        # Must have players from at least 3 teams
        @constraint(model, sum(y[j, :]) >= 3)

        for game in slate.games
            # If no players are selected from a game z is set to 0
            @constraint(model, z[j, game] <= sum(x[j, player.ID] for player in slate.players if player.Game == game))
        end
        # Must select players from at least 2 games
        @constraint(model, sum(z[j, :]) >= 2)
    end

    mu_x1 = @expression(model, x.data[1, :]' * slate.μ)
    mu_x2 = @expression(model, x.data[2, :]' * slate.μ)
    @constraint(model, mu_x1 >= mu_x2)

    var_x1 = @expression(model, x.data[1, :]' * slate.Σ * x.data[1, :])
    var_x2 = @expression(model, x.data[2, :]' * slate.Σ * x.data[2, :])
    cov = @expression(model, x.data[1, :]' * slate.Σ * x.data[2, :])
    theta = @expression(model, mu_x1 + (1 / sqrt(2 * pi)) * (1 + var_x1 + var_x2 - 2 * cov))

    @objective(model, Max, theta)
    optimize!(model)
    return value(theta)
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
    # Run for maximum three minutes
    model = Model(CPLEX.Optimizer)
    set_optimizer_attribute(model, "CPXPARAM_MIP_Display", 0)

    # The 2 different lineups are along axis 1
    # Players selected
    @variable(model, x[1:2, [player.ID for player in slate.players]], binary = true)
    # Games selected
    @variable(model, y[1:2, slate.teams], binary = true)
    # Teams selected
    @variable(model, z[1:2, slate.games], binary = true)

    for j in 1:2
        # Total salary must be <= $35,000
        @constraint(model, sum(player.Salary * x[j, player.ID] for player in slate.players) <= 35000)
        # Must select 9 total players
        @constraint(model, sum(x[j, :]) == 9)
        # Constraints for each position, must always select 1 pitcher, but can select 1 additional
        # player for each other position to fill the utility slot
        @constraint(model, sum(x[j, player.ID] for player in slate.players if player.Position == "P") == 1)
        @constraint(model, 1 <= sum(x[j, player.ID] for player in slate.players if player.Position == "C/1B") <= 2)
        @constraint(model, 1 <= sum(x[j, player.ID] for player in slate.players if player.Position == "2B") <= 2)
        @constraint(model, 1 <= sum(x[j, player.ID] for player in slate.players if player.Position == "3B") <= 2)
        @constraint(model, 1 <= sum(x[j, player.ID] for player in slate.players if player.Position == "SS") <= 2)
        @constraint(model, 3 <= sum(x[j, player.ID] for player in slate.players if player.Position == "OF") <= 4)

        for team in slate.teams
            # Excluding the pitcher, we can select a maximum of 4 players per team
            @constraint(model, sum(x[j, player.ID] for player in slate.players if player.Team == team && player.Position != "P") <= 4)
            # If no players are selected from a team, y is set to 0
            @constraint(model, y[j, team] <= sum(x[j, player.ID] for player in slate.players if player.Team == team))
        end
        # Must have players from at least 3 teams
        @constraint(model, sum(y[j, :]) >= 3)

        for game in slate.games
            # If no players are selected from a game z is set to 0
            @constraint(model, z[j, game] <= sum(x[j, player.ID] for player in slate.players if player.Game == game))
        end
        # Must select players from at least 2 games
        @constraint(model, sum(z[j, :]) >= 2)
    end

    mu_x1 = @expression(model, x.data[1, :]' * slate.μ)
    mu_x2 = @expression(model, x.data[2, :]' * slate.μ)
    @constraint(model, mu_x1 >= mu_x2)

    obj = @expression(model, mu_x1 - mu_x2)

    @objective(model, Max, obj)
    optimize!(model)
    return value(obj)
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


thetasq_upper_bound = find_thetasq_upper_bound(slate)
thetasq_intervals = make_thetasq_intervals(thetasq_upper_bound, 100)
theta_upper_intervals = make_theta_upper_intervals(thetasq_intervals)
theta_lower_intervals = make_theta_lower_intervals(thetasq_intervals)
delta_upper_bound = find_delta_upper_bound(slate)
delta_intervals = make_delta_intervals(delta_upper_bound, 100)
cdf_constants = make_cdf_constants(theta_lower_intervals, delta_intervals)

model = Model(CPLEX.Optimizer)
set_optimizer_attribute(model, "CPXPARAM_MIP_Display", 4)
set_optimizer_attribute(model, "CPXPARAM_ScreenOutput", 1)
set_optimizer_attribute(model, "CPXPARAM_Emphasis_MIP", 3)
# set_optimizer_attribute(model, "CPXPARAM_TimeLimit", 180)

# The 2 different lineups are along axis 1
# Players selected
@variable(model, x[1:2, [player.ID for player in slate.players]], binary = true)
# Games selected
@variable(model, y[1:2, slate.teams], binary = true)
# Teams selected
@variable(model, z[1:2, slate.games], binary = true)

for j in 1:2
    # Total salary must be <= $35,000
    @constraint(model, sum(player.Salary * x[j, player.ID] for player in slate.players) <= 35000)
    # Must select 9 total players
    @constraint(model, sum(x[j, :]) == 9)
    # Constraints for each position, must always select 1 pitcher, but can select 1 additional
    # player for each other position to fill the utility slot
    @constraint(model, sum(x[j, player.ID] for player in slate.players if player.Position == "P") == 1)
    @constraint(model, 1 <= sum(x[j, player.ID] for player in slate.players if player.Position == "C/1B") <= 2)
    @constraint(model, 1 <= sum(x[j, player.ID] for player in slate.players if player.Position == "2B") <= 2)
    @constraint(model, 1 <= sum(x[j, player.ID] for player in slate.players if player.Position == "3B") <= 2)
    @constraint(model, 1 <= sum(x[j, player.ID] for player in slate.players if player.Position == "SS") <= 2)
    @constraint(model, 3 <= sum(x[j, player.ID] for player in slate.players if player.Position == "OF") <= 4)

    for team in slate.teams
        # Excluding the pitcher, we can select a maximum of 4 players per team
        @constraint(model, sum(x[j, player.ID] for player in slate.players if player.Team == team && player.Position != "P") <= 4)
        # If no players are selected from a team, y is set to 0
        @constraint(model, y[j, team] <= sum(x[j, player.ID] for player in slate.players if player.Team == team))
    end
    # Must have players from at least 3 teams
    @constraint(model, sum(y[j, :]) >= 3)

    for game in slate.games
        # If no players are selected from a game z is set to 0
        @constraint(model, z[j, game] <= sum(x[j, player.ID] for player in slate.players if player.Game == game))
    end
    # Must select players from at least 2 games
    @constraint(model, sum(z[j, :]) >= 2)
end

@variable(model, w[1:100], binary = true)
@constraint(model, sum(w) == 1)
@variable(model, r[1:100], binary = true)
@constraint(model, sum(r) == 1)
@variable(model, u_prime)

mu_x1 = @expression(model, x.data[1, :]' * slate.μ)
mu_x2 = @expression(model, x.data[2, :]' * slate.μ)
@constraint(model, mu_x1 >= mu_x2)
delta = @expression(model, mu_x1 - mu_x2)
for k in 1:100
    @constraint(model, delta_intervals[k] * r[k] <= delta)
    @constraint(model, delta <= delta_intervals[k+1] + delta_intervals[end] * (1 - r[k]))
end

var_x1 = @expression(model, x.data[1, :]' * slate.Σ * x.data[1, :])
var_x2 = @expression(model, x.data[2, :]' * slate.Σ * x.data[2, :])
cov = @expression(model, x.data[1, :]' * slate.Σ * x.data[2, :])
thetasq = @expression(model, var_x1 + var_x2 - 2 * cov)

for q in 1:100
    @constraint(model, thetasq_intervals[q] * w[q] <= thetasq)
    @constraint(model, thetasq <= thetasq_intervals[q+1] + thetasq_intervals[end] * (1 - w[q]))
end

s_prime = @expression(model, sum(theta_upper_intervals[q] * w[q] for q in 1:100))

for q in 1:100
    for k in 1:100
        @constraint(model, u_prime <= mu_x1 * cdf_constants[q, k] + mu_x2 * (1 - cdf_constants[q, k]) + 250 * (2 - w[q] - r[k]))
    end
end

@objective(model, Max, u_prime + 1 / (sqrt(2 * pi)) * s_prime)
optimize!(model)