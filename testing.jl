include("./src/solve.jl")


function emax(mu_x1, mu_x2, var_x1, var_x2, cov)
    Φ = x -> cdf(Normal(), x)
    ϕ = x -> pdf(Normal(), x)
    θ = sqrt(var_x1 + var_x2 - 2 * cov)
    return mu_x1 * Φ((mu_x1 - mu_x2) / θ) + mu_x2 * Φ((mu_x2 - mu_x1) / θ) + θ * ϕ((mu_x1 - mu_x2) / θ)
end


function get_mlb_slate()
    players = CSV.read("./data/slates/slate_2022-07-13.csv", Tables.rowtable)
    μ = [player.Projection for player in players]
    Σ = makeposdef(Symmetric(CSV.read("./data/slates/cov_2022-07-13.csv", header=false, Tables.matrix)))
    games = unique([player.Game for player in players])
    teams = unique([player.Team for player in players])
    return MLBSlate(players, games, teams, μ, Σ)
end


function do_optim(slate::MLBSlate, cuts::Vector{JuMP.Containers.DenseAxisArray})
    model = Model(() -> Xpress.Optimizer(logfile="optimlog.log", HEUREMPHASIS=1, MIPRELSTOP=0.25))

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
        @constraint(model, sum(cut .* x) <= 6)
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


slate = get_mlb_slate()

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






function find_theta_upper_bound(slate)
    # Run for maximum three minutes
    model = Model(() -> Xpress.Optimizer(logfile="optimlog.log", HEUREMPHASIS=1, MAXTIME=180))

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


function make_theta_intervals(upper_bound, n)
    intervals = Vector{Float64}(undef, n + 1)
    intervals[1] = 0
    intervals[2] = 1
    for i in 3:n+1
        intervals[i] = intervals[i-1] + (upper_bound - 1) / (n - 1)
    end
    return intervals
end


function theta_interval_bounds(intervals, q)
    if q == 1
        theta_upper = 1
        theta_lower = 0
        return (theta_upper, theta_lower)
    else
        theta_upper = sqrt(intervals[q+1])
        theta_lower = sqrt(intervals[q])
        return (theta_upper, theta_lower)
    end
end


function find_delta_upper_bound(slate)
    # Run for maximum three minutes
    model = Model(() -> Xpress.Optimizer(logfile="optimlog.log", HEUREMPHASIS=1, MAXTIME=180))

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


function delta_interval_bounds(intervals, q)
    delta_upper = intervals[q+1]
    delta_lower = intervals[q]
    return (delta_upper, delta_lower)
end