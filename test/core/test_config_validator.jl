using Test

@testset "config validator errors and normalizations" begin
    # Base valid config
    base = deepcopy(SMOKE_CFG)

    # Missing top-level key
    @test begin
        cfg = deepcopy(base)
        delete!(cfg, :solver)
        try
            validate_config(cfg)
            false
        catch err
            occursin("Missing top-level key: solver", sprint(showerror, err))
        end
    end

    # Unknown solver.method
    @test begin
        cfg = deepcopy(base)
        cfg[:solver][:method] = "Foo"
        try
            validate_config(cfg)
            false
        catch err
            occursin("solver.method must be one of", sprint(showerror, err))
        end
    end

    # interp_kind invalid
    @test begin
        cfg = deepcopy(base)
        cfg[:solver][:interp_kind] = "spline"
        try
            validate_config(cfg)
            false
        catch err
            occursin("solver.interp_kind must be one of", sprint(showerror, err))
        end
    end

    # warm_start invalid
    @test begin
        cfg = deepcopy(base)
        cfg[:solver][:warm_start] = :weird
        try
            validate_config(cfg)
            false
        catch err
            occursin("solver.warm_start invalid", sprint(showerror, err))
        end
    end

    # reject ASCII shorthand parameter keys
    @test begin
        cfg = deepcopy(base)
        cfg[:params][:s] = cfg[:params][:σ]
        delete!(cfg[:params], :σ)
        try
            validate_config(cfg)
            false
        catch err
            occursin("params.σ missing", sprint(showerror, err))
        end
    end

    @test begin
        cfg = deepcopy(base)
        cfg[:params][:beta] = cfg[:params][:β]
        delete!(cfg[:params], :β)
        try
            validate_config(cfg)
            false
        catch err
            occursin("params.β missing", sprint(showerror, err))
        end
    end

    # β out of range
    @test begin
        cfg = deepcopy(base)
        cfg[:params][:β] = 1.5
        try
            validate_config(cfg)
            false
        catch err
            occursin("0 < β < 1", sprint(showerror, err))
        end
    end

    # grids: Na too small
    @test begin
        cfg = deepcopy(base)
        cfg[:grids][:Na] = 1
        try
            validate_config(cfg)
            false
        catch err
            occursin("grids.Na must be > 1", sprint(showerror, err))
        end
    end

    # grids: a_max <= a_min
    @test begin
        cfg = deepcopy(base)
        cfg[:grids][:a_max] = cfg[:grids][:a_min]
        try
            validate_config(cfg)
            false
        catch err
            occursin("grids.a_max must be > a_min", sprint(showerror, err))
        end
    end

    # shocks: invalid method when active
    @test begin
        cfg = deepcopy(base)
        cfg[:shocks] = Dict{Symbol,Any}(
            :active => true,
            :method => "unknown",
            :ρ_shock => 0.5,
            :s_shock => 0.1,
            :Nz => 5,
        )
        try
            validate_config(cfg)
            false
        catch err
            occursin("shocks.method must be tauchen or rouwenhorst", sprint(showerror, err))
        end
    end

    # shocks: ρ out of range
    @test begin
        cfg = deepcopy(base)
        cfg[:shocks] = Dict{Symbol,Any}(
            :active => true,
            :method => "tauchen",
            :ρ_shock => 1.2,
            :s_shock => 0.1,
            :Nz => 5,
        )
        try
            validate_config(cfg)
            false
        catch err
            occursin("must be Real in (-1,1)", sprint(showerror, err))
        end
    end

    # init.c shape mismatch (deterministic)
    @test begin
        cfg = deepcopy(base)
        Na = cfg[:grids][:Na]
        cfg[:init] = Dict{Symbol,Any}(:c => ones(Na - 1))
        try
            validate_config(cfg)
            false
        catch err
            occursin("init.c must be a vector of length", sprint(showerror, err))
        end
    end

    # random.seed invalid type
    @test begin
        cfg = deepcopy(base)
        cfg[:random] = Dict{Symbol,Any}(:seed => "abc")
        try
            validate_config(cfg)
            false
        catch err
            occursin("random.seed must be an integer", sprint(showerror, err))
        end
    end
end
