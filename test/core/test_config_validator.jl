using Test

@testset "config validator errors and normalizations" begin
    # Base valid config
    base = deepcopy(SMOKE_CFG)

    # Missing top-level key
    @test begin
        cfg = cfg_without(base, :solver)
        try
            validate_config(cfg)
            false
        catch err
            msg = lowercase(sprint(showerror, err))
            occursin("missing", msg) && occursin("solver", msg)
        end
    end

    # Unknown solver.method
    @test begin
        cfg = cfg_patch(base, (:solver, :method) => "Foo")
        try
            validate_config(cfg)
            false
        catch err
            msg = lowercase(sprint(showerror, err))
            occursin("solver.method", msg) && occursin("invalid", msg)
        end
    end

    # interp_kind invalid
    @test begin
        cfg = cfg_patch(base, (:solver, :interp_kind) => "spline")
        try
            validate_config(cfg)
            false
        catch err
            msg = lowercase(sprint(showerror, err))
            occursin("interp", msg) && occursin("invalid", msg)
        end
    end

    # warm_start invalid
    @test begin
        cfg = cfg_patch(base, (:solver, :warm_start) => :weird)
        try
            validate_config(cfg)
            false
        catch err
            msg = lowercase(sprint(showerror, err))
            occursin("warm_start", msg) && occursin("invalid", msg)
        end
    end

    # reject ASCII shorthand parameter keys
    @test begin
        cfg = cfg_patch(base, (:params, :s) => cfg_get(base, :params, Symbol("σ")))
        cfg = cfg_without(cfg, (:params, Symbol("σ")))
        try
            validate_config(cfg)
            false
        catch err
            msg = lowercase(sprint(showerror, err))
            occursin("params", msg) && occursin("σ", msg)
        end
    end

    @test begin
        cfg = cfg_patch(base, (:params, :beta) => cfg_get(base, :params, Symbol("β")))
        cfg = cfg_without(cfg, (:params, Symbol("β")))
        try
            validate_config(cfg)
            false
        catch err
            msg = lowercase(sprint(showerror, err))
            occursin("params", msg) && occursin("β", msg)
        end
    end

    # β out of range
    @test begin
        cfg = cfg_patch(base, (:params, Symbol("β")) => 1.5)
        try
            validate_config(cfg)
            false
        catch err
            msg = lowercase(sprint(showerror, err))
            occursin("β", msg) && occursin("range", msg)
        end
    end

    # grids: Na too small
    @test begin
        cfg = cfg_patch(base, (:grids, :Na) => 1)
        try
            validate_config(cfg)
            false
        catch err
            msg = lowercase(sprint(showerror, err))
            occursin("grids", msg) && occursin("na", msg)
        end
    end

    # grids: a_max <= a_min
    @test begin
        cfg = cfg_patch(base, (:grids, :a_max) => cfg_get(base, :grids, :a_min))
        try
            validate_config(cfg)
            false
        catch err
            msg = lowercase(sprint(showerror, err))
            occursin("a_max", msg) && occursin("a_min", msg)
        end
    end

    # shocks: invalid method when active
    @test begin
        cfg = cfg_patch(
            base,
            :shocks => Dict{Symbol,Any}(
                :active => true,
                :method => "unknown",
                :ρ_shock => 0.5,
                :s_shock => 0.1,
                :Nz => 5,
            ),
        )
        try
            validate_config(cfg)
            false
        catch err
            msg = lowercase(sprint(showerror, err))
            occursin("shocks", msg) && occursin("method", msg)
        end
    end

    # shocks: ρ out of range
    @test begin
        cfg = cfg_patch(
            base,
            :shocks => Dict{Symbol,Any}(
                :active => true,
                :method => "tauchen",
                :ρ_shock => 1.2,
                :s_shock => 0.1,
                :Nz => 5,
            ),
        )
        try
            validate_config(cfg)
            false
        catch err
            msg = lowercase(sprint(showerror, err))
            occursin("ρ", msg) || occursin("rho", msg)
        end
    end

    # init.c shape mismatch (deterministic)
    @test begin
        Na = cfg_get(base, :grids, :Na)
        cfg = cfg_patch(base, :init => Dict{Symbol,Any}(:c => ones(Na - 1)))
        try
            validate_config(cfg)
            false
        catch err
            msg = lowercase(sprint(showerror, err))
            occursin("init", msg) && occursin("length", msg)
        end
    end

    # random.seed invalid type
    @test begin
        cfg = cfg_patch(base, :random => Dict{Symbol,Any}(:seed => "abc"))
        try
            validate_config(cfg)
            false
        catch err
            msg = lowercase(sprint(showerror, err))
            occursin("random", msg) && occursin("seed", msg)
        end
    end
end
