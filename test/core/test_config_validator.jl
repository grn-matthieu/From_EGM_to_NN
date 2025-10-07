# test/utils/test_utils_config.jl
using Test
using ThesisProject.UtilsConfig
const UC = ThesisProject.UtilsConfig

# -------------------------
# helpers: yaml_to_namedtuple
# -------------------------
@testset "yaml_to_namedtuple" begin
    nt = UC.yaml_to_namedtuple(
        Dict("a" => 1, "b" => Dict("c" => 2), "d" => [1, 2, Dict("e" => 3)]),
    )
    @test nt.a == 1
    @test nt.b.c == 2
    @test nt.d[3].e == 3

    @test UC.yaml_to_namedtuple([1, 2, 3]) == [1, 2, 3]
    @test UC.yaml_to_namedtuple((a = 1,)) == (a = 1,)  # passthrough
end

# -------------------------
# maybe() API
# -------------------------
@testset "maybe() API" begin
    cfg = (a = (b = (c = 42,), k = true), x = 1.5, y = nothing)
    # single key, keyword default
    @test UC.maybe(cfg, :missing; default = 99) == 99
    @test UC.maybe(cfg, :x; default = 0) == 1.5
    # nested traversal (all Symbols)
    @test UC.maybe(cfg, :a, :b, :c; default = -1) == 42
    @test UC.maybe(cfg, :a, :missing, -1) == -1
    # positional default when 3rd arg is a value
    @test UC.maybe(cfg, :missing, 123) == 123
    # positional default even if the value is a Symbol (heuristic path)
    @test UC.maybe(cfg, :missing, :fallback) == :fallback
    # trailing Symbol as nested key when present
    @test UC.maybe(cfg, :a, :b) == (c = 42,)
    # cfg === nothing → return positional default
    @test UC.maybe(nothing, :whatever, false) == false
    # mixed: keys + positional default at end
    @test UC.maybe(cfg, :a, :missing, 7) == 7
    # nothing helpers
    @test UC.maybe(nothing; default = :d) === :d
end

# -------------------------
# validate_config: happy paths
# -------------------------
function base_cfg(;
    method = "egm",
    β = 0.95,
    σ = 2.0,
    r = 0.02,
    y = 1.0,
    Na = 5,
    a_min = 0.0,
    a_max = 2.0,
)
    return (
        model = (name = "consumer_saving",),
        params = (; (Symbol("β") => β), (Symbol("σ") => σ), r = r, y = y),
        grids = (Na = Na, a_min = a_min, a_max = a_max),
        solver = (
            method = method,
            tol = 1e-6,
            tol_pol = 1e-6,
            maxit = 1000,
            verbose = false,
            relax = 0.5,
            warm_start = :default,
            egm = (interp_kind = :linear,),
            time_iteration = (interp_kind = :linear,),
            projection = (orders = [2], Nval = max(Na, 3), basis_orders = [2]),
            perturbation = (
                order = 1,
                a_bar = nothing,
                h_a = nothing,
                h_z = nothing,
                tol_fit = 1e-8,
                maxit_fit = 25,
            ),
            nn = (
                epochs = 10,
                batch = 32,
                lr = 1e-3,
                hid1 = 8,
                hid2 = 8,
                samples_per_epoch = 64,
                objective = :euler_fb_aio,
                v_h = 0.5,
                w_min = 0.1,
                w_max = 4.0,
                sigma_shocks = nothing,
                target_loss = 1e-10,
                use_cuda = false,
            ),
        ),
    )
end

@testset "validate_config happy paths" begin
    @test UC.validate_config(base_cfg()) === true             # egm
    proj_cfg = base_cfg(method = "projection")
    proj_solver = merge(
        proj_cfg.solver,
        (projection = merge(proj_cfg.solver.projection, (orders = [0, 2], Nval = 3)),),
    )
    proj_cfg = merge(proj_cfg, (solver = proj_solver,))
    @test UC.validate_config(proj_cfg) === true

    pert_cfg = base_cfg(method = "perturbation")
    pert_solver = merge(
        pert_cfg.solver,
        (perturbation = merge(pert_cfg.solver.perturbation, (order = 1,)),),
    )
    pert_cfg = merge(pert_cfg, (solver = pert_solver,))
    @test UC.validate_config(pert_cfg) === true
    # optional utility
    cfgU = merge(base_cfg(), (utility = (u_type = "crra",),))
    @test UC.validate_config(cfgU) === true
    # Dict delegation path
    @test UC.validate_config(Dict(pairs(cfgU))) === true
end

# -------------------------
# validate_config: errors (cover branches)
# -------------------------
@testset "validate_config errors" begin
    update_solver(cfg, nt::NamedTuple) = merge(cfg, (solver = merge(cfg.solver, nt),))
    function update_solver_block(cfg, block::Symbol, nt::NamedTuple)
        solver = cfg.solver
        block_cfg = getproperty(solver, block)
        new_block = merge(block_cfg, nt)
        block_tuple = NamedTuple{(block,)}((new_block,))
        solver = merge(solver, block_tuple)
        return merge(cfg, (solver = solver,))
    end

    # missing sections / wrong types
    @test_throws ErrorException UC.validate_config((
        params = (;),
        grids = (;),
        solver = (;),
    ))
    bad = base_cfg()
    @test_throws ErrorException UC.validate_config(merge(bad, (model = (:not_a_nt,),)))
    # params checks
    @test_throws ErrorException UC.validate_config(base_cfg(β = 1.2))
    @test_throws ErrorException UC.validate_config(base_cfg(σ = 0.0))
    @test_throws ErrorException UC.validate_config(base_cfg(r = -1.1))
    @test_throws ErrorException UC.validate_config(base_cfg(y = 0.0))
    # grids checks
    @test_throws ErrorException UC.validate_config(base_cfg(Na = 1))
    @test_throws ErrorException UC.validate_config(base_cfg(a_max = 0.0))
    # solver method invalid
    @test_throws ErrorException UC.validate_config(
        update_solver(base_cfg(), (; method = "nope")),
    )
    # tol / maxit / verbose type checks
    @test_throws ErrorException UC.validate_config(
        update_solver(base_cfg(), (; tol = -1.0)),
    )
    @test_throws ErrorException UC.validate_config(update_solver(base_cfg(), (; maxit = 0)))
    @test_throws ErrorException UC.validate_config(
        update_solver(base_cfg(), (; verbose = 1)),
    )
    # EGM: interp_kind
    @test_throws ErrorException UC.validate_config(
        update_solver_block(base_cfg(), :egm, (; interp_kind = "cubic")),
    )
    # warm_start variants + steady_state requirements
    @test UC.validate_config(update_solver(base_cfg(), (; warm_start = "default")))
    @test_throws ErrorException UC.validate_config(
        update_solver(base_cfg(y = nothing), (; warm_start = "steady_state")),
    )
    # projection: orders and Nval
    @test_throws ErrorException UC.validate_config(
        update_solver_block(
            base_cfg(method = "projection"),
            :projection,
            (; orders = Int[]),
        ),
    )
    @test_throws ErrorException UC.validate_config(
        update_solver_block(
            base_cfg(method = "projection"),
            :projection,
            (; orders = [-1]),
        ),
    )
    @test_throws ErrorException UC.validate_config(
        update_solver_block(base_cfg(method = "projection"), :projection, (; Nval = 1)),
    )
    # perturbation: order / a_bar / 2nd order h_a/h_z / tol_fit / maxit_fit
    @test_throws ErrorException UC.validate_config(
        update_solver_block(
            base_cfg(method = "perturbation"),
            :perturbation,
            (; order = 0),
        ),
    )
    @test_throws ErrorException UC.validate_config(
        update_solver_block(
            base_cfg(method = "perturbation"),
            :perturbation,
            (; a_bar = 9.0),
        ),
    )
    # active shocks require h_z>0 when order≥2
    cfgP2 = merge(
        base_cfg(method = "perturbation"),
        (
            shocks = (
                active = true,
                method = "tauchen",
                Symbol("ρ_shock") => 0.5,
                Symbol("σ_shock") => 0.1,
                Nz = 3,
                m = 3.0,
            ),
        ),
    )
    cfgP2 = update_solver_block(cfgP2, :perturbation, (; order = 2, h_a = 0.1, h_z = 0.0))
    @test_throws ErrorException UC.validate_config(cfgP2)
    # tol_fit / maxit_fit invalid
    @test_throws ErrorException UC.validate_config(
        update_solver_block(
            base_cfg(method = "perturbation"),
            :perturbation,
            (; tol_fit = 0.0),
        ),
    )
    @test_throws ErrorException UC.validate_config(
        update_solver_block(
            base_cfg(method = "perturbation"),
            :perturbation,
            (; maxit_fit = 0),
        ),
    )
    # shocks block validations
    baseS = merge(
        base_cfg(),
        (
            shocks = (
                active = true,
                method = "tauchen",
                Symbol("ρ_shock") => 0.5,
                Symbol("σ_shock") => 0.1,
                Nz = 3,
                m = 3.0,
            ),
        ),
    )
    @test UC.validate_config(baseS) === true
    @test_throws ErrorException UC.validate_config(
        merge(base_cfg(), (shocks = (active = true, method = "nope"),)),
    )
    @test_throws ErrorException UC.validate_config(
        merge(
            base_cfg(),
            (shocks = (active = true, Symbol("σ_shock") => 0.1, Nz = 3, m = 3.0),),
        ),
    )
    @test_throws ErrorException UC.validate_config(
        merge(
            base_cfg(),
            (
                shocks = (
                    active = true,
                    Symbol("ρ_shock") => 1.2,
                    Symbol("σ_shock") => 0.1,
                    Nz = 3,
                    m = 3.0,
                ),
            ),
        ),
    )
    @test_throws ErrorException UC.validate_config(
        merge(
            base_cfg(),
            (
                shocks = (
                    active = true,
                    Symbol("ρ_shock") => 0.5,
                    Symbol("σ_shock") => -0.1,
                    Nz = 3,
                    m = 3.0,
                ),
            ),
        ),
    )
    @test_throws ErrorException UC.validate_config(
        merge(
            base_cfg(),
            (
                shocks = (
                    active = true,
                    Symbol("ρ_shock") => 0.5,
                    Symbol("σ_shock") => "x",
                    Nz = 3,
                    m = 3.0,
                ),
            ),
        ),
    )
    @test_throws ErrorException UC.validate_config(
        merge(
            base_cfg(),
            (
                shocks = (
                    active = true,
                    Symbol("ρ_shock") => 0.5,
                    Symbol("σ_shock") => 0.1,
                    Nz = 1,
                    m = 3.0,
                ),
            ),
        ),
    )
    @test_throws ErrorException UC.validate_config(
        merge(
            base_cfg(),
            (
                shocks = (
                    active = true,
                    method = "tauchen",
                    Symbol("ρ_shock") => 0.5,
                    Symbol("σ_shock") => 0.1,
                    Nz = 3,
                    m = -1.0,
                ),
            ),
        ),
    )
    @test_throws ErrorException UC.validate_config(
        merge(
            base_cfg(),
            (
                shocks = (
                    active = true,
                    Symbol("ρ_shock") => 0.5,
                    Symbol("σ_shock") => 0.1,
                    Nz = 3,
                    validate = "yes",
                ),
            ),
        ),
    )
    # init block (deterministic and stochastic shapes + positivity)
    bad_init_len = merge(base_cfg(), (init = (c = fill(-1.0, base_cfg().grids.Na),),))
    @test_throws ErrorException UC.validate_config(bad_init_len)
    st_cfg = merge(
        base_cfg(),
        (
            shocks = (
                active = true,
                method = "tauchen",
                Symbol("ρ_shock") => 0.5,
                Symbol("σ_shock") => 0.0,
                Nz = 2,
                m = 3.0,
            ),
        ),
    )
    @test_throws ErrorException UC.validate_config(merge(st_cfg, (init = (c = ones(3),),)))
    @test_throws ErrorException UC.validate_config(
        merge(st_cfg, (init = (; c = fill(-1.0, (st_cfg.grids.Na, st_cfg.shocks.Nz))),)),
    )
    # random seed type
    @test_throws ErrorException UC.validate_config(
        merge(base_cfg(), (random = (seed = "abc",),)),
    )
end

# -------------------------
# load_config (via temp file)
# -------------------------
@testset "load_config roundtrip" begin
    cfg = base_cfg()
    yaml_str = """
model:
  name: consumer_saving
params:
  β: 0.95
  σ: 2.0
  r: 0.02
  y: 1.0
grids:
  Na: 5
  a_min: 0.0
  a_max: 2.0
solver:
  method: egm
"""
    tmp = mktemp()[1]
    open(tmp, "w") do io
        write(io, yaml_str)
    end
    nt = UC.load_config(tmp)
    @test nt.model.name == "consumer_saving"
    @test nt.params.β ≈ 0.95
    @test nt.solver.method == "egm"
end
