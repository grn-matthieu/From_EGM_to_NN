using Test
using ThesisProject
import ThesisProject.Determinism: canonicalize_cfg, hash_hex

@testset "Config determinism" begin
    cfg_a = """
model:
  name: baseline
params:
  β: 0.96
  σ: 2.0
grids:
  Na: 5
  a_min: 0.0
  a_max: 10.0
solver:
  method: EGM
"""

    cfg_b = """
solver:
  method: EGM
grids:
  a_max: 10.0
  a_min: 0.0
  Na: 5
params:
  σ: 2.0
  β: 0.96
model:
  name: baseline
"""

    mktemp() do path_a, io_a
        write(io_a, cfg_a)
        close(io_a)
        mktemp() do path_b, io_b
            write(io_b, cfg_b)
            close(io_b)
            cfg1 = load_config(path_a)
            cfg2 = load_config(path_b)
            bytes1 = canonicalize_cfg(cfg1)
            bytes2 = canonicalize_cfg(cfg2)
            seed1 = hash_hex(bytes1)
            seed2 = hash_hex(bytes2)
            @test bytes1 == bytes2
            @test seed1 == seed2
        end
    end
end
