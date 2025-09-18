using Test
using ThesisProject: NNMixedPrecision, NNTrain
using .NNMixedPrecision:
    UseFP16,
    UseBF16,
    to_mp,
    to_fp32,
    cast_params,
    cast_params!,
    cast_batch,
    with_mixed_precision

@testset "mixed-precision helpers" begin
    # to_mp and to_fp32 conversions
    a32 = Float32[1.0, 2.5, -3.0]
    a16 = to_mp(a32, UseFP16())
    @test eltype(a16) == Float16
    @test all(Float32.(a16) .== a32)

    a_bf = to_mp(a32, UseBF16())
    @test eltype(a_bf) == NNMixedPrecision.eltype_from(UseBF16())
    @test all(Float32.(a_bf) .== a32)

    a_fp32 = to_fp32(a16)
    @test eltype(a_fp32) == Float32
    @test all(a_fp32 .== a32)

    # cast_params and cast_params!
    params = [Float32[1.0 2.0; 3.0 4.0], Float32[5.0, 6.0]]
    p16 = cast_params(params, UseFP16())
    @test eltype.(p16) == [Float16, Float16]
    p32 = cast_params(p16, Float32)
    @test eltype.(p32) == [Float32, Float32]

    # cast_params! roundtrip: prepare dst and src
    dst = cast_params(params, Float32)
    src = cast_params(params, UseFP16())
    cast_params!(dst, src, UseFP16())
    # after cast_params!, dst should hold converted values (Float16 converted into dst elements)
    @test eltype.(dst) == [Float32, Float32]

    # cast_batch: simple arrays and NamedTuple
    batch = (x = Float32[1.0, 2.0], y = 3.5f0)
    batch16 = cast_batch(batch, UseFP16())
    @test eltype(batch16.x) == Float16
    @test typeof(batch16.y) == Float16

    # with_mixed_precision should call the provided function with converted params
    function fn(params, batch)
        return (sum(Float32.(params[1])) + Float32(batch.x[1]))
    end
    res = with_mixed_precision(
        nothing,
        params,
        batch;
        mp = UseFP16(),
        loss_scale = 1.0,
        f = fn,
    )
    @test isfinite(res)
end
