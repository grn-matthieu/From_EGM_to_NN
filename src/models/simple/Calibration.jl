module SimpleCalibration
export SimpleParams, default_simple_params

Base.@kwdef struct SimpleParams
    β::Float64 = 0.96
    σ::Float64 = 2.0
    r::Float64 = 0.02
    y::Float64 = 1.0
    a_min::Float64 = 0.0
    a_max::Float64 = 20.0
end

default_simple_params() = SimpleParams()

end