# precompile.jl
# This file imports declared deps and weak/extras so PackageCompiler includes them in the sysimage.

# Try to load the package itself (if available in dev environment)
try
    using ThesisProject
catch
end

# Strong dependencies
try
    using JSON3
    using Logging
    using Random
    using SHA
    using SpecialFunctions
    using StableRNGs
    using Statistics
    using YAML
catch
end

# Weak dependencies / extras
for pkg in (:Plots, :Test)
    try
        eval(:(using $(pkg)))
    catch
        # ignore optional deps that are not installed
    end
end

# end
