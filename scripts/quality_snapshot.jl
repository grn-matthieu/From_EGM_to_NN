#!/usr/bin/env julia
using Pkg
using Dates
using Printf
using InteractiveUtils

function ensure_outputs_dir()
    outdir = joinpath(@__DIR__, "..", "outputs/diagnostics")
    isdir(outdir) || mkpath(outdir)
    return normpath(outdir)
end

function capture_pkg_status()
    # Run in a subprocess to capture output reliably across platforms
    cmd = `$(Base.julia_cmd()) --project=. -e "using Pkg; Pkg.status()"`
    return try
        read(cmd, String)
    catch err
        sprint(showerror, err)
    end
end

function run_tests_and_get_coverage()
    # Run tests with coverage enabled
    tests_passed = true
    test_output = ""
    # Capture to a temporary file to avoid stream quirks
    tmp = tempname()
    try
        open(tmp, "w") do io
            cmd = `$(Base.julia_cmd()) --project=. --code-coverage=user test/runtests.jl`
            try
                run(pipeline(cmd; stdout = io, stderr = io))
            catch err
                tests_passed = false
                println(io, "\n[ERROR running tests] ", sprint(showerror, err))
            end
        end
        test_output = read(tmp, String)
    finally
        isfile(tmp) && rm(tmp; force = true)
    end

    # Compute coverage percentage using Coverage.jl in a subprocess
    pct = 0.0
    try
        code = join(
            [
                "using Pkg;",
                "Pkg.activate(temp=true);",
                "Pkg.add(\"Coverage\");",
                "using Coverage: process_folder, get_summary;",
                @sprintf(
                    "cov = process_folder(\"%s\");",
                    replace(joinpath(@__DIR__, "..", "src"), "\\" => "/")
                ),
                "covered, total = get_summary(cov);",
                "pct = total == 0 ? 0.0 : covered / total * 100;",
                "println(pct)",
            ],
            " ",
        )
        out = read(`$(Base.julia_cmd()) -e $(code)`, String)
        pct = try
            parse(Float64, strip(out))
        catch
            0.0
        end
    catch
        pct = 0.0
    end

    return (tests_passed, pct, test_output)
end

function run_smoke_and_capture()
    # Run the CI smoke script in a separate Julia process and capture output
    smoke_path = joinpath(@__DIR__, "ci", "smoke.jl")
    if !isfile(smoke_path)
        return (false, "Smoke script not found at $(smoke_path)")
    end
    cmd = `$(Base.julia_cmd()) --project=. $(smoke_path)`
    output = ""
    ok = true
    try
        output = read(pipeline(cmd; stderr = stdout), String)
    catch err
        ok = false
        # Attempt to capture any partial output by rerunning with pipeline capturing
        try
            p = open(cmd, "r")
            output = read(p, String)
            close(p)
        catch
            output = sprint(showerror, err)
        end
    end
    return (ok, output)
end

function find_slowest_from_test_output(test_output::AbstractString)
    # Parse lines like:
    # "Test Summary:        | Pass  Total   Time"
    # "Deterministic config |   29     29  53.4s"
    slow_name = "<unknown>"
    slow_time = -Inf
    per = Dict{String,Float64}()
    lines = split(test_output, '\n')
    for i = 1:length(lines)-1
        if occursin("Test Summary:", lines[i]) && occursin("Time", lines[i])
            name_line = strip(lines[i+1])
            m = match(r"^(.*)\s\|.*\s([0-9]+\.?[0-9]*)s$", name_line)
            if m !== nothing
                name = strip(m.captures[1])
                t = parse(Float64, m.captures[2])
                per[name] = t
                if t > slow_time
                    slow_time = t
                    slow_name = name
                end
            end
        end
    end
    return (slow_name, slow_time, per)
end

function main()
    cd(joinpath(@__DIR__, ".."))  # repo root
    outdir = ensure_outputs_dir()
    log_path = joinpath(outdir, "smoke_full_log.txt")
    ts = Dates.format(now(), dateformat"yyyy-mm-dd HH:MM:SS")

    # 1) Pkg.status
    status_txt = capture_pkg_status()

    # 2) Tests + coverage
    tests_passed, coverage_pct, test_output = run_tests_and_get_coverage()

    # 3) Smoke
    smoke_ok, smoke_log = run_smoke_and_capture()

    # 4) Slowest test identification (from test output)
    slow_name, slow_time, times = find_slowest_from_test_output(test_output)

    open(log_path, "w") do io
        println(io, "Quality Snapshot @ ", ts)
        println(io, repeat("=", 80))

        println(io, "[Pkg.status()]")
        println(io, status_txt)

        println(io, repeat("-", 80))
        println(io, "[Tests]")
        println(io, "Passed: ", tests_passed)
        @printf(io, "Coverage: %.2f%%\n", coverage_pct)
        println(io, "Test output (truncated to 10k chars):")
        if ncodeunits(test_output) > 10_000
            println(io, first(test_output, 10_000))
            println(io, "\n...\n[truncated]")
        else
            print(io, test_output)
        end

        println(io, repeat("-", 80))
        println(io, "[CI Smoke]")
        println(io, "Command: julia --project=. scripts/ci/smoke.jl")
        println(io, "Succeeded: ", smoke_ok)
        println(io, "Output:")
        print(io, smoke_log)

        println(io, repeat("-", 80))
        println(io, "[Slowest Test]")
        if isfinite(slow_time) && slow_time >= 0
            @printf(io, "Slowest testset: %s (%.3fs)\n", slow_name, slow_time)
        else
            println(io, "Slowest testset: <unknown>")
        end
        println(io, "Per-testset timings (s):")
        for (k, v) in sort(collect(times); by = x -> x[2], rev = true)
            if isfinite(v) && v >= 0
                @printf(io, "  %-30s %8.3f\n", k, v)
            else
                @printf(io, "  %-30s      NaN\n", k)
            end
        end
    end

    println("Wrote smoke log to: ", log_path)
end

main()
