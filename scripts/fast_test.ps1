<#
.\scripts\fast_test.ps1
Runs a quick test cycle: precompile (once) then run tests with FAST_TEST=1
Usage: Open PowerShell in repo root and run:
    .\scripts\fast_test.ps1
Optionally pass -NoPrecompile to skip precompilation step if already done.
#>

param(
    [switch]$NoPrecompile
)

Push-Location $PSScriptRoot/.. | Out-Null
try {
    if (-not $NoPrecompile) {
        Write-Host "Precompiling project (this speeds up subsequent runs)..."
        julia --project=. scripts/precompile.jl
    } else {
        Write-Host "Skipping precompile step as requested."
    }

    Write-Host "Running fast tests (FAST_TEST=1)..."
    $env:FAST_TEST = '1'
    julia --project=. -e "using Pkg; Pkg.test()"
} finally {
    Pop-Location | Out-Null
}
