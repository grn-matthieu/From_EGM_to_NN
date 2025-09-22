<#
Parallel test runner for Windows PowerShell.
Usage:
  - Run all tests in parallel (default):
      .\scripts\parallel_run_tests.ps1 -Throttle 4
  - Run specific tests:
      .\scripts\parallel_run_tests.ps1 -Files test/core/test_residuals.jl,test/stochastic/test_sim.jl

Notes:
- The script runs each test file in its own Julia process to avoid shared-state conflicts.
- Make sure you precompiled the project first to reduce per-process overhead:
    julia --project=. scripts/precompile.jl
#>
param(
    [int]$Throttle = 4,
    [string[]]$Files
)

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root
$proj = Resolve-Path ".." | Select-Object -ExpandProperty Path

if (-not $Files) {
    # find test files
    $Files = Get-ChildItem -Path "$proj\test" -Filter "*.jl" -Recurse | Where-Object { $_.Name -ne 'runtests.jl' } | ForEach-Object { $_.FullName }
}

Write-Host "Running tests in parallel (throttle=$Throttle) - $($Files.Count) files"

$jobs = @()
foreach ($f in $Files) {
    $rel = Resolve-Path $f | ForEach-Object { $_.Path }
    $script = "julia --project=$proj `"$proj\scripts\run_single_test.jl`" $rel"
    $job = Start-Job -ScriptBlock { param($cmd) Invoke-Expression $cmd } -ArgumentList $script
    $jobs += $job
    # throttle: wait while the number of running jobs is >= $Throttle
    while (($jobs | Where-Object { $_.State -eq 'Running' }).Count -ge $Throttle) {
        Start-Sleep -Seconds 1
    }
}

# Wait for all
Write-Host "Waiting for jobs to finish..."
$jobs | Wait-Job

# Collect results
$failed = 0
foreach ($j in $jobs) {
    $out = Receive-Job $j -Keep
    $state = $j.State
    if ($state -ne 'Completed') {
        Write-Host "Job $($j.Id) failed with state $state" -ForegroundColor Red
        $failed++
    } else {
        Write-Host "Job $($j.Id) completed" -ForegroundColor Green
    }
}

if ($failed -gt 0) {
    Write-Host "$failed jobs failed" -ForegroundColor Red
    exit 1
} else {
    Write-Host "All jobs completed successfully" -ForegroundColor Green
}
