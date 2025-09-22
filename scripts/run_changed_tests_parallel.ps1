<#
.\scripts\run_changed_tests_parallel.ps1

Detect changed files and run matching tests in parallel.

Usage:
  .\scripts\run_changed_tests_parallel.ps1 -Throttle 4
  .\scripts\run_changed_tests_parallel.ps1 -NoPrecompile -Throttle 6

Behavior:
  - Detects changed files (origin/main, main, or last commit), modified and untracked files.
  - Maps changed files to test files using the same heuristics as `run_changed_tests.ps1`.
  - Runs all discovered test files in parallel Julia processes, throttled by `-Throttle`.
#>

param(
    [int]$Throttle = 4,
    [switch]$NoPrecompile
)

Push-Location $PSScriptRoot/.. | Out-Null
try {
    Write-Host "Detecting changed files (git)..."

    $changed = @()

    # Try to diff against origin/main, fall back to main, then to last commit
    $baseRefs = @('origin/main','main')
    $diffList = @()
    foreach ($b in $baseRefs) {
        try {
            git rev-parse --verify $b > $null 2>&1
            if ($LASTEXITCODE -eq 0) {
                $diffList = git diff --name-only $b...HEAD 2>$null | Where-Object { $_ -ne '' }
                break
            }
        } catch { }
    }
    if (-not $diffList -or $diffList.Count -eq 0) {
        # fallback to last commit
        $diffList = git diff --name-only HEAD~1..HEAD 2>$null | Where-Object { $_ -ne '' }
    }

    $changed += $diffList
    $changed += git ls-files -m 2>$null | Where-Object { $_ -ne '' }
    $changed += git ls-files --others --exclude-standard 2>$null | Where-Object { $_ -ne '' }

    $changed = $changed | Select-Object -Unique

    if (-not $changed -or $changed.Count -eq 0) {
        Write-Host "No changed files detected. Nothing to run."
        exit 0
    }

    Write-Host "Changed files:`n$($changed -join "`n")"

    $testsToRun = New-Object System.Collections.Generic.List[string]
    $testFiles = Get-ChildItem -Path (Join-Path $PWD 'test') -Filter "*.jl" -Recurse | Where-Object { $_.Name -ne 'runtests.jl' } | ForEach-Object { $_.FullName }

    foreach ($f in $changed) {
        if ($f -like 'test/*.jl' -or $f -like 'test\\*.jl') {
            $testsToRun.Add((Resolve-Path $f).Path)
            continue
        }

        if ($f -notlike '*.jl') { continue }

        $basename = [System.IO.Path]::GetFileNameWithoutExtension($f)

        # Search test files for the basename (word boundary)
        $pattern = "\b$basename\b"
        try {
            $matches = Select-String -Path (Join-Path $PWD 'test' -ChildPath '**\*.jl') -Pattern $pattern -SimpleMatch -ErrorAction SilentlyContinue | ForEach-Object { $_.Path }
        } catch {
            $matches = @()
        }
        if ($matches) { $matches | ForEach-Object { $testsToRun.Add((Resolve-Path $_).Path) } ; continue }

        # fallback: search for any path segment occurrences
        $parts = $f -split '[\\/]'
        foreach ($p in $parts) {
            if ($p.Length -lt 3) { continue }
            try {
                $m2 = Select-String -Path (Join-Path $PWD 'test' -ChildPath '**\*.jl') -Pattern $p -SimpleMatch -ErrorAction SilentlyContinue | ForEach-Object { $_.Path }
            } catch { $m2 = @() }
            if ($m2) { $m2 | ForEach-Object { $testsToRun.Add((Resolve-Path $_).Path) }; break }
        }
    }

    $testsToRun = $testsToRun | Select-Object -Unique

    if (-not $testsToRun -or $testsToRun.Count -eq 0) {
        Write-Host "No matching tests found for changed files. Consider running fast_test.ps1 or full test suite."
        exit 0
    }

    Write-Host "Tests to run in parallel:`n$($testsToRun -join "`n")"

    if (-not $NoPrecompile) {
        Write-Host "Precompiling project to speed test runs..."
        julia --project=. scripts/precompile.jl
    }

    # Start jobs
    $jobs = @()
    $scriptPath = (Resolve-Path (Join-Path $PWD 'scripts\run_single_test.jl')).Path
    $proj = Resolve-Path ".." | Select-Object -ExpandProperty Path
    foreach ($t in $testsToRun) {
        $full = Resolve-Path $t | ForEach-Object { $_.Path }
        $job = Start-Job -ScriptBlock { param($proj,$scriptPath,$test) & julia --project=$proj $scriptPath $test } -ArgumentList $proj, $scriptPath, $full
        $jobs += $job
        while (($jobs | Where-Object { $_.State -eq 'Running' }).Count -ge $Throttle) { Start-Sleep -Seconds 1 }
    }

    Write-Host "Waiting for jobs to finish..."
    $jobs | Wait-Job

    # Collect results
    $failed = 0
    foreach ($j in $jobs) {
        $state = $j.State
        if ($state -ne 'Completed') {
            Write-Host "Job $($j.Id) failed with state $state" -ForegroundColor Red
            $failed++
        } else {
            Write-Host "Job $($j.Id) completed" -ForegroundColor Green
        }
    }

    if ($failed -gt 0) { Write-Host "$failed jobs failed" -ForegroundColor Red; exit 1 } else { Write-Host "All jobs completed successfully" -ForegroundColor Green }

} finally {
    Pop-Location | Out-Null
}
