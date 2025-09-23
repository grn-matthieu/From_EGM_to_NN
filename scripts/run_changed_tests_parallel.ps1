<#
.\scripts\run_changed_tests_parallel.ps1

Detect changed files and run matching tests in parallel.

Usage:
  .\scripts\run_changed_tests_parallel.ps1 -Throttle 4
  .\scripts\run_changed_tests_parallel.ps1 -NoPrecompile -Throttle 6

Behavior:
  - Detects changed files (origin/main, main, last commit), modified, staged, and untracked files.
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

    # Include staged (indexed) but not committed changes (A/C/M/R)
    $staged = git diff --name-only --cached --diff-filter=ACMR 2>$null | Where-Object { $_ -ne '' }
    $changed += $staged

    # Include modified in working tree and untracked
    $changed += git ls-files -m 2>$null | Where-Object { $_ -ne '' }
    $changed += git ls-files --others --exclude-standard 2>$null | Where-Object { $_ -ne '' }

    # Unique and keep only paths that exist (some staged deletions may not)
    $changed = $changed | Select-Object -Unique | Where-Object { Test-Path $_ }

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

        # Search test files for the basename as a word. Escape the basename to
        # avoid regex metacharacters interfering with the pattern.
        $escaped = [Regex]::Escape($basename)
        $pattern = "\b$escaped\b"
        try {
            $matches = Select-String -Path (Join-Path $PWD 'test' -ChildPath '**\*.jl') -Pattern $pattern -ErrorAction SilentlyContinue | ForEach-Object { $_.Path }
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
        & julia --project=. scripts/precompile.jl
        if ($LASTEXITCODE -ne 0) {
            Write-Host "Precompile failed with exit code $LASTEXITCODE" -ForegroundColor Red
            exit $LASTEXITCODE
        }
    }

    # Start jobs and capture per-test exit codes
    $jobs = @()
    $entryScript = (Resolve-Path (Join-Path $PWD 'scripts\julia_test_entry.jl')).Path
    # Use the current directory (repository root after Push-Location) as the
    # Julia project path. Using ".." could point to the parent folder and
    # cause Julia to run with the wrong project environment.
    $proj = (Resolve-Path ".").Path

    foreach ($t in $testsToRun) {
        $full = (Resolve-Path $t).Path
        $job = Start-Job -ScriptBlock {
            param($proj,$entry,$testPath)
            $output = & julia --project=$proj $entry $testPath 2>&1
            $code = $LASTEXITCODE
            [pscustomobject]@{ TestPath = $testPath; ExitCode = $code; Output = $output -join "`n" }
        } -ArgumentList $proj, $entryScript, $full
            $jobs += $job
            while (($jobs | Where-Object { $_.State -eq 'Running' }).Count -ge $Throttle) { Start-Sleep -Seconds 1 }
        }

    Write-Host "Waiting for jobs to finish..."
    $jobs | Wait-Job

    # Collect results and fail on any non-zero exit code
    $failed = 0
    foreach ($j in $jobs) {
        $res = Receive-Job -Job $j
        if ($null -eq $res) {
            Write-Host "Job $($j.Id) produced no result" -ForegroundColor Red
            $failed++
            continue
        }

        if ($res.ExitCode -ne 0) {
            Write-Host "[FAIL] $($res.TestPath) (exit $($res.ExitCode))" -ForegroundColor Red
            Write-Host $res.Output
            $failed++
        } else {
            Write-Host "[PASS] $($res.TestPath)" -ForegroundColor Green
            # Optionally show summarized output:
            # Write-Host $res.Output
        }
    }

    if ($failed -gt 0) {
        Write-Host "$failed test job(s) failed" -ForegroundColor Red
        exit 1
    } else {
        Write-Host "All jobs completed successfully" -ForegroundColor Green
    }
} finally {
    Pop-Location | Out-Null
}