<#
.\scripts\run_changed_tests.ps1
Detect changed Julia files (unstaged, staged, or committed vs main) and run related tests.
Behavior:
  - Gathers changed files from git (tries `origin/main` then `main` then `HEAD~1..HEAD`) and includes modified/untracked files.
  - Maps changed files to test files by:
      * If a changed file is under `test/` and is a `.jl` file -> run it
      * Otherwise, search `test/` for occurrences of the changed file's basename and path components
  - Runs each discovered test file with `scripts/run_single_test.jl`

Usage:
  .\scripts\run_changed_tests.ps1 [-NoPrecompile]

#>

param(
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

    # include modified and untracked files
    $changed += git ls-files -m 2>$null | Where-Object { $_ -ne '' }
    $changed += git ls-files --others --exclude-standard 2>$null | Where-Object { $_ -ne '' }

    $changed = $changed | Select-Object -Unique

    if (-not $changed -or $changed.Count -eq 0) {
        Write-Host "No changed files detected. Nothing to run."
        exit 0
    }

    Write-Host "Changed files:`n$($changed -join "`n")"

    $testsToRun = New-Object System.Collections.Generic.List[string]

    foreach ($f in $changed) {
        if ($f -like 'test/*.jl' -or $f -like 'test\\*.jl') {
            # direct test file
            $testsToRun.Add($f)
            continue
        }

        if ($f -notlike '*.jl') { continue }

        $basename = [System.IO.Path]::GetFileNameWithoutExtension($f)

        # Search test files for the basename (word boundary) and for path fragments
        $pattern = "\b$basename\b"
        try {
            $matches = Select-String -Path (Join-Path $PWD 'test' -ChildPath '**\*.jl') -Pattern $pattern -SimpleMatch -ErrorAction SilentlyContinue | ForEach-Object { $_.Path }
        } catch {
            $matches = @()
        }
        if ($matches) { $matches | ForEach-Object { $testsToRun.Add($_) } ; continue }

        # fallback: search for any path segment occurrences
        $parts = $f -split '[\\/]'
        foreach ($p in $parts) {
            if ($p.Length -lt 3) { continue }
            try {
                $m2 = Select-String -Path (Join-Path $PWD 'test' -ChildPath '**\*.jl') -Pattern $p -SimpleMatch -ErrorAction SilentlyContinue | ForEach-Object { $_.Path }
            } catch { $m2 = @() }
            if ($m2) { $m2 | ForEach-Object { $testsToRun.Add($_) }; break }
        }
    }

    $testsToRun = $testsToRun | Select-Object -Unique

    if (-not $testsToRun -or $testsToRun.Count -eq 0) {
        Write-Host "No matching tests found for changed files. Consider running fast_test.ps1 or full test suite."
        exit 0
    }

    Write-Host "Tests to run:`n$($testsToRun -join "`n")"

    if (-not $NoPrecompile) {
        Write-Host "Precompiling project to speed test runs..."
        julia --project=. scripts/precompile.jl
    }

    $exitCode = 0
    foreach ($t in $testsToRun) {
        Write-Host "Running test: $t"
        julia --project=. scripts/run_single_test.jl $t
        if ($LASTEXITCODE -ne 0) { $exitCode = $LASTEXITCODE }
    }

    exit $exitCode

} finally {
    Pop-Location | Out-Null
}
