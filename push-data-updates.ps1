# Fantasy Model — auto-push data files to GitHub
# Watches the Fantasy Model folder for CSV changes and commits + pushes automatically.
# Run once; keep the window open (or set up via Task Scheduler to run on login).

$RepoPath  = "C:\Users\mjgin\Fantasy Model"
$WatchExts = @("*.csv")   # file types to watch

Write-Host "📡 Watching for CSV changes in $RepoPath" -ForegroundColor Cyan
Write-Host "    Press Ctrl+C to stop." -ForegroundColor Gray
Write-Host ""

# Set up FileSystemWatcher
$watcher = New-Object System.IO.FileSystemWatcher
$watcher.Path                = $RepoPath
$watcher.Filter              = "*.csv"
$watcher.IncludeSubdirectories = $false
$watcher.NotifyFilter        = [System.IO.NotifyFilters]::LastWrite -bor [System.IO.NotifyFilters]::FileName

# Debounce: track last-push time to avoid double-firing
$script:lastPush = [datetime]::MinValue
$DebounceSeconds = 5

$action = {
    $path = $Event.SourceEventArgs.FullPath
    $name = $Event.SourceEventArgs.Name
    $type = $Event.SourceEventArgs.ChangeType

    # Skip temp files (Excel lock files start with ~$)
    if ($name -like "~`$*") { return }

    $now = [datetime]::Now
    if (($now - $script:lastPush).TotalSeconds -lt $DebounceSeconds) { return }
    $script:lastPush = $now

    Write-Host ""
    Write-Host "[$($now.ToString('HH:mm:ss'))] Detected $type : $name" -ForegroundColor Yellow
    Write-Host "  Staging..." -ForegroundColor Gray

    Set-Location $RepoPath

    # Stage only CSV files that are tracked or new (not gitignored)
    & git add "*.csv" 2>&1 | Out-Null

    # Check if there's actually anything to commit
    $status = & git status --porcelain
    if (-not $status) {
        Write-Host "  No changes to commit (file identical to last version)." -ForegroundColor DarkGray
        return
    }

    $msg = "Auto-update data: $name"
    & git commit -m $msg 2>&1 | Out-Null

    Write-Host "  Pushing to GitHub..." -ForegroundColor Gray
    $pushResult = & git push origin main 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  ✅ Pushed — Streamlit will rebuild in ~2 min." -ForegroundColor Green
    } else {
        Write-Host "  ❌ Push failed:" -ForegroundColor Red
        Write-Host $pushResult -ForegroundColor Red
    }
}

# Register events
Register-ObjectEvent $watcher "Changed" -Action $action | Out-Null
Register-ObjectEvent $watcher "Created" -Action $action | Out-Null
Register-ObjectEvent $watcher "Renamed" -Action $action | Out-Null

$watcher.EnableRaisingEvents = $true

# Keep alive
try {
    while ($true) { Start-Sleep -Seconds 1 }
} finally {
    $watcher.EnableRaisingEvents = $false
    $watcher.Dispose()
    Write-Host "Watcher stopped." -ForegroundColor Gray
}
