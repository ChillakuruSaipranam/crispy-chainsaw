# PowerShell Script: Keep Render Site Awake
$Url = "https://biz-zip.onrender.com/"
$LogFile = "C:\Users\My PC\crispy-cricket\poll.log"
$IntervalSeconds = 60 # Set the interval in seconds

Write-Host "Polling $Url every $IntervalSeconds seconds. Press Ctrl+C to stop."
Write-Host "Logging output to $LogFile"


try {
    $response = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 10
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $logMessage = "[$timestamp] Pinged successfully: Status $($response.StatusCode)"
    Write-Host $logMessage
    Add-Content -Path $LogFile -Value $logMessage
} catch {
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $logMessage = "[$timestamp] Failed to ping $Url - $($_.Exception.Message)"
    Write-Warning $logMessage
    Add-Content -Path $LogFile -Value $logMessage
}


