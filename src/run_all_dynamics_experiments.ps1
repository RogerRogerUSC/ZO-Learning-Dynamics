# PowerShell script to run all configs in dynamics_experiments directory
# Usage: .\run_all_dynamics_experiments.ps1 [-NumTestSamples N]

param(
    [int]$NumTestSamples = 5
)

# Get the directory where this script is located
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ConfigsDir = Join-Path $ScriptDir "zo_llm\configs\text_classification\dynamics_experiments"

# Check if configs directory exists
if (-not (Test-Path $ConfigsDir)) {
    Write-Host "Error: Configs directory not found: $ConfigsDir" -ForegroundColor Red
    exit 1
}

# Find all YAML files in dynamics_experiments
$ConfigFiles = Get-ChildItem -Path $ConfigsDir -Filter "*.yaml" | Sort-Object Name

# Check if any config files were found
if ($ConfigFiles.Count -eq 0) {
    Write-Host "Error: No YAML config files found in $ConfigsDir" -ForegroundColor Red
    exit 1
}

# Count total configs
$TotalConfigs = $ConfigFiles.Count
Write-Host "Found $TotalConfigs config files to run"
Write-Host "Number of test samples: $NumTestSamples"
Write-Host "=========================================="
Write-Host ""

# Track success and failures
$SuccessCount = 0
$FailureCount = 0
$FailedConfigs = @()

# Run each config file
$Progress = 0
foreach ($ConfigFile in $ConfigFiles) {
    $Progress++
    $ConfigName = $ConfigFile.Name
    $ConfigRelativePath = "text_classification\dynamics_experiments\$ConfigName"
    
    Write-Host "[$Progress/$TotalConfigs] Running: $ConfigName"
    Write-Host "----------------------------------------"
    
    # Run the experiment
    $PythonScript = Join-Path $ScriptDir "llm_dynamics_main.py"
    $Result = & uv run $PythonScript --config-path $ConfigRelativePath --num-test-samples $NumTestSamples
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ Successfully completed: $ConfigName" -ForegroundColor Green
        $SuccessCount++
    } else {
        Write-Host "✗ Failed: $ConfigName" -ForegroundColor Red
        $FailureCount++
        $FailedConfigs += $ConfigName
    }
    
    Write-Host ""
}

# Print summary
Write-Host "=========================================="
Write-Host "Summary:"
Write-Host "  Total configs: $TotalConfigs"
Write-Host "  Successful: $SuccessCount" -ForegroundColor Green
Write-Host "  Failed: $FailureCount" -ForegroundColor $(if ($FailureCount -gt 0) { "Red" } else { "Green" })
Write-Host ""

if ($FailureCount -gt 0) {
    Write-Host "Failed configs:" -ForegroundColor Red
    foreach ($FailedConfig in $FailedConfigs) {
        Write-Host "  - $FailedConfig"
    }
    exit 1
} else {
    Write-Host "All experiments completed successfully!" -ForegroundColor Green
    exit 0
}
