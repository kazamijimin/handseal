<#
.SYNOPSIS
    Train, evaluate, and export one HandSeal model run.

.DESCRIPTION
    This is the coworker handoff entry point. It calls the existing Python
    training modules in order and stops if any step fails. No full training is
    performed by repository validation; run this script on the training machine.

.EXAMPLE
    .\training.ps1

.EXAMPLE
    .\training.ps1 -Model mobilenetv3_small -Device cuda -RunName mobilenetv3_gpu_v1

.EXAMPLE
    .\training.ps1 -Resume artifacts\small_cnn_v1\last.pt -RunName small_cnn_v1_resume
#>

[CmdletBinding()]
param(
    [ValidateSet("small_cnn", "mobilenetv3_small")]
    [string]$Model = "small_cnn",

    [ValidateSet("auto", "cpu", "cuda")]
    [string]$Device = "auto",

    [string]$Config = "training/configs/default.yaml",

    [string]$RunName = "",

    [string]$Resume = "",

    [string]$PythonExe = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $projectRoot

function Invoke-PythonStep {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Label,

        [Parameter(Mandatory = $true)]
        [string[]]$Arguments
    )

    Write-Host "`n== $Label ==" -ForegroundColor Cyan
    & $script:pythonCommand @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Label failed with exit code $LASTEXITCODE."
    }
}

try {
    if (-not (Test-Path -LiteralPath $Config -PathType Leaf)) {
        throw "Training config not found: $Config"
    }
    if (-not (Test-Path -LiteralPath "handseal_dataset" -PathType Container)) {
        throw "Dataset directory not found: handseal_dataset"
    }

    if ($PythonExe) {
        $script:pythonCommand = $PythonExe
    } elseif (Test-Path -LiteralPath ".venv\Scripts\python.exe" -PathType Leaf) {
        $script:pythonCommand = (Resolve-Path ".venv\Scripts\python.exe").Path
    } else {
        $python = Get-Command python -ErrorAction SilentlyContinue
        if (-not $python) {
            throw "Python was not found. Activate the project virtual environment or pass -PythonExe."
        }
        $script:pythonCommand = $python.Source
    }

    Write-Host "Using Python: $script:pythonCommand"
    Invoke-PythonStep -Label "Checking training dependencies" -Arguments @(
        "-c",
        "import torch, torchvision, yaml; print('torch=' + torch.__version__); print('torchvision=' + torchvision.__version__); print('cuda_available=' + str(torch.cuda.is_available()))"
    )

    if ($Device -eq "cuda") {
        Invoke-PythonStep -Label "Checking CUDA availability" -Arguments @(
            "-c",
            "import torch; raise SystemExit(0 if torch.cuda.is_available() else 'CUDA was requested but is not available.')"
        )
    }

    if (-not $RunName) {
        $RunName = "{0}_{1}" -f $Model, (Get-Date -Format "yyyyMMdd_HHmmss")
    }

    $trainArguments = @(
        "-m", "training.train",
        "--config", $Config,
        "--model", $Model,
        "--run-name", $RunName,
        "--device", $Device
    )
    if ($Resume) {
        if (-not (Test-Path -LiteralPath $Resume -PathType Leaf)) {
            throw "Resume checkpoint not found: $Resume"
        }
        $trainArguments += @("--resume", $Resume)
    }

    Invoke-PythonStep -Label "Training $Model" -Arguments $trainArguments

    $runDirectory = Join-Path "artifacts" $RunName
    $bestCheckpoint = Join-Path $runDirectory "best.pt"
    if (-not (Test-Path -LiteralPath $bestCheckpoint -PathType Leaf)) {
        throw "Training completed without creating the best checkpoint: $bestCheckpoint"
    }

    Invoke-PythonStep -Label "Evaluating the best checkpoint" -Arguments @(
        "-m", "training.evaluate",
        "--config", (Join-Path $runDirectory "config.yaml"),
        "--checkpoint", $bestCheckpoint,
        "--split", "test",
        "--output-dir", "reports",
        "--device", $Device
    )

    Invoke-PythonStep -Label "Exporting TorchScript and metadata" -Arguments @(
        "-m", "training.export",
        "--checkpoint", $bestCheckpoint,
        "--output-dir", "artifacts/exports"
    )

    Write-Host "`nTraining pipeline complete." -ForegroundColor Green
    Write-Host "Checkpoint: $((Resolve-Path $bestCheckpoint).Path)"
    Write-Host "See artifacts/exports for the TorchScript model and metadata."
    Write-Host "Use the metadata file with: python -m app.run_hand_sign --mode model --metadata <metadata.json>"
} finally {
    Pop-Location
}
