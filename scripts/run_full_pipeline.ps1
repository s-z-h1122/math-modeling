param(
    [Parameter(Mandatory=$true)][string]$CacheDir,
    [string]$OutputDir = "run_outputs",
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $true
$repo = Split-Path -Parent $PSScriptRoot
$out = Join-Path $repo $OutputDir
$env:Q1_CACHE_DIR = (Resolve-Path $CacheDir).Path
$env:Q1_OUT_DIR = Join-Path $out "q1_v22"
& $Python (Join-Path $repo "src/q1/q1_reconstruct_all_v22.py")

$env:Q1_OUT_DIR = Join-Path $out "q1_v22_benchmark"
& $Python (Join-Path $repo "src/q1/q1_benchmark_v22.py")

$env:Q1_OUT_DIR = Join-Path $out "q1_v21_benchmark_rerun"
& $Python (Join-Path $repo "src/q1/q1_benchmark_v21.py")

$env:Q1_V22_ALIGNED_DIR = Join-Path $out "q1_v22/recon_aligned_q3_npz"
$env:Q2_OUT_DIR = Join-Path $out "q2"
& $Python (Join-Path $repo "src/q2/q2_reconstruct_delay.py")

$env:Q3_OUT_DIR = Join-Path $out "q3"
& $Python (Join-Path $repo "src/q3/q3_loho_compare.py")

$env:Q1_OUT_DIR = Join-Path $out "q1_v22"
$env:Q1_BENCH_DIR = Join-Path $out "q1_v22_benchmark"
$env:Q1_ABLATION_DIR = Join-Path $out "q1_v21_benchmark_rerun"
& $Python (Join-Path $repo "src/q1/q1_make_figures_v22.py")

$env:MODEL_FIG_DIR = Join-Path $out "model_figures"
& $Python (Join-Path $repo "src/q3/make_results_figures.py")
& $Python (Join-Path $repo "src/validate_full_pipeline.py") --output-dir $out
