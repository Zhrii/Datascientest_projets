# PowerShell equivalent of Makefile (Windows)
param(
    [Parameter(Position = 0)]
    [ValidateSet(
        "help", "install", "install-mlops", "api", "streamlit", "train",
        "up", "up-full", "down", "logs", "health", "train-docker", "ps",
        "export-fusion"
    )]
    [string]$Target = "help"
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

switch ($Target) {
    "help" {
        Write-Host @"
Targets:
  .\make.ps1 install        deps projet
  .\make.ps1 install-mlops  + FastAPI / Prometheus / MLflow
  .\make.ps1 api            FastAPI local :8000
  .\make.ps1 streamlit      Streamlit local :8501
  .\make.ps1 train          pipeline preprocess + train (local)
  .\make.ps1 up             docker compose — stack complète du schéma
  .\make.ps1 train-docker   docker compose run train
  .\make.ps1 export-fusion  entraîne/exporte artefacts fusion (CLIP)
  .\make.ps1 down           stop containers
  .\make.ps1 logs           logs API
  .\make.ps1 ps             statut containers
  .\make.ps1 health         GET /health
"@
    }
    "install" {
        pip install -r requirements.txt
        python -m spacy download fr_core_news_sm
    }
    "install-mlops" {
        pip install -r requirements.txt
        pip install -r requirements-mlops.txt
        python -m spacy download fr_core_news_sm
    }
    "api" {
        uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
    }
    "streamlit" {
        streamlit run app.py --server.port 8501
    }
    "train" {
        python pipelines/train_pipeline.py
    }
    "up" {
        docker compose up -d --build
        Write-Host ""
        Write-Host "URLs (schéma) :"
        Write-Host "  FastAPI     http://localhost:8000/docs"
        Write-Host "  Streamlit   http://localhost:8501"
        Write-Host "  ZenML       http://localhost:8888"
        Write-Host "  Prometheus  http://localhost:9092"
        Write-Host "  Grafana     http://localhost:3002  (admin / rakuten)"
        Write-Host "  MLflow      http://localhost:5000"
        Write-Host "  MinIO       http://localhost:9001  (minioadmin / minioadmin)"
    }
    "up-full" {
        docker compose up -d --build
    }
    "train-docker" {
        docker compose --profile train run --rm train
    }
    "export-fusion" {
        $env:FUSION_MAX_ROWS = if ($env:FUSION_MAX_ROWS) { $env:FUSION_MAX_ROWS } else { "4000" }
        python pipelines/export_fusion.py
    }
    "down" {
        docker compose --profile train down
    }
    "logs" {
        docker compose logs -f api
    }
    "ps" {
        docker compose ps
    }
    "health" {
        try {
            Invoke-RestMethod http://127.0.0.1:8000/health | ConvertTo-Json -Depth 5
        } catch {
            Write-Host "API unreachable: $_"
        }
    }
}
