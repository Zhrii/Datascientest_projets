# PowerShell equivalent of Makefile (Windows)
param(
    [Parameter(Position = 0)]
    [ValidateSet(
        "help", "install", "install-mlops", "api", "streamlit", "train", "train-local",
        "promote", "test", "up", "down", "logs", "health", "ps", "export-fusion",
        "data-push", "data-pull", "drift"
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
  .\make.ps1 up             docker compose — stack complète du schéma
  .\make.ps1 train          pipeline ZenML (Docker) : collect -> validate -> preprocess -> train -> promote
  .\make.ps1 train-local    sans Docker : collecte -> process -> training (candidat)
  .\make.ps1 promote        sans Docker : met le candidat en production s'il est meilleur
  .\make.ps1 test           tests pytest (dans l'image de l'API)
  .\make.ps1 data-push      envoie les CSV versionnés par DVC dans MinIO
  .\make.ps1 data-pull      récupère les CSV depuis MinIO (version du commit courant)
  .\make.ps1 drift          lance tout de suite une analyse de dérive
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
        docker compose run --rm train
        if ($LASTEXITCODE -ne 0) { throw "le pipeline a échoué" }
        Write-Host "Si le modèle est promu, les répliques de l'API le chargent toutes seules (< 1 min)."
    }
    "train-local" {
        python scripts/collecte.py --data-dir "data brut"
        python scripts/process.py
        python scripts/training.py --mlflow --models-dir models/candidate
    }
    "promote" {
        python scripts/promote_model.py
    }
    "up" {
        docker compose up -d --build
        Write-Host ""
        Write-Host "URLs :"
        Write-Host "  FastAPI     http://localhost:8000/docs  (via nginx, 2 répliques)"
        Write-Host "  Streamlit   http://localhost:8501"
        Write-Host "  ZenML       http://localhost:8888  (default / zenml)"
        Write-Host "  MLflow      http://localhost:5000"
        Write-Host "  MinIO       http://localhost:9001  (minioadmin / minioadmin)"
        Write-Host "  Prometheus  http://localhost:9092"
        Write-Host "  Grafana     http://localhost:3002  (admin / rakuten)"
        Write-Host ""
        Write-Host "API protégée : admin / rakuten (rôle admin), demo / demo (rôle user)"
    }
    "test" {
        docker compose run --rm --no-deps api pytest -q
    }
    "data-push" {
        python -m dvc push "data brut/X_train_update.csv.dvc" "data brut/Y_train.csv.dvc" "data brut/X_test_update.csv.dvc"
    }
    "data-pull" {
        python -m dvc pull "data brut/X_train_update.csv.dvc" "data brut/Y_train.csv.dvc" "data brut/X_test_update.csv.dvc"
    }
    "drift" {
        docker compose exec drift-monitor python pipelines/drift_monitor.py --once
    }
    "export-fusion" {
        $env:FUSION_MAX_ROWS = if ($env:FUSION_MAX_ROWS) { $env:FUSION_MAX_ROWS } else { "4000" }
        python scripts/export_fusion.py
    }
    "down" {
        docker compose --profile jobs down
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
