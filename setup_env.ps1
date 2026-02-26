# Script de configuration de l'environnement pour les notebooks
# Usage : .\setup_env.ps1
# À exécuter depuis la racine du projet (Datascientest_projets)

$ErrorActionPreference = "Stop"
$ProjectRoot = $PSScriptRoot
Set-Location $ProjectRoot

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Configuration environnement projet   " -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# 1. Vérifier si Conda est disponible
$condaExists = Get-Command conda -ErrorAction SilentlyContinue
if ($condaExists) {
    Write-Host "[1/4] Conda detecte - Creation de l'environnement..." -ForegroundColor Green
    conda env create -f environment.yml
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Environnement deja existant ? Tentative de mise a jour..." -ForegroundColor Yellow
        conda env update -f environment.yml --prune
    }
    conda run -n datascientest_projet python -m spacy download fr_core_news_sm
    conda run -n datascientest_projet python -c "import nltk; nltk.download('stopwords', quiet=True); nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True)"
    conda run -n datascientest_projet python -m ipykernel install --user --name=datascientest_projet --display-name="Python (Datascientest Projet)"
    Write-Host ""
    Write-Host "Environnement pret !" -ForegroundColor Green
    Write-Host "  Activation : conda activate datascientest_projet" -ForegroundColor White
    Write-Host "  Dans Jupyter : Kernel > Changer de noyau > Python (Datascientest Projet)" -ForegroundColor White
    exit 0
}

# 2. Sinon, utiliser venv
Write-Host "[1/4] Creation de l'environnement virtuel (venv)..." -ForegroundColor Green
$venvPath = Join-Path $ProjectRoot "venv"
if (-not (Test-Path $venvPath)) {
    python -m venv venv
    Write-Host "  Environnement cree dans venv/" -ForegroundColor Gray
} else {
    Write-Host "  Environnement existant detecte" -ForegroundColor Gray
}

$pythonExe = Join-Path $venvPath "Scripts\python.exe"
$pipExe = Join-Path $venvPath "Scripts\pip.exe"

if (-not (Test-Path $pythonExe)) {
    Write-Host "ERREUR : Python introuvable dans venv. Verifiez que Python 3.10 est installe." -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "[2/4] Installation des dependances (peut prendre plusieurs minutes)..." -ForegroundColor Green
& $pipExe install --upgrade pip
& $pipExe install -r requirements.txt

Write-Host ""
Write-Host "[3/4] Telechargement des modeles NLP..." -ForegroundColor Green
& $pythonExe -m spacy download fr_core_news_sm
& $pythonExe -c "import nltk; nltk.download('stopwords', quiet=True); nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True)"

Write-Host ""
Write-Host "[4/4] Enregistrement du noyau Jupyter..." -ForegroundColor Green
& $pythonExe -m ipykernel install --user --name=datascientest_projet --display-name="Python (Datascientest Projet)"

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "  Environnement pret !                 " -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Pour activer l'environnement :" -ForegroundColor White
Write-Host "  .\venv\Scripts\Activate.ps1" -ForegroundColor Cyan
Write-Host ""
Write-Host "Pour lancer Jupyter :" -ForegroundColor White
Write-Host "  jupyter notebook   OU   jupyter lab" -ForegroundColor Cyan
Write-Host ""
Write-Host "Dans les notebooks : Kernel > Changer de noyau > Python (Datascientest Projet)" -ForegroundColor White
Write-Host ""
