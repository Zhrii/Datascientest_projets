"""
data_version.py : identifie la version des données brutes utilisée pour un entraînement

Les CSV de "data brut/" sont versionnés avec DVC : à côté de chaque fichier,
un petit fichier X.csv.dvc (commité dans Git) contient son empreinte md5.
On relit cette empreinte (ou on la calcule si le .dvc n'existe pas : même
algorithme que DVC), puis training.py l'enregistre dans le run MLflow.

=> pour un modèle donné, MLflow dit avec quelles données il a été entraîné,
   et `git checkout <commit> && dvc pull` permet de retrouver exactement ces données.
"""
import hashlib
import re
from pathlib import Path

DATA_FILES = ["X_train_update.csv", "Y_train.csv"]
_MD5_IN_DVC = re.compile(r"md5:\s*([0-9a-f]{32})")


def file_md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def data_version(data_dir) -> dict:
    """{"data_X_train_update_md5": "...", ..., "data_versioning": "dvc" | "md5"}."""
    data_dir = Path(data_dir)
    version, source = {}, "dvc"
    for name in DATA_FILES:
        dvc_file = data_dir / f"{name}.dvc"
        match = _MD5_IN_DVC.search(dvc_file.read_text(encoding="utf-8")) if dvc_file.exists() else None
        if match:
            md5 = match.group(1)
        else:
            md5, source = file_md5(data_dir / name), "md5"
        version[f"data_{Path(name).stem}_md5"] = md5
    version["data_versioning"] = source
    return version
