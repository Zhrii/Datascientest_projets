"""Tests de l'identification de la version des données (scripts/data_version.py)."""

from scripts.data_version import data_version, file_md5


def write_csvs(directory):
    (directory / "X_train_update.csv").write_text("id,designation\n1,chaise\n")
    (directory / "Y_train.csv").write_text("id,prdtypecode\n1,1560\n")


def test_reads_md5_from_dvc_files(tmp_path):
    write_csvs(tmp_path)
    (tmp_path / "X_train_update.csv.dvc").write_text("outs:\n- md5: " + "a" * 32 + "\n  path: X_train_update.csv\n")
    (tmp_path / "Y_train.csv.dvc").write_text("outs:\n- md5: " + "b" * 32 + "\n  path: Y_train.csv\n")
    version = data_version(tmp_path)
    assert version == {"data_X_train_update_md5": "a" * 32, "data_Y_train_md5": "b" * 32, "data_versioning": "dvc"}


def test_computes_md5_without_dvc(tmp_path):
    write_csvs(tmp_path)
    version = data_version(tmp_path)
    assert version["data_versioning"] == "md5"
    assert version["data_X_train_update_md5"] == file_md5(tmp_path / "X_train_update.csv")


def test_version_changes_with_data(tmp_path):
    write_csvs(tmp_path)
    before = data_version(tmp_path)
    (tmp_path / "Y_train.csv").write_text("id,prdtypecode\n1,10\n")
    assert data_version(tmp_path)["data_Y_train_md5"] != before["data_Y_train_md5"]
