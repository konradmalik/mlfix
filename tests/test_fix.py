import tempfile
import shutil
from pathlib import Path
import yaml
from mlfix import fix
from tests import TEST_MLRUNS_PATH


def test_fix_path_in_uri():
    u = "file:///Users/konrad/Code/mlflow/mlruns/0"
    p = Path("/Users/another/mlfix")
    fixed = fix._fix_path_in_uri(u, "mlruns", p)
    assert fixed == "file:///Users/another/mlfix/0"


def test_fix():
    with tempfile.TemporaryDirectory() as tmpdir:
        wd = Path(tmpdir)
        # copy folder from tests to tmp as testruns (changed from mlruns)
        path_to_store = wd.joinpath("testruns")
        shutil.copytree(TEST_MLRUNS_PATH, path_to_store)
        # provide mlruns here because this is in metadata, current folder name should be autodetected
        assert fix.fix_meta(path_to_store, "mlruns")

        with open(path_to_store.joinpath(*["0", "meta.yaml"]), "r") as f:
            c = yaml.full_load(f)
            assert str(path_to_store) in c["artifact_location"]
            assert "mlruns" not in c["artifact_location"]
            assert "meta.yaml" not in c["artifact_location"]
