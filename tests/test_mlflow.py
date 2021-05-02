import tempfile
import shutil
from pathlib import Path
from mlflow.tracking import MlflowClient
from tests import TEST_MLRUNS_PATH
from mlfix import fix


def test_invalid_load():
    client = MlflowClient(tracking_uri=str(TEST_MLRUNS_PATH))
    with tempfile.TemporaryDirectory() as td:
        # should crash
        try:
            client.download_artifacts(
                "6e6280f331a94bf388fa9d0de0ecee99", "model/model.pkl", td
            )
            crashed = False
        except:
            crashed = True
    assert crashed


def test_fixed_load():
    # first fix in tempdir
    with tempfile.TemporaryDirectory() as tmpdir:
        wd = Path(tmpdir)
        # copy folder from tests to tmp as testruns (changed from mlruns)
        path_to_store = wd.joinpath("testruns")
        shutil.copytree(TEST_MLRUNS_PATH, path_to_store)
        # provide mlruns here because this is in metadata, current folder name should be autodetected
        assert fix.fix_meta(path_to_store, "mlruns")

        # now try to read
        client = MlflowClient(tracking_uri=str(path_to_store))

        # should run without error (file must be found)
        client.download_artifacts(
            "6e6280f331a94bf388fa9d0de0ecee99", "model/model.pkl", tmpdir
        )
