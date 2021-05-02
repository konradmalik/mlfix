from tests import TEST_MLRUNS_PATH
from mlfix import path_utils


def test_find_file():
    metas = path_utils.find_file(TEST_MLRUNS_PATH, "meta.yaml")
    assert len(metas) == 2
    assert metas[0].name == "meta.yaml"
    assert metas[1].name == "meta.yaml"


def test_last_index_where():
    line = "something: asd/test/asd/test/asd.txt"
    assert path_utils.last_index_where(line.split("/"), lambda x: x == "test") == 3


def test_uri_to_path():
    p = "file:///Users/konrad/Code/mlflow/mlruns/0"
    assert str(path_utils.uri_to_path(p)) == "/Users/konrad/Code/mlflow/mlruns/0"
