import runpy
from unittest.mock import patch


def test_setup_metadata_is_loadable():
    with patch("setuptools.setup") as setup:
        runpy.run_path("setup.py", run_name="__main__")

    kwargs = setup.call_args.kwargs
    assert kwargs["name"] == "ganglia-common"
    assert "gTTS>=2.5.0" not in kwargs["install_requires"]
