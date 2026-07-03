"""Basic repository readiness checks for public presentation."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_readme_exists_and_mentions_strix_halo_setup():
    """README should explain the project and the AMD Strix Halo focus."""
    readme = ROOT / "README.md"

    assert readme.exists()

    text = readme.read_text(encoding="utf-8").lower()
    assert "ultragui" in text
    assert "strix halo" in text
    assert "rocm" in text
    assert "wsl" in text


def test_gitignore_excludes_heavy_local_artifacts():
    """Heavy local build/runtime artifacts should not be committed."""
    gitignore = ROOT / ".gitignore"

    assert gitignore.exists()

    patterns = gitignore.read_text(encoding="utf-8").splitlines()
    required = {
        ".venv-rocm/",
        ".venv-rocm-win/",
        "runs/",
        "__pycache__/",
        ".pytest_cache/",
        "*.pt",
    }

    assert required.issubset(set(patterns))


def test_pytest_config_does_not_use_machine_specific_path():
    """Test imports should work relative to the repository checkout."""
    conftest = ROOT / "conftest.py"
    text = conftest.read_text(encoding="utf-8")

    assert "/c/Projects/UltraGUI" not in text
    assert "C:\\Projects\\UltraGUI" not in text


def test_launcher_scripts_do_not_contain_personal_absolute_paths():
    """Public launcher scripts should be portable across checkouts."""
    checked_files = [
        ROOT / "run_wsl_gui.ps1",
        ROOT / "scripts" / "gpu_smoke_train_wsl.sh",
    ]

    for path in checked_files:
        text = path.read_text(encoding="utf-8")
        assert "C:\\Users\\" not in text
        assert "/mnt/c/Users/" not in text
        assert "C:\\Projects\\UltraGUI" not in text
        assert "/mnt/c/Projects/UltraGUI" not in text
