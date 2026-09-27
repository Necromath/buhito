"""Keep the public quick start executable."""

from pathlib import Path


def test_readme_quick_start() -> None:
    readme = (Path(__file__).parents[1] / "README.md").read_text()
    quick_start = readme.split("## Runnable quick start", maxsplit=1)[1]
    quick_start = quick_start.split("## The strongest parts of the API", maxsplit=1)[0]
    code = quick_start.split("```python", maxsplit=1)[1].split("```", maxsplit=1)[0]

    exec(compile(code, "README.md", "exec"), {"__name__": "__main__"})
