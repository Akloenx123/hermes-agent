"""Supply the image's fixed runtime paths to the shared offline assembler."""
from __future__ import annotations

from pathlib import Path
import sysconfig

from scripts.build.agent import assemble
from scripts.build.inputs import AgentInputs, RESOURCE_ENV
from pm.features import installed_extras, write_features
from pm.store import current_target


def assemble_image(root: Path) -> None:
    environment = root / ".venv"
    site = Path(sysconfig.get_path("purelib", vars={"base": str(environment), "platbase": str(environment)}))
    manifest = assemble(AgentInputs(
        project=root / "pyproject.toml", code=root, repo=".", placement="fixed",
        target=current_target(), python=(environment / "bin/python").absolute(),
        site_packages=site, environment=environment, tools=root / "tools",
        pm_runtime=root / "pm-runtime", bin_dir="libexec",
        resources={name: root / name for name in RESOURCE_ENV},
        frontends={"tui": root / "ui-tui", "web": root / "hermes_cli/web_dist"},
    ), root)
    # Preserve the venv command paths used by s6 and the privilege-drop shim.
    for name, command in manifest["runtime"]["commands"].items():
        link = environment / "bin" / name
        link.unlink(missing_ok=True)
        link.symlink_to(f"../../{command}")
    # PM's shipped baseline, as in a native bundle: the first writable generation
    # keeps the image's extras instead of only the ones requested at that moment.
    write_features(installed_extras(root, environment, python_exe=environment / "bin/python"), root)


if __name__ == "__main__":
    assemble_image(Path("/opt/hermes"))
