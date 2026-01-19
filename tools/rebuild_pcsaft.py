import argparse
import subprocess
from pathlib import Path
import sys
import time


def run(cmd, cwd=None):
    print(f"+ {cmd}")
    subprocess.check_call(cmd, cwd=cwd, shell=True)


def main():
    parser = argparse.ArgumentParser(description="Rebuild PC-SAFT in-place (Cython -> .pyd).")
    parser.add_argument("--python", dest="python", default=sys.executable,
                        help="Path to Python interpreter for the target env (default: current python).")
    args = parser.parse_args()

    pcsaft_dir = Path(__file__).resolve().parent.parent
    if not pcsaft_dir.exists():
        raise SystemExit(f"PC-SAFT repo not found at: {pcsaft_dir}")

    # If the extension is loaded, Windows will lock it and build_ext --inplace will fail.
    pyd = next(iter(pcsaft_dir.glob("pcsaft*.pyd")), None)
    if pyd is not None:
        bak = pyd.with_suffix(pyd.suffix + f".bak.{int(time.time())}")
        try:
            pyd.rename(bak)
            print(f"Renamed existing extension to: {bak.name}")
        except Exception:
            raise SystemExit(
                f"Could not rename existing {pyd.name}. "
                "It is likely in use by Python/PyCharm. "
                "Close all Python processes using pcsaft and retry."
            )

        # keep only the newest backup
        backups = sorted(pcsaft_dir.glob("pcsaft*.pyd.bak.*"), key=lambda p: p.stat().st_mtime, reverse=True)
        for old in backups[1:]:
            try:
                old.unlink()
            except Exception:
                pass

    py = f'"{args.python}"'
    run(f"{py} setup.py build_ext --inplace", cwd=pcsaft_dir)


if __name__ == "__main__":
    main()
