import importlib
import sys
print("Checking environment setup...\n")

packages = [
    "torch",
    "torchvision",
    "torchaudio",
    "numpy",
    "pandas",
    "PIL",
    "matplotlib",
    "sklearn",
    "tqdm",
]

print(f"Python version: {sys.version}")
print("-" * 50)

failed = []

for pkg in packages:
    try:
        importlib.import_module(pkg)
        print(f"[OK] {pkg}")
    except Exception as e:
        print(f"[FAIL] {pkg}: {e}")
        failed.append(pkg)

print("-" * 50)

if failed:
    raise SystemExit(f"Setup check failed. Missing/broken packages: {failed}")

print("Setup check passed.")