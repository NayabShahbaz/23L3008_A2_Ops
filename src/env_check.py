"""Task 1.1 - print exact library versions and whether TensorFlow sees a GPU.

Run:  python -m src.env_check
Screenshot the terminal output (shows your prompt + versions + GPU line).
"""
import platform
import sys
from importlib.metadata import PackageNotFoundError, version

PACKAGES = [
    "tensorflow", "keras", "keras-tuner", "numpy", "pandas", "pillow",
    "matplotlib", "scikit-learn", "mlflow", "dagshub", "dvc", "flask", "sqlalchemy",
]


def main() -> None:
    print(f"Python      : {sys.version.split()[0]}  ({sys.executable})")
    print(f"Platform    : {platform.platform()}")
    for pkg in PACKAGES:
        try:
            print(f"{pkg:<12}: {version(pkg)}")
        except PackageNotFoundError:
            print(f"{pkg:<12}: NOT INSTALLED")

    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    print(f"TF built with CUDA: {tf.test.is_built_with_cuda()}")
    print(f"GPUs visible      : {len(gpus)} -> {gpus if gpus else 'none (CPU only)'}")


if __name__ == "__main__":
    main()
