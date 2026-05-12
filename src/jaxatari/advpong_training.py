import argparse
from pathlib import Path
import sys
import importlib.util

from omegaconf import OmegaConf


def _load_config(root: Path, alg_path: str) -> dict:
    base_cfg = OmegaConf.load(root / "scripts" / "benchmarks" / "config" / "config.yaml")
    alg_cfg = OmegaConf.load(root / alg_path)
    base_cfg["alg"] = alg_cfg
    return OmegaConf.to_container(base_cfg)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run adversarial PQN training for Pong using benchmark configs."
    )
    parser.add_argument(
        "--alg",
        default="scripts/benchmarks/config/alg/pqn_jaxatari_object_adv_random.yaml",
        help="Path to the PQN adversarial alg config.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))

    module_path = root / "scripts" / "benchmarks" / "pqn_agent_adv_random.py"
    benchmarks_path = str(module_path.parent)
    if benchmarks_path not in sys.path:
        sys.path.insert(0, benchmarks_path)
    spec = importlib.util.spec_from_file_location("pqn_agent_adv_random", module_path)
    pqn_agent_adv_random = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pqn_agent_adv_random)

    config = _load_config(root, args.alg)
    pqn_agent_adv_random.single_run(config)


if __name__ == "__main__":
    main()
