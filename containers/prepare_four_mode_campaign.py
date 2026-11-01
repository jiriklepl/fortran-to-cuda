"""Import native/be58/7a612/candidate authorities for the existing generic campaign runner."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "elmm-pipeline"))
    from elmm_pipeline.campaigns.prepared_modes import import_modes
    from reproduce_e2e import validate_prepared_builds

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-prepared", type=Path, required=True)
    parser.add_argument("--candidate-prepared", type=Path, required=True)
    parser.add_argument("--baseline-identity", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import_modes(args.baseline_prepared, args.candidate_prepared, args.baseline_identity, args.output,
                 validate_prepared=validate_prepared_builds)
    print(args.output / "modes.json")


if __name__ == "__main__":
    main()
