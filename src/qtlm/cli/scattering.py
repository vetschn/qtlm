from threadpoolctl import threadpool_limits  # isort:skip
from pathlib import Path

from qtlm.config import parse_config
from qtlm.scattering.device import Device
from qtlm.scattering.scba import SCBA

device = Device()


def main(config_file: Path):
    """Calculates transport through the structure.

    Parameters
    ----------
    config_file : Path
        The configuration for the transport calculation.

    """
    config = parse_config(config_file)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    print("Calculating electron-photon scattering...", flush=True)

    device.configure(config)

    scba = SCBA(config)

    with threadpool_limits(limits=2, user_api="blas"):
        scba.run()

    print(
        f"Transport calculation finished. Results written to {scba.config.output_dir}.",
        flush=True,
    )
