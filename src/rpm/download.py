"""Script to download the RPM dataset from google drive."""

from os.path import join as pjoin
from typing import Annotated

import tyro
from gdown.download_folder import download_folder

# Google drive folder id of each trace.
TRACES = {
    "c_corridor": "1Cf_b4eGJdWj5G4qGrl9V-Fmd8MbjcjtL",
    "garden": "1x96lXxuzWgzPJByUwQPySGGZ_DmAGWFe",
    "nsh": "1EkEEjE4VW0UFpYwrTr4JjuoBBstsUr_x",
    "nsh_a": "1TqrEwB-e_nA1s9TIjCvYSOfBtKDbiTJH",
    "nsh_a1": "1Ymoz79KyimF_YSEahYNncCEFnqqfljZR",
    "nsh_b": "1Ql_pUFcfa1tOnRshi6x4PQEqiIi--0ka",
    "nsh_short": "1Ohuiz1bnprk8AF6jFfdkEI2a5-M8V-yw",
    "parking": "1DfTdodq5lo7P9qk_YMc8geWMH-I4WWkp",
    "square1": "14RHizcqibbaJI57I-UGB6QltByrToX-W",
    "square2": "1aweCEyyT1r7KAbfsTSbXCd-WntUcnUjN",
    "tepper": "1znQirjHvPXE_TOI2N9NBodk47MMpiyN3",
    "wean": "1o_Y12BaK3U_OJyBsCsfVYGhjMny9jMi1",
    "z_shape1": "1iggkdgI_-GkTtPco9UkYlCaP3f-wDldD",
    "z_shape2": "1XWwJdd_gFaqkIjHYMjoVLjaVsCbrH3e8",
}


def download_dataset(
    traces: tyro.conf.Positional[tuple[str, ...]] = (),
    download_all: Annotated[bool, tyro.conf.arg(name="all")] = False,
    output: str = "data/rpm",
) -> None:
    """Download traces from google drive.

    Args:
        traces: Names of the traces to download.
        download_all: Download all traces (~1.1TB).
        output: Directory to download into; each trace lands in
            `<output>/<trace>`.
    """
    available = ", ".join(TRACES)
    if download_all == bool(traces):
        raise SystemExit(
            f"Specify either trace names or --all. Available traces: {available}"
        )
    unknown = [t for t in traces if t not in TRACES]
    if unknown:
        raise SystemExit(
            f"Unknown traces: {', '.join(unknown)}. Available traces: {available}"
        )

    for trace in TRACES if download_all else traces:
        download_folder(id=TRACES[trace], output=pjoin(output, trace), resume=True)


def main() -> None:
    """Entry point for the `download-dataset` script."""
    tyro.cli(download_dataset)


if __name__ == "__main__":
    main()
