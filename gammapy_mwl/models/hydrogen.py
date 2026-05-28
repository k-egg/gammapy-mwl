"""Module to calculate Galactic HI column density (nH) and generate/load TBabs absorption models."""

import logging
from pathlib import Path
from typing import Optional, Tuple, Union

import astropy.units as u
import numpy as np
import requests
from astropy.coordinates import SkyCoord
from astropy.io import fits as pyfits
from astropy.table import Table

# Gammapy 2.0+ imports
from gammapy.maps import MapAxis, RegionNDMap
from gammapy.modeling.models import TemplateNDSpectralModel

# Setup logger for cleaner professional feedback instead of print statements
logger = logging.getLogger(__name__)

# Determine paths relative to this file's location
MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_TBABS_FILE = MODULE_DIR / "data" / "tbabs_tau_factor_vs_nH_energy.ecsv"

SWIFT_URL = "https://www.swift.ac.uk/analysis/nhtot/donhtot.php"


def parse_response(html_response: str) -> float:
    """Parses Swift's HTML response and extracts the weighted total NH value."""
    akey = "headers='htotw'>"
    try:
        a = html_response.index(akey) + len(akey)
        # Find the end tag or next marker cleanly
        l = html_response[a:].index("</td>")
        part = html_response[a : a + l].strip()

        if "×10" in part:
            base, expo = part.split(" ×10")
            # Handle superscripts if present in raw html strings
            expo = (
                expo.replace("<sup>", "")
                .replace("</sup>", "")
                .replace(" ", "")
            )
            return float(base) * 10 ** float(expo)
        return float(part)
    except (ValueError, IndexError) as e:
        raise ValueError(
            f"Failed to parse NH value from HTML response. Error: {e}"
        )


def get_gal_nh_from_radec(ra: float, dec: float) -> float:
    """Queries swift.ac.uk for the total NH value at (RA, Dec) in degrees."""
    payload = {
        "equinox": 2000,
        "Coords": f"{ra} {dec}",
        "submit": "Calculate NH",
    }
    response = requests.post(SWIFT_URL, data=payload, timeout=15)
    response.raise_for_status()
    return parse_response(response.text)


def get_ra_dec(
    infile: Optional[Union[str, Path]] = None,
    srcname: Optional[str] = None,
    src: Optional[SkyCoord] = None,
) -> Tuple[float, float]:
    """Resolves Right Ascension and Declination from file header, name, or SkyCoord."""
    if infile is not None:
        with pyfits.open(infile) as hdul:
            header = hdul[0].header
            ra = header["RA_OBJ"]
            dec = header["DEC_OBJ"]
    elif srcname is not None:
        coord = SkyCoord.from_name(srcname)
        ra, dec = coord.ra.deg, coord.dec.deg
    elif src is not None:
        ra, dec = src.ra.deg, src.dec.deg
    else:
        raise ValueError(
            "Must specify either 'infile', 'srcname', or an astropy SkyCoord 'src'"
        )

    return ra, dec


def get_gal_nh(
    infile: Optional[Union[str, Path]] = None,
    srcname: Optional[str] = None,
    src: Optional[SkyCoord] = None,
) -> Optional[float]:
    """Safely retrieves Galactic NH column density; returns None if resolution fails."""
    try:
        ra, dec = get_ra_dec(infile=infile, srcname=srcname, src=src)
        return get_gal_nh_from_radec(ra, dec)
    except Exception as e:
        logger.error(f"Could not retrieve Galactic nH: {e}")
        return None


def sherpa_xtbabs_model(
    infile: Optional[Union[str, Path]] = None,
    srcname: Optional[str] = None,
    src: Optional[SkyCoord] = None,
):
    """Generates a wrapper around Sherpa's XSTBabs model.

    Requires 'sherpa' and 'gammapy_ogip' packages to be installed.
    """
    from gammapy_ogip.models import SherpaSpectralModel
    from sherpa.astro.xspec import XSTBabs

    nhgal = get_gal_nh(infile, srcname, src)
    if nhgal is None:
        raise ValueError("Cannot initialize Sherpa model without valid nH.")

    abs_model = XSTBabs()
    abs_model.nh = nhgal / 1e22
    abs_model.nh.frozen = True

    sherpa_wrapped = SherpaSpectralModel(abs_model, default_units=(u.keV, 1))
    sherpa_wrapped.tag = "sherpa.astro.xspec.XSTBabs"
    return sherpa_wrapped


def generate_tbabs_interp_table(outfile: Union[str, Path]) -> None:
    """Generates the 2D interpolation ECSV table using local Sherpa/XSpec installation."""
    from gammapy_ogip.models import SherpaSpectralModel
    from sherpa.astro.xspec import XSTBabs

    out_path = Path(outfile)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    nH_array = np.round(np.linspace(-4, 2.5, 10 * 60), decimals=5)
    en_array = np.round(np.linspace(-2, 2.5, 10 * 10), decimals=5)

    tbabs_table_data = []

    for nH in nH_array:
        model = XSTBabs()
        model.nh.val = 10**nH
        wrapped_model = SherpaSpectralModel(model, default_units=(u.keV, 1))

        # Calculate optical depth tau
        tau = -np.log(wrapped_model(10**en_array * u.keV))
        tau = np.where(np.isinf(tau), 100.0, tau)
        tau = np.where(tau < 0, 0.0, tau)

        tbabs_table_data.append(list(np.round(tau, 5)))

    tbabs_table = Table(data=tbabs_table_data)
    tbabs_table.meta["log10_E_type"] = "rows"
    tbabs_table.meta["log10_E_values"] = list(en_array)
    tbabs_table.meta["log10_nH_type"] = "columns"
    tbabs_table.meta["log10_nH_values"] = list(nH_array)
    tbabs_table.meta["table data"] = "tau factor (natural log of absorption)"

    tbabs_table.write(out_path, format="ascii.ecsv", overwrite=True)
    logger.info(f"Interpolation table written to {out_path}")


def get_tbabs_template_model(
    tbabsfile: Optional[Union[str, Path]] = None,
    infile: Optional[Union[str, Path]] = None,
    srcname: Optional[str] = None,
    src: Optional[SkyCoord] = None,
    freeze: bool = True,
) -> TemplateNDSpectralModel:
    """Creates a Gammapy TemplateNDSpectralModel for TBabs absorption.

    Uses relative directory pathways to find the data files if not explicitly provided.
    """
    # Use relative path if none provided
    if tbabsfile is None:
        tbabsfile = DEFAULT_TBABS_FILE
    else:
        tbabsfile = Path(tbabsfile)

    if not tbabsfile.exists():
        raise FileNotFoundError(
            f"Absorption template file not found at: {tbabsfile.resolve()}"
        )

    tbabs_table = Table.read(tbabsfile, format="ascii.ecsv")

    # Clean multi-dimensional array extraction from ECSV columns
    tbabs_data = tbabs_table.as_array()
    tbabs_data = tbabs_data.view(np.float64).reshape(tbabs_data.shape + (-1,))

    log_nh_array_20 = np.asarray(tbabs_table.meta["log10_nH_values"])
    log_en_array = np.asarray(tbabs_table.meta["log10_E_values"])

    # Node mappings for Gammapy 2.0+
    energy_axis = MapAxis.from_nodes(
        10**log_en_array * u.keV, name="energy_true", interp="log"
    )
    nh_axis = MapAxis.from_nodes(
        10**log_nh_array_20 * 1e22 * u.Unit("cm-2"), name="nH", interp="log"
    )

    # Convert optical depth to transmission, preserving numerical thresholds
    transmission = np.exp(-np.transpose(tbabs_data))
    transmission = np.clip(transmission, 1e-5, None)

    region_ndmap = RegionNDMap.create(
        region=None,
        axes=[energy_axis, nh_axis],
        data=transmission,
    )

    template_abs_model = TemplateNDSpectralModel(
        map=region_ndmap,
        interp_kwargs={"method": "linear", "fill_value": 1e-5},
    )

    # Handle parameters assignment safely
    nhgal = get_gal_nh(infile, srcname, src)
    if nhgal is not None:
        template_abs_model.nH.quantity = nhgal * u.Unit("cm-2")
        if freeze:
            template_abs_model.nH.frozen = True
    else:
        logger.warning(
            "Generating generic template absorption model with default nH parameters."
        )

    return template_abs_model
