"""Module to calculate Galactic dust extinction and generate/load xredden template models."""

import logging
from pathlib import Path
from typing import Optional, Tuple, Union

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits as pyfits
from astropy.table import Table

# Gammapy 2.0+ imports
from gammapy.maps import MapAxis, RegionNDMap
from gammapy.modeling.models import TemplateNDSpectralModel

# Setup logger for professional feedback instead of print statements
logger = logging.getLogger(__name__)

# Determine paths relative to this file's location
MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_XREDDEN_FILE = MODULE_DIR / "data" / "xredden_tau_factor_vs_EBV_energy.ecsv"


def get_ebv_from_radec(ra: float, dec: float) -> float:
    """Calculates E(B-V) dust extinction value at (RA, Dec) using dustmaps."""
    try:
        from dustmaps.config import config
        from dustmaps.sfd import SFDQuery
    except ImportError:
        raise ImportError(
            "The 'dustmaps' package is required for this function. "
            "Please install it or provide E(B-V) via another method."
        )

    coords = SkyCoord(ra * u.deg, dec * u.deg, frame="fk5")
    sfd = SFDQuery()
    ebv = sfd(coords)
    return float(ebv)


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


def get_gal_ebv(
    infile: Optional[Union[str, Path]] = None,
    srcname: Optional[str] = None,
    src: Optional[SkyCoord] = None,
) -> Optional[float]:
    """Safely retrieves Galactic E(B-V) value; returns None if resolution fails."""
    try:
        ra, dec = get_ra_dec(infile=infile, srcname=srcname, src=src)
        return get_ebv_from_radec(ra, dec)
    except Exception as e:
        logger.error(f"Could not retrieve Galactic E(B-V): {e}")
        return None


def generate_xredden_interp_table(outfile: Union[str, Path]) -> None:
    """Generates the 2D interpolation ECSV table using local Sherpa/XSpec installation."""
    from gammapy_ogip.models import SherpaSpectralModel
    from sherpa.astro.xspec import XSxredden

    out_path = Path(outfile)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ebv_array = np.round(np.linspace(-4, 1.5, 10 * 60), decimals=5)
    en_array = np.round(np.linspace(-2, 2.5, 10 * 10), decimals=5)

    xredden_table_data = []

    for ebv in ebv_array:
        model = XSxredden()
        model.E_B_V.val = 10**ebv
        wrapped_model = SherpaSpectralModel(model, default_units=(u.keV, 1))

        # Calculate optical depth tau
        tau = -np.log(wrapped_model(10**en_array * u.keV))
        tau = np.where(np.isinf(tau), 100.0, tau)
        tau = np.where(tau < 0, 0.0, tau)

        xredden_table_data.append(list(np.round(tau, 5)))

    xredden_table = Table(data=xredden_table_data)
    xredden_table.meta["log10_E_type"] = "rows"
    xredden_table.meta["log10_E_values"] = list(en_array)
    xredden_table.meta["log10_EBV_type"] = "columns"
    xredden_table.meta["log10_EBV_values"] = list(ebv_array)
    xredden_table.meta["table data"] = "tau factor (natural log of absorption)"

    xredden_table.write(out_path, format="ascii.ecsv", overwrite=True)
    logger.info(f"Xredden interpolation table written to {out_path}")


def get_xredden_template_model(
    xreddenfile: Optional[Union[str, Path]] = None,
    infile: Optional[Union[str, Path]] = None,
    srcname: Optional[str] = None,
    src: Optional[SkyCoord] = None,
    freeze: bool = True,
) -> TemplateNDSpectralModel:
    """Creates a Gammapy TemplateNDSpectralModel for dust extinction (xredden).

    Uses relative directory pathways to find the data files if not explicitly provided.
    """
    # Use relative path if none provided
    if xreddenfile is None:
        xreddenfile = DEFAULT_XREDDEN_FILE
    else:
        xreddenfile = Path(xreddenfile)

    if not xreddenfile.exists():
        raise FileNotFoundError(
            f"Extinction template file not found at: {xreddenfile.resolve()}"
        )

    xredden_table = Table.read(xreddenfile, format="ascii.ecsv")

    # Clean multi-dimensional array extraction from ECSV columns
    xredden_data = xredden_table.as_array()
    xredden_data = xredden_data.view(np.float64).reshape(
        xredden_data.shape + (-1,)
    )

    log_ebv_array = np.asarray(xredden_table.meta["log10_EBV_values"])
    log_en_array = np.asarray(xredden_table.meta["log10_E_values"])

    # Node mappings optimized for Gammapy 2.0+ 
    # Note: Named the map axes 'energy_true' and 'ebv' to keep parameters logical
    energy_axis = MapAxis.from_nodes(
        10**log_en_array * u.keV, name="energy_true", interp="log"
    )
    ebv_axis = MapAxis.from_nodes(
        10**log_ebv_array * u.dimensionless_unscaled, name="ebv", interp="log"
    )

    # Convert optical depth to transmission safely
    transmission = np.exp(-np.transpose(xredden_data))
    transmission = np.clip(transmission, 1e-5, None)

    region_ndmap = RegionNDMap.create(
        region=None,
        axes=[energy_axis, ebv_axis],
        data=transmission,
    )
