# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Module to calculate Galactic dust extinction and generate/load xredden template models.

To manually query/obtain the Galactic E(B-V) dust extinction value for a specific coordinate:
1. Use the IRSA Dust Query tool: https://irsa.ipac.caltech.edu/applications/DUST/
2. Use the python dustmaps package:
   >>> from dustmaps.sfd import SFDQuery
   >>> from astropy.coordinates import SkyCoord
   >>> import astropy.units as u
   >>> # First-time setup (downloading maps):
   >>> # import dustmaps.sfd
   >>> # dustmaps.sfd.fetch()
   >>> sfd = SFDQuery()
   >>> coords = SkyCoord(ra * u.deg, dec * u.deg, frame="fk5")
   >>> ebv = sfd(coords)
"""

import logging
from pathlib import Path

import astropy.units as u
import numpy as np

# Gammapy 2.0+ imports
from gammapy.maps import MapAxis, RegionNDMap
from gammapy.modeling.models import TemplateNDSpectralModel

# Setup logger for professional feedback instead of print statements
logger = logging.getLogger(__name__)

# Determine paths relative to this file's location
MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_XREDDEN_FILE = MODULE_DIR / "data" / "xredden_tau_factor_vs_EBV_energy.ecsv"


def generate_xredden_interp_table(outfile):
    """Generates the 2D interpolation ECSV table using local Sherpa/XSpec installation."""
    from gammapy_ogip.models import SherpaSpectralModel
    from sherpa.astro.xspec import XSxredden
    from astropy.table import Table

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
    ebv=None,
    xreddenfile=None,
    freeze=True,
):
    """Creates a Gammapy TemplateNDSpectralModel for dust extinction (xredden).

    Uses relative directory pathways to find the data files if not explicitly provided.

    Parameters
    ----------
    ebv : float or `~astropy.units.Quantity`, optional
        Galactic E(B-V) dust extinction. If float, dimensionless unit is assumed.
    xreddenfile : str or `~pathlib.Path`, optional
        Path to the interpolation ECSV file.
    freeze : bool, optional
        Whether to freeze the ebv parameter. Default is True.
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

    from astropy.table import Table
    xredden_table = Table.read(xreddenfile, format="ascii.ecsv")

    # Clean multi-dimensional array extraction from ECSV columns
    xredden_data = np.stack(
        [xredden_table[col].astype(np.float64) for col in xredden_table.colnames],
        axis=1
    )

    ebv_array = np.asarray(xredden_table.meta["ebv_values"])
    log_en_array = np.asarray(xredden_table.meta["log10_E_values"])

    # Node mappings optimized for Gammapy 2.0+ 
    # Note: Named the map axes 'energy_true' and 'ebv' to keep parameters logical
    energy_axis = MapAxis.from_nodes(
        10**log_en_array * u.keV, name="energy_true", interp="log"
    )
    ebv_axis = MapAxis.from_nodes(
        ebv_array * u.dimensionless_unscaled, name="ebv", interp="linear"
    )

    # Convert optical depth to transmission safely
    transmission = np.exp(-np.transpose(xredden_data))
    transmission = np.clip(transmission, 1e-5, None)

    region_ndmap = RegionNDMap.create(
        region=None,
        axes=[energy_axis, ebv_axis],
        data=transmission,
    )

    template_abs_model = TemplateNDSpectralModel(
        map=region_ndmap,
        interp_kwargs={"method": "linear", "fill_value": 1e-5},
    )

    # Handle parameters assignment safely
    if ebv is not None:
        template_abs_model.ebv.quantity = u.Quantity(ebv, u.dimensionless_unscaled)
        if freeze:
            template_abs_model.ebv.frozen = True
    else:
        logger.warning(
            "Generating generic template absorption model with default EBV parameters."
        )

    return template_abs_model
