# `gammapy_mwl.models` — Spectral absorption models

This subpackage provides spectral absorption models for use within the
[Gammapy](https://gammapy.org/) modeling and fitting framework, covering X-ray
absorption by the interstellar medium (hydrogen column density) and optical/UV
dust extinction.

Two complementary implementations are offered for each physical component:

| Implementation | Accuracy | Requires |
|---|---|---|
| Pre-computed interpolation table (`TemplateNDSpectralModel`) | ★★★ (grid resolution) | Gammapy only |
| Live Sherpa/XSpec evaluation (`SherpaSpectralModel` wrapper) | ★★★★★ (native) | Gammapy + Sherpa |

The table-based models are the recommended default for fitting: they have no
extra runtime dependency and are fast to evaluate. The Sherpa wrappers are
useful for validation, for regenerating the reference tables, or when access to
the full XSpec parameter space is needed.

---

## Modules

### [`sherpa.py`](sherpa.py)

Defines `SherpaSpectralModel`, a thin Gammapy `SpectralModel` wrapper around
any Sherpa model instance (from `sherpa.models` or `sherpa.astro.xspec`). It
bridges Sherpa's evaluation convention (bin-integrated fluxes) into Gammapy's
energy-point-evaluation scheme and exposes all Sherpa parameters as Gammapy
`Parameter` objects.

```python
from gammapy_mwl.models.sherpa import SherpaSpectralModel
from sherpa.astro.xspec import XSTBabs

sherpa_model = XSTBabs()
sherpa_model.nh.val = 0.2        # in units of 1e22 cm-2
wrapped = SherpaSpectralModel(sherpa_model, default_units=(u.keV, 1))
```

### [`hydrogen.py`](hydrogen.py)

X-ray absorption by neutral hydrogen (ISM), based on the
[TBabs](https://pulsar.sternwarte.uni-erlangen.de/wilms/research/tbabs/)
cross-sections (Wilms et al. 2000).

| Function | Description |
|---|---|
| `sherpa_xtbabs_model(nh)` | Live Sherpa `XSTBabs` wrapper |
| `generate_tbabs_interp_table(outfile)` | Regenerate the reference ECSV table |
| `get_tbabs_template_model(nh, ...)` | Load the table as a `TemplateNDSpectralModel` |

```python
from gammapy_mwl.models.hydrogen import get_tbabs_template_model
import astropy.units as u

model = get_tbabs_template_model(nh=2e21 * u.Unit("cm-2"), freeze=True)
```

The column density `nH` spans 10⁻⁴–10²·⁵ × 10²² cm⁻² and the energy grid
covers 0.01–316 keV (both in log-steps of 0.1 dex).

### [`dustextinction.py`](dustextinction.py)

Optical/UV/soft-X-ray extinction by Galactic dust, based on Sherpa's
`XSxredden` (Cardelli, Clayton & Mathis 1989 extinction law).

| Function | Description |
|---|---|
| `sherpa_xredden_model(ebv)` | Live Sherpa `XSxredden` wrapper |
| `generate_xredden_interp_table(outfile)` | Regenerate the reference ECSV table |
| `get_xredden_template_model(ebv, ...)` | Load the table as a `TemplateNDSpectralModel` |

```python
from gammapy_mwl.models.dustextinction import get_xredden_template_model

model = get_xredden_template_model(ebv=0.3, freeze=True)
```

The E(B-V) reddening grid spans 10⁻⁴–10¹·⁵ mag and the energy grid covers
0.01–316 keV.

---

## Querying ISM parameters for a source

**Hydrogen column density (nH)**
- [Swift NH tool](https://www.swift.ac.uk/analysis/nhtot/)
- [HEASARC NH tool](https://heasarc.gsfc.nasa.gov/cgi-bin/Tools/w3nh/w3nh.pl)

**Galactic dust extinction E(B-V)**
- [IRSA Dust Query](https://irsa.ipac.caltech.edu/applications/DUST/)
- Python `dustmaps` package (SFD map):
  ```python
  from dustmaps.sfd import SFDQuery
  from astropy.coordinates import SkyCoord
  import astropy.units as u

  sfd = SFDQuery()
  coords = SkyCoord(ra * u.deg, dec * u.deg, frame="fk5")
  ebv = sfd(coords)
  ```

---

## Credits and acknowledgements

This subpackage was developed within the `gammapy-mwl` project. The following
people have contributed:

- **Fabio Acero** ([@facero](https://github.com/facero)) — development
- **Mireia Nievas Rosillo** ([@mireianievas](https://github.com/mireianievas)) — development

The implementation builds on ideas and code from
[`gammapyXray`](https://github.com/gammapy/gammapy-ogip-spectra):

- **Luca Giunti** ([@luca-giunti](https://github.com/luca-giunti)) — original `gammapyXray`
- **Régis Terrier** ([@registerrier](https://github.com/registerrier)) — original `gammapyXray`
- **Bruno Khélifi** ([@bkhelifi](https://github.com/bkhelifi)) — validation

The `SherpaSpectralModel` wrapper in [`sherpa.py`](sherpa.py) was inspired by
work from:

- **Katharina Egg** ([@k-egg](https://github.com/k-egg))
