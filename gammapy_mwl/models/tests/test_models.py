import pytest
import numpy as np
from numpy.testing import assert_allclose
import astropy.units as u

from gammapy.modeling.models import PowerLawSpectralModel, SkyModel
from gammapy_mwl.models.sherpa import SherpaSpectralModel


def test_SherpaSpectralModel():
    sherpa = pytest.importorskip("sherpa")
    from sherpa.models import basic

    energy_grid = np.linspace(0.5, 10.0, 10) * u.keV
    plaw = basic.PowLaw1D()
    plaw.ampl = 1e-3
    plaw.gamma = 2

    #abs_model = sherpa.astro.xspec.XSwabs()
    #abs_model.nH = 5

    # Gammapy wrapper
    f1 = SherpaSpectralModel(plaw)
    #f2 = SherpaSpectralModel(abs_model, default_units=(u.keV, 1))
    f3 = f1  #* f2

    # Plain sherpa
    plaw_with_abs = plaw #* abs_model

    assert_allclose(f3(energy_grid).value[:-1], plaw_with_abs(energy_grid.value)[:-1])
    assert_allclose(f3.evaluate(energy_grid,2,1,1e-3,5).value[:-1], plaw_with_abs(energy_grid.value)[:-1])

    SkyModel(spectral_model=f3)  # Test evaluate on simple geom
    #with pytest.raises(AttributeError):
    #    SkyModel(spectral_model=f2)  # Wrong units, f2 is an absorption model


def test_SherpaSpectralModel_multicomponent():
    # test multicomponent wrapping of sherpa models
    #sherpa = pytest.importorskip("sherpa")

    energy_grid = np.linspace(0.5, 10.0, 10) * u.keV
    plaw = PowLaw1D()
    plaw.ampl = 1e-3
    plaw.gamma = 2

    #abs_model = sherpa.astro.xspec.XSwabs()
    #abs_model.nH = 5
    plaw2 = PowLaw1D()
    plaw2.ampl = 5e-3
    plaw2.gamma = 3

    # multicomponent sherpa model:

    sherpa_model = plaw + plaw2

    # Gammapy wrapper
    gammapy_model = SherpaSpectralModel(sherpa_model)

    assert_allclose(gammapy_model(energy_grid).value[:-1], sherpa_model(energy_grid.value)[:-1])
    assert_allclose(gammapy_model.evaluate(energy_grid,2,1,1e-3,5).value[:-1], sherpa_model(energy_grid.value)[:-1])

    m = SkyModel(spectral_model=gammapy_model)  # Test evaluate on simple geom

    # Check that parameter names are unique
    for i in m.parameters.names:
        if m.parameters.names.count(i) > 1:
            pytest.raises(AttributeError)