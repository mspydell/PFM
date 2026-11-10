"""Folders of the shared QC code, found next to this folder.

In the PFM repository: ../ModelQC/Moorings/common (formerly ../Hydro_QC/Moorings/common) and ../ddPCR_ENT;
in the home directory (~/modelQC_DA next to ~/modelQC): ../modelQC/Moorings/common and
../modelQC/Dye_vs_ddPCRENT.
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def _first(*options):
    for parts in options:
        p = os.path.normpath(os.path.join(HERE, '..', *parts))
        if os.path.isdir(p):
            return p
    raise SystemExit(f'none of these folders exist: {options}')


# mooring obs and model code (extract_model.py, plot_model_vs_obs.py)
MOORINGS_COMMON = _first(('ModelQC', 'Moorings', 'common'), ('Hydro_QC', 'Moorings', 'common'),
                         ('modelQC', 'Moorings', 'common'))
# dye vs ddPCR code (plot_dye_ddpcr.py, station_dye.py)
DYE_DDPCR = _first(('ddPCR_ENT',), ('modelQC', 'Dye_vs_ddPCRENT'))
