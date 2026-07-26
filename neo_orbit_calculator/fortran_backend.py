"""ctypes wrapper for the real128 Fortran orbit integrator."""

from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np

from .build_fortran import build
from .core import DAY_S, ForceModel, PERTURBER_KEYS, DE440Environment


class FortranIntegrator:
    def __init__(self, environment: DE440Environment):
        library_path = Path(__file__).resolve().parent / "fortran" / "libneo_integrator.so"
        source_path = Path(__file__).resolve().parent / "fortran" / "neo_integrator.f90"
        if not library_path.exists() or library_path.stat().st_mtime < source_path.stat().st_mtime:
            library_path = build()
        self.library = ctypes.CDLL(str(library_path))
        self.environment = environment
        self.library.neo_real_precision.restype = ctypes.c_int
        self.precision_digits = int(self.library.neo_real_precision())

    def propagate(
        self,
        initial_state: np.ndarray,
        epoch_et: float,
        output_seconds: np.ndarray,
        model: ForceModel,
        rtol: float,
        atol_position_km: float,
        atol_velocity_kms: float,
        max_step_days: float,
    ) -> tuple[np.ndarray, int]:
        initial = np.ascontiguousarray(initial_state, dtype=np.float64)
        times = np.ascontiguousarray(output_seconds, dtype=np.float64)
        gm = np.ascontiguousarray(
            [self.environment.gm[body] for body in PERTURBER_KEYS],
            dtype=np.float64,
        )
        zonal_radius = np.zeros(len(PERTURBER_KEYS), dtype=np.float64)
        zonal_coefficients = np.zeros(
            (3, len(PERTURBER_KEYS)),
            dtype=np.float64,
            order="F",
        )
        pole_ra = np.zeros(
            (3, len(PERTURBER_KEYS)),
            dtype=np.float64,
            order="F",
        )
        pole_dec = np.zeros(
            (3, len(PERTURBER_KEYS)),
            dtype=np.float64,
            order="F",
        )
        for index, body in enumerate(PERTURBER_KEYS):
            if body not in self.environment.zonal:
                continue
            zonal = self.environment.zonal[body]
            zonal_radius[index] = float(zonal["radius_km"])
            zonal_coefficients[:, index] = zonal["coefficients"]
            pole_ra[:, index] = zonal["pole_ra_deg"]
            pole_dec[:, index] = zonal["pole_dec_deg"]
        options = np.ascontiguousarray(
            [
                int(model.relativity_1pn),
                int(model.poynting_robertson),
                int(model.solar_wind_drag),
                int(model.planetary_zonal_harmonics),
                int(model.nongrav_law == "marsden"),
                int(model.full_multibody_1pn),
            ],
            dtype=np.int32,
        )
        parameters = np.ascontiguousarray(
            [
                model.area_mass_m2_kg,
                model.radiation_coefficient,
                model.solar_wind_density_cm3,
                model.solar_wind_speed_km_s,
                model.solar_wind_momentum_factor,
                model.a1_au_day2,
                model.a2_au_day2,
                model.a3_au_day2,
                model.outgassing_r0_au,
                model.outgassing_m,
                model.outgassing_n,
                model.outgassing_k,
                model.outgassing_alpha,
                model.outgassing_lag_days,
                0.0,
                0.0,
            ],
            dtype=np.float64,
        )
        output = np.empty((6, len(times)), dtype=np.float64, order="F")
        nfev = ctypes.c_int(0)
        status = ctypes.c_int(0)
        function = self.library.propagate_neo
        double_pointer = ctypes.POINTER(ctypes.c_double)
        int_pointer = ctypes.POINTER(ctypes.c_int)
        function.argtypes = [
            double_pointer,
            ctypes.POINTER(ctypes.c_double),
            double_pointer,
            ctypes.c_int,
            double_pointer,
            double_pointer,
            double_pointer,
            double_pointer,
            double_pointer,
            int_pointer,
            double_pointer,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            double_pointer,
            int_pointer,
            int_pointer,
        ]
        epoch = ctypes.c_double(epoch_et)
        rtol_value = ctypes.c_double(rtol)
        atol_position = ctypes.c_double(atol_position_km)
        atol_velocity = ctypes.c_double(atol_velocity_kms)
        max_step = ctypes.c_double(max_step_days * DAY_S)
        function(
            initial.ctypes.data_as(double_pointer),
            ctypes.byref(epoch),
            times.ctypes.data_as(double_pointer),
            len(times),
            gm.ctypes.data_as(double_pointer),
            zonal_radius.ctypes.data_as(double_pointer),
            zonal_coefficients.ctypes.data_as(double_pointer),
            pole_ra.ctypes.data_as(double_pointer),
            pole_dec.ctypes.data_as(double_pointer),
            options.ctypes.data_as(int_pointer),
            parameters.ctypes.data_as(double_pointer),
            ctypes.byref(rtol_value),
            ctypes.byref(atol_position),
            ctypes.byref(atol_velocity),
            ctypes.byref(max_step),
            output.ctypes.data_as(double_pointer),
            ctypes.byref(nfev),
            ctypes.byref(status),
        )
        if status.value != 0:
            raise RuntimeError(f"Fortran integrator failed with status {status.value}.")
        return output.T.copy(), nfev.value
