"""Integrated desktop interface for CODES."""

from __future__ import annotations

import json
import shlex
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = (Path(__file__).resolve().parent / "output").resolve()


class CODESApplication(tk.Tk):
    """Run the CODES command-line modes from one responsive desktop window."""

    def __init__(self) -> None:
        super().__init__()
        self.title("CODES | Orbit Dynamics and Ephemeris")
        self.geometry("1120x900")
        self.minsize(960, 720)
        self.configure(bg="#0e1b18")
        self._buttons: list[ttk.Button] = []
        self._configure_style()
        self._build()

    def _configure_style(self) -> None:
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure(
            ".",
            background="#0e1b18",
            foreground="#f4f1e8",
            font=("DejaVu Sans", 11),
        )
        style.configure("TFrame", background="#0e1b18")
        style.configure("TLabel", background="#0e1b18", foreground="#f4f1e8")
        style.configure(
            "Section.TLabel",
            foreground="#56d6c2",
            font=("DejaVu Sans", 12, "bold"),
        )
        style.configure(
            "TButton",
            background="#24483e",
            foreground="#f4f1e8",
            padding=(10, 7),
            font=("DejaVu Sans", 10, "bold"),
        )
        style.map(
            "TButton",
            background=[("active", "#34705f"), ("disabled", "#263934")],
            foreground=[("disabled", "#7f918b")],
        )
        style.configure(
            "TCheckbutton",
            background="#0e1b18",
            foreground="#f4f1e8",
        )
        style.map(
            "TCheckbutton",
            background=[("active", "#0e1b18")],
            foreground=[("active", "#ffd166")],
        )
        style.configure(
            "TNotebook",
            background="#0e1b18",
            borderwidth=0,
        )
        style.configure(
            "TNotebook.Tab",
            background="#172923",
            foreground="#c9d4cf",
            padding=(16, 9),
            font=("DejaVu Sans", 10, "bold"),
        )
        style.map(
            "TNotebook.Tab",
            background=[("selected", "#34705f")],
            foreground=[("selected", "#ffffff")],
        )
        style.configure(
            "TEntry",
            fieldbackground="#f7f7f2",
            foreground="#101713",
            padding=5,
        )

    def _build(self) -> None:
        shell = ttk.Frame(self, padding=(24, 18, 24, 20))
        shell.pack(fill="both", expand=True)
        shell.columnconfigure(0, weight=1)
        shell.rowconfigure(2, weight=1)
        shell.rowconfigure(4, weight=1)

        header = ttk.Frame(shell)
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)
        ttk.Label(
            header,
            text="CODES",
            font=("DejaVu Sans", 23, "bold"),
            foreground="#ffd166",
        ).grid(row=0, column=0, sticky="w")
        ttk.Label(
            header,
            text="Close-approach Orbit Dynamics and Ephemeris System",
            font=("DejaVu Sans", 12),
            foreground="#f4f1e8",
        ).grid(row=1, column=0, sticky="w", pady=(1, 0))
        self.status = tk.StringVar(value="Ready")
        ttk.Label(
            header,
            textvariable=self.status,
            foreground="#56d6c2",
            font=("DejaVu Sans", 11, "bold"),
        ).grid(row=0, column=1, rowspan=2, sticky="e")

        ttk.Separator(shell).grid(row=1, column=0, sticky="ew", pady=(14, 12))

        notebook = ttk.Notebook(shell)
        notebook.grid(row=2, column=0, sticky="nsew")
        self._build_neo_tab(notebook)
        self._build_comet_tab(notebook)
        self._build_sky_tab(notebook)
        self._build_historical_tab(notebook)
        self._build_validation_tab(notebook)

        log_header = ttk.Frame(shell)
        log_header.grid(row=3, column=0, sticky="ew", pady=(13, 6))
        log_header.columnconfigure(0, weight=1)
        ttk.Label(
            log_header,
            text="Run log",
            style="Section.TLabel",
        ).grid(row=0, column=0, sticky="w")
        ttk.Button(
            log_header,
            text="Clear",
            command=lambda: self.log.delete("1.0", "end"),
        ).grid(row=0, column=1, sticky="e")

        log_frame = ttk.Frame(shell)
        log_frame.grid(row=4, column=0, sticky="nsew")
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        self.log = tk.Text(
            log_frame,
            height=11,
            bg="#172923",
            fg="#f4f1e8",
            insertbackground="#f4f1e8",
            selectbackground="#34705f",
            relief="flat",
            font=("DejaVu Sans Mono", 10),
            wrap="word",
            padx=12,
            pady=10,
        )
        scrollbar = ttk.Scrollbar(
            log_frame,
            orient="vertical",
            command=self.log.yview,
        )
        self.log.configure(yscrollcommand=scrollbar.set)
        self.log.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")

    def _new_tab(self, notebook: ttk.Notebook, title: str) -> ttk.Frame:
        tab = ttk.Frame(notebook, padding=(18, 16))
        notebook.add(tab, text=title)
        tab.columnconfigure(1, weight=1)
        tab.columnconfigure(3, weight=1)
        return tab

    @staticmethod
    def _field(
        parent: ttk.Frame,
        label: str,
        variable: tk.StringVar,
        row: int,
        column: int,
        width: int = 24,
    ) -> None:
        ttk.Label(parent, text=label).grid(
            row=row,
            column=column,
            sticky="w",
            padx=(0, 8),
            pady=5,
        )
        ttk.Entry(parent, textvariable=variable, width=width).grid(
            row=row,
            column=column + 1,
            sticky="ew",
            padx=(0, 20),
            pady=5,
        )

    def _output_field(
        self,
        parent: ttk.Frame,
        variable: tk.StringVar,
        row: int,
        columnspan: int = 4,
    ) -> None:
        ttk.Label(parent, text="Output directory").grid(
            row=row,
            column=0,
            sticky="w",
            pady=5,
        )
        ttk.Entry(parent, textvariable=variable).grid(
            row=row,
            column=1,
            columnspan=columnspan - 2,
            sticky="ew",
            padx=(0, 8),
            pady=5,
        )
        ttk.Button(
            parent,
            text="Choose",
            command=lambda: self._choose_output(variable),
        ).grid(row=row, column=columnspan - 1, sticky="ew", pady=5)

    def _action_button(
        self,
        parent: ttk.Frame,
        text: str,
        command: object,
        row: int,
        column: int,
        columnspan: int = 1,
    ) -> None:
        button = ttk.Button(parent, text=text, command=command)
        button.grid(
            row=row,
            column=column,
            columnspan=columnspan,
            sticky="ew",
            padx=(0, 10),
            pady=(14, 4),
        )
        self._buttons.append(button)

    def _build_neo_tab(self, notebook: ttk.Notebook) -> None:
        tab = self._new_tab(notebook, "NEO dynamics")
        self.neo = {
            "designation": tk.StringVar(value="99942"),
            "start": tk.StringVar(value="2026-01-01"),
            "stop": tk.StringVar(value="2126-01-01"),
            "samples": tk.StringVar(value="401"),
            "clones": tk.StringVar(value="100"),
            "seed": tk.StringVar(value="42"),
            "area_mass": tk.StringVar(value="0"),
            "wind_density": tk.StringVar(value="5.0"),
            "wind_speed": tk.StringVar(value="400.0"),
            "a1": tk.StringVar(value="0"),
            "a2": tk.StringVar(value="0"),
            "a3": tk.StringVar(value="0"),
            "nongrav_law": tk.StringVar(value="inverse_square"),
            "outgassing_lag": tk.StringVar(value="0.0"),
            "output": tk.StringVar(value=str(DEFAULT_OUTPUT / "neo")),
        }
        ttk.Label(
            tab,
            text="Authoritative JPL products and local force-model tests",
            style="Section.TLabel",
        ).grid(row=0, column=0, columnspan=4, sticky="w", pady=(0, 8))
        self._field(tab, "Designation", self.neo["designation"], 1, 0)
        self._field(tab, "Samples", self.neo["samples"], 1, 2)
        self._field(tab, "Start epoch, TDB", self.neo["start"], 2, 0)
        self._field(tab, "Stop epoch, TDB", self.neo["stop"], 2, 2)
        self._field(
            tab,
            "Area / mass [m2 kg-1]",
            self.neo["area_mass"],
            3,
            0,
        )
        self._field(tab, "A1 [au d-2]", self.neo["a1"], 3, 2)
        self._field(tab, "A2 [au d-2]", self.neo["a2"], 4, 0)
        self._field(tab, "A3 [au d-2]", self.neo["a3"], 4, 2)
        self._field(
            tab,
            "Solar-wind density [cm-3]",
            self.neo["wind_density"],
            5,
            0,
        )
        self._field(
            tab,
            "Solar-wind speed [km s-1]",
            self.neo["wind_speed"],
            5,
            2,
        )
        ttk.Label(tab, text="Non-gravitational law").grid(
            row=6,
            column=0,
            sticky="w",
            padx=(0, 8),
            pady=5,
        )
        ttk.Combobox(
            tab,
            textvariable=self.neo["nongrav_law"],
            values=("inverse_square", "marsden"),
            state="readonly",
            width=22,
        ).grid(
            row=6,
            column=1,
            sticky="ew",
            padx=(0, 20),
            pady=5,
        )
        self._field(
            tab,
            "Outgassing lag [day]",
            self.neo["outgassing_lag"],
            6,
            2,
        )
        self._field(tab, "Covariance clones", self.neo["clones"], 7, 0)
        self._field(tab, "Random seed", self.neo["seed"], 7, 2)
        self._output_field(tab, self.neo["output"], 8)

        self.relativity = tk.BooleanVar(value=True)
        self.pr_drag = tk.BooleanVar(value=True)
        self.solar_wind = tk.BooleanVar(value=True)
        self.zonal_harmonics = tk.BooleanVar(value=True)
        self.large_asteroids = tk.BooleanVar(value=True)
        self.jupiter_system = tk.BooleanVar(value=False)
        checks = (
            ("Full multi-body 1PN", self.relativity),
            ("Poynting-Robertson drag", self.pr_drag),
            ("Solar-wind drag", self.solar_wind),
            ("Planetary J2/J4/J6", self.zonal_harmonics),
        )
        for index, (label, variable) in enumerate(checks):
            ttk.Checkbutton(tab, text=label, variable=variable).grid(
                row=9,
                column=index,
                sticky="w",
                pady=(9, 0),
            )
        ttk.Checkbutton(
            tab,
            text="16 large asteroids",
            variable=self.large_asteroids,
        ).grid(row=10, column=0, sticky="w", pady=(5, 0))
        ttk.Checkbutton(
            tab,
            text="Resolved Jupiter system",
            variable=self.jupiter_system,
        ).grid(row=10, column=1, sticky="w", pady=(5, 0))

        self._action_button(
            tab,
            "Download JPL SPK",
            lambda: self._run(self._neo_command("spk"), "JPL SPK download"),
            11,
            0,
        )
        self._action_button(
            tab,
            "Fetch Horizons vectors",
            lambda: self._run(
                self._neo_command("vectors"),
                "Horizons vector retrieval",
            ),
            11,
            1,
        )
        self._action_button(
            tab,
            "Run local propagation",
            lambda: self._run(
                self._neo_command("propagate"),
                "local NEO propagation",
            ),
            11,
            2,
        )
        self._action_button(
            tab,
            "Run covariance ensemble",
            lambda: self._run(
                self._neo_command("virtual-asteroids"),
                "virtual-asteroid covariance propagation",
            ),
            11,
            3,
        )

    def _build_comet_tab(self, notebook: ttk.Notebook) -> None:
        tab = self._new_tab(notebook, "Comet evolution")
        self.comet = {
            "designation": tk.StringVar(value="1P"),
            "start_year": tk.StringVar(value="800"),
            "stop_year": tk.StringVar(value="2100"),
            "return_years": tk.StringVar(value=""),
            "output": tk.StringVar(value=str(DEFAULT_OUTPUT / "comet_orbits")),
        }
        ttk.Label(
            tab,
            text="Apparition-to-apparition orbital-element history",
            style="Section.TLabel",
        ).grid(row=0, column=0, columnspan=4, sticky="w", pady=(0, 8))
        self._field(tab, "Comet designation", self.comet["designation"], 1, 0)
        self._field(tab, "Start year", self.comet["start_year"], 1, 2)
        self._field(tab, "Stop year", self.comet["stop_year"], 2, 0)
        self._field(
            tab,
            "Known return years",
            self.comet["return_years"],
            2,
            2,
        )
        ttk.Label(
            tab,
            text="For non-Halley objects, enter space-separated perihelion years when JPL aliases are incomplete.",
            foreground="#c9d4cf",
        ).grid(row=3, column=0, columnspan=4, sticky="w", pady=(5, 9))
        self._output_field(tab, self.comet["output"], 4)
        self._action_button(
            tab,
            "Retrieve returns and plot orbit evolution",
            lambda: self._run(
                self._comet_command(),
                "comet apparition analysis",
            ),
            5,
            0,
            columnspan=4,
        )

    def _build_sky_tab(self, notebook: ttk.Notebook) -> None:
        tab = self._new_tab(notebook, "Sky positions")
        self.sky = {
            "designation": tk.StringVar(value="1P"),
            "observer": tk.StringVar(value="500@399"),
            "output": tk.StringVar(value=str(DEFAULT_OUTPUT / "comet_sky")),
        }
        ttk.Label(
            tab,
            text="Apparent coordinates, distance, and IAU constellation",
            style="Section.TLabel",
        ).grid(row=0, column=0, columnspan=4, sticky="w", pady=(0, 8))
        self._field(tab, "Comet designation", self.sky["designation"], 1, 0)
        self._field(tab, "Horizons observer", self.sky["observer"], 1, 2)
        ttk.Label(tab, text="UTC epochs").grid(
            row=2,
            column=0,
            sticky="nw",
            pady=5,
        )
        self.sky_epochs = tk.Text(
            tab,
            height=5,
            bg="#f7f7f2",
            fg="#101713",
            insertbackground="#101713",
            relief="flat",
            font=("DejaVu Sans Mono", 10),
            padx=8,
            pady=6,
        )
        self.sky_epochs.insert(
            "1.0",
            "2061-07-28\n2061-08-15\n2061-09-01",
        )
        self.sky_epochs.grid(
            row=2,
            column=1,
            columnspan=3,
            sticky="ew",
            pady=5,
        )
        self._output_field(tab, self.sky["output"], 3)
        self._action_button(
            tab,
            "Retrieve and plot sky positions",
            lambda: self._run(
                self._sky_command(),
                "comet sky-position retrieval",
            ),
            4,
            0,
            columnspan=4,
        )

    def _build_historical_tab(self, notebook: ttk.Notebook) -> None:
        tab = self._new_tab(notebook, "Historical identification")
        self.historical = {
            "designation": tk.StringVar(value="1P"),
            "record": tk.StringVar(value="joseon-halley-1759-03-11"),
            "epoch": tk.StringVar(value="1759-04-06T20:30:00"),
            "span_days": tk.StringVar(value="4"),
            "samples": tk.StringVar(value="17"),
            "apparition_record": tk.StringVar(value=""),
            "longitude": tk.StringVar(value="126.9780"),
            "latitude": tk.StringVar(value="37.5665"),
            "elevation": tk.StringVar(value="0.05"),
            "field_radius": tk.StringVar(value="12"),
            "output": tk.StringVar(
                value=str(DEFAULT_OUTPUT / "historical_comet")
            ),
        }
        ttk.Label(
            tab,
            text="Historical sky chart and record-constraint comparison",
            style="Section.TLabel",
        ).grid(row=0, column=0, columnspan=4, sticky="w", pady=(0, 8))
        self._field(
            tab,
            "Comet designation",
            self.historical["designation"],
            1,
            0,
        )
        ttk.Label(tab, text="Bundled record").grid(
            row=1,
            column=2,
            sticky="w",
            padx=(0, 8),
            pady=5,
        )
        ttk.Combobox(
            tab,
            textvariable=self.historical["record"],
            values=("", "joseon-halley-1759-03-11"),
            state="readonly",
            width=27,
        ).grid(
            row=1,
            column=3,
            sticky="ew",
            padx=(0, 20),
            pady=5,
        )
        self._field(
            tab,
            "Center epoch, UTC",
            self.historical["epoch"],
            2,
            0,
        )
        self._field(
            tab,
            "JPL apparition record",
            self.historical["apparition_record"],
            2,
            2,
        )
        self._field(
            tab,
            "Track span [day]",
            self.historical["span_days"],
            3,
            0,
        )
        self._field(
            tab,
            "Track samples",
            self.historical["samples"],
            3,
            2,
        )
        self._field(
            tab,
            "Longitude east [deg]",
            self.historical["longitude"],
            4,
            0,
        )
        self._field(
            tab,
            "Latitude [deg]",
            self.historical["latitude"],
            4,
            2,
        )
        self._field(
            tab,
            "Elevation [km]",
            self.historical["elevation"],
            5,
            0,
        )
        self._field(
            tab,
            "Chart radius [deg]",
            self.historical["field_radius"],
            5,
            2,
        )
        ttk.Label(
            tab,
            text=(
                "The bundled Joseon record selects the 1759 Halley "
                "apparition and Hanyang observing metadata. Clear the "
                "record field to use the manual target, epoch, and site."
            ),
            wraplength=860,
            justify="left",
            foreground="#c9d4cf",
        ).grid(row=6, column=0, columnspan=4, sticky="w", pady=(7, 5))
        self._output_field(tab, self.historical["output"], 7)
        self._action_button(
            tab,
            "Generate historical finder chart",
            lambda: self._run(
                self._historical_command(),
                "historical comet identification",
            ),
            8,
            0,
            columnspan=4,
        )

    def _build_validation_tab(self, notebook: ttk.Notebook) -> None:
        tab = self._new_tab(notebook, "Validation")
        ttk.Label(
            tab,
            text="Reproducibility checks against NASA/JPL and historical records",
            style="Section.TLabel",
        ).grid(row=0, column=0, columnspan=4, sticky="w", pady=(0, 12))
        ttk.Label(
            tab,
            text=(
                "The NEO suite compares ten close approaches with official "
                "JPL CNEOS results. The Halley suite compares apparition "
                "solutions and historical perihelion timing."
            ),
            wraplength=860,
            justify="left",
            foreground="#c9d4cf",
        ).grid(row=1, column=0, columnspan=4, sticky="w", pady=(0, 10))
        self._action_button(
            tab,
            "Run ten-object NEO validation",
            lambda: self._run(
                [sys.executable, "validate_neo_orbits.py"],
                "ten-object NEO validation",
            ),
            2,
            0,
            columnspan=2,
        )
        self._action_button(
            tab,
            "Run Halley return validation",
            lambda: self._run(
                [sys.executable, "validate_halley_returns.py"],
                "Halley return validation",
            ),
            2,
            2,
            columnspan=2,
        )

    def _choose_output(self, variable: tk.StringVar) -> None:
        initial = variable.get() or str(DEFAULT_OUTPUT)
        chosen = filedialog.askdirectory(initialdir=initial)
        if chosen:
            variable.set(chosen)

    def _neo_command(self, mode: str) -> list[str]:
        if mode == "virtual-asteroids":
            command = [
                sys.executable,
                "-m",
                "neo_orbit_calculator.cli",
                mode,
                self.neo["designation"].get().strip(),
                "--stop",
                self.neo["stop"].get().strip(),
                "--clones",
                self.neo["clones"].get().strip(),
                "--samples",
                self.neo["samples"].get().strip(),
                "--seed",
                self.neo["seed"].get().strip(),
                "--output-dir",
                self.neo["output"].get().strip(),
                "--kernel-dir",
                str(Path(__file__).resolve().parent / "kernels"),
            ]
            if not self.relativity.get():
                command.append("--no-relativity")
            if not self.zonal_harmonics.get():
                command.append("--no-zonal-harmonics")
            if not self.large_asteroids.get():
                command.append("--major-bodies-only")
            if self.jupiter_system.get():
                command.append("--jupiter-system")
            return command
        command = [
            sys.executable,
            "-m",
            "neo_orbit_calculator.cli",
            mode,
            self.neo["designation"].get().strip(),
            "--start",
            self.neo["start"].get().strip(),
            "--stop",
            self.neo["stop"].get().strip(),
        ]
        output_dir = self.neo["output"].get().strip()
        if mode == "spk":
            return command + ["--output-dir", output_dir]
        command += ["--samples", self.neo["samples"].get().strip()]
        if mode == "vectors":
            return command + [
                "--output",
                str(Path(output_dir) / "horizons_vectors.csv"),
            ]
        command += [
            "--output-dir",
            output_dir,
            "--kernel-dir",
            str(Path(__file__).resolve().parent / "kernels"),
            "--area-mass",
            self.neo["area_mass"].get().strip(),
            "--solar-wind-density",
            self.neo["wind_density"].get().strip(),
            "--solar-wind-speed",
            self.neo["wind_speed"].get().strip(),
            "--a1",
            self.neo["a1"].get().strip(),
            "--a2",
            self.neo["a2"].get().strip(),
            "--a3",
            self.neo["a3"].get().strip(),
            "--nongrav-law",
            self.neo["nongrav_law"].get().strip(),
            "--outgassing-lag-days",
            self.neo["outgassing_lag"].get().strip(),
            "--backend",
            "fortran",
        ]
        if not self.relativity.get():
            command.append("--no-relativity")
        if not self.pr_drag.get():
            command.append("--no-pr")
        if not self.solar_wind.get():
            command.append("--no-solar-wind")
        if not self.zonal_harmonics.get():
            command.append("--no-zonal-harmonics")
        if not self.large_asteroids.get():
            command.append("--major-bodies-only")
        if self.jupiter_system.get():
            command.append("--jupiter-system")
        return command

    def _comet_command(self) -> list[str]:
        command = [
            sys.executable,
            "-m",
            "neo_orbit_calculator.cli",
            "comet-orbits",
            self.comet["designation"].get().strip(),
            "--start-year",
            self.comet["start_year"].get().strip(),
            "--stop-year",
            self.comet["stop_year"].get().strip(),
            "--output-dir",
            self.comet["output"].get().strip(),
        ]
        years = self.comet["return_years"].get().split()
        if years:
            command.extend(["--return-years", *years])
        return command

    def _sky_command(self) -> list[str]:
        epochs = self.sky_epochs.get("1.0", "end").replace(",", " ").split()
        if not epochs:
            raise ValueError("At least one UTC epoch is required.")
        return [
            sys.executable,
            "-m",
            "neo_orbit_calculator.cli",
            "comet-sky",
            self.sky["designation"].get().strip(),
            "--epochs",
            *epochs,
            "--observer",
            self.sky["observer"].get().strip(),
            "--output-dir",
            self.sky["output"].get().strip(),
        ]

    def _historical_command(self) -> list[str]:
        command = [
            sys.executable,
            "-m",
            "neo_orbit_calculator.cli",
            "historical-comet",
            self.historical["designation"].get().strip(),
        ]
        record = self.historical["record"].get().strip()
        if record:
            command.extend(["--record", record])
        else:
            command.extend(
                [
                    "--epoch",
                    self.historical["epoch"].get().strip(),
                    "--observer-lon",
                    self.historical["longitude"].get().strip(),
                    "--observer-lat",
                    self.historical["latitude"].get().strip(),
                    "--observer-elevation-km",
                    self.historical["elevation"].get().strip(),
                ]
            )
        apparition_record = self.historical["apparition_record"].get().strip()
        if apparition_record:
            command.extend(["--apparition-record", apparition_record])
        command.extend(
            [
                "--span-days",
                self.historical["span_days"].get().strip(),
                "--samples",
                self.historical["samples"].get().strip(),
                "--field-radius",
                self.historical["field_radius"].get().strip(),
                "--output-dir",
                self.historical["output"].get().strip(),
            ]
        )
        return command

    def _run(self, command: list[str], label: str) -> None:
        if any(not token for token in command):
            messagebox.showerror(
                "Missing input",
                "Every required field must contain a value.",
            )
            return
        self._set_running(True, label)
        self.log.insert(
            "end",
            f"\n$ {shlex.join(command)}\n",
        )
        self.log.see("end")

        def worker() -> None:
            result = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                text=True,
                capture_output=True,
            )
            self.after(0, lambda: self._finish(result, label))

        threading.Thread(target=worker, daemon=True).start()

    def _set_running(self, running: bool, label: str = "") -> None:
        self.status.set(f"Running {label}..." if running else "Ready")
        state = "disabled" if running else "normal"
        for button in self._buttons:
            button.configure(state=state)

    def _finish(
        self,
        result: subprocess.CompletedProcess[str],
        label: str,
    ) -> None:
        self._set_running(False)
        text = result.stdout if result.returncode == 0 else result.stderr
        try:
            text = json.dumps(json.loads(text), indent=2)
        except (json.JSONDecodeError, TypeError):
            pass
        self.log.insert("end", text.rstrip() + "\n")
        self.log.see("end")
        if result.returncode == 0:
            self.status.set(f"Completed {label}")
            return
        self.status.set(f"Failed {label}")
        messagebox.showerror("CODES calculation failed", text[-1600:])


# Compatibility for code that imported the former GUI class.
OrbitCalculator = CODESApplication


def main() -> None:
    CODESApplication().mainloop()


if __name__ == "__main__":
    main()
