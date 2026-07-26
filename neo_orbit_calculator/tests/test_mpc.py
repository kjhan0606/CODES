from neo_orbit_calculator.mpc import summarize_mpc_ades


def test_summary_prefers_discovery_marker_over_precovery():
    rows = [
        {
            "Obstype": "optical",
            "obstime": "2014-01-01T00:00:00Z",
            "ra": "10.0",
            "dec": "20.0",
            "stn": "A00",
            "disc": None,
        },
        {
            "Obstype": "optical",
            "obstime": "2024-01-01T00:00:00Z",
            "ra": "11.0",
            "dec": "21.0",
            "stn": "B00",
            "disc": "*",
        },
    ]
    summary = summarize_mpc_ades(rows)
    assert summary["observation_count"] == 2
    assert summary["station_count"] == 2
    assert summary["first_observation_utc"].year == 2014
    assert summary["discovery_observation_utc"].year == 2024
    assert summary["discovery_marker_available"] is True


def test_summary_uses_first_observation_without_discovery_marker():
    rows = [
        {
            "Obstype": "optical",
            "obstime": "2024-01-02T00:00:00Z",
            "ra": "10.0",
            "dec": "20.0",
            "stn": "A00",
        }
    ]
    summary = summarize_mpc_ades(rows)
    assert (
        summary["discovery_observation_utc"]
        == summary["first_observation_utc"]
    )
    assert summary["discovery_marker_available"] is False
