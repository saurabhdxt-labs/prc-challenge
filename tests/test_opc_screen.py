"""OPC peer feature contract: no labels, no self, no cross-month/runway joins."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

P = Path(__file__).resolve().parents[1] / 'scripts/analysis/opc_screen.py'
SPEC = importlib.util.spec_from_file_location('opc_screen_test', P)
OPC = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(OPC)


def frame():
    times = pd.to_datetime(['2025-01-01T10:00Z', '2025-01-01T10:01Z', '2025-01-01T10:01Z',
                            '2025-01-01T10:02Z', '2025-01-01T12:00Z', '2025-02-01T10:01Z'])
    return pd.DataFrame({OPC.ID: np.arange(6), 'PHASE_mvt': 'DEP', 'ADEP_mvt': 'AAA',
                         'RUNWAY_mvt': ['01','01','02','01','01','01'], 'STAND_mvt': 'S1',
                         'MVT_TIME_UTC_mvt': times, 'AOBT_3_flt': times-pd.Timedelta(minutes=10),
                         'SCHED_TIME_UTC_mvt': times-pd.Timedelta(minutes=20),
                         'EOBT_1_flt': times-pd.Timedelta(minutes=12),
                         'LOBT_flt': times-pd.Timedelta(minutes=11)})


def test_strict_neighbors_ties_and_no_self():
    out = OPC.peer_features(frame()).set_index(OPC.ID)
    assert out.loc[1, 'opc_ap_prev1_gap'] == -60
    assert out.loc[1, 'opc_ap_next1_gap'] == 60
    assert out.loc[2, 'opc_ap_prev1_gap'] == -60
    assert out.loc[2, 'opc_ap_next1_gap'] == 60
    assert out.loc[0, 'opc_ap_next1_proxy'] == 600
    assert out.loc[0, 'opc_ap_next1_aobt_eobt'] == 120
    assert out.loc[0, 'opc_ap_next1_eobt_lobt'] == -60


def test_runway_month_and_distance_boundaries():
    out = OPC.peer_features(frame()).set_index(OPC.ID)
    assert np.isnan(out.loc[2, 'opc_rwy_prev1_gap'])
    assert np.isnan(out.loc[2, 'opc_rwy_next1_gap'])
    assert np.isnan(out.loc[5, 'opc_ap_prev1_gap'])
    assert np.isnan(out.loc[5, 'opc_ap_next1_gap'])
    assert np.isnan(out.loc[4, 'opc_ap_prev1_gap'])


def test_label_and_fractional_timestamp_invariance():
    a = frame()
    a['TAXITIME_SEC_mvt'] = 42
    a['BLOCK_TIME_UTC_mvt'] = a.MVT_TIME_UTC_mvt-pd.Timedelta(minutes=2)
    b = a.copy()
    b['TAXITIME_SEC_mvt'] = np.arange(len(b))*100000
    b['BLOCK_TIME_UTC_mvt'] = pd.Timestamp('1900-01-01', tz='UTC')
    for c in ['MVT_TIME_UTC_mvt','AOBT_3_flt','EOBT_1_flt','LOBT_flt','SCHED_TIME_UTC_mvt']:
        b[c] += pd.to_timedelta(np.arange(len(b))*1000, unit='us')
    pd.testing.assert_frame_equal(OPC.peer_features(a), OPC.peer_features(b))


def test_missing_stand_is_not_a_match():
    a = frame()
    a.loc[0, 'STAND_mvt'] = None
    out = OPC.peer_features(a).set_index(OPC.ID)
    assert np.isnan(out.loc[0, 'opc_ap_next1_same_stand'])
    assert len(out.columns) == 96
