"""Tests for simulator SQLite store — schema init, CRUD, joins."""

import pytest
from advisor.simulator.db import SimulatorStore
from advisor.simulator.models import CalibrationRecord, PCSCandidate, SimResult


@pytest.fixture
def store(tmp_path):
    """Create a store with a temp database."""
    db_path = tmp_path / "test_simulator.db"
    s = SimulatorStore(db_path=db_path)
    yield s
    s.close()


@pytest.fixture
def sample_chain_records():
    return [
        {
            "expiration": "2026-04-03",
            "strike": 45.0,
            "dte": 35,
            "bid": 0.80,
            "ask": 0.90,
            "mid": 0.85,
            "delta": -0.20,
            "gamma": 0.03,
            "theta": -0.05,
            "vega": 0.10,
            "iv": 0.35,
            "underlying_price": 50.0,
            "iv_rank": 55.0,
            "iv_percentile": 60.0,
        },
        {
            "expiration": "2026-04-03",
            "strike": 42.0,
            "dte": 35,
            "bid": 0.30,
            "ask": 0.40,
            "mid": 0.35,
            "delta": -0.10,
            "gamma": 0.02,
            "theta": -0.03,
            "vega": 0.06,
            "iv": 0.38,
            "underlying_price": 50.0,
            "iv_rank": 55.0,
            "iv_percentile": 60.0,
        },
    ]


@pytest.fixture
def sample_candidate():
    return PCSCandidate(
        symbol="TEST",
        expiration="2026-04-03",
        dte=35,
        short_strike=45.0,
        long_strike=42.0,
        width=3.0,
        short_bid=0.80,
        short_ask=0.90,
        long_bid=0.30,
        long_ask=0.40,
        net_credit=0.40,
        mid_credit=0.45,
        short_delta=-0.20,
        short_gamma=0.03,
        short_theta=-0.05,
        short_vega=0.10,
        short_iv=0.35,
        long_delta=-0.10,
        long_iv=0.38,
        underlying_price=50.0,
        iv_percentile=60.0,
        iv_rank=55.0,
        pop_estimate=0.80,
        sell_score=65.0,
        buying_power=260.0,
    )


@pytest.fixture
def sample_result():
    return SimResult(
        symbol="TEST",
        short_strike=45.0,
        long_strike=42.0,
        dte=35,
        net_credit=0.40,
        ev=12.50,
        pop=0.78,
        touch_prob=0.25,
        cvar_95=-180.0,
        stop_prob=0.08,
        avg_hold_days=22.5,
        ev_per_bp=0.048,
        pnl_p5=-200.0,
        pnl_p25=-50.0,
        pnl_p50=30.0,
        pnl_p75=40.0,
        pnl_p95=40.0,
        exit_profit_target=0.55,
        exit_stop_loss=0.08,
        exit_dte=0.22,
        exit_expiration=0.15,
    )


class TestSchemaInit:
    def test_tables_created(self, store):
        cursor = store._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = {row["name"] for row in cursor.fetchall()}
        assert "chain_snapshots" in tables
        assert "candidates" in tables
        assert "sim_results" in tables

    def test_indices_created(self, store):
        cursor = store._conn.execute("SELECT name FROM sqlite_master WHERE type='index'")
        indices = {row["name"] for row in cursor.fetchall()}
        assert "idx_snapshots_symbol" in indices
        assert "idx_candidates_symbol" in indices
        assert "idx_results_ev_bp" in indices

    def test_idempotent_init(self, tmp_path):
        """Creating store twice on same DB should not fail."""
        db_path = tmp_path / "test.db"
        s1 = SimulatorStore(db_path=db_path)
        s1.close()
        s2 = SimulatorStore(db_path=db_path)
        s2.close()


class TestChainSnapshots:
    def test_save_and_load(self, store, sample_chain_records):
        count = store.save_chain_snapshot(sample_chain_records, "TEST")
        assert count == 2

        loaded = store.get_chain_snapshots("TEST")
        assert len(loaded) == 2
        assert loaded[0]["symbol"] == "TEST"
        assert loaded[0]["strike"] in (45.0, 42.0)

    def test_empty_load(self, store):
        loaded = store.get_chain_snapshots("NONEXISTENT")
        assert loaded == []

    def test_limit(self, store, sample_chain_records):
        store.save_chain_snapshot(sample_chain_records, "TEST")
        loaded = store.get_chain_snapshots("TEST", limit=1)
        assert len(loaded) == 1


class TestCandidates:
    def test_save_batch(self, store, sample_candidate):
        ids = store.save_candidates_batch([sample_candidate])
        assert len(ids) == 1
        assert isinstance(ids[0], str)

    def test_save_multiple(self, store, sample_candidate):
        ids = store.save_candidates_batch([sample_candidate, sample_candidate])
        assert len(ids) == 2
        assert ids[0] != ids[1]


class TestSimResults:
    def test_save_result(self, store, sample_result):
        rid = store.save_sim_result(sample_result)
        assert isinstance(rid, str)
        assert len(rid) > 0

    def test_save_with_candidate_id(self, store, sample_candidate, sample_result):
        cand_ids = store.save_candidates_batch([sample_candidate])
        rid = store.save_sim_result(sample_result, candidate_id=cand_ids[0])
        assert isinstance(rid, str)


class TestTopResults:
    def test_join_query(self, store, sample_candidate, sample_result):
        cand_ids = store.save_candidates_batch([sample_candidate])
        sample_result.candidate_id = cand_ids[0]
        store.save_sim_result(sample_result, candidate_id=cand_ids[0])

        top = store.get_top_results(limit=10)
        assert len(top) == 1
        assert top[0]["symbol"] == "TEST"
        assert top[0]["ev_per_bp"] == 0.048

    def test_empty_results(self, store):
        top = store.get_top_results()
        assert top == []

    def test_ranking_order(self, store, sample_candidate):
        # Save two results with different EV/BP
        cand_ids = store.save_candidates_batch([sample_candidate, sample_candidate])

        r1 = SimResult(
            symbol="TEST",
            short_strike=45.0,
            long_strike=42.0,
            dte=35,
            net_credit=0.40,
            ev=10.0,
            pop=0.75,
            touch_prob=0.20,
            cvar_95=-150.0,
            stop_prob=0.05,
            avg_hold_days=20.0,
            ev_per_bp=0.038,
            pnl_p5=-150.0,
            pnl_p25=-30.0,
            pnl_p50=25.0,
            pnl_p75=40.0,
            pnl_p95=40.0,
            exit_profit_target=0.50,
            exit_stop_loss=0.05,
            exit_dte=0.25,
            exit_expiration=0.20,
        )
        r2 = SimResult(
            symbol="TEST",
            short_strike=45.0,
            long_strike=42.0,
            dte=35,
            net_credit=0.40,
            ev=20.0,
            pop=0.82,
            touch_prob=0.15,
            cvar_95=-100.0,
            stop_prob=0.03,
            avg_hold_days=18.0,
            ev_per_bp=0.077,
            pnl_p5=-100.0,
            pnl_p25=10.0,
            pnl_p50=35.0,
            pnl_p75=40.0,
            pnl_p95=40.0,
            exit_profit_target=0.60,
            exit_stop_loss=0.03,
            exit_dte=0.20,
            exit_expiration=0.17,
        )
        store.save_sim_result(r1, candidate_id=cand_ids[0])
        store.save_sim_result(r2, candidate_id=cand_ids[1])

        top = store.get_top_results(limit=10)
        assert len(top) == 2
        # Higher EV/BP should be first
        assert top[0]["ev_per_bp"] > top[1]["ev_per_bp"]


class TestCalibrationTracking:
    def test_save_and_compute_brier_perfect(self, store):
        """Perfect predictions should give Brier score = 0."""
        # Save records with perfect predictions
        for i in range(10):
            record = CalibrationRecord(
                candidate_id=f"cand_{i}",
                symbol="TEST",
                predicted_pop=1.0,
                predicted_touch=0.0,
                predicted_stop=0.0,
                predicted_ev=10.0,
            )
            store.save_calibration_record(record)
            # Fill in actuals that match predictions exactly
            store.update_calibration_outcome(
                candidate_id=f"cand_{i}",
                actual_profit=1.0,
                actual_touch=0.0,
                actual_stop=0.0,
                actual_pnl=10.0,
            )

        brier = store.compute_brier_scores("TEST")
        assert brier["n_samples"] == 10
        assert brier["pop_brier"] == pytest.approx(0.0, abs=1e-6)
        assert brier["touch_brier"] == pytest.approx(0.0, abs=1e-6)
        assert brier["stop_brier"] == pytest.approx(0.0, abs=1e-6)

    def test_worst_case_brier(self, store):
        """All-wrong predictions should give Brier score = 1."""
        for i in range(10):
            record = CalibrationRecord(
                candidate_id=f"bad_{i}",
                symbol="BAD",
                predicted_pop=1.0,
                predicted_touch=0.0,
                predicted_stop=0.0,
            )
            store.save_calibration_record(record)
            store.update_calibration_outcome(
                candidate_id=f"bad_{i}",
                actual_profit=0.0,  # Opposite of prediction
                actual_touch=1.0,
                actual_stop=1.0,
                actual_pnl=-100.0,
            )

        brier = store.compute_brier_scores("BAD")
        assert brier["n_samples"] == 10
        assert brier["pop_brier"] == pytest.approx(1.0, abs=1e-6)
        assert brier["touch_brier"] == pytest.approx(1.0, abs=1e-6)
        assert brier["stop_brier"] == pytest.approx(1.0, abs=1e-6)

    def test_empty_returns_none(self, store):
        """No calibration data should return n_samples=0."""
        brier = store.compute_brier_scores("EMPTY")
        assert brier["n_samples"] == 0
        assert brier["pop_brier"] is None

    def test_update_outcome(self, store):
        """Fill in actuals and verify they're reflected in query."""
        record = CalibrationRecord(
            candidate_id="upd_1",
            symbol="UPD",
            predicted_pop=0.80,
            predicted_touch=0.25,
            predicted_stop=0.10,
        )
        store.save_calibration_record(record)

        # Actuals should be None initially
        cursor = store._conn.execute(
            "SELECT actual_profit FROM calibration_tracking WHERE candidate_id = ?",
            ("upd_1",),
        )
        row = cursor.fetchone()
        assert row["actual_profit"] is None

        # Update
        store.update_calibration_outcome("upd_1", 1.0, 0.0, 0.0, 25.0)

        cursor = store._conn.execute(
            "SELECT actual_profit, actual_pnl FROM calibration_tracking WHERE candidate_id = ?",
            ("upd_1",),
        )
        row = cursor.fetchone()
        assert row["actual_profit"] == 1.0
        assert row["actual_pnl"] == 25.0
