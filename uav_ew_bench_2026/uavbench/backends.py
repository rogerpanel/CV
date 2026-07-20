"""Execution backends.

Both backends expose one method::

    fly(flight, defense_id, js_db, rng) -> bool   # True == mission completed

* ``SimLiteBackend`` draws the completion outcome from the calibrated
  :class:`~uavbench.model.CompletionModel`.  Deterministic given the RNG.
* ``AirSimBackend`` drives a real AirSim / PX4 SITL flight, applies the
  adversarial contour to the sensor streams, and applies the DO-326A
  safe-completion label.  Requires the ``airsim`` package and a running
  simulator; imported lazily so sim-lite works with no extra dependencies.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from .config import BenchmarkConfig
from .corpus import Flight
from .model import AdversaryContour, CompletionModel


class Backend(Protocol):
    def fly(
        self, flight: Flight, defense_id: str, js_db: float, rng: np.random.Generator
    ) -> bool: ...


class SimLiteBackend:
    """Analytical Monte-Carlo backend (runs anywhere)."""

    name = "sim-lite"

    def __init__(self, cfg: BenchmarkConfig, model: CompletionModel):
        self.cfg = cfg
        self.model = model

    def fly(
        self, flight: Flight, defense_id: str, js_db: float, rng: np.random.Generator
    ) -> bool:
        p = self.model.probability(
            defense_id, js_db, mission=flight.mission, receiver=flight.receiver
        )
        return bool(rng.random() < p)


class AirSimBackend:
    """Full-fidelity AirSim / PX4 SITL backend.

    This is a functional driver skeleton against the AirSim Python API.  It
    is intentionally conservative: it wires up the flight, hands the
    adversarial contour to the sensor pipeline, and reads back a DO-326A
    completion label.  The site-specific pieces (Unreal environment name,
    PX4 connection string, attack injection hooks) are marked ``TODO`` and
    documented in ``docs/AIRSIM_BACKEND.md``.
    """

    name = "airsim"

    def __init__(
        self,
        cfg: BenchmarkConfig,
        contour: AdversaryContour,
        connection: str = "127.0.0.1",
    ):
        self.cfg = cfg
        self.contour = contour
        self.connection = connection
        self._client = None

    def _ensure_client(self):
        if self._client is not None:
            return
        try:
            import airsim  # noqa: F401
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "The airsim backend requires the 'airsim' package and a "
                "running AirSim/PX4 SITL simulator. Install with "
                "'pip install airsim msgpack-rpc-python' and see "
                "docs/AIRSIM_BACKEND.md."
            ) from exc
        import airsim

        client = airsim.MultirotorClient(ip=self.connection)
        client.confirmConnection()
        client.enableApiControl(True)
        client.armDisarm(True)
        self._client = client

    def fly(  # pragma: no cover - requires external simulator
        self, flight: Flight, defense_id: str, js_db: float, rng: np.random.Generator
    ) -> bool:
        self._ensure_client()
        import airsim

        client = self._client
        # 1) reset & take off ------------------------------------------------
        client.reset()
        client.enableApiControl(True)
        client.armDisarm(True)
        client.takeoffAsync().join()

        # 2) install the adversarial contour on the sensor streams ----------
        #    TODO(site): inject GNSS spoofing at the configured J/S, apply
        #    PGD perturbation to the camera frames, and BIM perturbation to
        #    the DRL policy observations.  See docs/AIRSIM_BACKEND.md.
        #    attack.install(client, js_db=js_db, contour=self.contour)

        # 3) fly the mission to the target waypoint under `defense_id` -------
        target = airsim.Vector3r(flight.waypoint_km * 1000.0, 0.0, -20.0)
        try:
            client.moveToPositionAsync(target.x_val, target.y_val, target.z_val, 8).join()
        except Exception:
            return False

        # 4) DO-326A safe-completion label ----------------------------------
        state = client.getMultirotorState()
        collided = client.simGetCollisionInfo().has_collided
        pos = state.kinematics_estimated.position
        reached = abs(pos.x_val - target.x_val) < 10.0 and abs(pos.y_val) < 10.0
        # stabilisation / course-deviation checks would read the certified
        # bound from the M7 certificate; omitted here (TODO(site)).
        return bool(reached and not collided)
