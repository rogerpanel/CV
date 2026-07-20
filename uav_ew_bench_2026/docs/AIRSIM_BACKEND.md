# Full-fidelity AirSim / PX4 SITL backend

The `sim-lite` backend produces the published benchmark artifact by Monte-Carlo
sampling from the calibrated completion model. The `airsim` backend flies the
same corpus in a real 3-D simulator so the aggregate curves can be reproduced
from actual UAV dynamics, sensor streams, and attacks. This document explains
how to wire it up on a GPU workstation.

## 1. Prerequisites

* Ubuntu 20.04/22.04 or Windows 10/11 with an NVIDIA GPU (≥ 8 GB VRAM).
* **Unreal Engine 4.27** (AirSim's supported UE version).
* **AirSim** built with the Blocks or a custom landscape environment:
  <https://github.com/microsoft/AirSim>.
* **PX4-Autopilot** SITL (v1.14+): <https://github.com/PX4/PX4-Autopilot>.
* Python deps:

  ```bash
  pip install airsim msgpack-rpc-python
  ```

## 2. Simulator wiring

1. Launch the Unreal environment containing the three landscapes used by the
   three missions (mixed terrain, perimeter, search-and-rescue). Set
   `settings.json` to `"SimMode": "Multirotor"` and enable the GNSS, camera,
   and IMU sensors.
2. Start PX4 SITL and connect it to AirSim (MAVLink bridge). Confirm you can
   arm and take off with the stock `moveToPositionAsync` call.

## 3. Attack-injection hooks (the three `TODO(site)` points)

`uavbench/backends.py::AirSimBackend.fly` marks three integration points.
Implement them against your attack tooling:

* **GNSS spoofing at a target J/S.** Convert the requested `js_db` into a
  jammer power and drive your spoofing model (e.g. the cross-ambiguity /
  Seq2Seq references `borhani_2024_uav`, `aigner_2025_uav`). Corrupt the
  simulated GNSS fix stream that PX4 consumes.
* **PGD on the camera frames.** Intercept `simGetImages`, apply an ε = 8/255,
  20-step PGD perturbation to the RGB frame before it reaches the visual
  detector, per `lian_2022_uav`, `hickling_2023_uav`.
* **BIM on the DRL policy observations.** If the mission uses a DRL waypoint
  policy, apply a 10-step BIM perturbation to its observation vector.

Each hook should be a no-op when its flag in `config/benchmark.yaml`
(`adversary.*.enabled`) is false, so you can ablate attack channels.

## 4. DO-326A safe-completion label

A flight is `completed = True` iff **all** of:

1. reaches the target waypoint (‖pos − target‖ < tolerance),
2. no collision (`simGetCollisionInfo().has_collided == False`),
3. no loss of stabilisation (attitude within the airframe envelope for the
   whole flight),
4. certified course deviation within the M7 bound (read the certificate
   radius from your M7 module and check max cross-track error).

The skeleton implements (1) and (2); add (3) and (4) from your telemetry.

## 5. Run

```bash
# with the simulator running:
python scripts/generate_benchmark.py --backend airsim --out ./artifact_airsim \
    --airsim-ip 127.0.0.1
```

Because real flights are slow, start small: reduce `sampling.repeats_per_point`
and `ew_sweep.js_levels` in a copy of `benchmark.yaml`, validate the pipeline,
then scale up. The full 93,600-evaluation sweep is intended to run as a batch
job over several days on one workstation, or in parallel across several.

## 6. Cross-check against sim-lite

After an `airsim` run, compare its `report_points.csv` against the sim-lite
artifact. Agreement within the Wilson intervals is your validation that the
calibrated model and the 3-D simulator tell the same story.
