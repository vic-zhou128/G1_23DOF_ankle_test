"""Focused DDS transport and recorder for the G1 ankle semantics experiment.

This module intentionally has no RL policy, InEKF, DCM, or MuJoCo dependency.
"""
from __future__ import annotations

from collections import deque
import csv
from dataclasses import dataclass
from enum import Enum
import json
import math
import os
from pathlib import Path
import queue
import threading
import time
import traceback

import numpy as np

try:
    from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
    from unitree_sdk2py.core.channel import (ChannelFactoryInitialize, ChannelPublisher,
                                             ChannelSubscriber)
    from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import IMUState_, LowCmd_, LowState_
    from unitree_sdk2py.utils.crc import CRC
except ImportError as exc:
    raise SystemExit(
        "Unitree SDK is unavailable. Install unitree_sdk2_python and CycloneDDS "
        "in the active Python environment; see README.md.\n"
        f"Import error: {exc}"
    ) from exc

try:
    from .core import MODE_MACHINE, MODE_PR, MODEL_MOTOR_COUNT
except ImportError:  # Direct execution from this directory.
    from core import MODE_MACHINE, MODE_PR, MODEL_MOTOR_COUNT


def json_ready(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Enum):
        return value.value
    raise TypeError(f"cannot encode {type(value).__name__}")


def uint8_array(value, expected=None):
    if isinstance(value, (bytes, bytearray, memoryview)):
        result = np.frombuffer(value, dtype=np.uint8).copy()
    else:
        result = np.asarray(value, dtype=np.uint8).copy()
    if expected is not None and result.shape != (expected,):
        raise ValueError(f"expected {expected} uint8 values, got {result.shape}")
    return result


def remote_lt_b(wireless_remote):
    data = uint8_array(wireless_remote)
    return len(data) >= 4 and bool(data[2] & (1 << 5)) and bool(data[3] & (1 << 1))


@dataclass(frozen=True)
class StateSnapshot:
    sequence: int
    monotonic_ns: int
    wall_ns: int
    tick: int
    mode_pr: int
    mode_machine: int
    q: np.ndarray
    dq: np.ndarray
    ddq: np.ndarray
    tau_est: np.ndarray
    motor_mode: np.ndarray
    motorstate: np.ndarray
    temperature: np.ndarray
    wireless_remote: np.ndarray


@dataclass(frozen=True)
class LowCmdSnapshot:
    sequence: int
    monotonic_ns: int
    wall_ns: int
    mode_pr: int
    mode_machine: int
    motor_mode: np.ndarray
    q: np.ndarray
    dq: np.ndarray
    tau: np.ndarray
    kp: np.ndarray
    kd: np.ndarray
    motor_reserve: np.ndarray
    reserve: np.ndarray
    crc_valid: bool


class Recorder:
    """Nonblocking, chunked recorder for every input and output of the test."""

    CHUNK_SIZES = {"state": 5000, "secondary_imu": 5000,
                   "observed_lowcmd": 5000, "command": 5000, "analysis": 500}

    def __init__(self, root, run_name, metadata, queue_capacity=10000):
        self.run_dir = Path(root).resolve() / run_name
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self.metadata = dict(metadata)
        self.metadata.update(run_name=run_name, created_wall_ns=time.time_ns())
        (self.run_dir / "metadata.json").write_text(
            json.dumps(self.metadata, indent=2, ensure_ascii=False,
                       default=json_ready) + "\n")
        self.queue = queue.Queue(maxsize=queue_capacity)
        self.failed = threading.Event()
        self.failure_reason = ""
        self.counts = {name: 0 for name in self.CHUNK_SIZES}
        self.dropped = 0
        self._thread = threading.Thread(target=self._run, name="ankle-recorder", daemon=True)
        self._thread.start()

    def submit(self, stream, record):
        if stream not in self.CHUNK_SIZES:
            raise ValueError(f"unknown recorder stream {stream}")
        try:
            self.queue.put_nowait((stream, record))
            return True
        except queue.Full:
            self.dropped += 1
            self.fail("recorder queue overflow")
            return False

    def event(self, kind, **values):
        record = {"monotonic_ns": time.monotonic_ns(), "wall_ns": time.time_ns(),
                  "kind": kind, **values}
        try:
            self.queue.put_nowait(("__event__", record))
        except queue.Full:
            self.dropped += 1
            self.fail("recorder queue overflow while writing an event")

    def fail(self, reason):
        if not self.failed.is_set():
            self.failure_reason = str(reason)
            self.failed.set()

    @staticmethod
    def _arrays(records):
        if any(record.keys() != records[0].keys() for record in records):
            raise ValueError("recorder schema changed inside a chunk")
        result = {}
        for key in records[0]:
            values = [record[key] for record in records]
            result[key] = (np.stack(values) if isinstance(values[0], np.ndarray)
                           else np.asarray(values))
        return result

    def _flush(self, stream, records, chunk_index):
        if not records:
            return
        destination = self.run_dir / f"{stream}_{chunk_index:06d}.npz"
        temporary = self.run_dir / f".{stream}_{chunk_index:06d}.tmp.npz"
        np.savez(temporary, **self._arrays(records))
        os.replace(temporary, destination)
        self.counts[stream] += len(records)

    def _run(self):
        buffers = {name: [] for name in self.CHUNK_SIZES}
        chunks = {name: 0 for name in self.CHUNK_SIZES}
        event_file = (self.run_dir / "events.jsonl").open("a", buffering=1)
        csv_file = (self.run_dir / "analysis.csv").open("w", newline="", buffering=1)
        csv_writer = None
        try:
            while True:
                item = self.queue.get()
                if item is None:
                    break
                stream, record = item
                if stream == "__event__":
                    event_file.write(json.dumps(record, ensure_ascii=False,
                                                default=json_ready) + "\n")
                    continue
                if stream == "analysis":
                    if csv_writer is None:
                        csv_writer = csv.DictWriter(csv_file, fieldnames=list(record))
                        csv_writer.writeheader()
                    csv_writer.writerow(record)
                buffers[stream].append(record)
                if len(buffers[stream]) >= self.CHUNK_SIZES[stream]:
                    self._flush(stream, buffers[stream], chunks[stream])
                    buffers[stream].clear()
                    chunks[stream] += 1
            for stream, records in buffers.items():
                self._flush(stream, records, chunks[stream])
        except Exception as exc:
            self.fail(f"recorder writer failed: {exc}")
            traceback.print_exc()
        finally:
            event_file.close()
            csv_file.close()

    def close(self, outcome, extra=None):
        try:
            self.queue.put(None, timeout=2.0)
        except queue.Full:
            self.fail("could not stop recorder queue cleanly")
        self._thread.join(timeout=30.0)
        if self._thread.is_alive():
            self.fail("recorder writer did not stop")
        summary = {"outcome": outcome, "counts": self.counts, "dropped": self.dropped,
                   "failed": self.failed.is_set(), "failure_reason": self.failure_reason,
                   "closed_wall_ns": time.time_ns()}
        if extra:
            summary.update(extra)
        (self.run_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False,
                       default=json_ready) + "\n")


class HardwareTransport:
    def __init__(self, network, recorder):
        ChannelFactoryInitialize(0, network)
        self.recorder = recorder
        self.crc = CRC()
        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)
        self._latest = None
        self.state_history = deque(maxlen=6000)
        self.lowcmd_history = deque(maxlen=6000)
        self.invalid_state_times = deque(maxlen=6000)
        self.sequence = 0
        self.valid_times = deque(maxlen=5000)
        self.invalid_count = 0
        self.existing_lowcmd_count = 0
        self.publisher = None
        self.state_sub = ChannelSubscriber("rt/lowstate", LowState_)
        self.state_sub.Init(self._on_state, 20)
        self.secondary_sub = ChannelSubscriber("rt/secondary_imu", IMUState_)
        self.secondary_sub.Init(self._on_secondary_imu, 20)
        self.lowcmd_watch = ChannelSubscriber("rt/lowcmd", LowCmd_)
        self.lowcmd_watch.Init(self._on_existing_lowcmd, 10)

    def _on_existing_lowcmd(self, sample):
        now = time.monotonic_ns()
        sequence = self.existing_lowcmd_count
        self.existing_lowcmd_count += 1
        motors = sample.motor_cmd
        computed_crc = int(self.crc.Crc(sample))
        raw = dict(
            sequence=sequence, monotonic_ns=now, wall_ns=time.time_ns(),
            mode_pr=int(sample.mode_pr), mode_machine=int(sample.mode_machine),
            motor_mode=np.asarray([m.mode for m in motors], dtype=np.uint8),
            motor_q=np.asarray([m.q for m in motors], dtype=np.float32),
            motor_dq=np.asarray([m.dq for m in motors], dtype=np.float32),
            motor_tau=np.asarray([m.tau for m in motors], dtype=np.float32),
            motor_kp=np.asarray([m.kp for m in motors], dtype=np.float32),
            motor_kd=np.asarray([m.kd for m in motors], dtype=np.float32),
            motor_reserve=np.asarray([m.reserve for m in motors], dtype=np.uint32),
            reserve=np.asarray(sample.reserve, dtype=np.uint32),
            message_crc=int(sample.crc), computed_crc=computed_crc,
            crc_valid=int(int(sample.crc) == computed_crc))
        self.recorder.submit("observed_lowcmd", raw)
        snapshot = LowCmdSnapshot(
            sequence, now, raw["wall_ns"], raw["mode_pr"], raw["mode_machine"],
            raw["motor_mode"].copy(), raw["motor_q"].astype(np.float64),
            raw["motor_dq"].astype(np.float64), raw["motor_tau"].astype(np.float64),
            raw["motor_kp"].astype(np.float64), raw["motor_kd"].astype(np.float64),
            raw["motor_reserve"].copy(), raw["reserve"].copy(), bool(raw["crc_valid"]))
        with self.condition:
            self.lowcmd_history.append(snapshot)
            self.condition.notify_all()

    def _on_secondary_imu(self, msg):
        self.recorder.submit("secondary_imu", dict(
            monotonic_ns=time.monotonic_ns(), wall_ns=time.time_ns(),
            quaternion=np.asarray(msg.quaternion, dtype=np.float32),
            gyroscope=np.asarray(msg.gyroscope, dtype=np.float32),
            accelerometer=np.asarray(msg.accelerometer, dtype=np.float32),
            rpy=np.asarray(msg.rpy, dtype=np.float32), temperature=int(msg.temperature)))

    def _on_state(self, msg):
        monotonic_ns = time.monotonic_ns()
        wall_ns = time.time_ns()
        computed_crc = int(self.crc.Crc(msg))
        crc_valid = int(msg.crc) == computed_crc
        motors = msg.motor_state
        raw = dict(
            sequence=self.sequence, monotonic_ns=monotonic_ns, wall_ns=wall_ns,
            version=np.asarray(msg.version, dtype=np.uint32), mode_pr=int(msg.mode_pr),
            mode_machine=int(msg.mode_machine), tick=int(msg.tick),
            imu_quaternion=np.asarray(msg.imu_state.quaternion, dtype=np.float32),
            imu_gyroscope=np.asarray(msg.imu_state.gyroscope, dtype=np.float32),
            imu_accelerometer=np.asarray(msg.imu_state.accelerometer, dtype=np.float32),
            imu_rpy=np.asarray(msg.imu_state.rpy, dtype=np.float32),
            imu_temperature=int(msg.imu_state.temperature),
            motor_mode=np.asarray([m.mode for m in motors], dtype=np.uint8),
            motor_q=np.asarray([m.q for m in motors], dtype=np.float32),
            motor_dq=np.asarray([m.dq for m in motors], dtype=np.float32),
            motor_ddq=np.asarray([m.ddq for m in motors], dtype=np.float32),
            motor_tau_est=np.asarray([m.tau_est for m in motors], dtype=np.float32),
            motor_temperature=np.asarray([m.temperature for m in motors], dtype=np.int16),
            motor_voltage=np.asarray([m.vol for m in motors], dtype=np.float32),
            motor_sensor=np.asarray([m.sensor for m in motors], dtype=np.uint32),
            motor_state=np.asarray([m.motorstate for m in motors], dtype=np.uint32),
            motor_reserve=np.asarray([m.reserve for m in motors], dtype=np.uint32),
            wireless_remote=uint8_array(msg.wireless_remote, 40),
            reserve=np.asarray(msg.reserve, dtype=np.uint32), message_crc=int(msg.crc),
            computed_crc=computed_crc, crc_valid=int(crc_valid))
        self.sequence += 1
        self.recorder.submit("state", raw)
        finite = all(np.isfinite(raw[key]).all() for key in (
            "imu_quaternion", "imu_gyroscope", "imu_accelerometer", "motor_q",
            "motor_dq", "motor_ddq", "motor_tau_est"))
        quat_norm = float(np.linalg.norm(raw["imu_quaternion"]))
        if not crc_valid or not finite or abs(quat_norm - 1.0) > 0.001:
            self.invalid_count += 1
            with self.condition:
                self.invalid_state_times.append(monotonic_ns)
            return
        snapshot = StateSnapshot(
            raw["sequence"], monotonic_ns, wall_ns, raw["tick"], raw["mode_pr"],
            raw["mode_machine"], raw["motor_q"].astype(np.float64),
            raw["motor_dq"].astype(np.float64), raw["motor_ddq"].astype(np.float64),
            raw["motor_tau_est"].astype(np.float64), raw["motor_mode"], raw["motor_state"],
            raw["motor_temperature"], raw["wireless_remote"])
        with self.condition:
            self._latest = snapshot
            self.state_history.append(snapshot)
            self.valid_times.append(monotonic_ns)
            self.condition.notify_all()

    def latest(self):
        with self.lock:
            return self._latest

    def histories(self):
        with self.lock:
            return (list(self.state_history), list(self.lowcmd_history),
                    list(self.invalid_state_times))

    def wait_preflight(self, active_slots, duration=2.0, timeout=15.0,
                       max_temperature=75.0):
        deadline = time.monotonic() + timeout
        first = None
        while time.monotonic() < deadline:
            with self.condition:
                snapshot = self._latest
                if snapshot is not None:
                    first = snapshot.monotonic_ns if first is None else first
                    if snapshot.monotonic_ns - first >= duration * 1e9:
                        break
                self.condition.wait(0.05)
        snapshot = self.latest()
        if snapshot is None or first is None or snapshot.monotonic_ns - first < duration * 1e9:
            raise TimeoutError("did not receive two seconds of valid LowState")
        if (snapshot.mode_machine, snapshot.mode_pr) != (MODE_MACHINE, MODE_PR):
            raise ValueError(f"expected mode_machine=4/mode_pr=0, got "
                             f"{snapshot.mode_machine}/{snapshot.mode_pr}")
        active = np.asarray(active_slots, dtype=np.int64)
        errors = active[np.asarray(snapshot.motorstate)[active] != 0]
        if errors.size:
            raise ValueError(f"motorstate errors in slots {errors.tolist()}")
        maximum_temperature = float(np.max(np.asarray(snapshot.temperature)[active]))
        if maximum_temperature > max_temperature:
            raise ValueError(f"motor temperature {maximum_temperature:.1f} exceeds "
                             f"{max_temperature:.1f}")
        times = np.asarray(self.valid_times, dtype=np.int64)
        intervals = np.diff(times) * 1e-6
        return dict(valid_samples=len(times), invalid_samples=self.invalid_count,
                    median_interval_ms=float(np.median(intervals)) if len(intervals) else math.nan,
                    p99_interval_ms=float(np.quantile(intervals, 0.99)) if len(intervals) else math.nan,
                    max_temperature=maximum_temperature)

    def check_motion_mode(self):
        switcher = MotionSwitcherClient()
        switcher.SetTimeout(5.0)
        switcher.Init()
        code, result = switcher.CheckMode()
        self.recorder.event("motion_switcher_observe", code=code, result=result)
        if code != 0 or result is None:
            raise RuntimeError(f"MotionSwitcher CheckMode failed: code={code}")
        return result

    def enable_commands(self, expected_service, handoff_factory, released_callback):
        """Validate handoff, release the service, and immediately write LowCmd."""
        switcher = MotionSwitcherClient()
        switcher.SetTimeout(5.0)
        switcher.Init()
        code, result = switcher.CheckMode()
        self.recorder.event("motion_switcher_initial", code=code, result=result,
                            observed_lowcmd=self.existing_lowcmd_count)
        if code != 0 or result is None:
            raise RuntimeError(f"MotionSwitcher CheckMode failed: code={code}")
        if not expected_service or result.get("name") != expected_service:
            raise RuntimeError(f"expected service {expected_service!r}, got {result.get('name')!r}")
        baseline = handoff_factory()
        publisher = ChannelPublisher("rt/lowcmd", LowCmd_)
        publisher.Init()
        released = False
        try:
            release_code, _ = switcher.ReleaseMode()
            self.recorder.event("motion_switcher_release", code=release_code)
            if release_code != 0:
                raise RuntimeError(f"MotionSwitcher ReleaseMode failed: code={release_code}")
            released = True
            self.publisher = publisher
            # ReleaseMode may stop the official LowCmd stream immediately.  Close the
            # observer and publish the selected hold/damping command before waiting
            # on another RPC, so confirmation latency cannot create an open-loop gap.
            self.lowcmd_watch.Close()
            released_callback(baseline)
            deadline = time.monotonic() + 2.0
            while True:
                code, result = switcher.CheckMode()
                if code != 0 or result is None:
                    raise RuntimeError(f"CheckMode failed after release: code={code}")
                if not result.get("name"):
                    break
                if time.monotonic() >= deadline:
                    raise RuntimeError(f"MotionSwitcher release timed out: {result}")
                time.sleep(0.002)
            return baseline
        except Exception:
            if not released:
                publisher.Close()
            raise

    def write(self, command):
        if self.publisher is None:
            raise RuntimeError("LowCmd publisher is not enabled")
        return bool(self.publisher.Write(command))

    def disable_commands(self):
        publisher = self.publisher
        self.publisher = None
        if publisher is not None:
            publisher.Close()
            self.recorder.event("lowcmd_publisher_closed")

    def close(self):
        for channel in (self.state_sub, self.secondary_sub, self.lowcmd_watch):
            try:
                channel.Close()
            except Exception:
                pass
        self.disable_commands()


def new_lowcmd():
    return unitree_hg_msg_dds__LowCmd_()
