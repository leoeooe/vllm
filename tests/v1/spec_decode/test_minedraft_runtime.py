# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validate the public topology API and transport against real child processes.

The external engine service is replaced, not the controller or multiprocessing
transport. No model inference or accelerator correctness is asserted here.
"""

import importlib.util
import multiprocessing as mp
import os
import select
import signal
import sys
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

SPAWN_VISIBLE_DEVICES = os.environ.get("ASCEND_RT_VISIBLE_DEVICES")
SPAWN_MODEL_RUNNER = os.environ.get("VLLM_USE_V2_MODEL_RUNNER")
SPAWN_HCCL_PORT_RANGE = os.environ.get("HCCL_NPU_SOCKET_PORT_RANGE")

if __name__ in ("__main__", "__mp_main__"):
    repository = Path(__file__).resolve().parents[3]
    for name, relative_path in (
        ("vllm.v1.spec_decode.minedraft", "vllm/v1/spec_decode/minedraft.py"),
        ("minedraft_runtime", "vllm/entrypoints/minedraft.py"),
    ):
        spec = importlib.util.spec_from_file_location(name, repository / relative_path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    runtime = sys.modules["minedraft_runtime"]
else:
    from vllm.entrypoints import minedraft as runtime


def _engine_service(connection, config):
    """External engine fixture with real connection, exit and error behavior."""
    if config.model != "before-setsid":
        os.setsid()
    try:
        if config.model == "startup-error":
            connection.send(("error", "test engine startup failure"))
            return
        if config.model == "disconnect":
            return
        if config.model == "environment":
            connection.send(
                (
                    "ready",
                    SPAWN_VISIBLE_DEVICES,
                    SPAWN_MODEL_RUNNER,
                    SPAWN_HCCL_PORT_RANGE,
                )
            )
        else:
            connection.send(("ready", "same-tokenizer", 100))
        while True:
            command, _ = connection.recv()
            if command == "close" and config.model != "before-setsid":
                return
            if command == "diagnostics":
                if config.model == "diagnostics-error":
                    connection.send(("error", "test diagnostics failure"))
                    return
                connection.send(("diagnostics", {"pid": os.getpid()}))
    finally:
        connection.close()


def _config(model="normal"):
    return runtime._EngineConfig(model, (0,), "npu", True, 2, {}, None)


def _orphaning_engine_service(connection, config):
    """Leave a TERM-resistant worker behind after the engine leader exits."""
    os.setsid()
    read_fd, write_fd = os.pipe()
    child_pid = os.fork()
    if child_pid == 0:
        connection.close()
        os.close(read_fd)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        os.write(write_fd, b"ready")
        os.close(write_fd)
        # Drop inherited multiprocessing sentinel handles before waiting.
        os.closerange(3, 65536)
        while True:
            signal.pause()
    os.close(write_fd)
    os.read(read_fd, 5)
    os.close(read_fd)
    connection.send(("ready", child_pid))
    connection.close()


class TestMineDraftRuntime(unittest.TestCase):
    def setUp(self):
        self.service_patch = patch.object(runtime, "_engine_process", _engine_service)
        self.service_patch.start()
        self.addCleanup(self.service_patch.stop)

    def test_public_constructor_rejects_invalid_configuration(self):
        base: dict[str, Any] = dict(
            draft_model="draft",
            target_model="target",
            draft_devices=[0],
            verifier_devices=[[1, 2], [3, 4]],
        )
        invalid: list[dict[str, Any]] = [
            {"draft_devices": []},
            {"verifier_devices": []},
            {"verifier_devices": [[0, 1]]},
            {"draft_devices": [False]},
            {"device": "cpu"},
            {"num_batches": 1},
            {"num_batches": 2.5},
            {"num_speculative_tokens": True},
            {"num_speculative_tokens": 0},
            {"num_speculative_tokens": 1.5},
            {"timeout": float("nan")},
            {"timeout": float("inf")},
            {"timeout": True},
            {"timeout": 0},
            {"max_model_len": -1},
            {"max_num_seqs": 1.5},
            {"stream_interval": 2},
            {"draft_engine_kwargs": {"stream_interval": 2}},
            {"draft_engine_kwargs": {"max_model_len": 1024}},
            {"draft_engine_kwargs": {"tensor_parallel_size": 2}},
            {"draft_engine_kwargs": {"runner": "pooling"}},
            {"draft_engine_kwargs": {"convert": "embed"}},
            {"draft_engine_kwargs": {"enable_dbo": True}},
            {"async_scheduling": True},
        ]
        for kwargs in invalid:
            with (
                self.subTest(kwargs=kwargs),
                self.assertRaises(ValueError),
                runtime.MineDraftLLM(**(base | kwargs)),
            ):
                self.fail("Invalid configuration started external engines")

    def test_actor_timeout_preserves_connection_for_shutdown(self):
        actor = runtime._Actor(_config())
        try:
            self.assertEqual(actor.receive(10)[0], "ready")
            with self.assertRaises(TimeoutError):
                actor.receive(0.01)
        finally:
            actor.close()
        self.assertFalse(actor.process.is_alive())
        actor.close()

    def test_spawn_inherits_device_environment_before_module_import(self):
        original = {
            name: os.environ.get(name)
            for name in runtime._actor_environment(_config("environment"))
        }
        actor = runtime._Actor(_config("environment"))
        try:
            expected_port_range = original["HCCL_NPU_SOCKET_PORT_RANGE"]
            if expected_port_range is None:
                expected_port_range = "auto"
            self.assertEqual(
                actor.receive(10),
                ("ready", "0", "1", expected_port_range),
            )
            self.assertEqual(
                {name: os.environ.get(name) for name in original}, original
            )
        finally:
            actor.close()

    def test_actor_propagates_service_error_and_disconnect(self):
        for mode, expected in (
            ("startup-error", "test engine startup failure"),
            ("disconnect", "closed its connection|exited"),
        ):
            with self.subTest(mode=mode):
                actor = runtime._Actor(_config(mode))
                try:
                    with self.assertRaisesRegex(RuntimeError, expected):
                        actor.receive(10)
                finally:
                    actor.close()
                self.assertIsNotNone(actor.process.exitcode)
                self.assertFalse(actor.process.is_alive())

    def test_shutdown_terminates_child_before_process_group_setup(self):
        actor = runtime._Actor(_config("before-setsid"))
        self.assertEqual(actor.receive(10)[0], "ready")
        try:
            actor.close()
        finally:
            if actor.process.is_alive():
                actor.process.kill()
                actor.process.join(timeout=5)
        self.assertFalse(actor.process.is_alive())
        self.assertNotEqual(actor.process.exitcode, 0)

    def test_shutdown_reaps_workers_after_engine_leader_exited(self):
        with patch.object(runtime, "_engine_process", _orphaning_engine_service):
            actor = runtime._Actor(_config())
        worker_pid = actor.receive(10)[1]
        worker_fd = os.pidfd_open(worker_pid)
        try:
            actor.process.join(timeout=10)
            self.assertFalse(actor.process.is_alive())
            actor.close()
            self.assertEqual(select.select([worker_fd], [], [], 5)[0], [worker_fd])
        finally:
            if not select.select([worker_fd], [], [], 0)[0]:
                os.kill(worker_pid, signal.SIGKILL)
            os.close(worker_fd)
            actor.close()

    def test_diagnostics_failure_closes_all_engine_actors(self):
        existing = {child.pid for child in mp.active_children()}
        with runtime.MineDraftLLM(
            "diagnostics-error", "target", [0], [[1]], num_batches=2, timeout=10
        ) as llm:
            with self.assertRaisesRegex(RuntimeError, "test diagnostics failure"):
                llm.get_diagnostics()
            with self.assertRaisesRegex(RuntimeError, "idle, open"):
                llm.get_diagnostics()
        self.assertFalse(
            [child for child in mp.active_children() if child.pid not in existing]
        )

    def test_startup_failure_closes_previously_started_engine_actors(self):
        existing = {child.pid for child in mp.active_children()}
        with self.assertRaisesRegex(RuntimeError, "test engine startup failure"):
            runtime.MineDraftLLM("draft", "startup-error", [0], [[1], [2]], timeout=10)
        self.assertFalse(
            [child for child in mp.active_children() if child.pid not in existing]
        )


if __name__ == "__main__":
    unittest.main()
