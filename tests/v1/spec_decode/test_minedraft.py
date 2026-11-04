# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real controller transitions; model correctness belongs to the LLM e2e test.

The contract is immutable accepted-prefix work in, matching actor results out.
These CPU tests guard races, lost work, and starvation without simulating a model.
Run directly with Python when vLLM's torch-dependent pytest fixtures are unavailable.
"""

import unittest
from dataclasses import replace

if __name__ == "__main__":
    import importlib.util
    import sys
    from pathlib import Path

    module_path = (
        Path(__file__).resolve().parents[3] / "vllm/v1/spec_decode/minedraft.py"
    )
    spec = importlib.util.spec_from_file_location("minedraft_controller", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    BatchRequest = module.BatchRequest
    MineDraftController = module.MineDraftController
else:
    from vllm.v1.spec_decode.minedraft import BatchRequest, MineDraftController


class TestMineDraftController(unittest.TestCase):
    def test_two_batches_overlap_on_one_verifier(self):
        controller = MineDraftController(
            [[BatchRequest("a", (1,), 8)], [BatchRequest("b", (2,), 8)]], 1, 2
        )
        (prefill_a,) = controller.next_work()
        controller.complete(prefill_a, {"a": [10]})
        prefill_b, draft_a = controller.next_work()
        self.assertEqual((prefill_b.kind, draft_a.kind), ("prefill", "draft"))
        self.assertEqual(draft_a.requests[0].token_ids, (1, 10))
        self.assertEqual(draft_a.requests[0].max_tokens, 7)
        self.assertEqual(controller.next_work(), [])

        controller.complete(draft_a, {"a": [11, 12]})
        self.assertEqual(controller.next_work(), [])  # Verifier still owns B.
        controller.complete(prefill_b, {"b": [20]})
        verify_a, draft_b = controller.next_work()
        self.assertEqual((verify_a.kind, draft_b.kind), ("verify", "draft"))
        self.assertEqual(verify_a.draft_token_ids, ((11, 12),))
        self.assertNotEqual(verify_a.batch_id, draft_b.batch_id)

    def test_one_drafter_feeds_two_independent_verifiers(self):
        controller = MineDraftController(
            [[BatchRequest("a", (1,), 8)], [BatchRequest("b", (2,), 8)]], 2, 2
        )
        prefill_a, prefill_b = controller.next_work()
        self.assertEqual({prefill_a.verifier_id, prefill_b.verifier_id}, {0, 1})
        controller.complete(prefill_a, {"a": [10]})
        controller.complete(prefill_b, {"b": [20]})
        (draft_a,) = controller.next_work()
        controller.complete(draft_a, {"a": [11, 12]})
        draft_b, verify_a = controller.next_work()
        self.assertEqual((draft_b.kind, verify_a.kind), ("draft", "verify"))
        controller.complete(draft_b, {"b": [21, 22]})
        (verify_b,) = controller.next_work()
        self.assertEqual(verify_b.verifier_id, 1)
        self.assertEqual({item.kind for item in controller.inflight}, {"verify"})
        self.assertEqual(len(controller.inflight), 2)

    def test_only_accepted_outputs_enter_next_draft_prefix(self):
        controller = MineDraftController([[BatchRequest("a", (1,), 8)]], 1, 3)
        (prefill,) = controller.next_work()
        controller.complete(prefill, {"a": [10]})
        (draft,) = controller.next_work()
        controller.complete(draft, {"a": [11, 12, 13]})
        self.assertEqual(controller.output_token_ids, {"a": [10]})
        (verify,) = controller.next_work()
        controller.complete(verify, {"a": [11, 42]})
        (next_draft,) = controller.next_work()
        self.assertEqual(next_draft.requests[0].token_ids, (1, 10, 11, 42))
        self.assertEqual(controller.output_token_ids, {"a": [10, 11, 42]})
        self.assertEqual(next_draft.requests[0].max_tokens, 5)

    def test_replayed_and_forged_completions_cannot_release_actor(self):
        controller = MineDraftController([[BatchRequest("a", (1,), 8)]], 1, 2)
        (prefill,) = controller.next_work()
        with self.assertRaisesRegex(ValueError, "Stale"):
            controller.complete(replace(prefill, epoch=prefill.epoch + 1), {"a": [2]})
        self.assertEqual(controller.inflight, (prefill,))
        controller.complete(prefill, {"a": [2]})
        (draft,) = controller.next_work()
        with self.assertRaisesRegex(ValueError, "Stale"):
            controller.complete(prefill, {"a": [9]})
        self.assertEqual(controller.inflight, (draft,))
        self.assertEqual(controller.output_token_ids, {"a": [2]})

    def test_invalid_result_is_atomic_for_all_batch_members(self):
        controller = MineDraftController(
            [[BatchRequest("a", (1,), 8), BatchRequest("b", (2,), 8)]], 1, 2
        )
        (prefill,) = controller.next_work()
        for result in ({"a": [3]}, {"a": [3], "b": [4, 5]}, {"a": [3], "b": [-1]}):
            with self.subTest(result=result), self.assertRaises(ValueError):
                controller.complete(prefill, result)
            self.assertEqual(controller.output_token_ids, {"a": [], "b": []})
            self.assertEqual(controller.inflight, (prefill,))
        controller.complete(prefill, {"a": [3], "b": [4]})
        self.assertEqual(controller.output_token_ids, {"a": [3], "b": [4]})

    def test_cancellation_waits_for_inflight_actor_before_reuse(self):
        controller = MineDraftController(
            [[BatchRequest("a", (1,), 8)], [BatchRequest("b", (2,), 1)]], 1, 2
        )
        (prefill_a,) = controller.next_work()
        self.assertTrue(controller.cancel("a"))
        self.assertFalse(controller.cancel("a"))
        self.assertEqual(controller.next_work(), [])
        self.assertFalse(controller.done)
        controller.complete(prefill_a, {"a": [99]})
        (prefill_b,) = controller.next_work()
        controller.complete(prefill_b, {"b": [3]})
        self.assertTrue(controller.done)
        self.assertEqual(controller.output_token_ids, {"a": [], "b": [3]})

    def test_cancelled_rows_do_not_shift_proposals(self):
        controller = MineDraftController(
            [[BatchRequest("a", (1,), 8), BatchRequest("b", (2,), 8)]], 1, 2
        )
        (prefill,) = controller.next_work()
        controller.complete(prefill, {"a": [10], "b": [20]})
        (draft,) = controller.next_work()
        controller.cancel("a")
        controller.complete(draft, {"a": [11, 12], "b": [21, 22]})
        (verify,) = controller.next_work()
        self.assertEqual([request.request_id for request in verify.requests], ["b"])
        self.assertEqual(verify.draft_token_ids, ((21, 22),))
        controller.complete(verify, {"b": [21, 22, 23]}, finished={"b"})
        self.assertTrue(controller.done)
        self.assertEqual(
            controller.output_token_ids, {"a": [10], "b": [20, 21, 22, 23]}
        )

    def test_tail_and_empty_proposals_still_make_progress(self):
        controller = MineDraftController(
            [[BatchRequest("a", (1,), 3)], [BatchRequest("b", (2,), 1)]], 1, 2
        )
        (prefill_a,) = controller.next_work()
        controller.complete(prefill_a, {"a": [10]})
        prefill_b, draft_a = controller.next_work()
        controller.complete(prefill_b, {"b": [20]})
        controller.complete(draft_a, {"a": []})
        (verify_a,) = controller.next_work()
        controller.complete(verify_a, {"a": [11]})
        (draft_a,) = controller.next_work()
        self.assertEqual(draft_a.requests[0].max_tokens, 1)
        controller.complete(draft_a, {"a": [12]})
        (verify_a,) = controller.next_work()
        controller.complete(verify_a, {"a": [12, 13]})
        self.assertTrue(controller.done)
        self.assertEqual(controller.output_token_ids, {"a": [10, 11, 12], "b": [20]})

    def test_fifo_fairness_and_actor_exclusivity_across_four_batches(self):
        controller = MineDraftController(
            [[BatchRequest(str(i), (i,), 4)] for i in range(4)], 2, 1
        )
        drafts = []
        for _ in range(40):
            work_items = controller.next_work()
            self.assertTrue(work_items or controller.done)
            active = controller.inflight
            self.assertEqual(len({item.batch_id for item in active}), len(active))
            self.assertLessEqual(sum(item.kind == "draft" for item in active), 1)
            verifier_ids = [item.verifier_id for item in active if item.kind != "draft"]
            self.assertEqual(len(set(verifier_ids)), len(verifier_ids))
            for work in reversed(work_items):
                request_id = work.requests[0].request_id
                if work.kind == "draft":
                    drafts.append(work.batch_id)
                controller.complete(work, {request_id: [10]})
            if controller.done:
                break
        self.assertTrue(controller.done)
        self.assertEqual(set(drafts[:4]), {0, 1, 2, 3})
        self.assertEqual(
            controller.output_token_ids, {str(i): [10] * 4 for i in range(4)}
        )

    def test_bad_topology_and_duplicate_ids_fail_early(self):
        request = BatchRequest("a", (1,), 2)
        for args in (
            ([[request]], 0, 1),
            ([[request]], 1, 0),
            ([[request], [request]], 2, 1),
        ):
            with self.subTest(args=args), self.assertRaises(ValueError):
                MineDraftController(*args)


if __name__ == "__main__":
    unittest.main()
