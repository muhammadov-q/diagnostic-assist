import unittest

from diagnostic_assist.labeling.canonicalize import Cause
from diagnostic_assist.labeling.quality import same_fix_parts
from test_canonicalize import outcome


class SameFixPartsTest(unittest.TestCase):
    def test_two_causes_fixed_with_the_same_part_are_merge_candidates(self):
        by_id = {
            "1": outcome("1", "boom cylinder rod seal", {"SEALKIT-BC"}),
            "2": outcome("2", "hydraulic seal", {"SEALKIT-BC"}),
            "3": outcome("3", "boom cylinder", {"CYL-BOOM"}),
        }
        causes = [
            Cause("ap/rod-seal", "AP", "rod seal", members=["1"]),
            Cause("ap/hydraulic-seal", "AP", "hydraulic seal", members=["2"]),
            Cause("ap/cylinder", "AP", "boom cylinder", members=["3"]),
        ]
        self.assertEqual(
            same_fix_parts(causes, by_id), [("AP", frozenset({"SEALKIT-BC"}), ["ap/rod-seal", "ap/hydraulic-seal"])]
        )

    def test_no_part_causes_are_never_paired(self):
        by_id = {"1": outcome("1", "ground strap", set()), "2": outcome("2", "intake screen", set())}
        causes = [
            Cause("cx/strap", "CX", "ground strap", members=["1"]),
            Cause("cx/screen", "CX", "intake screen", members=["2"]),
        ]
        self.assertEqual(same_fix_parts(causes, by_id), [])


if __name__ == "__main__":
    unittest.main()
