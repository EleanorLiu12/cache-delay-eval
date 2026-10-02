"""Tests for whole-history selection and source sequence validation."""
import unittest

from cache_delay_eval.audit_wildchat import inspect_conversation, overlap_pairs, nonoverlap_representatives


def conversation(roles=("user","assistant")):
    return dict(turn=roles.count("user"),toxic=False,redacted=False,conversation=[
        dict(role=r,content="content",timestamp=None,toxic=False,redacted=False) for r in roles])


class WildChatAuditTests(unittest.TestCase):
    def test_role_order_is_audited_without_repair(self):
        source=conversation(("assistant","user"))
        before=[m.copy() for m in source["conversation"]]
        self.assertIn("nonalternating_or_incomplete_roles",inspect_conversation(source))
        self.assertEqual(before,source["conversation"])

    def test_empty_content_and_declared_turn_mismatch_are_separate(self):
        source=conversation()
        source["turn"]=2
        source["conversation"][0]["content"]="  "
        self.assertEqual(set(inspect_conversation(source)),{"missing_or_empty_content","declared_turn_count_mismatch"})

    def test_null_user_timestamps_do_not_disqualify_content(self):
        self.assertEqual(inspect_conversation(conversation()),[])

    def test_overlap_includes_source_ids_and_whole_history_prefixes(self):
        rows=[dict(sample_id="a",turn_identifiers=[1,1],message_hashes=["u","a"],messages=2,sample_rank=0),
              dict(sample_id="b",turn_identifiers=[2,2],message_hashes=["u","a","u2","a2"],messages=4,sample_rank=1),
              dict(sample_id="c",turn_identifiers=[2,3],message_hashes=["different","a3"],messages=2,sample_rank=2),
              dict(sample_id="d",turn_identifiers=[4,4],message_hashes=["different","b4"],messages=2,sample_rank=3)]
        pairs=overlap_pairs(rows)
        self.assertEqual({(p["left"],p["right"]) for p in pairs},{("a","b"),("b","c")})
        self.assertEqual(nonoverlap_representatives(rows,pairs),{"b","d"})


if __name__=="__main__":
    unittest.main()
