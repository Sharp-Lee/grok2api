from pathlib import Path
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/auto_update_deploy.sh"


class AutoUpdateDeployScriptTests(unittest.TestCase):
    def test_preserves_committed_local_changes_by_rebasing_onto_upstream(self):
        text = SCRIPT.read_text(encoding="utf-8")

        self.assertIn('git rebase "$UPSTREAM_REMOTE/$BRANCH"', text)
        self.assertNotIn("refusing unattended merge", text)

    def test_pushes_rebased_local_branch_with_lease(self):
        text = SCRIPT.read_text(encoding="utf-8")

        self.assertIn("--force-with-lease", text)
        self.assertIn('"HEAD:$BRANCH"', text)

    def test_refuses_uncommitted_changes(self):
        text = SCRIPT.read_text(encoding="utf-8")

        self.assertIn("working tree is dirty", text)
        self.assertNotIn("git stash push", text)


if __name__ == "__main__":
    unittest.main()
