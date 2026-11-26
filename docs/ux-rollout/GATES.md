# Gates: UX principles rollout (20 laws)

OWNS: docs/UX_PRINCIPLES.md, docs/ux-rollout/PLAN.md, docs/ux-rollout/GATES.md, .gitignore, CLAUDE.md, templates/base.html, scripts/check_ux.py, scripts/ux_baseline.json, scripts/run_all_gates.py

Scope: the 20 UX laws are written down, built into shared pieces, enforced by a ratcheting check, applied to every template, and verified in a browser. Phase 3 leaves get their own ledgers that inherit G7, G8 and G15. Refreshed 2026-11-26 against main 8ce69a8; no gate below G0 is met yet because scripts/check_ux.py does not exist.

- [x] G0: the rollout docs are tracked (gitignore exception) so they survive and reach worktrees by merge
  CHECK: git ls-files --cached --others --exclude-standard docs/ux-rollout
  EXPECT: docs/ux-rollout/PLAN\.md
  EVIDENCE: 2026-11-26, worktree ux-design-principles-c44154: `.gitignore` gained `!docs/ux-rollout/*.md`; git lists PLAN.md and GATES.md as addable (untracked, not ignored). Becomes "tracked" on commit.

- [x] G1: docs/UX_PRINCIPLES.md exists and names all 20 laws and the shared piece that implements each
  CHECK: /Users/sedriclouissaint/tools/Insights/venv/bin/python scripts/check_ux.py principles
  EXPECT: UX_PRINCIPLES_OK
  EVIDENCE: exit=0; shell=/bin/sh; cwd=/Users/sedriclouissaint/tools/Insights/.claude/worktrees/ux-design-principles-c44154; path=cac86490b0a1/25 entries; output=20 laws in order, a checklist item for each, and every shared piece documented | UX_PRINCIPLES_OK

- [x] G2: CLAUDE.md tells every chat to read the doc, forbids new alert/confirm/toast copies, and names the check
  CHECK: /Users/sedriclouissaint/tools/Insights/venv/bin/python scripts/check_ux.py claude-md
  EXPECT: UX_CLAUDE_MD_OK
  EVIDENCE: exit=0; shell=/bin/sh; cwd=/Users/sedriclouissaint/tools/Insights/.claude/worktrees/ux-design-principles-c44154; path=cac86490b0a1/25 entries; output=CLAUDE.md sends UI work to the doc, the shared pieces and the ratchet | UX_CLAUDE_MD_OK

- [x] G3: base.html defines Insights.toast, confirm, busy, explain, fieldError, deferred once, with an aria-live region, showToast kept as a wrapper, and Bootstrap JS still loaded exactly once
  CHECK: /Users/sedriclouissaint/tools/Insights/venv/bin/python scripts/check_ux.py shell
  EXPECT: UX_SHELL_OK
  EVIDENCE: exit=0; shell=/bin/sh; cwd=/Users/sedriclouissaint/tools/Insights/.claude/worktrees/ux-design-principles-c44154; path=cac86490b0a1/25 entries; output=base.html loads Insights once after Bootstrap; 6 pages render it with one live region | UX_SHELL_OK

- [x] G4: the shared pieces behave under node (toast error stays, undo fires, confirm resolves true/false and focuses Cancel for danger, busy restores the button and ignores a second click, deferred sends after the window, Undo cancels the request, a failed request restores the item)
  CHECK: /Users/sedriclouissaint/tools/Insights/venv/bin/python scripts/check_ux.py behavior
  EXPECT: UX_BEHAVIOR_OK
  EVIDENCE: exit=0; shell=/bin/sh; cwd=/Users/sedriclouissaint/tools/Insights/.claude/worktrees/ux-design-principles-c44154; path=cac86490b0a1/25 entries; output=17 behavior checks pass; 8 broken copies are each caught | UX_BEHAVIOR_OK

- [x] G5: the scanner fails on a fixture full of violations (known positive control) and passes on a clean one
  CHECK: /Users/sedriclouissaint/tools/Insights/venv/bin/python scripts/check_ux.py control
  EXPECT: UX_CONTROL_OK
  EVIDENCE: exit=0; shell=/bin/sh; cwd=/Users/sedriclouissaint/tools/Insights/.claude/worktrees/ux-design-principles-c44154; path=cac86490b0a1/25 entries; output=bad fixture trips all 7 rules; clean fixture trips none | UX_CONTROL_OK

- [x] G6: the ratchet is registered in run_all_gates.py and no template exceeds or undercuts its baseline
  CHECK: /Users/sedriclouissaint/tools/Insights/venv/bin/python scripts/check_ux.py ratchet && /Users/sedriclouissaint/tools/Insights/venv/bin/python scripts/check_ux.py registered
  EXPECT: UX_REGISTERED_OK
  EVIDENCE: exit=0; shell=/bin/sh; cwd=/Users/sedriclouissaint/tools/Insights/.claude/worktrees/ux-design-principles-c44154; path=cac86490b0a1/25 entries; output=all 8 check_ux.py modes are registered under their tokens | UX_REGISTERED_OK

- [ ] G7: no template contains alert( or native confirm( or a local showToast definition (baseline is zero)
  CHECK: /Users/sedriclouissaint/tools/Insights/venv/bin/python scripts/check_ux.py zero
  EXPECT: UX_ZERO_DEBT_OK
  EVIDENCE: pending

- [x] G8: every gate passes (148 before this work, plus the 8 UX gates), and later migrations keep it so
  CHECK: /Users/sedriclouissaint/tools/Insights/venv/bin/python scripts/run_all_gates.py
  EXPECT: ALL_GATES_OK
  EVIDENCE: exit=0; shell=/bin/sh; cwd=/Users/sedriclouissaint/tools/Insights/.claude/worktrees/ux-design-principles-c44154; path=cac86490b0a1/25 entries; output=All 156 gates passed. | ALL_GATES_OK

- [ ] G9: impossible actions are disabled with a stated reason on compose, schedule and accounts (no platform ticked, account not connected)
  CHECK: /Users/sedriclouissaint/tools/Insights/venv/bin/python scripts/check_ux.py disabled-states
  EXPECT: UX_DISABLED_STATES_OK
  EVIDENCE: pending

- [x] G10: coarse-pointer CSS gives every icon-only and small button a 44px target
  CHECK: /Users/sedriclouissaint/tools/Insights/venv/bin/python scripts/check_ux.py targets
  EXPECT: UX_TARGETS_OK
  EVIDENCE: exit=0; shell=/bin/sh; cwd=/Users/sedriclouissaint/tools/Insights/.claude/worktrees/ux-design-principles-c44154; path=cac86490b0a1/25 entries; output=40px floor for icon-only and close buttons; 44px for small buttons by touch | UX_TARGETS_OK

- [ ] G11: first-run checklist renders from live data, hides when complete, and the queue/publish flow ends on a confirmation naming what happened and what happens next
  CHECK: /Users/sedriclouissaint/tools/Insights/venv/bin/python scripts/check_ux.py flows
  EXPECT: UX_FLOWS_OK
  EVIDENCE: pending

- [ ] G12: compose, schedule and accounts walked in the browser at 375, 768 and 1280 px and by keyboard only; an error and an undo path exercised for real; screenshots saved as evidence
  EVIDENCE: pending

- [ ] G13: the rules reached chats already running in other worktrees (owner chose the global pointer 2026-11-01; each active worktree checked for the doc or the pointer)
  EVIDENCE: partial, 2026-11-26: the pointer is in ~/.claude/CLAUDE.md and the doc exists at the absolute path it names; work shipped since (inbox.html with 0 alert/confirm, the Nov 25 connect-error commits) follows laws 14 and 15. Stays open until the ratchet (G6) is on main.

- [ ] G14: every delete that stays inside Insights uses Insights.deferred with Undo; every action that leaves Insights (publish, disconnect, send) uses Insights.confirm naming the count and destination
  CHECK: /Users/sedriclouissaint/tools/Insights/venv/bin/python scripts/check_ux.py undo
  EXPECT: UX_UNDO_OK
  EVIDENCE: pending

- [ ] G15: harness parity: every gate script that stubbed confirm/alert/showToast also stubs Insights.* into the same log, and no assertion line was removed or loosened
  CHECK: /Users/sedriclouissaint/tools/Insights/venv/bin/python scripts/check_ux.py harness
  EXPECT: UX_HARNESS_PARITY_OK
  EVIDENCE: pending
