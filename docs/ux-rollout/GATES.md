# Gates: UX principles rollout (20 laws)

OWNS: docs/UX_PRINCIPLES.md, docs/ux-rollout/PLAN.md, docs/ux-rollout/GATES.md, .gitignore, CLAUDE.md, templates/base.html, scripts/check_ux.py, scripts/ux_baseline.json, scripts/run_all_gates.py

Scope: the 20 UX laws are written down, built into shared pieces, enforced by a ratcheting check, applied to every template, and verified in a browser. Refreshed 2026-11-26 against main 8ce69a8. Phase 3 was done by four parallel leaves on disjoint templates (compose+accounts, schedule, article pages, the small pages) in this one checkout; each leaf was verified with zero/undo/parse on its files and its harness gates, then the lead re-ran every gate here.

- [x] G0: the rollout docs are tracked (gitignore exception) so they survive and reach worktrees by merge
  CHECK: git ls-files docs/ux-rollout
  EXPECT: docs/ux-rollout/PLAN.md
  EVIDENCE: exit=0; shell=/bin/sh; cwd=/Users/sedriclouissaint/tools/Insights/.claude/worktrees/ux-design-principles-c44154; path=cac86490b0a1/25 entries; output=docs/ux-rollout/GATES.md | docs/ux-rollout/PLAN.md

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
  EVIDENCE: exit=0; shell=/bin/sh; cwd=/Users/sedriclouissaint/tools/Insights/.claude/worktrees/ux-design-principles-c44154; path=cac86490b0a1/25 entries; output=22 behavior checks pass; 12 broken copies are each caught | UX_BEHAVIOR_OK

- [x] G5: the scanner fails on a fixture full of violations (known positive control) and passes on a clean one
  CHECK: /Users/sedriclouissaint/tools/Insights/venv/bin/python scripts/check_ux.py control
  EXPECT: UX_CONTROL_OK
  EVIDENCE: exit=0; shell=/bin/sh; cwd=/Users/sedriclouissaint/tools/Insights/.claude/worktrees/ux-design-principles-c44154; path=cac86490b0a1/25 entries; output=bad fixture trips all 7 rules; clean fixture trips none | UX_CONTROL_OK

- [x] G6: the ratchet is registered in run_all_gates.py and no template exceeds or undercuts its baseline
  CHECK: /Users/sedriclouissaint/tools/Insights/venv/bin/python scripts/check_ux.py ratchet && /Users/sedriclouissaint/tools/Insights/venv/bin/python scripts/check_ux.py registered
  EXPECT: UX_REGISTERED_OK
  EVIDENCE: exit=0; shell=/bin/sh; cwd=/Users/sedriclouissaint/tools/Insights/.claude/worktrees/ux-design-principles-c44154; path=cac86490b0a1/25 entries; output=all 13 check_ux.py modes are registered under their tokens | UX_REGISTERED_OK

- [x] G7: no template contains alert( or native confirm( or a local showToast definition (baseline is zero)
  CHECK: /Users/sedriclouissaint/tools/Insights/venv/bin/python scripts/check_ux.py zero
  EXPECT: UX_ZERO_DEBT_OK
  EVIDENCE: exit=0; shell=/bin/sh; cwd=/Users/sedriclouissaint/tools/Insights/.claude/worktrees/ux-design-principles-c44154; path=cac86490b0a1/25 entries; output=no alert(, confirm( or local showToast in 33 templates | UX_ZERO_DEBT_OK

- [x] G8: every gate passes (148 before this work, plus the 8 UX gates), and later migrations keep it so
  CHECK: /Users/sedriclouissaint/tools/Insights/venv/bin/python scripts/run_all_gates.py
  EXPECT: ALL_GATES_OK
  EVIDENCE: exit=0; shell=/bin/sh; cwd=/Users/sedriclouissaint/tools/Insights/.claude/worktrees/ux-design-principles-c44154; path=cac86490b0a1/25 entries; output=All 161 gates passed. | ALL_GATES_OK

- [x] G9: impossible actions are disabled with a stated reason on compose, schedule and accounts (platform not connected, nothing pending to clear, platform without API keys); unticking a card's only platform deletes it behind Undo instead of outright (check_accounts_js.py compose)
  CHECK: /Users/sedriclouissaint/tools/Insights/venv/bin/python scripts/check_ux.py disabled-states
  EXPECT: UX_DISABLED_STATES_OK
  EVIDENCE: exit=0; shell=/bin/sh; cwd=/Users/sedriclouissaint/tools/Insights/.claude/worktrees/ux-design-principles-c44154; path=cac86490b0a1/25 entries; output=compose, schedule and accounts disable what cannot work yet and say why, next to it, with the fix | UX_DISABLED_STATES_OK

- [x] G10: coarse-pointer CSS gives every icon-only and small button a 44px target
  CHECK: /Users/sedriclouissaint/tools/Insights/venv/bin/python scripts/check_ux.py targets
  EXPECT: UX_TARGETS_OK
  EVIDENCE: exit=0; shell=/bin/sh; cwd=/Users/sedriclouissaint/tools/Insights/.claude/worktrees/ux-design-principles-c44154; path=cac86490b0a1/25 entries; output=40px floor for icon-only and close buttons; 44px for small buttons by touch | UX_TARGETS_OK

- [x] G11: first-run checklist renders from live data, hides when complete, and the queue/publish flow ends on a confirmation naming what happened and what happens next
  CHECK: /Users/sedriclouissaint/tools/Insights/venv/bin/python scripts/check_ux.py flows
  EXPECT: UX_FLOWS_OK
  EVIDENCE: exit=0; shell=/bin/sh; cwd=/Users/sedriclouissaint/tools/Insights/.claude/worktrees/ux-design-principles-c44154; path=cac86490b0a1/25 entries; output=the getting-started card counts real progress and leaves when done; queue and publish end on the next step | UX_FLOWS_OK

- [x] G12: compose, schedule and accounts walked in the browser at 375, 768 and 1280 px and by keyboard only; an error and an undo path exercised for real; screenshots saved as evidence
  EVIDENCE: 2026-11-26, isolated server (throwaway DB, no workers) on 127.0.0.1:5078, browser pane. No horizontal scroll on /compose, /schedule, /accounts at 375, 768, 1280 (scrollWidth == innerWidth). Compose: getting-started card 3 of 4 with one primary button; unconnected LinkedIn card shows "Connect LinkedIn in Accounts to post or queue it." and its Post now / Schedule / queue controls disabled pointing at it; delete card -> hidden + "Deleted this post. Undo" -> Undo restored it and the server log shows no delete; a real mouse click on delete -> one POST /compose/post/3/delete after the window. Keyboard only: skip link, Open Schedule, Hide for now (Enter hides, focus moves to Undo, Enter restores, focus back). Schedule: Edit time with an empty date -> field error under the input, modal kept open (error path); Cancel -> row hidden, Undo restored, no /cancel request; Post now dialog names the platform, Escape sent nothing. Accounts: Disconnect dialog names the account, red action, focus on Cancel, Escape sent nothing. 14 other migrated pages load with window.Insights and no console or server errors. Screenshots in the session transcript.

- [ ] G13: the rules reached chats already running in other worktrees (owner chose the global pointer 2026-11-01; each active worktree checked for the doc or the pointer)
  EVIDENCE: partial, 2026-11-26: the pointer is in ~/.claude/CLAUDE.md and the doc exists at the absolute path it names; since 2026-11-26 the laws are also a global rule for every project (~/.claude/UX_PRINCIPLES.md, read first by every session). Work shipped since (inbox.html with 0 alert/confirm, the Nov 25 connect-error commits) follows laws 14 and 15. Stays open until PR l0lsec/Insights#27 is on main: the ratchet binds a chat only once its branch carries check_ux.py.

- [x] G14: every delete that stays inside Insights uses Insights.deferred with Undo; every action that leaves Insights (publish, disconnect, send) uses Insights.confirm naming the count and destination
  CHECK: /Users/sedriclouissaint/tools/Insights/venv/bin/python scripts/check_ux.py undo
  EXPECT: UX_UNDO_OK
  EVIDENCE: exit=0; shell=/bin/sh; cwd=/Users/sedriclouissaint/tools/Insights/.claude/worktrees/ux-design-principles-c44154; path=cac86490b0a1/25 entries; output=every delete inside Insights is deferred with Undo and every exit is confirmed by name; 2 marked exceptions | UX_UNDO_OK

- [x] G15: harness parity: every gate script that stubbed confirm/alert/showToast also stubs Insights.* into the same log, and no assertion line was removed or loosened
  CHECK: /Users/sedriclouissaint/tools/Insights/venv/bin/python scripts/check_ux.py harness
  EXPECT: UX_HARNESS_PARITY_OK
  EVIDENCE: exit=0; shell=/bin/sh; cwd=/Users/sedriclouissaint/tools/Insights/.claude/worktrees/ux-design-principles-c44154; path=cac86490b0a1/25 entries; output=10 harnesses stub the dialogs and Insights into one log; none lost an assertion | UX_HARNESS_PARITY_OK
