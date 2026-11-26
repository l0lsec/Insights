# UX rollout plan: the 20 laws across Insights

Status (2026-11-26, main at 8ce69a8): **Phases 0, 1 and 2 built on branch `claude/ux-design-principles-c44154`** (`static/insights-ui.js`, base.html CSS, `scripts/check_ux.py` with 8 registered gates, baseline of 663 retired patterns). **Phases 3 to 5 not started.** Gates are in `GATES.md`
next to this file. The laws and the Insights checklist are `docs/UX_PRINCIPLES.md`.

This file and `GATES.md` are tracked from branch `claude/ux-design-principles-c44154` on (a `.gitignore`
exception, since `*.md` is ignored). Older untracked copies from 2026-11-01 sit at the same paths in the main
checkout; see step 1.0 for the merge trap that creates.

## Where things stand (re-measured 2026-11-26)

The shell is in good shape: `base.html` owns tokens, nav (one table), the Ctrl/Cmd+K palette, skip link, focus
ring, reduced-motion handling, one loading overlay. The weak layer is **feedback, confirmation and recovery
inside the big pages**, which is where the owner spends time. The table below is the survey before Phase 1; the
ratchet's baseline (`scripts/ux_baseline.json`) is now the live count.

| Template | `alert(` | `confirm(` | own `showToast` | `btn-sm` | `onclick=` | `btn-primary` | lines |
|---|---|---|---|---|---|---|---|
| compose.html | 91 | 29 | yes | 64 | 119 | 18 | 9452 |
| article.html | 53 | 12 | yes (copy) | 28 | 71 | 7 | 3540 |
| schedule.html | 44 | 12 | no | 50 | 27 | 8 | 2823 |
| sources.html | 9 | 1 | no | 19 | 16 | 13 | 876 |
| library.html | 6 | 3 | no | 17 | 19 | 3 | 830 |
| tickets.html | 6 | 2 | no | 4 | 2 | 1 | 231 |
| outreach.html | 5 | 2 | no | 31 | 23 | 10 | 970 |
| article_edit.html | 4 | 0 | no | 8 | 13 | 2 | 299 |
| briefs.html | 2 | 1 | no | 4 | 6 | 3 | 408 |
| 4 x `*_configure.html` | 1 each | 1 each | no | 0 | 1 each | 0-1 | ~220 |
| inbox.html (new since Nov 1) | 0 | 0 | no | 18 | 0 | 3 | 1021 |
| partials/post_item.html | 0 | 0 | no | 16 | 39 | 1 | 455 |
| status, feeds, accounts, articles, thumbnails | 0 | 1-3 | no | 0-5 | 0-3 | 0-2 | |
| **all templates** | **224** | **~74** | 2 copies | **277** | **348** | | 24458 |

What changed since the Nov 1 survey: the numbers barely moved, but the new work followed the rules. `inbox.html`
shipped with zero `alert(`/`confirm(`, and the Nov 25 commits ("Connect errors: every platform says what happened
and how to fix it") apply law 15. So the global pointer reaches chats; what is missing is the shared pieces that
make the right thing easy and the ratchet that makes the wrong thing fail.

Consequences, by law (unchanged from the survey):

- **Doherty / Recoverable errors:** a failed save is an `alert('Error: ...')`; the message is often raw
  `error.message`; the toast exists only in compose and article as two diverging copies and vanishes after 3 s.
- **Prevent errors / Undo:** deletes, bulk delete, "publish all", "clear all" are `confirm()` only. No Undo
  anywhere. Impossible actions (no platform ticked, account not connected) stay clickable and scold afterwards.
- **Fitts:** `btn-sm` and bare `btn-close` icons are the main targets on every card; no coarse-pointer sizing.
- **Hick / Tesler / Von Restorff:** compose has 18 `btn-primary` and ~148 buttons; cards compete; advanced
  controls are visible by default.
- **Zeigarnik / Goal-gradient / Peak-end:** no first-run checklist; queue and publish end on a toast plus reload,
  not on "here is what happened and what happens next".
- **Similarity / Jakob:** four `*_configure.html` pages repeat one pattern four ways; nav links carry inline
  hex colors; `.badge-audio` and friends are off-palette.

## How the other chats are kept in line

Three layers, from soft to hard. All three exist once this branch merges.

1. **Reachable rules (done).** `~/.claude/CLAUDE.md` names `docs/UX_PRINCIPLES.md` by absolute path in the main
   checkout, so every Insights chat in any worktree, on any branch, reads it before UI work. The project
   `CLAUDE.md` carries the short version. Memory entry `insights-ux-principles` points future sessions at it.
2. **Shared pieces (Phase 1).** When `Insights.toast/confirm/busy/explain` exist, following the rules is less
   code than breaking them. `UX_PRINCIPLES.md` flips its status line from PLANNED to SHIPPED in the same commit,
   so every chat learns the API from the doc it already reads.
3. **Ratchet (Phase 2).** `scripts/check_ux.py` registered in `run_all_gates.py`: any new `alert(`, `confirm(`,
   local toast, untyped `<button>`, icon-only button without a label, or hard-coded color fails the gate run
   every chat already does before committing. This is the layer that binds a chat that skimmed the doc.

Ship 2 and 3 together, as one small PR, before any page migration. That PR is the highest-leverage step in
the whole plan.

## Phase 1: shared pieces in `base.html` (one owner, one small diff)

1.0 **Track the rollout docs (done on this branch).** `.gitignore` gains `!docs/ux-rollout/*.md`.
    Trap: the main checkout holds older untracked copies at the same paths, so `git merge`/`pull` in main will
    refuse ("untracked working tree files would be overwritten"). Before that merge, the owner (or a session in
    the main checkout, with the owner's OK) diffs the untracked copies against the branch and removes them.

1.1 One namespace, `window.Insights`, defined once in `base.html` (same rule as Bootstrap JS: never per page):

- `Insights.toast(message, {type, undo, sticky, action})`: one `aria-live="polite"` region (errors use
  `role="alert"`). Success auto-dismisses after ~5 s; **errors stay until dismissed**; optional **Undo** and
  **Retry** buttons. `showToast(message, type)` stays as a one-line wrapper because 16 gate scripts and both
  pages call that name.
- `Insights.confirm({title, body, confirmLabel, danger})` returns a Promise. Uses the Bootstrap modal already
  loaded once. Names what will happen and how many items; focus starts on Cancel when `danger`; Esc cancels.
- `Insights.busy(button, promise, {label})`: disables and shows a spinner plus "Saving..." immediately, restores
  on settle, never leaves a dead button; a second click while busy does nothing (law 6, and the double-press
  class of bug the inbox fix just closed).
- `Insights.explain(errorOrResponse)`: maps HTTP status, network failure and known server error keys to plain
  "what happened, what to do" text. Never shows raw `error.message` or a stack.
- `Insights.fieldError(el, message)`: inline error under a field, keeps the input, clears on edit.
- `Insights.deferred(run, {message, undoMs})`: **Undo without a schema change.** Every delete in `database.py`
  is a hard `DELETE` (no soft-delete column exists). So the client hides the item at once (optimistic), shows
  "Deleted 3 posts. Undo" for ~6 s, and only then sends the request; Undo cancels it and restores the DOM. On
  `pagehide` pending requests flush with `fetch(..., {keepalive: true})`. If the request then fails, the item
  comes back with an error toast and Retry. Use it for deletes, clear, remove-from-queue. Do **not** use it
  where the effect leaves Insights (publish to a platform, disconnect a token, send a message): those keep a
  confirm that states the count and the destination.

1.2 CSS in the existing token block:

- `@media (pointer: coarse)`: `.btn-sm`, `.btn-close`, tick-chips and checkboxes get a 44 px target (padding or
  a pseudo-element hit area, so layouts do not jump); a 40 px floor everywhere for icon-only buttons.
- `.btn-danger-quiet` for destructive secondary actions so they stop looking like primaries.
- Nav link colors and `.badge-audio/text/unknown`, `.processed` move onto tokens (law 16).
- (Dropped: no template renders Flask flash messages, so there is nothing to route through the toast.)

## Phase 2: the ratchet

`scripts/check_ux.py` scans `templates/**/*.html` and counts, per file: `alert(`, native `confirm(`, a local
`function showToast`, `onclick="return confirm`, `<button` without `type`, icon-only `<button>` without
`aria-label`, `<img` without `alt`, hex colors in `style=` attributes. `scripts/ux_baseline.json` records today's
counts.

- A count **above** baseline fails: new debt is blocked in every chat.
- A count **below** baseline also fails until the baseline is lowered with `python scripts/check_ux.py --lower`,
  in the same commit. That locks progress in. Two sessions lowering the same file conflict only on one JSON line.
- Modes for the gates: `principles`, `claude-md`, `shell`, `behavior` (runs the `Insights.*` code under node with
  a fake DOM: toast error persists, Undo cancels a deferred request, confirm resolves true/false and focuses
  Cancel, busy restores), `control` (a fixture full of violations must fail, a clean one must pass), `zero`,
  `targets`, `disabled-states`, `flows`.
- Registered in `run_all_gates.py` with token `UX_RATCHET_OK`, so its existing "every check_*.py is registered"
  guard covers it.

## Phase 3: migrate pages (one leaf per template, never two writers on one file)

Each leaf: `alert`/`confirm` to Phase 1 pieces; disable impossible actions and say why next to the control
(law 14); exactly one `btn-primary` per card or section (law 7); advanced controls behind a disclosure (law 19);
group by proximity; keep every id and badge string `partials/connection_strip.html` and the gates depend on;
lower the baseline in the same commit.

| Leaf | Owns | Focus |
|---|---|---|
| 3a Compose | `compose.html`, `partials/post_item.html` | 91 alerts, 29 confirms; disabled Queue/Post with reason; card hierarchy; bulk-action progress. Three commits: generate region, cards, bulk bar |
| 3b Schedule | `schedule.html` | 44 alerts, 12 confirms; reorder feedback; deferred delete; empty states |
| 3c Accounts | `accounts.html`, 4 `*_configure.html` | one shared configure partial (law 16); disconnect names what stops working |
| 3d Create | `article.html`, `article_edit.html`, `briefs.html` | article's toast copy removed; 53 alerts |
| 3e Sources and library | `sources.html`, `library.html`, `outreach.html`, `inbox.html` | outreach keeps "worker never sends"; inbox only needs targets |
| 3f Small pages | login, signup, feeds, feed, status, usage, activity, tickets, thumbnails, result, errors | headers, empty states, error copy |

**Compose can start now.** On 2026-11-26 no Insights session is running and no live branch touches
`compose.html` (compose-queue-filter and bulk-post are at or behind main). Re-check with `git worktree list` and
the session list at the start of each leaf, because that changes daily.

**Harness rule (the biggest trap).** 16 gate scripts touch `confirm`/`alert`/`showToast` (heaviest:
`check_queue_groups.py` 58 hits, `check_accounts_js.py` 45, `check_find_results.py` 18, `_video_ui_gate.py` 17,
`check_link_comment.py` 17, `check_outreach.py` 17). Their `world()` stubs record `log.confirms` and return
`opts.confirm !== false` synchronously. When a call site moves to `await Insights.confirm(...)`, the same commit
adds an `Insights` stub to that harness's world that pushes to the **same** `log.confirms` and returns
`Promise.resolve(opts.confirm !== false)`, and the harness awaits a tick before asserting. Assertions such as
"declining does nothing at all" stay word for word. Weakening one is a failed leaf, not a fix.

## Phase 4: whole-flow fixes (the laws that live between screens)

- **First-run checklist (Zeigarnik, Goal-gradient):** a card on Compose while incomplete: Connect an account, Make
  your first post, Queue it, Schedule it. Derived from existing data (accounts, posts, queue), no new table. Shows
  "2 of 4 done"; hides when complete; reachable again from the palette.
- **Completion (Peak-end):** after queue or publish, a state that names what happened and what is next ("Queued
  for Tue 9:00 on LinkedIn and X. Edit it in Schedule") with that next step as the primary button, instead of a
  toast plus reload.
- **Save progress:** compose and outreach drafts survive reload; "Restore draft" on return.
- **Defaults and shortcuts (Hick, Parkinson):** platforms preselected from the last successful post, never a
  default that publishes; Cmd/Ctrl+Enter submits the focused card.

## Phase 5: verify like a user

Browser pane at 375, 768 and 1280 px on compose, schedule, accounts; keyboard-only walk of the primary path; an
error path and an Undo path exercised for real against the isolated test DB (memory `insights-isolated-testing`);
screenshots as evidence; `python scripts/run_all_gates.py` green.

## Order of work

1. Phase 1 + 2, one PR from this branch (`base.html`, `scripts/check_ux.py`, `scripts/ux_baseline.json`,
   `run_all_gates.py`, the `UX_PRINCIPLES.md` status line, plus the tracked rollout docs). After it merges,
   every chat is bound by the ratchet.
2. 3a Compose (largest debt, most used), then 3b Schedule, then 3c Accounts. One PR per leaf.
3. 3d, 3e, 3f in any order; these are the ones a parallel session could take, one template set each.
4. Phase 4 flows, then Phase 5 verification.

## Risks and traps

1. Harness stubs (above). Same commit, same assertions.
2. Undo is client-side deferral, not a soft delete. If the tab dies mid-window and `keepalive` also fails, the
   item is simply not deleted: the safe side.
3. Never load Bootstrap JS twice; `Insights.confirm` uses the one already in `<head>`.
4. Connection strip ids and badge strings (`Connected`, `Needs Setup`, ...) stay stable.
5. `youthful-fermat-0237a2` (web deployment) holds an **uncommitted** one-line `base.html` nav change and is a
   pre-rewrite branch (memory: never push pre-rewrite branches). `agentic-content-prep-8d5438` is pre-rewrite too.
   Whoever revives them ports their changes onto current main by hand, not by merge; Phase 1 touches the same
   file but not the same lines.
6. Every push goes through the concurrent-editing check (global CLAUDE.md); no attribution trailers; author
   `l0lsec` with the no-reply address; no em dashes.
7. Laws conflict (Hick vs Parkinson; fewer confirmations vs error prevention). Tiebreak is the owner's: clarity,
   accessibility, user control, task completion. Confirm only what leaves Insights or cannot be undone; use
   deferred Undo for the rest.

## Decisions

1. How chats learn the rules: **decided 2026-11-01**, global `~/.claude/CLAUDE.md` pointer. Working (see above).
2. Ratchet strictness: planned as "fail above baseline, and fail below until lowered". Default unless the owner
   says otherwise.
3. Compose timing: no longer blocked; start after the Phase 1+2 PR merges.
4. **Open:** the owner pasted the same 20 laws into sessions in other repositories (fire-calculator,
   loan-serv-app) on 2026-11-26. Should the laws become a global rule for every project (one
   `~/.claude/UX_PRINCIPLES.md` plus a generic pointer, with Insights keeping its own checklist), or stay
   per-repo? Those sessions may edit `~/.claude/CLAUDE.md` too, so only one session should make that change.
