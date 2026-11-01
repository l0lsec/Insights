# UX principles for Insights

Read this before you design, build or change any template, page script or flow in Insights. It applies to
every chat and every worktree. The rollout plan is `docs/ux-rollout/PLAN.md`.

Your goal is to minimize confusion, reduce effort, prevent mistakes, and help the user finish their task fast.
When laws conflict, prioritize clarity, accessibility, user control and task completion. Do not apply them
mechanically.

## Status of the shared pieces (update this line when they ship)

`window.Insights` (toast, confirm, busy, fieldError) is **PLANNED, not built** (rollout Phase 1). Until it exists:

- do not add new `alert(` calls; show the problem inline next to the control, or with the page's existing `showToast`;
- do not write a new toast helper or copy `showToast` into another page;
- a native `confirm()` is allowed only for an irreversible or high-blast-radius action, and its text must say
  what will happen and how many items (existing examples in compose.html);
- leave the existing `showToast(message, type)` signature alone: gate scripts stub it by name;
- when `Insights.*` ships, use it everywhere and the `scripts/check_ux.py` ratchet will fail any new `alert(`/`confirm(`.

## Checklist: what each law means here

Check each item against what you are about to ship.

1. **Hick (choices):** one purpose per screen; hide low-priority options; split big decisions into steps; recommend one option.
2. **Fitts (size):** targets at least 40px, 44px on touch; no icon-only target without padding; `btn-sm` is never the only way to do a main task.
3. **Jakob (familiar):** use Bootstrap/standard patterns and the nav table in `base.html`; no novel controls without a reason.
4. **Proximity:** related label, control and hint sit together; separate groups with space, not boxes.
5. **Miller (chunks):** headings and short groups; long forms become steps; never make the user remember something from another screen.
6. **Doherty (400 ms):** every click shows a response at once (pressed, busy, or optimistic update); slow work shows progress.
7. **Von Restorff:** exactly one `btn-primary` per card or section; the rest are outline or link.
8. **Fitts (nearby):** the action sits beside the content it affects; Submit sits under the last field.
9. **Serial position:** the most important item first, the closing action last, lower priority in the middle (nav, lists, forms).
10. **Peak-end:** finishing a flow shows what was done and what happens next (for example "Queued for Tue 9:00 on LinkedIn and X"), never a blank or ambiguous screen.
11. **Zeigarnik:** show done and not-done steps; save progress (drafts); make resuming easy.
12. **Prägnanz:** simple structure, clear hierarchy, no decoration that does not carry meaning.
13. **Defaults:** preselect the safest and most common option; prefill from existing context; never default into an unexpected commitment (a publish, a spend); every default is changeable.
14. **Prevent errors:** accept reasonable input variations; state requirements before submit; **disable impossible actions and say why** (for example "Connect LinkedIn to post there"); warn before risky or destructive actions.
15. **Recoverable errors:** keep the user's input; say what went wrong in plain language and how to fix it, never raw `error.message`; offer undo, retry or cancel.
16. **Similarity:** same look and behavior for same-purpose controls; reuse components; use design tokens from `base.html`, not hard-coded colors; do not give different actions the same treatment.
17. **Uniform connectedness:** containers or shared state connect a control to what it affects; keep unrelated things apart.
18. **Parkinson (time):** fewest steps; drop confirmations that do not protect anything (prefer Undo); prefill; offer shortcuts for repeat actions.
19. **Tesler (complexity):** essential controls first, advanced ones behind disclosure; let the system handle internals; do not expose technical details.
20. **Goal-gradient:** show progress and milestones; emphasize what is done and make the remainder specific.

## Before you finish a UI change

1. Name the user's primary goal on the screen and the shortest path to it.
2. Make the next action visually obvious and the only dominant one.
3. Give immediate feedback after every interaction, including failures.
4. Prevent the error first; if it still happens, keep the user's work and say how to fix it.
5. Confirm clearly when the goal is done.
6. Keep what the app already depends on: ids and badge strings used by `partials/connection_strip.html`, one Bootstrap JS load in `<head>`, function names the `scripts/check_*.py` gates stub.
7. Run `python scripts/run_all_gates.py`. Once it exists, `python scripts/check_ux.py` must pass too.

---

## Source text (the owner's instructions, verbatim)

# UX Design Instructions

Follow these instructions whenever you design, build, or revise a user interface. Apply them across layouts, navigation, onboarding, forms, settings, dashboards, and interactive flows.

Your goal is to minimize confusion, reduce effort, prevent mistakes, and help users complete their intended task as quickly as possible.

## 1. Reduce Choices per Screen: Hick's Law

The time required to make a decision increases with the number and complexity of available choices.

* Give each screen one clear purpose.
* Remove irrelevant or low-priority options.
* Break complicated decisions into smaller steps.
* Recommend an option when users may struggle to choose.

## 2. Make Targets Large: Fitts's Law

Large, nearby targets are faster and easier to interact with.

* Make buttons and controls easy to click or tap.
* Give interactive elements sufficient spacing.
* Avoid tiny icons as the only interaction target.
* Increase the clickable area around important controls.

## 3. Follow Familiar Patterns: Jakob's Law

Users expect your product to work like products they already understand.

* Use established interface conventions.
* Place navigation, search, settings, and account controls where users expect them.
* Use familiar icons and interaction patterns.
* Do not invent a new pattern unless it provides a meaningful advantage.

## 4. Group Related Information: Law of Proximity

Elements positioned near one another are perceived as related.

* Place related labels, controls, and information together.
* Use spacing to communicate relationships.
* Separate unrelated groups with additional space.
* Do not rely on borders when spacing can establish the hierarchy.

## 5. Break Content Into Chunks: Miller's Law

Working memory can only process a limited amount of information at once.

* Divide long content into small, meaningful groups.
* Break complex forms and tasks into manageable steps.
* Use headings, sections, and concise labels.
* Avoid asking users to remember information between screens.

## 6. Respond Within 400 Milliseconds: Doherty Threshold

Interfaces feel more productive when feedback appears within approximately 400 milliseconds.

* Acknowledge every user action immediately.
* Show loading, processing, or success states when results are not instant.
* Use optimistic updates when they are safe.
* Never leave users wondering whether their action registered.

## 7. Highlight the Primary Action: Von Restorff Effect

An element that visually differs from surrounding elements receives more attention.

* Give the primary action the strongest visual emphasis.
* Use one dominant call to action per section.
* Keep secondary actions visually quieter.
* Avoid making every button compete for attention.

## 8. Place Key Actions Nearby: Fitts's Law

Interaction becomes faster when important targets are close to the user's current focus.

* Place actions beside the content they affect.
* Keep form submission near the final input.
* Position frequent actions within easy reach.
* Avoid forcing unnecessary cursor or eye movement.

## 9. Put Essentials First: Serial Position Effect

People remember the first and last items in a sequence most clearly.

* Put the most important information first.
* Place the final action or takeaway at the end.
* Keep lower-priority information in the middle.
* Order navigation and lists according to user importance.

## 10. End Flows Memorably: Peak-End Rule

Users judge an experience largely by its most intense moment and how it ends.

* Create a clear and satisfying completion state.
* Confirm what the user accomplished.
* Explain what happens next.
* Avoid ending flows on an empty or ambiguous screen.

## 11. Show Visible Progress: Zeigarnik Effect

Incomplete tasks remain mentally active and encourage users to return.

* Clearly show completed and unfinished steps.
* Save progress whenever possible.
* Make it easy to resume interrupted tasks.
* Use checklists or completion states for multi-step work.

## 12. Simplify Complex Interfaces: Law of Prägnanz

People interpret complex or ambiguous designs in the simplest form possible.

* Prefer simple structures and recognizable shapes.
* Remove unnecessary decoration and visual noise.
* Create an obvious visual hierarchy.
* Make the interface understandable at a glance.

## 13. Use Sensible Defaults: Hick's Law

Helpful defaults reduce the number of decisions users must make.

* Preselect the safest and most common option.
* Use existing context to reduce unnecessary input.
* Never use defaults that create unexpected commitments.
* Make every default easy to change.

## 14. Prevent Errors Proactively: Postel's Law

Interfaces should accept reasonable variations in user input while producing clear, predictable results.

* Accept common input formats and variations.
* Explain requirements before submission.
* Disable impossible or unavailable actions.
* Warn users before risky or destructive actions.

## 15. Make Errors Recoverable: Postel's Law

The interface should handle user mistakes gracefully without creating unnecessary failure.

* Preserve the user's work after an error.
* Explain what went wrong in plain language.
* Tell the user exactly how to fix it.
* Provide undo, retry, restore, or cancel options where appropriate.

## 16. Maintain Pattern Consistency: Law of Similarity

Elements that look similar are perceived as having related purposes.

* Give similar components the same appearance and behavior.
* Use consistent colors, labels, icons, spacing, and interaction states.
* Do not use the same visual treatment for different actions.
* Reuse established components before creating new ones.

## 17. Connect Related Elements Visually: Law of Uniform Connectedness

Visually connected elements are perceived as more closely related.

* Use containers, lines, backgrounds, or shared states to show relationships.
* Visually connect controls to the content they affect.
* Keep unrelated elements visually separate.
* Use connection deliberately, not decoratively.

## 18. Reduce Task Completion Time: Parkinson's Law

Tasks tend to expand to consume the time made available for them.

* Minimize the number of steps required.
* Remove unnecessary confirmations and screens.
* Prefill information the user has already provided.
* Offer shortcuts for frequent or repeat actions.

## 19. Reveal Complexity Gradually: Tesler's Law

Every system contains some complexity that cannot be removed, only managed or transferred.

* Show essential controls first.
* Reveal advanced options only when relevant.
* Let the system handle complexity whenever possible.
* Do not force users to understand internal technical details.

## 20. Make Completion Feel Closer: Goal-Gradient Effect

Motivation increases as users perceive themselves getting closer to a goal.

* Show progress throughout multi-step flows.
* Divide long tasks into visible milestones.
* Emphasize progress already made.
* Make the remaining work feel specific and achievable.

## Implementation Requirements

When creating or revising an interface:

1. Identify the user's primary goal.
2. Design the shortest clear path to that goal.
3. Make the next action visually obvious.
4. Remove anything that distracts from task completion.
5. Provide immediate feedback after every interaction.
6. Prevent errors before they occur.
7. Preserve user work when something goes wrong.
8. Confirm clearly when the goal has been completed.

When laws appear to conflict, prioritize clarity, accessibility, user control, and successful task completion. Do not apply these laws mechanically. Use them to make deliberate decisions based on the user's context and goal.
