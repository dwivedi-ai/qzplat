# Survey wording history

## Baseline: commit b26f52c (before the guided survey update)

The original interface displayed each Superset as a heading and hid Subsets behind
“Show Details.” The instructions were:

> Please annotate the following terms based on your perception. For each term,
> indicate if you consider it a stereotype, not a stereotype, or if you are unsure.

The options were “Stereotype,” “Not a Stereotype,” and “Not sure.” Participants
marking “Stereotype” were asked “How offensive is this stereotype?” with endpoints
0 (Not at all offensive) and 5 (Very offensive).

The export reviewed for this update, raw_quiz_results(7).csv, contains 5,661 rows
dated 2025-04-28 through 2025-07-02. The baseline implementation remains available
in Git history, including templates/quiz.html and static/js/script.js.

## Guided survey update: prepared 2026-10-10

This date records preparation, not deployment or the beginning of data collection.
Record the actual deployment date and commit here when this version goes live.

The updated interface defines stereotype as a generalised idea that can be positive,
negative, or neutral. It asks whether a participant recognises each group as a
stereotype associated with people from their state. Recognition does not imply
agreement, truth, or offensiveness. Associated terms are visible and participants
are instructed to answer for the group overall, using “Not sure” if they cannot
confidently judge it. A fictional Riverland example is instructional only; it is
not recorded.

The exact participant-facing text is in templates/quiz.html. The response mapping is:

| Visible response | Stored annotation |
| --- | --- |
| Yes, I recognise this stereotype | Stereotype |
| No, I don’t recognise this stereotype | Not a Stereotype |
| Not sure | Not sure |

The offensiveness follow-up asks how offensive the participant personally finds
the stereotype about people from their state. The numeric values and endpoints
remain 0–5. Non-Stereotype answers continue to store -1 through the existing backend.
The interface clears ratings when an answer changes from Yes to No or Not sure.

No CSV columns, group labels, associated terms, item order, database schema,
submission field names, or aggregation logic changed. The existing aggregation
copies each group answer and rating to its associated terms; those are inherited
group annotations, not separately collected judgments of those terms.

With JavaScript, participants see one group at a time, can navigate backward,
and review/edit their answers before submitting. Without JavaScript, the same
questions appear in a single form, with the conditional rating instructions visible.
Answers remain in the page while navigating; they are not saved across refreshes
and reach the server only on submission.

Although export structure is identical, the wording clarifies the judgment being
collected and may affect response patterns. Keep collection periods distinguishable
in research records and pilot the revised wording before interpreting old and new
answers together. The existing export does not identify how earlier participants
interpreted “stereotype.”

### Implementation validation

- Rendered the actual Jinja templates for all 37 state/UT definitions and the
  empty-question case.
- Checked the guided flow in Chrome at desktop and 390 px mobile widths: required
  answers, conditional ratings, backward navigation, cleared ratings after changing
  Yes, completion of all 22 Maharashtra groups, review/edit, and submission
  validation after leaving an edit incomplete.
- Checked original field names and annotation/rating values in browser FormData,
  and exercised the existing submit function using that payload with an in-memory
  database substitute. All 22 rows preserved the expected database fields and -1
  sentinel. The live database was not contacted or tested.
- Checked the JavaScript-disabled form fallback, JavaScript syntax, and whitespace.
