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

### Full runtime validation on 2026-10-10

The later full application test used an isolated MySQL 8.4 database. Integration
and real-browser submissions, CSV exports, and aggregation passed. All generated
response rows were deleted and the preview database was confirmed empty. See
[testing and preview](testing-and-preview.md) for scope, commands, and hosting status.

## Group explanation update: prepared 2026-10-10

The survey now displays a plain-language explanation beneath each original group
label. All 762 explanations live in data/survey_descriptions.csv, keyed by the
exact State, Category, and Superset from data/stereotypes.csv. This separate file
can be reviewed or edited without changing research definitions. Both app.py and
app_dev.py load these explanations for display. The loader requires complete,
unique coverage and rejects missing, duplicate, empty, or unexpected entries.

The explanations describe the image being judged rather than assert that it is
true of people from a place. Short explanations clarify the group label; the
original associated terms remain visible and are still part of the judgment.
For mixed groups, some explanations explicitly distinguish the included ideas,
such as Marathi cultural pride versus hostility towards North Indians, and
Gujarat's careful spending versus being described as stingy. State-specific
examples distinguish Kerala's migrant-work association from Bihar's farming and
manual-labour association under the same Hardworking label.

The exact participant-facing definitions and prose are versioned in the display
CSV. They are not hidden form values and are not stored in response rows or CSV
exports. Original group labels, categories, state names, associated terms,
annotation values, offensiveness ratings, and aggregation are unchanged. Explain
the full original group when interpreting results; associated terms are not all
synonyms and should not be treated as separate participant judgments.

Coverage checks and the full Flask/MySQL integration suite passed with the new
explanations. Original definitions and schema were also checked to be unchanged.
Any collection using this wording should record its start date and commit,
and compare it with earlier wording when interpreting continuation data.

## Voting layout update: prepared 2026-10-11

This update changes presentation and short interface prompts. The research
labels, display-description CSV, associated terms, response mapping, rating scale,
submission fields, and exports are unchanged.

The welcome and practice content now appear only before the guided survey.
During voting, a compact bar shows the state, position, and a How to answer button.
The instructions remain available in a keyboard-accessible dialog. Each screen
presents the group title, unchanged explanation, visible associated terms, the
recognition question, and three equally styled answers in that order.

The question is now “Do you recognise this as a stereotype about people from
[state]?” Answer cards show Yes / I recognise it, No / I don’t recognise it, and
Not sure / I cannot confidently decide. The shorter supporting text retains the
recognition task and all three original saved annotation values. The reminder
explains that recognition does not imply agreement and asks for one group answer.

Colour indicates selection consistently across all three options; Yes and No
are not assigned success or error colours. The numeric offensiveness question
appears after Yes, with the same original wording, values, and endpoints. The
browser brings that follow-up into view when needed. Mobile navigation stays at
the bottom with content padding so the page can scroll clear of the controls.
Review, editing, native radio keyboard behaviour, and the no-JavaScript form
fallback remain available.

Implementation followed the GOV.UK Design System question-page and radio patterns:
https://design-system.service.gov.uk/patterns/question-pages/
https://design-system.service.gov.uk/components/radios/

Seven coverage/integration tests passed. Chrome checks covered mobile/desktop
layout, the help dialog and Escape/focus restoration, all answer/rating/navigation
paths, review/edit, a real submission, and the no-JavaScript fallback. Synthetic
response rows were deleted. These checks verify functionality; participant
comprehension should still be assessed in a small usability pilot.
