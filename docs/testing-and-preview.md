# Testing and local preview

## Validation performed on 2026-10-10

The real Flask application was tested with MySQL 8.4 in a separate, localhost-only
container. No production database was available or modified.

Five integration tests passed, covering required registration fields, all 37
state/UT pages, submissions for all 762 definitions, persistence of all three
annotation values and ratings 0–5/-1, raw and processed CSV headers, group/subset
aggregation, admin rendering, and database failure handling.

Chrome checks passed at desktop and 390 px mobile widths for the explanation,
one-group navigation, required answers/ratings, backward navigation, cleared
ratings after changing Yes, review/edit/revalidation, form field compatibility,
JavaScript-disabled fallback, and an actual POST followed by the thank-you page.
The browser submission created 22 marked rows; these were deleted. The integration
tests remove their individually marked rows through unittest cleanup callbacks.
After validation, the isolated database contained zero response rows.

Fresh-database testing found and fixed two startup issues: python-dotenv was not
listed in requirements.txt, and the connector no longer accepts multi=True.
The existing schema.sql is now tracked so fresh deployments can create the table.

## Repeating integration tests

Use a dedicated database, never the research database. The test module explicitly
uses localhost and qzsql_test, and overrides the project's environment before
importing the app.

```sh
docker run --rm --name qzsql-survey-test \
  -e MYSQL_ROOT_PASSWORD=qzsql-isolated-test-only \
  -e MYSQL_DATABASE=qzsql_test \
  -p 127.0.0.1:33079:3306 -d mysql:8.4
```

Wait until MySQL is ready. Install requirements into a virtual environment, then:

```sh
QZSQL_TEST_MYSQL_PORT=33079 python -m unittest discover -s tests -v
```

QZSQL_TEST_MYSQL_PASSWORD can override the container's test password. Without
QZSQL_TEST_MYSQL_PORT, the integration test module skips rather than using .env.

## Local preview

The preview runs the actual application at http://localhost:5000 with the separate
qzsql_test database. It supports real submissions, but they are preview responses
and do not join the research database. During this session it uses a temporary
virtual environment at /tmp/qzsql-runtime.

```sh
MYSQL_HOST=127.0.0.1 MYSQL_PORT=33079 MYSQL_USER=root \
MYSQL_PASSWORD=qzsql-isolated-test-only MYSQL_DB=qzsql_test \
SECRET_KEY=local-preview-only \
gunicorn --bind 127.0.0.1:5000 --workers 1 app:app
```

Stop the preview server before removing the container. The container is temporary
and removes its database volume when stopped:

```sh
docker stop qzsql-survey-test
```

## Public deployment status

Public hosting has not been configured for this update. Railway is excluded as a
deployment destination. The local preview is not a public deployment. A future
public deployment needs a chosen host and working production database settings.

## Final end-to-end check — 2026-10-11

Application revision tested: 9355e36. Tested using the actual Gunicorn app:app
entry point, Python 3.10.20, and a fresh MySQL 8.4 container on localhost:33080.
The ordinary localhost:5000 preview and the research database were not modified.

Results:

- All seven integration/description tests passed, including first-start schema
  creation, 37 state/UT pages, 762 group submissions, exports, aggregation,
  registration validation, and database failure handling.
- A real Chrome journey began at registration and completed Maharashtra on
  desktop, Goa on mobile, and a native-form submission with JavaScript disabled.
  It exercised all three annotations, all six ratings, required-answer/rating
  checks, backward navigation, clearing an old rating, review/edit, incomplete-edit
  revalidation, help/Escape/focus restoration, and the thank-you response.
- All 62 browser responses matched the exact stored MySQL rows and raw CSV.
  The 146 attributes in the processed CSV had independently checked vote counts
  and mean offensiveness ratings. CSV columns and saved annotation values were
  unchanged.
- No browser JavaScript exceptions were observed. Mobile checks included 390 px
  and a 320 px fallback with no horizontal overflow.
- Every marked test response was deleted. The disposable database contained zero
  remaining rows before its container was removed.

The browser regression check is now retained in tests/e2e_survey.mjs, with database
and export verification/cleanup in tests/verify_e2e.py. They only target localhost.
The verifier deletes exact recorded participant markers in a finally block.

To repeat, start a separate MySQL container and run the app with its credentials;
never aim the server at the research database. Chrome must run with remote debugging
on localhost:9222, and Node 22+ is required for the browser check. After running the
integration suite against the fresh database, run:

```sh
QZSQL_E2E_BASE_URL=http://127.0.0.1:5001 \
QZSQL_E2E_ARTIFACT_DIR=/tmp/qzsql-final-e2e \
node tests/e2e_survey.mjs

python tests/verify_e2e.py --port 33080 --artifacts /tmp/qzsql-final-e2e
```

Stop that test server and remove its container afterward, including if a test
fails. The test artifacts contain synthetic responses and screenshots only.

### Deployment preparation still required

The survey's functional path passed; a public deployment was not performed.
Choose the hosting destination, supply a working production MySQL connection,
and set a strong SECRET_KEY. Use Gunicorn with app:app from the repository root
and bind to the host's assigned port, for example:

```sh
gunicorn --bind "0.0.0.0:${PORT:-8000}" app:app
```

The admin page and both download routes are currently unauthenticated. The browser
check confirmed the exports can be downloaded without credentials. Protect these
three routes through application authentication or hosting access controls before
public launch, because the raw export includes participant names and demographics.
Neither the admin-access policy nor production secrets were changed by this test.
