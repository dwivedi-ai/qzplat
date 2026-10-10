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
