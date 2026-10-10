"""Full Flask/MySQL flow tests; run only against a dedicated qzsql_test database.

QZSQL_TEST_MYSQL_PORT=33079 python -m unittest discover -s tests -v
The tests never load the project's production database credentials.
"""
import csv
import io
import logging
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
from uuid import uuid4

if 'QZSQL_TEST_MYSQL_PORT' not in os.environ:
    raise unittest.SkipTest('Set QZSQL_TEST_MYSQL_PORT for an isolated MySQL test server.')

os.environ.update(
    MYSQL_HOST='127.0.0.1',
    MYSQL_PORT=os.environ['QZSQL_TEST_MYSQL_PORT'],
    MYSQL_USER='root',
    MYSQL_PASSWORD=os.environ.get('QZSQL_TEST_MYSQL_PASSWORD', 'qzsql-isolated-test-only'),
    MYSQL_DB='qzsql_test',
    SECRET_KEY='isolated-survey-test-secret',
)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
logging.disable(logging.CRITICAL)
import mysql.connector
import app as survey

RAW_COLUMNS = ['id', 'timestamp', 'user_name', 'user_state', 'user_age',
               'user_sex', 'category', 'attribute_superset', 'annotation',
               'offensiveness_rating']
PROCESSED_COLUMNS = ['Stereotype_State', 'Category', 'Attribute', 'Stereotype_Votes',
                     'Not_Stereotype_Votes', 'Not_Sure_Votes', 'Average_Offensiveness']


def connect():
    return mysql.connector.connect(
        host='127.0.0.1', port=int(os.environ['QZSQL_TEST_MYSQL_PORT']),
        user='root', password=os.environ['MYSQL_PASSWORD'], database='qzsql_test')


class SurveyFlowTests(unittest.TestCase):
    def setUp(self):
        self.name = 'QZSQL_TEST_' + uuid4().hex
        self.client = survey.app.test_client()
        self.addCleanup(self.delete_test_rows)

    def delete_test_rows(self):
        with connect() as conn:
            cur = conn.cursor()
            cur.execute('DELETE FROM results WHERE user_name = %s', (self.name,))
            conn.commit()
            cur.execute('SELECT COUNT(*) FROM results WHERE user_name = %s', (self.name,))
            self.assertEqual(cur.fetchone()[0], 0)
            cur.close()

    def payload(self, state='Maharashtra'):
        data = dict(user_name=self.name, user_state=state, user_age='25', user_sex='Female')
        items = [item for item in survey.ALL_STEREOTYPE_DATA if item['state'] == state]
        for index, item in enumerate(items):
            data[f'superset_{index}'] = item['superset']
            data[f'category_{index}'] = item['category']
            # Exercise all annotation values and all six rating values.
            value = ['Stereotype', 'Not a Stereotype', 'Not sure'][index % 3]
            data[f'annotation_{index}'] = value
            if value == 'Stereotype':
                data[f'offensiveness_{index}'] = str((index // 3) % 6)
        return data, items

    def test_registration_and_required_information(self):
        self.assertEqual(self.client.get('/').status_code, 200)
        invalid = self.client.post('/', data={})
        self.assertEqual(invalid.status_code, 200)
        self.assertIn(b'Name is required', invalid.data)
        invalid = self.client.post('/', data=dict(name=self.name, user_state='Maharashtra', age='0', sex='Female'))
        self.assertIn(b'valid positive number', invalid.data)
        valid = self.client.post('/', data=dict(name=self.name, user_state='Maharashtra', age='25', sex='Female'))
        self.assertEqual(valid.status_code, 302)
        self.assertIn('/quiz?', valid.location)
        self.assertEqual(self.client.get('/quiz').status_code, 302)

    def test_every_state_renders_only_its_groups(self):
        for state in survey.INDIAN_STATES:
            with self.subTest(state=state):
                response = self.client.get('/quiz', query_string=dict(name=self.name, state=state, age='25', sex='Female'))
                self.assertEqual(response.status_code, 200)
                expected = sum(item['state'] == state for item in survey.ALL_STEREOTYPE_DATA)
                self.assertEqual(response.data.count(b'class="quiz-item"'), expected)
                self.assertIn(b'A quick example', response.data)
                self.assertIn(b'Review your answers', response.data)

    def test_submission_persistence_and_raw_export(self):
        data, items = self.payload()
        response = self.client.post('/submit', data=data, follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Your responses have been successfully recorded', response.data)
        with connect() as conn:
            cur = conn.cursor(dictionary=True)
            cur.execute('SELECT * FROM results WHERE user_name = %s ORDER BY id', (self.name,))
            saved = cur.fetchall()
            cur.close()
        self.assertEqual(len(saved), len(items))
        for index, row in enumerate(saved):
            self.assertEqual(row['attribute_superset'], items[index]['superset'])
            self.assertEqual(row['category'], items[index]['category'])
            self.assertEqual(row['annotation'], data[f'annotation_{index}'])
            self.assertEqual(row['offensiveness_rating'], int(data.get(f'offensiveness_{index}', '-1')))
        exported = self.client.get('/admin/download_raw')
        self.assertEqual(exported.status_code, 200)
        reader = csv.DictReader(io.StringIO(exported.data.decode()))
        self.assertEqual(reader.fieldnames, RAW_COLUMNS)
        self.assertEqual(len([row for row in reader if row['user_name'] == self.name]), len(items))
        self.assertEqual(self.client.get('/admin').status_code, 200)

    def test_every_state_submission_and_group_aggregation(self):
        # Cover all 762 definitions, including punctuation and unusual group names.
        for state in survey.INDIAN_STATES:
            with self.subTest(state=state):
                data, items = self.payload(state)
                response = self.client.post('/submit', data=data)
                self.assertEqual(response.status_code, 302)
                self.assertTrue(response.location.endswith('/thank_you'))
                with connect() as conn:
                    cur = conn.cursor()
                    cur.execute('SELECT COUNT(*) FROM results WHERE user_name=%s AND user_state=%s', (self.name, state))
                    self.assertEqual(cur.fetchone()[0], len(items))
                    cur.close()
        exported = self.client.get('/admin/download_processed')
        self.assertEqual(exported.status_code, 200)
        reader = csv.DictReader(io.StringIO(exported.data.decode()))
        self.assertEqual(reader.fieldnames, PROCESSED_COLUMNS)
        rows = list(reader)
        group = next(row for row in rows if row['Stereotype_State'] == 'Maharashtra' and row['Attribute'] == 'Proud of Marathi identity')
        self.assertEqual(group['Stereotype_Votes'], '1')
        self.assertEqual(group['Average_Offensiveness'], '0.0')
        subset = next(row for row in rows if row['Stereotype_State'] == 'Maharashtra' and row['Attribute'] == 'anti-North Indian sentiment')
        self.assertEqual(subset['Stereotype_Votes'], '1')
        self.assertEqual(subset['Average_Offensiveness'], '0.0')

    def test_database_failure_and_missing_participant_do_not_save(self):
        data, _ = self.payload()
        data.pop('user_name')
        response = self.client.post('/submit', data=data)
        self.assertEqual(response.status_code, 302)
        self.assertFalse(response.location.endswith('/thank_you'))
        data, _ = self.payload()
        with patch.object(survey, 'get_db', return_value=None):
            response = self.client.post('/submit', data=data)
        self.assertEqual(response.status_code, 302)
        self.assertFalse(response.location.endswith('/thank_you'))
        with connect() as conn:
            cur = conn.cursor()
            cur.execute('SELECT COUNT(*) FROM results WHERE user_name=%s', (self.name,))
            self.assertEqual(cur.fetchone()[0], 0)
            cur.close()


if __name__ == '__main__':
    unittest.main()
