"""Verify browser payloads against MySQL/CSV exports, then delete their exact markers.

Run only against the disposable localhost qzsql_test database used by e2e_survey.mjs.
"""
import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path

import mysql.connector


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--artifacts', type=Path, required=True)
    args = parser.parse_args()
    cases = json.loads((args.artifacts / 'cases.json').read_text())
    markers = [case['marker'] for case in cases]
    assert markers and all(marker.startswith('QZSQL_E2E_') for marker in markers)
    assert len(set(markers)) == len(markers)
    with open(Path(__file__).resolve().parents[1] / 'data/stereotypes.csv', encoding='utf-8-sig', newline='') as source:
        definitions = {(row['State'], row['Category'], row['Superset']): row for row in csv.DictReader(source)}
    conn = mysql.connector.connect(host='127.0.0.1', port=args.port, user='root',
                                   password='qzsql-isolated-test-only', database='qzsql_test')
    cursor = conn.cursor(dictionary=True)
    try:
        with open(args.artifacts / 'raw.csv', newline='') as source:
            reader = csv.DictReader(source)
            columns = reader.fieldnames
            exported = {int(row['id']): row for row in reader}
        grouped = defaultdict(list)
        total = 0
        for case in cases:
            payload = case['payload']
            cursor.execute('SELECT * FROM results WHERE user_name=%s ORDER BY id', (case['marker'],))
            saved = cursor.fetchall()
            indexes = sorted(int(key.removeprefix('annotation_')) for key in payload if key.startswith('annotation_'))
            assert len(saved) == len(indexes)
            total += len(saved)
            for row, index in zip(saved, indexes):
                assert list(row) == columns
                assert row['user_state'] == payload['user_state']
                assert row['user_age'] == int(payload['user_age'])
                assert row['user_sex'] == payload['user_sex']
                assert row['attribute_superset'] == payload[f'superset_{index}']
                assert row['category'] == payload[f'category_{index}']
                assert row['annotation'] == payload[f'annotation_{index}']
                rating = int(payload[f'offensiveness_{index}']) if row['annotation'] == 'Stereotype' else -1
                assert row['offensiveness_rating'] == rating
                assert {key: str(value) for key, value in row.items()} == exported[row['id']]
                key = (row['user_state'], row['category'], row['attribute_superset'])
                definition = definitions[key]
                attributes = [definition['Superset']] + [term.strip() for term in definition['Subsets'].split(',') if term.strip()]
                for attribute in attributes:
                    grouped[(key[0], key[1], attribute)].append((row['annotation'], rating))
        assert total == 62, f'Expected 62 browser rows; found {total}'
        with open(args.artifacts / 'processed.csv', newline='') as source:
            processed = {(row['Stereotype_State'], row['Category'], row['Attribute']): row for row in csv.DictReader(source)}
        assert processed.keys() == grouped.keys()
        for key, values in grouped.items():
            row = processed[key]
            for annotation, column in [('Stereotype', 'Stereotype_Votes'), ('Not a Stereotype', 'Not_Stereotype_Votes'), ('Not sure', 'Not_Sure_Votes')]:
                assert int(row[column]) == sum(value[0] == annotation for value in values)
            ratings = [value[1] for value in values if value[1] >= 0]
            if ratings:
                assert math.isclose(float(row['Average_Offensiveness']), round(sum(ratings) / len(ratings), 2), abs_tol=.001)
            else:
                assert row['Average_Offensiveness'] == ''
        print(f'PASS: {total} browser responses exactly match persisted rows and raw CSV; {len(grouped)} aggregated attributes have correct counts and mean ratings.')
    finally:
        # Cleanup runs even if a data assertion fails and never uses a broad name prefix.
        for marker in markers:
            cursor.execute('DELETE FROM results WHERE user_name=%s', (marker,))
        conn.commit()
        for marker in markers:
            cursor.execute('SELECT COUNT(*) AS count FROM results WHERE user_name=%s', (marker,))
            assert cursor.fetchone()['count'] == 0
        cursor.execute('SELECT COUNT(*) AS count FROM results')
        print('Remaining responses in disposable database:', cursor.fetchone()['count'])
        cursor.close()
        conn.close()


if __name__ == '__main__':
    main()
