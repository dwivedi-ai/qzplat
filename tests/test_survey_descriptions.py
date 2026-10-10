"""Check explanation coverage without importing the app or contacting a database."""
import csv
from pathlib import Path
import tempfile
import unittest

from survey_descriptions import load_descriptions

ROOT = Path(__file__).resolve().parents[1]


class SurveyDescriptionTests(unittest.TestCase):
    def test_every_original_group_has_an_explanation(self):
        with open(ROOT / 'data/stereotypes.csv', encoding='utf-8-sig', newline='') as source:
            groups = list(csv.DictReader(source))
        keys = {(group['State'], group['Category'], group['Superset']) for group in groups}
        descriptions = load_descriptions(ROOT / 'data/survey_descriptions.csv', keys)
        self.assertEqual(len(groups), 762)
        self.assertEqual(descriptions.keys(), keys)
        self.assertIn('hostility', descriptions[('Maharashtra', 'Personality', 'Proud of Marathi identity')])
        self.assertIn('stingy', descriptions[('Gujarat', 'Personality', 'Business-minded')])
        self.assertIn('Gulf countries', descriptions[('Kerala', 'Personality', 'Hardworking')])
        self.assertNotEqual(descriptions[('Kerala', 'Personality', 'Hardworking')],
                            descriptions[('Bihar', 'Personality', 'Hardworking')])

    def test_missing_duplicate_empty_and_unexpected_descriptions_are_rejected(self):
        key = ('Example', 'Personality', 'Example label')
        header = ['State', 'Category', 'Superset', 'Description']
        cases = [[], [list(key) + ['']], [list(key) + ['Explanation']] * 2,
                 [['Other state', 'Personality', 'Example label', 'Explanation']]]
        for rows in cases:
            with self.subTest(rows=rows), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'descriptions.csv'
                with open(path, 'w', newline='') as destination:
                    writer = csv.writer(destination)
                    writer.writerow(header)
                    writer.writerows(rows)
                with self.assertRaises(ValueError):
                    load_descriptions(path, {key})
