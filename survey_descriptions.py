"""Load participant-facing explanations separately from research labels."""
import csv


def load_descriptions(filepath, expected_keys):
    """Require one explanation for each exact (state, category, superset) key."""
    expected_keys = set(expected_keys)
    descriptions = {}
    with open(filepath, encoding='utf-8-sig', newline='') as source:
        reader = csv.DictReader(source)
        required = {'State', 'Category', 'Superset', 'Description'}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError('Survey descriptions must have State, Category, Superset, and Description columns.')
        for row in reader:
            key = tuple(row[column].strip() for column in ['State', 'Category', 'Superset'])
            description = row['Description'].strip()
            if not all(key) or not description:
                raise ValueError('Survey description keys and explanations must not be empty.')
            if key in descriptions:
                raise ValueError(f'Duplicate survey description key: {key}')
            descriptions[key] = description
    missing = expected_keys - descriptions.keys()
    extra = descriptions.keys() - expected_keys
    if missing or extra:
        raise ValueError(f'Survey descriptions do not match definitions: {len(missing)} missing, {len(extra)} unexpected.')
    return descriptions
