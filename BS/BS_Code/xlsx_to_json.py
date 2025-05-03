import pandas as pd
import json
from pathlib import Path

def convert_xlsx_to_json_by_column(xlsx_path, json_path):
    df = pd.read_excel(xlsx_path)
    data = {}
    for col in df.columns:
        # Drop NaN and convert to string for consistency
        data[col] = [str(x) for x in df[col].dropna().tolist()]
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    base = Path(__file__).parent.parent / 'utterances_tests'
    convert_xlsx_to_json_by_column(base / 'tests.xlsx', base / 'tests.json')
    convert_xlsx_to_json_by_column(base / 'utterances.xlsx', base / 'utterances.json')
    print('Conversion complete.')
