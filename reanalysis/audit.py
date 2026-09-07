"""Read-only audit of the supplied datasets and saved-model provenance."""
from pathlib import Path
import hashlib
import json
import zipfile
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

def audit():
    frames = {n: pd.read_csv(ROOT / f'{n}.csv') for n in ['survey', 'data', 'new_train', 'new_test']}
    result = {}
    for name, df in frames.items():
        result[name] = {'shape': list(df.shape), 'columns': list(df.columns),
                        'missing': df.isna().sum()[lambda s: s > 0].to_dict(),
                        'target': df['Satisfaction'].value_counts().to_dict() if 'Satisfaction' in df else {}}
    result['overlap_ID'] = {}
    for a, b in [('survey', 'new_train'), ('survey', 'new_test'), ('new_train', 'new_test')]:
        if 'ID' in frames[a] and 'ID' in frames[b]:
            result['overlap_ID'][f'{a}/{b}'] = len(set(frames[a].ID) & set(frames[b].ID))
    result['models'] = {}
    for path in ROOT.glob('*.keras'):
        with zipfile.ZipFile(path) as z:
            config = json.loads(z.read('config.json'))
            result['models'][path.name] = {'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                'metadata': json.loads(z.read('metadata.json')),
                'layers': [{k: layer['config'].get(k) for k in ['name', 'units', 'rate', 'batch_shape', 'activation']} for layer in config['config']['layers']]}
    return result

if __name__ == '__main__':
    print(json.dumps(audit(), ensure_ascii=False, indent=2))
