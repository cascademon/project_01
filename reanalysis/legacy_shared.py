"""Evaluate the ACTUAL supplied shared checkpoint with a fixed 78-column schema.

Historical imputation/encoder objects were not saved. Reconstruct them from survey.csv
and the original notebook's rules, never from the inference batch. This is an explicit
compatibility assumption, not proof of the checkpoint's complete training lineage.
"""
from pathlib import Path
import json
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from reanalysis.pipeline import (ROOT, SEED, FEATURES, CATEGORICAL, RATINGS, CONTINUOUS,
                                 configure_tf, read_raw, targets, assert_disjoint,
                                 fit_dnn, evaluate, build_dnn)


class LegacyPreprocessor:
    def __init__(self, survey, scaler):
        self.scaler = scaler
        self.medians = survey[RATINGS + CONTINUOUS].median()
        self.modes = survey[CATEGORICAL].mode().iloc[0]
        self.mapping = {c: {v: i for i, v in enumerate(sorted(survey[c].dropna().unique()))} for c in CATEGORICAL}
        self.columns = list(scaler.feature_names_in_)

    def frame(self, df):
        data = df[FEATURES].copy()
        data[RATINGS + CONTINUOUS] = data[RATINGS + CONTINUOUS].fillna(self.medians)
        data[CATEGORICAL] = data[CATEGORICAL].fillna(self.modes)
        for col in CATEGORICAL:
            data[col] = data[col].map(self.mapping[col])
            if data[col].isna().any():
                raise ValueError(f'Unknown legacy category: {col}')
        for col in RATINGS:
            if not data[col].isin(range(6)).all():
                raise ValueError(f'Invalid legacy rating: {col}')
            data[col] = pd.Categorical(data[col], categories=range(6))
        return pd.get_dummies(data, columns=RATINGS, drop_first=True).reindex(columns=self.columns)

    def transform(self, df):
        return self.scaler.transform(self.frame(df)).astype(np.float32)


def run():
    tf = configure_tf()
    survey, extra, test = [read_raw(n) for n in ['survey', 'new_train', 'new_test']]
    assert_disjoint(survey, extra, test)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        scaler = joblib.load(ROOT / 'scaler.pkl')
    prep = LegacyPreprocessor(survey, scaler)
    # Audit the scaler against the visible notebook's exact 80/20 split.
    historic_train, _ = train_test_split(prep.frame(survey), test_size=0.2, random_state=1)
    reconstructed = MinMaxScaler().fit(historic_train)
    scaler_matches = all(np.allclose(getattr(scaler, k), getattr(reconstructed, k)) for k in ['scale_', 'min_', 'data_min_', 'data_max_'])
    if not scaler_matches:
        raise ValueError('Saved scaler does not match visible historical preprocessing; do not guess')
    # Use the verified, freshly constructed current-version scaler for portability.
    prep.scaler = reconstructed
    train, val = train_test_split(extra, test_size=0.2, stratify=targets(extra), random_state=SEED)
    assert_disjoint(train, val, test)
    xtr, xv, xt = [prep.transform(df) for df in [train, val, test]]
    ytr, yv, yt = [targets(df) for df in [train, val, test]]
    baseline_path = ROOT / 'base_model.keras'
    original = tf.keras.models.load_model(baseline_path, compile=False)
    models = {'Original_shared_no_update': original}
    histories = {}
    for name in ['Scratch_same_architecture', 'Continue_all_layers', 'Fine_tune_head']:
        tf.keras.utils.set_random_seed(SEED)
        if name == 'Scratch_same_architecture':
            model = build_dnn(tf, 78, [64, 32, 16, 8, 4], 0.3)
        else:
            model = tf.keras.models.load_model(baseline_path, compile=False)
            if name == 'Fine_tune_head':
                for layer in model.layers[:-3]:
                    layer.trainable = False
            model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])
        histories[name] = fit_dnn(tf, model, xtr, ytr, xv, yv, 80 if name.startswith('Scratch') else 40, True)
        models[name] = model
    validation = {n: evaluate(m, xv, yv) for n, m in models.items()}
    selected = max(validation, key=lambda n: (validation[n]['macro_f1'], validation[n]['recall_0']))
    results = {n: {'validation': validation[n], 'test': evaluate(m, xt, yt)} for n, m in models.items()}
    output = ROOT / 'reanalysis' / 'results'
    output.mkdir(exist_ok=True)
    record = {'seed': SEED, 'selected_by_validation': selected, 'threshold': 0.5,
              'model_source': 'Original root base_model.keras, not the rebuilt 83-input model',
              'input_features': 78, 'scaler_matches_original_split': bool(scaler_matches),
              'load_warnings': [str(w.message) for w in caught],
              'provenance_limit': 'Imputers/encoders reconstructed from survey and original code; exact model training lineage unverified. Historical test has been examined before; not a new blind benchmark.',
              'models': results, 'histories': histories}
    (output / 'legacy_shared_results.json').write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
    rows = [{'model': n, 'partition': s, **{k: v for k, v in m.items() if k != 'confusion_matrix'}} for n, d in results.items() for s, m in d.items()]
    pd.DataFrame(rows).to_csv(output / 'legacy_shared_metrics.csv', index=False)
    print(json.dumps({'selected': selected, 'results': results}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    run()
