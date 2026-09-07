"""Train-only preprocessing and fresh predictions for every model.

Run from repository root: python -m reanalysis.pipeline
This is a corrected reconstruction, NOT a reproduction of April's reported scores.
Original shared checkpoints are never overwritten or silently reused with a new scaler.
"""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')
os.environ.setdefault('TF_DETERMINISTIC_OPS', '1')

from pathlib import Path
import argparse
import hashlib
import importlib.metadata
import json
import platform
import time
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
TARGET = 'Satisfaction'
LABELS = {'Neutral or Dissatisfied': 0, 'Satisfied': 1}
CATEGORICAL = ['Gender', 'Customer Type', 'Type of Travel', 'Class']
RATINGS = ['Inflight wifi service', 'Departure/Arrival time convenient',
           'Ease of Online booking', 'Gate location', 'Food and drink',
           'Online boarding', 'Seat comfort', 'Inflight entertainment',
           'On-board service', 'Leg room service', 'Baggage handling',
           'Checkin service', 'Inflight service', 'Cleanliness']
CONTINUOUS = ['Age', 'Flight Distance', 'Departure Delay in Minutes', 'Arrival Delay in Minutes']
FEATURES = CATEGORICAL + RATINGS + CONTINUOUS


def read_raw(name):
    df = pd.read_csv(ROOT / f'{name}.csv')
    if df.ID.isna().any() or df.ID.duplicated().any():
        raise ValueError(f'{name}: customer ID must be unique and non-null')
    if not set(df[TARGET]).issubset(LABELS):
        raise ValueError(f'{name}: unexpected target label')
    return df


def targets(df):
    mapped = df[TARGET].map(LABELS)
    if mapped.isna().any():
        raise ValueError('Unmapped target label')
    return mapped.to_numpy(dtype=np.int32)


def assert_disjoint(*frames):
    for i, a in enumerate(frames):
        for b in frames[i + 1:]:
            if set(a.ID) & set(b.ID):
                raise ValueError('Customer ID overlap between partitions')


def preprocessor():
    # No fit inside transform: medians, category modes and scaling stay fixed.
    return ColumnTransformer([
        ('continuous', Pipeline([('impute', SimpleImputer(strategy='median')),
                                  ('scale', MinMaxScaler())]), CONTINUOUS),
        ('ratings', Pipeline([('impute', SimpleImputer(strategy='most_frequent')),
                              ('encode', OneHotEncoder(categories=[np.arange(6)] * len(RATINGS),
                                                       drop='first', handle_unknown='error', sparse_output=False))]), RATINGS),
        ('category', Pipeline([('impute', SimpleImputer(strategy='most_frequent')),
                               ('encode', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), CATEGORICAL),
    ], remainder='drop', verbose_feature_names_out=True)


def scores(y, probability):
    probability = np.asarray(probability).reshape(-1)
    if len(y) != len(probability) or not np.isfinite(probability).all():
        raise ValueError('Invalid predictions')
    prediction = (probability >= 0.5).astype(int)
    return {'accuracy': float(accuracy_score(y, prediction)),
            'macro_f1': float(f1_score(y, prediction, average='macro', zero_division=0)),
            'recall_0': float(recall_score(y, prediction, pos_label=0, zero_division=0)),
            'recall_1': float(recall_score(y, prediction, pos_label=1, zero_division=0)),
            'confusion_matrix': confusion_matrix(y, prediction, labels=[0, 1]).tolist(),
            'n': int(len(y)), 'threshold': 0.5}


def predict_probability(model, x):
    # A call is mandatory for each model; no global y_pred shared across cells.
    if hasattr(model, 'predict_proba'):
        return model.predict_proba(x)[:, list(model.classes_).index(1)]
    return np.asarray(model.predict(x, batch_size=256, verbose=0)).reshape(-1)


def evaluate(model, x, y):
    return scores(y, predict_probability(model, x))


def configure_tf():
    import tensorflow as tf
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.keras.utils.set_random_seed(SEED)
    tf.config.experimental.enable_op_determinism()
    return tf


def build_dnn(tf, n_features, units, dropout=0.0, learning_rate=0.001):
    model = tf.keras.Sequential([tf.keras.layers.Input(shape=(n_features,))])
    for size in units:
        model.add(tf.keras.layers.Dense(size, activation='relu'))
        if dropout:
            model.add(tf.keras.layers.Dropout(dropout))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
    return model


def dataset(tf, x, y, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices((np.asarray(x, dtype=np.float32), y))
    if shuffle:
        ds = ds.shuffle(len(y), seed=SEED, reshuffle_each_iteration=True)
    options = tf.data.Options()
    options.threading.private_threadpool_size = 2
    return ds.batch(256).with_options(options)


def fit_dnn(tf, model, x, y, xv, yv, epochs, early_stop):
    # New callback per experiment; explicit untouched validation partition.
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=8,
                 min_delta=0.0001, restore_best_weights=True)] if early_stop else []
    history = model.fit(dataset(tf, x, y, True), validation_data=dataset(tf, xv, yv),
                        epochs=epochs, verbose=0, callbacks=callbacks, shuffle=False)
    return {k: [float(v) for v in values] for k, values in history.history.items()}


def run(output, epoch_cap=None):
    from reanalysis.audit import audit
    tf = configure_tf()
    output.mkdir(parents=True, exist_ok=True)
    artifacts = ROOT / 'reanalysis' / 'artifacts'
    if epoch_cap:
        artifacts = artifacts / 'smoke'
    artifacts.mkdir(parents=True, exist_ok=True)
    survey, additional, external = [read_raw(n) for n in ['survey', 'new_train', 'new_test']]
    assert_disjoint(survey, additional, external)
    train_val, test = train_test_split(survey, test_size=0.2, stratify=targets(survey), random_state=SEED)
    train, val = train_test_split(train_val, test_size=0.25, stratify=targets(train_val), random_state=SEED)
    add_train, add_val = train_test_split(additional, test_size=0.2, stratify=targets(additional), random_state=SEED)
    assert_disjoint(train, val, test, add_train, add_val, external)
    partitions = {'survey_train': train, 'survey_validation': val, 'survey_test': test,
                  'additional_train': add_train, 'additional_validation': add_val, 'additional_test': external}
    prep = preprocessor()
    prep.fit(train[FEATURES])
    x = {n: prep.transform(df[FEATURES]).astype(np.float32) for n, df in partitions.items()}
    y = {n: targets(df) for n, df in partitions.items()}
    joblib.dump(prep, artifacts / 'preprocessor.joblib')
    pd.DataFrame([(n, int(v)) for n, df in partitions.items() for v in df.ID],
                 columns=['partition', 'ID']).to_csv(output / 'split_ids.csv', index=False)
    started = time.monotonic()
    candidates, histories = {}, {}
    rf = RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=2)
    # RF uses the same raw rows and train-only preprocessing as the DNN comparison.
    rf.fit(x['survey_train'], y['survey_train'])
    candidates['RandomForest'] = rf
    specs = {'Logistic_Dense': ([], 0.0, 0.01, 20, False),
             'DNN_hidden': ([64, 32, 16, 8, 4], 0.0, 0.01, 20, False),
             'DNN_dropout': ([16, 8, 4], 0.3, 0.01, 20, False),
             'DNN_earlystop': ([64, 32, 8], 0.3, 0.001, 40, True),
             'Shared_architecture_rebuilt': ([64, 32, 16, 8, 4], 0.3, 0.001, 80, True)}
    for name, (units, dropout, lr, epochs, early) in specs.items():
        tf.keras.utils.set_random_seed(SEED)
        model = build_dnn(tf, x['survey_train'].shape[1], units, dropout, lr)
        histories[name] = fit_dnn(tf, model, x['survey_train'], y['survey_train'],
                                  x['survey_validation'], y['survey_validation'], min(epochs, epoch_cap or epochs), early)
        candidates[name] = model
        print(f'Trained {name}: {len(histories[name]["loss"])} epochs', flush=True)
    validation = {n: evaluate(m, x['survey_validation'], y['survey_validation']) for n, m in candidates.items()}
    selected = max(validation, key=lambda n: (validation[n]['macro_f1'], validation[n]['recall_0']))
    # Freeze selection BEFORE any held-out test evaluation.
    results = {n: {'validation': validation[n], 'test': evaluate(m, x['survey_test'], y['survey_test'])} for n, m in candidates.items()}
    joblib.dump(rf, artifacts / 'random_forest.joblib')
    baseline = candidates['Shared_architecture_rebuilt']
    baseline.save(artifacts / 'rebuilt_shared_base.keras')
    extensions = {'No_additional_training': baseline}
    for name in ['Scratch', 'Continue_all_layers', 'Fine_tune_head']:
        tf.keras.utils.set_random_seed(SEED)
        if name == 'Scratch':
            model = build_dnn(tf, x['survey_train'].shape[1], [64, 32, 16, 8, 4], 0.3)
        else:
            # Independent reload: one experiment cannot mutate the next one's baseline.
            model = tf.keras.models.load_model(artifacts / 'rebuilt_shared_base.keras', compile=False)
            if name == 'Fine_tune_head':
                for layer in model.layers[:-3]:
                    layer.trainable = False
            model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])
        epochs = 80 if name == 'Scratch' else 40
        histories[name] = fit_dnn(tf, model, x['additional_train'], y['additional_train'],
                                  x['additional_validation'], y['additional_validation'], min(epochs, epoch_cap or epochs), True)
        extensions[name] = model
        print(f'Trained {name}: {len(histories[name]["loss"])} epochs', flush=True)
    ext_validation = {n: evaluate(m, x['additional_validation'], y['additional_validation']) for n, m in extensions.items()}
    ext_selected = max(ext_validation, key=lambda n: (ext_validation[n]['macro_f1'], ext_validation[n]['recall_0']))
    ext_results = {n: {'validation': ext_validation[n], 'test': evaluate(m, x['additional_test'], y['additional_test'])} for n, m in extensions.items()}
    extensions[ext_selected].save(artifacts / 'selected_additional_model.keras')
    # Raw-feature permutation on validation only: grouped one-hot columns move together.
    rf_pipeline = Pipeline([('preprocess', prep), ('model', rf)])
    importance = permutation_importance(rf_pipeline, val[FEATURES], y['survey_validation'],
                                      scoring='f1_macro', n_repeats=5, random_state=SEED, n_jobs=2)
    important = pd.DataFrame({'feature': FEATURES, 'macro_f1_decrease': importance.importances_mean,
                              'std': importance.importances_std}).sort_values('macro_f1_decrease', ascending=False)
    important.to_csv(output / 'feature_importance.csv', index=False)
    result = {'run_at_utc': datetime.now(timezone.utc).isoformat(), 'seed': SEED,
              'status': 'smoke_only' if epoch_cap else 'completed_revalidation',
              'method': 'September reconstruction; not historical reported performance',
              'selection_rule': 'validation macro F1, tie-break recall class 0; fixed threshold 0.5',
              'feature_count': int(x['survey_train'].shape[1]), 'labels': LABELS,
              'class_counts': {n: {str(k): int(v) for k, v in pd.Series(y[n]).value_counts().items()} for n in partitions},
              'selected_survey_model': selected, 'selected_additional_model': ext_selected,
              'survey_models': results, 'additional_models': ext_results, 'histories': histories,
              'specifications': specs, 'audit': audit(),
              'file_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob('*.csv')},
              'versions': {p: importlib.metadata.version(p) for p in ['tensorflow-cpu', 'keras', 'scikit-learn', 'pandas', 'numpy']},
              'python': platform.python_version(), 'seconds': round(time.monotonic() - started, 2)}
    (output / 'results.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    for kind, table in [('survey', results), ('additional', ext_results)]:
        rows = [{'model': name, 'partition': split, **{k: v for k, v in metrics.items() if k != 'confusion_matrix'}}
                for name, splits in table.items() for split, metrics in splits.items()]
        pd.DataFrame(rows).to_csv(output / f'{kind}_metrics.csv', index=False)
    print(json.dumps({'selected_survey': selected, 'survey_test': results[selected]['test'],
                      'selected_additional': ext_selected, 'additional_test': ext_results[ext_selected]['test'],
                      'seconds': result['seconds']}, ensure_ascii=False, indent=2), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT / 'reanalysis' / 'results')
    parser.add_argument('--smoke', action='store_true', help='Two epochs only; never publish as final performance')
    args = parser.parse_args()
    run(args.output, 2 if args.smoke else None)
