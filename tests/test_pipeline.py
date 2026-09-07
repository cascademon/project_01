import numpy as np
import pandas as pd
import pytest
from reanalysis.pipeline import FEATURES, assert_disjoint, evaluate, preprocessor, read_raw, targets


def test_raw_datasets_disjoint():
    assert_disjoint(*(read_raw(n) for n in ['survey', 'new_train', 'new_test']))


def test_overlap_is_rejected():
    with pytest.raises(ValueError, match='overlap'):
        assert_disjoint(pd.DataFrame({'ID': [1, 2]}), pd.DataFrame({'ID': [2, 3]}))


def test_transform_does_not_refit_on_batch():
    train = read_raw('survey').iloc[:500]
    extra = read_raw('new_test').iloc[:10].copy()
    extra.loc[extra.index[0], 'Age'] = np.nan
    prep = preprocessor().fit(train[FEATURES])
    statistics = prep.named_transformers_['continuous']['impute'].statistics_.copy()
    solo = prep.transform(extra[FEATURES].iloc[:1])
    together = prep.transform(extra[FEATURES])[:1]
    np.testing.assert_allclose(solo, together)
    np.testing.assert_array_equal(statistics, prep.named_transformers_['continuous']['impute'].statistics_)


def test_unknown_category_and_column_order():
    train = read_raw('survey').iloc[:500]
    prep = preprocessor().fit(train[FEATURES])
    extra = train.iloc[:1].copy()
    extra.loc[extra.index[0], 'Class'] = 'Unseen'
    a = prep.transform(extra[FEATURES])
    b = prep.transform(extra[list(reversed(FEATURES))])
    np.testing.assert_allclose(a, b)
    assert np.isfinite(a).all()


def test_new_predictions_for_every_model():
    class Stub:
        classes_ = np.array([0, 1])
        def __init__(self, p):
            self.p, self.calls = p, 0
        def predict_proba(self, x):
            self.calls += 1
            return np.array([[1 - p, p] for p in self.p])
    a, b = Stub([0.1, 0.9]), Stub([0.9, 0.1])
    assert evaluate(a, [[0], [1]], [0, 1])['accuracy'] == 1
    assert evaluate(b, [[0], [1]], [0, 1])['accuracy'] == 0
    assert a.calls == b.calls == 1


def test_target_meaning_and_invalid_label():
    assert targets(pd.DataFrame({'Satisfaction': ['Neutral or Dissatisfied', 'Satisfied']})).tolist() == [0, 1]
    with pytest.raises(ValueError):
        targets(pd.DataFrame({'Satisfaction': ['Unknown']}))
