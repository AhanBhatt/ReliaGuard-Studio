from __future__ import annotations

from sklearn.linear_model import LogisticRegression

from reliaguard_studio.models.baselines import build_split, train_regression_baselines
from reliaguard_studio.models.sequence import RecurrentPredictor, predict_torch_model, split_sequence_data, train_torch_model


def test_tabular_baselines_train_on_tiny_fixture(small_modeling_frame) -> None:
    X_train, X_test, y_train, y_test, feature_columns, _, _ = build_split(small_modeling_frame, "overreliance_risk", seed=7)
    classifier = LogisticRegression(max_iter=600, solver="liblinear", random_state=7).fit(X_train, y_train)
    regression = train_regression_baselines(small_modeling_frame, "retention_gap", seed=7)

    assert classifier.predict_proba(X_test).shape[0] == len(y_test)
    assert "random_forest_regressor" in regression.models
    assert X_train.shape[1] == len(feature_columns)


def test_sequence_model_smoke_train_and_predict(small_config, small_modeling_frame) -> None:
    bundle = split_sequence_data(
        small_modeling_frame,
        target="overreliance_risk",
        sequence_length=small_config.model.sequence_length,
        seed=small_config.simulation.seed,
        target_type="classification",
    )
    model = RecurrentPredictor(len(bundle.feature_columns), hidden_dim=8, dropout=0.1, cell="gru")
    trained = train_torch_model(model, bundle, epochs=1)
    probs = predict_torch_model(trained, bundle.X_test, "classification")

    assert probs.shape[0] == bundle.X_test.shape[0]
    assert ((probs >= 0.0) & (probs <= 1.0)).all()
