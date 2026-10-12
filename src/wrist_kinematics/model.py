"""CNN-LSTM architecture and training settings from the original experiment."""

from pathlib import Path

from .config import ExperimentConfig


def build_model(config: ExperimentConfig):
    """Construct the final, nondilated Phase 4 model."""
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import (
        BatchNormalization, Conv1D, Dense, Dropout, GaussianNoise,
        GlobalAveragePooling1D, Input, LSTM, MaxPooling1D, SpatialDropout1D,
    )
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.regularizers import l2

    model = Sequential([
        Input(shape=(config.window_size, 12)),
        GaussianNoise(0.05),
        Conv1D(32, 5, activation="relu", kernel_regularizer=l2(0.005)),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        SpatialDropout1D(0.2),
        Conv1D(64, 3, activation="relu", kernel_regularizer=l2(0.005)),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        SpatialDropout1D(0.2),
        LSTM(64, return_sequences=True, kernel_regularizer=l2(0.005)),
        Dropout(0.3),
        GlobalAveragePooling1D(),
        Dense(64, activation="relu", kernel_regularizer=l2(0.005)),
        Dropout(0.3),
        Dense(3, activation="linear"),
    ])
    # The source compiles twice; 0.00025 is the rate used by model.fit.
    model.compile(
        optimizer=Adam(learning_rate=config.learning_rate),
        loss="mse", metrics=["mae"],
    )
    return model


def train_model(data, config: ExperimentConfig, checkpoint_path: Path):
    """Train to a new checkpoint; keep the supplied checkpoint intact."""
    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.resolve() == config.model_path.resolve():
        raise ValueError("Use a new checkpoint path to preserve the supplied trained model.")
    from tensorflow.keras.callbacks import (
        EarlyStopping, ModelCheckpoint, ReduceLROnPlateau,
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    model = build_model(config)
    callbacks = [
        ModelCheckpoint(
            str(checkpoint_path), monitor="val_loss", save_best_only=True, verbose=1,
        ),
        EarlyStopping(
            monitor="val_loss", patience=15, restore_best_weights=True, verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=0.00001, verbose=1,
        ),
    ]
    history = model.fit(
        data.X_train, data.y_train,
        epochs=config.epochs, batch_size=config.batch_size,
        validation_data=(data.X_test, data.y_test),
        callbacks=callbacks, verbose=1,
    )
    return model, history
