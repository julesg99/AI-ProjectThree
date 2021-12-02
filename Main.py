import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.ylim(ymin=0)
    plt.xlabel('Epoch')
    plt.ylabel('Mean Absolute Error for Test Scores')
    plt.legend()
    plt.grid(True)


if __name__ == '__main__':

    # data processing

    raw_dataset = pd.read_csv('StudentsPerformance.csv')
    dataset = raw_dataset.copy()

    dataset = pd.get_dummies(dataset, drop_first=True)

    train_dataset = dataset.sample(frac=0.85)
    test_dataset = dataset.drop(train_dataset.index)

    train_features = train_dataset.copy()
    test_features = test_dataset.copy()

    train_labels = train_features[['math score', 'reading score', 'writing score']].copy()
    train_features = train_features.drop(['math score', 'reading score', 'writing score'], axis=1)

    test_labels = test_features[['math score', 'reading score', 'writing score']].copy()
    test_features = test_features.drop(['math score', 'reading score', 'writing score'], axis=1)

    # regression

    normalizer = tf.keras.layers.Normalization()
    normalizer.adapt(np.array(train_features))
    first = np.array(train_features[:1])

    with np.printoptions(precision=2, suppress=True):
        print('First example: ', first)
        print()
        print('Normalized: ', normalizer(first).numpy())

    linear_model = tf.keras.Sequential([
        normalizer,
        layers.Dense(12, activation='relu'),
        layers.Dense(6, activation='relu'),
        layers.Dense(3)
    ])

    linear_model.compile(
        optimizer=tf.optimizers.Adam(learning_rate=0.01),
        loss='mean_absolute_error',
        metrics=[tf.keras.metrics.CategoricalAccuracy()]
    )

    history = linear_model.fit(
        train_features,
        train_labels,
        epochs=100,
        verbose=0,
        validation_split=0.15
    )

    results = linear_model.predict(test_features)
    print(results[:10])

    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    hist.tail()

    plot_loss(history)
