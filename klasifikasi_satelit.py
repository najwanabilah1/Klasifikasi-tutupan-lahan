from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def make_splits(
    labels: np.ndarray, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Buat indeks train/validation/test berstrata dengan proporsi 70/15/15."""
    indices = np.arange(len(labels))
    test_size = round(len(labels) * 0.15)
    train_val_indices, test_indices = train_test_split(
        indices,
        test_size=test_size,
        random_state=seed,
        stratify=labels,
    )
    validation_size = round(len(labels) * 0.15)
    train_indices, validation_indices = train_test_split(
        train_val_indices,
        test_size=validation_size,
        random_state=seed,
        stratify=labels[train_val_indices],
    )
    return train_indices, validation_indices, test_indices


def make_binary_labels(labels: np.ndarray, forest_label: int) -> np.ndarray:
    """Ubah label EuroSAT menjadi 1 untuk Forest dan 0 untuk kelas lain."""
    return (labels == forest_label).astype(np.int32)


def extract_color_features(images: np.ndarray, bins: int = 16) -> np.ndarray:
    """Ekstrak statistik kanal dan histogram warna sebagai fitur Random Forest."""
    features = []
    for image in images:
        image = image.astype(np.float32) / 255.0
        image_features = [
            *image.mean(axis=(0, 1)),
            *image.std(axis=(0, 1)),
        ]
        for channel in range(image.shape[-1]):
            histogram, _ = np.histogram(
                image[..., channel], bins=bins, range=(0.0, 1.0)
            )
            image_features.extend(histogram / histogram.sum())
        features.append(image_features)
    return np.asarray(features, dtype=np.float32)


def load_eurosat(
    data_dir: Path = Path("EuroSAT_RGB"),
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Muat EuroSAT lokal dari folder-folder yang masing-masing mewakili kelas."""
    if not data_dir.is_dir():
        raise FileNotFoundError(
            f"Folder dataset tidak ditemukan: {data_dir}. "
            "Gunakan --data-dir untuk menentukan folder yang berisi subfolder kelas."
        )

    class_directories = sorted(
        (path for path in data_dir.iterdir() if path.is_dir()),
        key=lambda path: path.name,
    )
    if not class_directories:
        raise ValueError(
            f"Tidak ada subfolder kelas di {data_dir}. "
            "Struktur yang diharapkan: EuroSAT_RGB/Forest/*.jpg."
        )

    supported_extensions = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    class_names = [directory.name for directory in class_directories]
    paths_by_class: list[list[Path]] = []
    for class_directory in class_directories:
        image_paths = sorted(
            path
            for path in class_directory.iterdir()
            if path.is_file() and path.suffix.lower() in supported_extensions
        )
        if not image_paths:
            raise ValueError(
                f"Tidak ada file citra yang didukung di {class_directory}."
            )
        paths_by_class.append(image_paths)

    if "Forest" not in class_names:
        raise ValueError(
            f"Subfolder kelas 'Forest' tidak ditemukan di {data_dir}. "
            f"Kelas yang ditemukan: {class_names}"
        )

    total_images = sum(len(paths) for paths in paths_by_class)
    images = np.empty((total_images, 64, 64, 3), dtype=np.uint8)
    labels = np.empty(total_images, dtype=np.int32)
    image_index = 0
    for label, image_paths in enumerate(paths_by_class):
        print(
            f"Memuat kelas {class_names[label]}: {len(image_paths)} citra...",
            flush=True,
        )
        for image_path in image_paths:
            with Image.open(image_path) as image:
                rgb_image = image.convert("RGB").resize((64, 64))
                images[image_index] = np.asarray(rgb_image, dtype=np.uint8)
                labels[image_index] = label
                image_index += 1
        print(
            f"Selesai memuat {class_names[label]} "
            f"({image_index}/{total_images} citra).",
            flush=True,
        )

    return images, labels, class_names


def build_cnn(input_shape: tuple[int, int, int]) -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.Rescaling(1.0 / 255.0),
            tf.keras.layers.Conv2D(32, 3, activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, 3, activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(128, 3, activation="relu"),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def score_predictions(
    true_labels: np.ndarray, predictions: np.ndarray
) -> dict[str, Any]:
    return {
        "accuracy": float(accuracy_score(true_labels, predictions)),
        "precision_macro": float(
            precision_score(true_labels, predictions, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(true_labels, predictions, average="macro", zero_division=0)
        ),
        "f1_macro": float(
            f1_score(true_labels, predictions, average="macro", zero_division=0)
        ),
        "confusion_matrix": confusion_matrix(
            true_labels, predictions, labels=[0, 1]
        ).tolist(),
    }


def save_metrics_plot(results: dict[str, Any], output: Path) -> None:
    """Simpan metrik model dan confusion matrix ke satu gambar PNG."""
    model_names = ("random_forest", "cnn")
    display_names = ("Random Forest", "CNN")
    metric_names = ("accuracy", "precision_macro", "recall_macro", "f1_macro")
    metric_labels = ("Accuracy", "Macro precision", "Macro recall", "Macro F1")
    colors = ("#3977b8", "#e07a35")

    figure = plt.figure(figsize=(13, 9))
    grid = figure.add_gridspec(2, 2, height_ratios=(0.8, 1))
    figure.suptitle("Perbandingan Model Klasifikasi EuroSAT", fontsize=16)

    metric_axis = figure.add_subplot(grid[0, :])
    positions = np.arange(len(metric_names))
    width = 0.36
    for index, (model_name, display_name, color) in enumerate(
        zip(model_names, display_names, colors)
    ):
        values = [
            results["models"][model_name][metric_name] * 100
            for metric_name in metric_names
        ]
        bars = metric_axis.bar(
            positions + (index - 0.5) * width,
            values,
            width,
            label=display_name,
            color=color,
        )
        metric_axis.bar_label(bars, fmt="%.1f%%", padding=2, fontsize=8)
    metric_axis.set_title("Metrik klasifikasi")
    metric_axis.set_ylabel("Skor (%)")
    metric_axis.set_xticks(positions, metric_labels, rotation=12, ha="right")
    metric_axis.set_ylim(0, 110)
    metric_axis.legend()
    metric_axis.grid(axis="y", alpha=0.25)

    matrix_labels = ("Non-Forest", "Forest")
    for column, model_name, display_name in zip(
        range(2), model_names, display_names
    ):
        axis = figure.add_subplot(grid[1, column])
        matrix = np.asarray(results["models"][model_name]["confusion_matrix"])
        image = axis.imshow(matrix, cmap="Blues")
        axis.set_title(f"Confusion matrix — {display_name}")
        axis.set_xlabel("Prediksi")
        axis.set_ylabel("Label sebenarnya")
        axis.set_xticks(range(2), matrix_labels, rotation=10, ha="right")
        axis.set_yticks(range(2), matrix_labels)
        threshold = matrix.max() / 2 if matrix.size else 0
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                axis.text(
                    column,
                    row,
                    f"{matrix[row, column]:,}",
                    ha="center",
                    va="center",
                    color="white" if matrix[row, column] > threshold else "black",
                )
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

    figure.tight_layout(rect=(0, 0, 1, 0.96))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def run_experiment(
    epochs: int = 5,
    batch_size: int = 32,
    estimators: int = 300,
    seed: int = 42,
    data_dir: Path = Path("EuroSAT_RGB"),
    output: Path = Path("results/metrics.json"),
) -> dict[str, Any]:
    tf.keras.utils.set_random_seed(seed)
    images, labels, class_names = load_eurosat(data_dir)
    try:
        forest_label = class_names.index("Forest")
    except ValueError as error:
        raise ValueError(
            f"Kelas 'Forest' tidak ditemukan dalam label EuroSAT: {class_names}"
        ) from error

    train_indices, validation_indices, test_indices = make_splits(labels, seed)
    binary_labels = make_binary_labels(labels, forest_label)
    train_labels = binary_labels[train_indices]
    validation_labels = binary_labels[validation_indices]
    test_labels = binary_labels[test_indices]

    train_images = images[train_indices]
    validation_images = images[validation_indices]
    test_images = images[test_indices]

    print(
        f"Dataset: {len(images)} citra; "
        f"train/validation/test: {len(train_indices)}/"
        f"{len(validation_indices)}/{len(test_indices)}"
    )
    class_counts = {
        name: int(np.count_nonzero(labels == index))
        for index, name in enumerate(class_names)
    }
    print(f"Jumlah citra per kelas: {class_counts}")
    if class_counts["Forest"] != 3000:
        print(
            "Peringatan: EuroSAT RGB lengkap biasanya memiliki 3.000 citra "
            f"Forest, tetapi folder ini berisi {class_counts['Forest']}. "
            "Periksa apakah ekstraksi dataset sudah lengkap."
        )
    print(f"Kelas target: Forest (1) vs non-Forest (0)")

    feature_start = time.perf_counter()
    train_features = extract_color_features(train_images)
    test_features = extract_color_features(test_images)
    feature_extraction_seconds = time.perf_counter() - feature_start

    random_forest = RandomForestClassifier(
        n_estimators=estimators,
        class_weight="balanced",
        n_jobs=-1,
        random_state=seed,
    )
    training_start = time.perf_counter()
    random_forest.fit(train_features, train_labels)
    random_forest_training_seconds = time.perf_counter() - training_start
    prediction_start = time.perf_counter()
    forest_predictions = random_forest.predict(test_features)
    random_forest_prediction_seconds = time.perf_counter() - prediction_start

    cnn = build_cnn(tuple(images.shape[1:]))
    train_data = (
        tf.data.Dataset.from_tensor_slices((train_images, train_labels))
        .shuffle(len(train_images), seed=seed)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    validation_data = (
        tf.data.Dataset.from_tensor_slices((validation_images, validation_labels))
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    test_data = (
        tf.data.Dataset.from_tensor_slices((test_images, test_labels))
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    class_weights = {
        int(class_label): float(len(train_labels) / (2 * class_count))
        for class_label, class_count in zip(
            *np.unique(train_labels, return_counts=True)
        )
    }
    training_start = time.perf_counter()
    cnn.fit(
        train_data,
        validation_data=validation_data,
        epochs=epochs,
        class_weight=class_weights,
        verbose=2,
    )
    cnn_training_seconds = time.perf_counter() - training_start
    prediction_start = time.perf_counter()
    cnn_probabilities = cnn.predict(test_data, verbose=0).ravel()
    cnn_predictions = (cnn_probabilities >= 0.5).astype(np.int32)
    cnn_prediction_seconds = time.perf_counter() - prediction_start

    results: dict[str, Any] = {
        "dataset": "EuroSAT RGB",
        "target": {"positive": "Forest", "negative": "non-Forest"},
        "class_names": class_names,
        "class_counts": class_counts,
        "split_sizes": {
            "train": len(train_indices),
            "validation": len(validation_indices),
            "test": len(test_indices),
        },
        "seed": seed,
        "models": {
            "random_forest": {
                **score_predictions(test_labels, forest_predictions),
                "feature_extraction_seconds": feature_extraction_seconds,
                "training_seconds": random_forest_training_seconds,
                "prediction_seconds": random_forest_prediction_seconds,
                "features": "mean/std per RGB channel and per-channel histograms",
            },
            "cnn": {
                **score_predictions(test_labels, cnn_predictions),
                "training_seconds": cnn_training_seconds,
                "prediction_seconds": cnn_prediction_seconds,
                "epochs": epochs,
            },
        },
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    plot_output = output.with_suffix(".png")
    save_metrics_plot(results, plot_output)
    print(json.dumps(results["models"], indent=2, ensure_ascii=False))
    print(f"Hasil disimpan ke: {output}")
    print(f"Grafik metrik disimpan ke: {plot_output}")
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bandingkan Random Forest dan CNN pada EuroSAT."
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--estimators", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("EuroSAT_RGB"),
        help="Folder dataset berisi subfolder kelas (default: EuroSAT_RGB).",
    )
    parser.add_argument("--output", type=Path, default=Path("results/metrics.json"))
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Buat PNG dari file metrik yang sudah ada tanpa melatih ulang model.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.plot_only:
        if not args.output.is_file():
            raise SystemExit(f"File metrik tidak ditemukan: {args.output}")
        results = json.loads(args.output.read_text(encoding="utf-8"))
        plot_output = args.output.with_suffix(".png")
        save_metrics_plot(results, plot_output)
        print(f"Grafik metrik disimpan ke: {plot_output}")
        return
    if args.epochs < 1 or args.batch_size < 1 or args.estimators < 1:
        raise SystemExit("--epochs, --batch-size, dan --estimators harus positif.")
    run_experiment(
        epochs=args.epochs,
        batch_size=args.batch_size,
        estimators=args.estimators,
        seed=args.seed,
        data_dir=args.data_dir,
        output=args.output,
    )


if __name__ == "__main__":
    main()
