from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split


DATA_DIR = Path("att_faces")
RANDOM_STATE = 0


def load_orl_faces(data_dir: Path):
    images = []
    labels = []

    subject_dirs = sorted(
        [p for p in data_dir.iterdir() if p.is_dir() and p.name.startswith("s")],
        key=lambda p: int(p.name[1:]),
    )

    for subject_dir in subject_dirs:
        label = int(subject_dir.name[1:]) - 1
        image_paths = sorted(subject_dir.glob("*.pgm"), key=lambda p: int(p.stem))

        for image_path in image_paths:
            img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise RuntimeError(f"Failed to read image: {image_path}")
            images.append(img)
            labels.append(label)

    return np.array(images), np.array(labels, dtype=np.int32)


def evaluate_model(model, X_train, y_train, X_test, y_test):
    model.train(list(X_train), y_train)

    correct = 0
    predictions = []
    confidences = []

    for img, label in zip(X_test, y_test):
        pred, confidence = model.predict(img)
        predictions.append(pred)
        confidences.append(confidence)
        if pred == label:
            correct += 1

    accuracy = correct / len(y_test)
    return accuracy, np.array(predictions), np.array(confidences)


def save_mean_face_and_eigenfaces(images):
    flat = images.reshape(images.shape[0], -1).astype(np.float64)
    mean = flat.mean(axis=0)
    centered = flat - mean

    # SVD gives principal directions. Rows of vh are eigenfaces.
    _, _, vh = np.linalg.svd(centered, full_matrices=False)

    h, w = images.shape[1], images.shape[2]
    mean_face = mean.reshape(h, w)

    plt.figure(figsize=(3, 4))
    plt.imshow(mean_face, cmap="gray")
    plt.title("Mean Face")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("mean_face.png", dpi=150)
    plt.close()

    plt.figure(figsize=(10, 4))
    for i in range(10):
        face = vh[i].reshape(h, w)
        plt.subplot(2, 5, i + 1)
        plt.imshow(face, cmap="gray")
        plt.title(f"Eigenface {i+1}")
        plt.axis("off")
    plt.tight_layout()
    plt.savefig("top10_eigenfaces.png", dpi=150)
    plt.close()

    return mean, vh


def reconstruct_examples(images, mean, eigenfaces, indices=(0, 1, 2), components=(5, 10, 30, 80)):
    h, w = images.shape[1], images.shape[2]

    for idx in indices:
        x = images[idx].reshape(-1).astype(np.float64)
        centered = x - mean

        plt.figure(figsize=(2 * (len(components) + 1), 3))
        plt.subplot(1, len(components) + 1, 1)
        plt.imshow(images[idx], cmap="gray")
        plt.title("Original")
        plt.axis("off")

        for j, r in enumerate(components, start=2):
            basis = eigenfaces[:r]
            coeffs = basis @ centered
            recon = mean + coeffs @ basis
            recon = recon.reshape(h, w)

            plt.subplot(1, len(components) + 1, j)
            plt.imshow(recon, cmap="gray")
            plt.title(f"r={r}")
            plt.axis("off")

        plt.tight_layout()
        plt.savefig(f"reconstruction_{idx}.png", dpi=150)
        plt.close()


def main():
    images, labels = load_orl_faces(DATA_DIR)
    print(f"Loaded images: {images.shape}")
    print(f"Loaded labels: {labels.shape}")
    print(f"Number of classes: {len(np.unique(labels))}")

    X_train, X_test, y_train, y_test = train_test_split(
        images,
        labels,
        test_size=0.4,
        stratify=labels,
        random_state=RANDOM_STATE,
    )

    print(f"Train size: {len(X_train)}")
    print(f"Test size: {len(X_test)}")

    eigen_model = cv2.face.EigenFaceRecognizer_create(num_components=80)
    eigen_acc, eigen_pred, eigen_conf = evaluate_model(
        eigen_model, X_train, y_train, X_test, y_test
    )

    # Fisherfaces has at most C - 1 effective components.
    fisher_model = cv2.face.FisherFaceRecognizer_create(num_components=39)
    fisher_acc, fisher_pred, fisher_conf = evaluate_model(
        fisher_model, X_train, y_train, X_test, y_test
    )

    print(f"Eigenfaces accuracy: {eigen_acc:.4f}")
    print(f"Fisherfaces accuracy: {fisher_acc:.4f}")

    mean, eigenfaces = save_mean_face_and_eigenfaces(images)
    reconstruct_examples(images, mean, eigenfaces)

    print("Saved figures:")
    print("  mean_face.png")
    print("  top10_eigenfaces.png")
    print("  reconstruction_0.png, reconstruction_1.png, reconstruction_2.png")


if __name__ == "__main__":
    main()