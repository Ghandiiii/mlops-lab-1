import argparse
from pathlib import Path

import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Food-11 classifier with MLflow"
    )

    parser.add_argument(
        "--dataset",
        choices=["mini", "full"],
        default="mini",
    )

    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=32)

    return parser.parse_args()


def main():
    args = parse_args()

    # MLflow configuration
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("food11")

    # Select dataset
    if args.dataset == "mini":
        dataset_root = Path("data/food11_processed_mini")
    else:
        dataset_root = Path("data/food11_processed")

    print("Dataset:", dataset_root)
    print("Epochs:", args.epochs)
    print("Learning rate:", args.lr)
    print("Batch size:", args.batch_size)

    # Image transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    # Load datasets
    train_dataset = datasets.ImageFolder(
        dataset_root / "training",
        transform=transform,
    )

    val_dataset = datasets.ImageFolder(
        dataset_root / "validation",
        transform=transform,
    )

    test_dataset = datasets.ImageFolder(
        dataset_root / "validation",
        transform=transform,
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )

    print("Classes:", train_dataset.classes)
    print("Training images:", len(train_dataset))
    print("Validation images:", len(val_dataset))
    print("Evaluation images:", len(test_dataset))

    # Select device
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    # Load pretrained ResNet18
    model = models.resnet18(
        weights=models.ResNet18_Weights.DEFAULT
    )

    # Replace final layer for 11 Food-11 classes
    model.fc = nn.Linear(
        model.fc.in_features,
        11,
    )

    model = model.to(device)

    print("Model: ResNet18")
    print("Output classes:", 11)

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
    )

    print("Loss function: CrossEntropyLoss")
    print("Optimizer: Adam")

    # Start MLflow run
    with mlflow.start_run():

        # Log experiment parameters
        mlflow.log_param("dataset", args.dataset)
        mlflow.log_param("epochs", args.epochs)
        mlflow.log_param("learning_rate", args.lr)
        mlflow.log_param("batch_size", args.batch_size)

        # Training loop
        for epoch in range(args.epochs):

            # -------------------------
            # Training
            # -------------------------
            model.train()

            running_loss = 0.0

            for images, labels in train_loader:

                images = images.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                outputs = model(images)

                loss = criterion(outputs, labels)

                loss.backward()

                optimizer.step()

                running_loss += (
                    loss.item() * images.size(0)
                )

            train_loss = (
                running_loss / len(train_dataset)
            )

            # -------------------------
            # Validation
            # -------------------------
            model.eval()

            val_loss = 0.0
            correct = 0
            total = 0

            with torch.no_grad():

                for images, labels in val_loader:

                    images = images.to(device)
                    labels = labels.to(device)

                    outputs = model(images)

                    loss = criterion(outputs, labels)

                    val_loss += (
                        loss.item() * images.size(0)
                    )

                    _, predicted = torch.max(
                        outputs,
                        1,
                    )

                    total += labels.size(0)

                    correct += (
                        (predicted == labels)
                        .sum()
                        .item()
                    )

            val_loss = (
                val_loss / len(val_dataset)
            )

            val_accuracy = correct / total

            # Log metrics to MLflow
            mlflow.log_metric(
                "train_loss",
                train_loss,
                step=epoch,
            )

            mlflow.log_metric(
                "val_loss",
                val_loss,
                step=epoch,
            )

            mlflow.log_metric(
                "val_accuracy",
                val_accuracy,
                step=epoch,
            )

            # Print results
            print(
                f"Epoch {epoch + 1}/{args.epochs} "
                f"- Train Loss: {train_loss:.4f} "
                f"- Val Loss: {val_loss:.4f} "
                f"- Val Accuracy: {val_accuracy:.4f}"
            )

        # -------------------------
        # Final test evaluation
        # -------------------------
        model.eval()

        correct = 0
        total = 0

        with torch.no_grad():

            for images, labels in test_loader:

                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)

                _, predicted = torch.max(
                    outputs,
                    1,
                )

                total += labels.size(0)

                correct += (
                    (predicted == labels)
                    .sum()
                    .item()
                )

        test_accuracy = correct / total

        # Log final test accuracy
        mlflow.log_metric(
            "test_accuracy",
            test_accuracy,
        )

        print(
            f"Final Test Accuracy: "
            f"{test_accuracy:.4f}"
        )

                # Save model to MLflow
        example_images, _ = next(iter(test_loader))
        example_images = example_images[:1].to(device)

        mlflow.pytorch.log_model(
    model,
    "model",
    input_example=example_images,
    serialization_format="pickle",
)

        print("Model logged to MLflow.")


if __name__ == "__main__":
    main()