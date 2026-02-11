import numpy as np
import plotly.express as px
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassConfusionMatrix,
)
from torchvision import datasets
from torchvision.transforms.v2 import Compose, Resize, ToDtype, ToImage


def get_data(data_folder, image_size, clean_run=True):
    # Get training data
    training_data = datasets.Flowers102(
        root=data_folder,
        split="train",
        download=clean_run,
        transform=Compose(
            [
                Resize((image_size, image_size)),
                ToImage(),
                ToDtype(torch.float32, scale=True),
            ]
        ),
    )

    # Get test data
    test_data = datasets.Flowers102(
        root=data_folder,
        split="test",
        download=clean_run,
        transform=Compose(
            [
                Resize((image_size, image_size)),
                ToImage(),
                ToDtype(torch.float32, scale=True),
            ]
        ),
    )

    return training_data, test_data


def plot_training_label_histogram(data):
    label_list = [label for _, label in data]
    fig0 = px.histogram(label_list)
    fig0.update_layout(
        xaxis=dict(
            title="Flower label",
            tickmode="array",
            tickvals=list(range(0, len(data.classes))),
            ticktext=data.classes,
        ),
    )
    fig0.show()


def make_data_loaders(training_data, test_data):
    # Make data loaders
    batch_size = 128
    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=1)

    return train_dataloader, test_dataloader


def make_model_architecture(image_size, num_classes):
    # Use GPU acceleration if available
    device = (
        torch.accelerator.current_accelerator().type
        if torch.accelerator.is_available()
        else "cpu"
    )

    # Define neural network model
    class NeuralNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.flatten = nn.Flatten()
            self.linear_relu_stack = nn.Sequential(
                nn.Linear(
                    image_size * image_size * 3, 128
                ),  # image size times 3 color channels
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, num_classes),  # 102 flower classes
            )

        def forward(self, x):
            x = self.flatten(x)
            logits = self.linear_relu_stack(x)
            return logits

    model = NeuralNetwork().to(device)
    print(model)

    return model, device


def train_model(model, train_dataloader, device):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    model.train()
    for batch, (X, y) in enumerate(train_dataloader):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Print progress - generally want to see loss decreasing
        # We wouldn't usually print every batch, but this is a small dataset
        loss, loss.item()
        print(f"Batch: {batch}, loss: {loss:>7f}")


def test_model(model, test_dataloader, classes, device):
    model.eval()

    # Go through each image in the test set in turn, pass it to the model with model(image)
    out = []  # used for generating the confusion matrix later
    for batch, (X, y) in enumerate(test_dataloader):
        with torch.no_grad():
            X = X.to(device)
            pred = model(X)
            pred_output = pred[0].argmax(0)
            predicted, actual = (
                classes[pred_output],
                classes[y],
            )
            print(f"Image no.: {batch} Predicted: {predicted}, Actual: {actual}")
            out.append((int(pred_output), int(y)))

    # Combine all predictions and actual labels into tensors for calculating overall metrics on
    y_true = torch.tensor([x[1] for x in out])
    y_pred = torch.tensor([x[0] for x in out])

    # Generate confusion matrix
    metric = MulticlassConfusionMatrix(num_classes=len(classes))
    print(metric(y_pred, y_true))

    # Calculate overall accuracy
    metric = MulticlassAccuracy(num_classes=len(classes))
    acc = metric(y_pred, y_true)  # overall accuracy
    print(f"Accuracy over test set: {acc:.4f}")
    metric = MulticlassAccuracy(num_classes=len(classes), average=None)
    per_class_acc = metric(y_pred, y_true)  # per-class accuracy
    print(f"Per-class accuracy: {per_class_acc}")


if __name__ == "__main__":
    # Settings
    data_folder = "data"  # put in data folder in project root
    image_size = 500  # images will be resized to 500x500

    # Run network steps
    training_data, test_data = get_data(data_folder, image_size, clean_run=False)
    plot_training_label_histogram(training_data)
    train_dataloader, test_dataloader = make_data_loaders(training_data, test_data)
    model, device = make_model_architecture(
        image_size, num_classes=len(training_data.classes)
    )
    train_model(model, train_dataloader, device)
    test_model(model, test_dataloader, test_data.classes, device)
