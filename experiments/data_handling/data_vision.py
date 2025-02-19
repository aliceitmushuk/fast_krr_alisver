import os
import pickle
import shutil
import torch
from torchvision import transforms, datasets
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from torch.utils.data import DataLoader

WEIGHTS = MobileNet_V2_Weights.IMAGENET1K_V1


# Define transformations for each dataset
def get_transform(dataset_name):
    """
    Returns the appropriate transform for the dataset.
    """
    if dataset_name in ["FashionMNIST", "MNIST"]:
        return transforms.Compose(
            [
                transforms.Grayscale(3),  # Convert grayscale to RGB
                transforms.Resize((224, 224)),  # Resize images to 224x224
                transforms.ToTensor(),  # Scales pixel values from [0, 255] to [0, 1]
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                ),  # ImageNet normalization
            ]
        )
    elif dataset_name in ["CIFAR10", "SVHN"]:
        return transforms.Compose(
            [
                transforms.Resize((224, 224)),  # Resize images to 224x224
                transforms.ToTensor(),  # Scales pixel values from [0, 255] to [0, 1]
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                ),  # ImageNet normalization
            ]
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


# Load dataset
def get_dataset(dataset_name, download_dir, train=True):
    """
    Loads the specified dataset.
    """
    transform = get_transform(dataset_name)
    if dataset_name == "CIFAR10":
        return datasets.CIFAR10(
            root=download_dir, train=train, download=True, transform=transform
        )
    elif dataset_name == "FashionMNIST":
        return datasets.FashionMNIST(
            root=download_dir, train=train, download=True, transform=transform
        )
    elif dataset_name == "MNIST":
        return datasets.MNIST(
            root=download_dir, train=train, download=True, transform=transform
        )
    elif dataset_name == "SVHN":
        split = "train" if train else "test"
        return datasets.SVHN(
            root=download_dir, split=split, download=True, transform=transform
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


# Create dataloader
def get_dataloader(dataset_name, download_dir, batch_size=64, train=True):
    """
    Returns a DataLoader for batch processing.
    """
    dataset = get_dataset(dataset_name, download_dir, train=train)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=False
    )  # shuffle=False preserves order


# Load MobileNet feature extractor
def get_feature_extractor():
    """
    Loads a pre-trained MobileNet model and removes its classification head.
    """
    model = mobilenet_v2(weights=WEIGHTS)
    feature_extractor = torch.nn.Sequential(
        *list(model.children())[:-1],  # Remove classification head
        torch.nn.AdaptiveAvgPool2d((1, 1)),  # Apply global average pooling
        torch.nn.Flatten(),  # Flatten output to vector form
    )
    feature_extractor.eval()
    return feature_extractor


# Extract features
def extract_features(feature_extractor, dataloader, device="cuda"):
    """
    Extracts features from a dataset using the pre-trained feature extractor.
    """
    features = []
    labels = []
    feature_extractor.to(device)
    with torch.no_grad():  # Disable gradient computation for efficiency
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            extracted_features = feature_extractor(inputs).cpu()  # Extract features
            features.append(extracted_features)
            labels.append(targets)
    return torch.cat(features, dim=0), torch.cat(labels, dim=0)


# Save separate features and labels
def save_separate_data(features, labels, dataset_name, output_dir):
    """
    Saves features and labels into separate .pkl files.

    File naming:
        - Features: [dataset_name]_data.pkl
        - Labels: [dataset_name]_target.pkl
    """
    os.makedirs(
        output_dir, exist_ok=True
    )  # Create output directory if it doesn't exist

    # Define file paths
    features_file = os.path.join(output_dir, f"{dataset_name.lower()}_data.pkl")
    labels_file = os.path.join(output_dir, f"{dataset_name.lower()}_target.pkl")

    # Save features
    with open(features_file, "wb") as f:
        pickle.dump(features.numpy(), f)  # Convert to NumPy before saving

    # Save labels
    with open(labels_file, "wb") as f:
        pickle.dump(labels.numpy(), f)

    print(f"Saved features to {features_file}")
    print(f"Saved labels to {labels_file}")


# Delete directory containing original data
def delete_original_data(download_dir):
    """
    Deletes the folder where the original data is downloaded.
    """
    shutil.rmtree(download_dir)  # Recursively delete the directory
    print(f"Deleted original data directory: {download_dir}")


# Process datasets
def process_all_datasets(output_dir, device, batch_size=64):
    """
    Processes CIFAR10, FashionMNIST, MNIST, and SVHN datasets.
    Saves features and labels into separate .pkl files for each dataset.
    """
    datasets_to_process = ["CIFAR10", "FashionMNIST", "MNIST", "SVHN"]
    feature_extractor = get_feature_extractor()  # Load feature extractor
    download_dir = os.path.join(output_dir, "vision")

    for dataset_name in datasets_to_process:
        print(f"Processing {dataset_name}...")

        # Process training set
        train_loader = get_dataloader(
            dataset_name, download_dir, batch_size=batch_size, train=True
        )
        train_features, train_labels = extract_features(
            feature_extractor, train_loader, device=device
        )

        # Process test set
        test_loader = get_dataloader(
            dataset_name, download_dir, batch_size=batch_size, train=False
        )
        test_features, test_labels = extract_features(
            feature_extractor, test_loader, device=device
        )

        # Combine training and test sets
        combined_features = torch.cat([train_features, test_features], dim=0)
        combined_labels = torch.cat([train_labels, test_labels], dim=0)

        # Save features and labels separately
        save_separate_data(
            combined_features, combined_labels, dataset_name, output_dir=output_dir
        )

        # Delete original downloaded data
        delete_original_data(download_dir)
