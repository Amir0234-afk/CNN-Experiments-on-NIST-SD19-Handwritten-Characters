import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
import seaborn as sns

# Load NIST SD19 'by_class' dataset (PNG-based)
def load_nist_data(data_dir, img_size=(28, 28), max_samples_per_class=None):
    class_folders = sorted(os.listdir(data_dir))
    class_to_label = {folder: i for i, folder in enumerate(class_folders) if os.path.isdir(os.path.join(data_dir, folder))}
    label_to_class = {i: folder for folder, i in class_to_label.items()}
    images = []
    labels = []

    for class_folder in class_folders:
        class_path = os.path.join(data_dir, class_folder)
        if not os.path.isdir(class_path):
            continue
        # Check if class folder contains hsf_ subfolders or direct PNGs
        subdirs = [class_path]
        hsf_folders = sorted(os.listdir(class_path))
        if any('hsf_' in f for f in hsf_folders):
            subdirs = [os.path.join(class_path, f) for f in hsf_folders if os.path.isdir(os.path.join(class_path, f)) and 'hsf_' in f]

        img_count = 0
        for subdir in subdirs:
            img_files = [f for f in os.listdir(subdir) if f.endswith('.png')]
            if max_samples_per_class:
                img_files = img_files[:max_samples_per_class - img_count]
            for img_file in img_files:
                img_path = os.path.join(subdir, img_file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    print(f"Warning: failed to load {img_path}")
                    continue
                img = cv2.resize(img, img_size)
                images.append(img)
                labels.append(class_to_label[class_folder])
                img_count += 1
                if max_samples_per_class and img_count >= max_samples_per_class:
                    break
            if max_samples_per_class and img_count >= max_samples_per_class:
                break

    return np.array(images), np.array(labels), class_to_label, label_to_class

# Load data
data_dir = './by_class'
images, labels, class_to_label, label_to_class = load_nist_data(data_dir, max_samples_per_class=1000)

# Normalize and reshape
images = images.astype('float32') / 255.0
images = images.reshape(-1, 28, 28, 1)

# One-hot encode labels
labels_cat = to_categorical(labels, num_classes=62)

# Train-test split
train_images, test_images, train_labels, test_labels = train_test_split(
    images, labels_cat, test_size=0.1, random_state=42
)

print(f"Training samples: {train_images.shape[0]}")
print(f"Testing samples: {test_images.shape[0]}")

# Function to build CNN model
def build_cnn_model(
    filter_sizes=(32, 64, 64),
    kernel_size=(3, 3),
    num_conv_layers=3,
    pool_type='max',
    activation='relu'
):
    model = models.Sequential()
    for i in range(num_conv_layers):
        if i == 0:
            model.add(layers.Conv2D(
                filter_sizes[i], kernel_size, activation=activation,
                input_shape=(28, 28, 1), padding='same'
            ))
        else:
            model.add(layers.Conv2D(
                filter_sizes[i], kernel_size, activation=activation, padding='same'
            ))
        if pool_type == 'max':
            model.add(layers.MaxPooling2D((2, 2)))
        else:
            model.add(layers.AveragePooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation=activation))
    model.add(layers.Dense(62, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Function to evaluate and report results
def evaluate_model(model, test_images, test_labels, label_to_class, config_name):
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=0)
    predictions = model.predict(test_images, verbose=0)
    predicted_labels = np.argmax(predictions, axis=1)
    true_labels = np.argmax(test_labels, axis=1)

    # F1 scores
    f1_micro = f1_score(true_labels, predicted_labels, average='micro')
    f1_macro = f1_score(true_labels, predicted_labels, average='macro')
    f1_weighted = f1_score(true_labels, predicted_labels, average='weighted')
    f1_classwise = f1_score(true_labels, predicted_labels, average=None)

    # Confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)
    misclassifications = []
    for i in range(62):
        for j in range(62):
            if i != j and cm[i, j] > 0:
                misclassifications.append(((i, j), cm[i, j]))
    misclassifications.sort(key=lambda x: x[1], reverse=True)

    # Prepare text output
    output_lines = []
    output_lines.append(f"--- Configuration: {config_name} ---")
    output_lines.append(f"Test Accuracy: {test_acc:.4f}")
    output_lines.append(f"F1 Micro: {f1_micro:.4f}")
    output_lines.append(f"F1 Macro: {f1_macro:.4f}")
    output_lines.append(f"F1 Weighted: {f1_weighted:.4f}\n")
    output_lines.append("Per-class F1 Scores:")
    for i, score in enumerate(f1_classwise):
        char = chr(int(label_to_class[i], 16)) if label_to_class[i] >= '41' else label_to_class[i][-1]
        output_lines.append(f"Class '{char}': F1 Score = {score:.4f}")
    output_lines.append("\nTop 5 misclassified pairs:")
    for (true, pred), count in misclassifications[:5]:
        true_char = chr(int(label_to_class[true], 16)) if label_to_class[true] >= '41' else label_to_class[true][-1]
        pred_char = chr(int(label_to_class[pred], 16)) if label_to_class[pred] >= '41' else label_to_class[pred][-1]
        output_lines.append(f"True: '{true_char}' predicted as '{pred_char}': {count} times")

    # Save text output to file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    text_filename = os.path.join(script_dir, f"{config_name}.txt")
    with open(text_filename, 'w') as f:
        f.write('\n'.join(output_lines))

    # Print results
    print('\n'.join(output_lines))

    # Plot and save confusion matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=False, cmap='Blues')
    plt.title(f"Confusion Matrix - {config_name}")
    plt.xlabel('Predicted')
    plt.ylabel('True')
    cm_filename = os.path.join(script_dir, f"{config_name}_cm.png")
    plt.savefig(cm_filename, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    return f1_micro, f1_macro, f1_weighted, f1_classwise, cm, predicted_labels, true_labels

# Visualize sample predictions
def plot_sample_predictions(images, true_labels, predicted_labels, label_map, config_name, num_samples=15):
    plt.figure(figsize=(15, 3))
    for i in range(num_samples):
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(images[i].reshape(28, 28), cmap='gray')
        true_char = chr(int(label_map[true_labels[i]], 16)) if label_map[true_labels[i]] >= '41' else label_map[true_labels[i]][-1]
        pred_char = chr(int(label_map[predicted_labels[i]], 16)) if label_map[predicted_labels[i]] >= '41' else label_map[predicted_labels[i]][-1]
        plt.title(f"T: {true_char}\nP: {pred_char}")
        plt.axis('off')
    plt.suptitle(f"Sample Predictions - {config_name}")
    plt.tight_layout()
    # Save the plot
    script_dir = os.path.dirname(os.path.abspath(__file__))
    samples_filename = os.path.join(script_dir, f"{config_name}_samples.png")
    plt.savefig(samples_filename, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

# Experiment configurations
experiments = [
    # 1. Effect of filter size and number
    {'name': 'Small Filters (16,32,32)', 'filter_sizes': (16, 32, 32), 'kernel_size': (3, 3), 'num_conv_layers': 3, 'pool_type': 'max', 'activation': 'relu'},
    {'name': 'Large Filters (64,128,128)', 'filter_sizes': (64, 128, 128), 'kernel_size': (3, 3), 'num_conv_layers': 3, 'pool_type': 'max', 'activation': 'relu'},
    {'name': 'Larger Kernel (5x5)', 'filter_sizes': (32, 64, 64), 'kernel_size': (5, 5), 'num_conv_layers': 3, 'pool_type': 'max', 'activation': 'relu'},
    # 2. Effect of number of conv layers
    {'name': '2 Conv Layers', 'filter_sizes': (32, 64), 'kernel_size': (3, 3), 'num_conv_layers': 2, 'pool_type': 'max', 'activation': 'relu'},
    {'name': '4 Conv Layers', 'filter_sizes': (32, 64, 64, 64), 'kernel_size': (3, 3), 'num_conv_layers': 4, 'pool_type': 'max', 'activation': 'relu'},
    # 3. Max vs Average Pooling
    {'name': 'Average Pooling', 'filter_sizes': (32, 64, 64), 'kernel_size': (3, 3), 'num_conv_layers': 3, 'pool_type': 'average', 'activation': 'relu'},
    # 4. Different activations
    {'name': 'Sigmoid Activation', 'filter_sizes': (32, 64, 64), 'kernel_size': (3, 3), 'num_conv_layers': 3, 'pool_type': 'max', 'activation': 'sigmoid'},
    {'name': 'Tanh Activation', 'filter_sizes': (32, 64, 64), 'kernel_size': (3, 3), 'num_conv_layers': 3, 'pool_type': 'max', 'activation': 'tanh'}
]

# Run experiments
results = []
for exp in experiments:
    print(f"\nRunning experiment: {exp['name']}")
    model = build_cnn_model(
        filter_sizes=exp['filter_sizes'],
        kernel_size=exp['kernel_size'],
        num_conv_layers=exp['num_conv_layers'],
        pool_type=exp['pool_type'],
        activation=exp['activation']
    )
    history = model.fit(
        train_images, train_labels,
        epochs=50,
        batch_size=64,
        validation_split=0.1,
        verbose=1
    )
    metrics = evaluate_model(model, test_images, test_labels, label_to_class, exp['name'])
    f1_micro, f1_macro, f1_weighted, f1_classwise, cm, predicted_labels, true_labels = metrics
    results.append((exp['name'], (f1_micro, f1_macro, f1_weighted, f1_classwise, cm)))
    # Plot sample predictions for this experiment
    plot_sample_predictions(test_images, true_labels, predicted_labels, label_to_class, exp['name'])