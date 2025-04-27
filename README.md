```markdown
# 🧠 CNN Experiments on NIST SD19 Dataset

![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

This project explores the performance of Convolutional Neural Networks (CNNs) on handwritten character recognition using the [NIST Special Database 19 (SD19)](https://www.nist.gov/srd/nist-special-database-19).  
Different CNN configurations are systematically compared to study the impact of:
- Filter sizes and numbers
- Kernel sizes
- Number of convolutional layers
- Pooling types (Max vs Average)
- Activation functions (ReLU, Sigmoid, Tanh)

---

## 📂 Project Structure

- **Dataset**  
  Expected at `./by_class`, where each character has its own subfolder containing PNG images (possibly inside `hsf_#` subfolders).

- **Script Workflow**  
  - Load and preprocess images
  - Build configurable CNN models
  - Train and evaluate multiple model configurations
  - Save confusion matrices, sample predictions, and evaluation metrics

---

## 🚀 Getting Started

### 1. Install Dependencies

```bash
pip install numpy opencv-python matplotlib seaborn scikit-learn tensorflow
```

### 2. Prepare the Dataset

You must have the NIST SD19 dataset available in PNG format, organized like:

```
by_class/
  |- 30/
      |- hsf_0/
          |- *.png
  |- 31/
      |- hsf_1/
          |- *.png
  ...
```

> ⚡ **Note:** Adjust the folder parsing logic if your dataset structure differs.

### 3. Run the Script

```bash
python your_script_name.py
```

All results (plots and text files) will be saved automatically in the script's directory.

---

## 🧪 Experiment Configurations

| Configuration             | Description                          |
| :------------------------- | :----------------------------------- |
| **Small Filters (16,32,32)** | Smaller filter sizes |
| **Large Filters (64,128,128)** | Larger filter sizes |
| **Larger Kernel (5x5)** | Larger convolution kernels |
| **2 Conv Layers** | Fewer convolutional layers |
| **4 Conv Layers** | More convolutional layers |
| **Average Pooling** | Using AveragePooling instead of MaxPooling |
| **Sigmoid Activation** | Using sigmoid activation function |
| **Tanh Activation** | Using tanh activation function |

Each model will output:
- Test Accuracy
- Micro, Macro, and Weighted F1 scores
- Per-class F1 scores
- Top 5 most common misclassifications
- Confusion matrix heatmap
- Sample prediction visualizations

---

## 📊 Example Outputs

### Confusion Matrix
> (Automatically saved as PNG)

![Confusion Matrix Example](https://github.com/Amir0234-afk/CNN-Experiments-on-NIST-SD19-Handwritten-Characters/blob/main/images/confusion%20matrix%20example.png)

### Sample Predictions
> (Automatically saved as PNG)

![Sample Predictions Example](https://github.com/Amir0234-afk/CNN-Experiments-on-NIST-SD19-Handwritten-Characters/blob/main/images/sample%20prediction%20example.png)

---

## 📌 Notes

- Misclassified examples and per-class F1 scores are also logged to text files.
- All experiments are trained for **50 epochs** with **batch size = 64**.
- Models use **Adam** optimizer and **categorical crossentropy** loss.

---

## 📄 License

This project is licensed under the **MIT License**.  
Feel free to use, modify, and distribute with credit. 🎓

---

## 🌟 Acknowledgments
- [NIST SD19 Dataset](https://www.nist.gov/srd/nist-special-database-19)
- TensorFlow, scikit-learn, and OpenCV communities

---
