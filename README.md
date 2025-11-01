# 🍎 Fruit Quality Classifier: Deep Learning Project

### 🎓 Introduction to Deep Learning at Albert School

This repository hosts a Deep Learning project developed as part of an introductory course at Albert School. The goal is to classify fruit quality using a computer vision model.

---

## 🚀 Try the App Live!

Experience the model's predictions instantly:

| Platform | Link |
| :--- | :--- |
| **Streamlit Web App** | **[https://workdeeplearningm1.streamlit.app](https://workdeeplearningm1.streamlit.app)** |

---

## 💡 Project Overview

The **Fruit Quality Classification Project** utilizes the **FruQ-DB** (Fruit Quality Dataset) to train an image classification model. This model is designed to determine the condition of a fruit from its image, categorizing it into one of three quality levels.

### Key Features
* **Transfer Learning:** Uses a pre-trained **MobileNetV2** model for high-performance classification.
* **Image Augmentation:** Employs techniques like random flips, rotation, zoom, brightness, and contrast to improve model robustness.
* **Web Application:** Features a simple, interactive **Streamlit app** for real-time testing with user-uploaded images.
* **Data Preparation:** Includes dedicated steps for image preprocessing, notably the removal of black backgrounds.

---

## 📊 Data Used: FruQ-DB

The project relies on the **Fruit Quality Dataset (FruQ-DB)**, a robust collection of images covering **11 fruit varieties** across three quality categories.

| Class | Number of Images | Description |
| :--- | :--- | :--- |
| **Fresh** | 2,182 | Ready for consumption |
| **Mild (Slightly Damaged)** | 1,364 | Minor signs of damage or decay |
| **Rotten** | 2,101 | Clear signs of spoilage |
| **Total** | **5,647** | |

**Fruit Varieties Included:** Banana, Cucumber, Grape, Persimmon, Papaya, Peach, Pear, Bell Pepper, Strawberry, Tomato, Watermelon.

| Key Training Detail | Value |
| :--- | :--- |
| **Image Size** | 224x224 pixels |
| **Training Sample** | 100 images per category (total of 300 images) |

### 📚 Sources
* **Dataset Link:** [https://zenodo.org/records/7224690](https://zenodo.org/records/7224690)
* *Note: Source YouTube time-lapse videos are linked on the Zenodo page.*

---

## ⚙️ Repository Structure

This project is organized into the following key files:

| File Name | Description |
| :--- | :--- |
| `app.py` | **Streamlit Web Application** script for interactive model testing. |
| `final_version.ipynb` | **Main Deep Learning Notebook** containing data loading, preprocessing, MobileNetV2 **Transfer Learning** setup, training, evaluation, and plotting (Accuracy/Loss curves, Confusion Matrix). |
| `cleaning_black_background.ipynb` | Dedicated notebook for initial **image cleaning** and black background removal steps. |
| `requirements.txt` | List of Python dependencies required to run the notebooks and the Streamlit app. |
| `fruit_classifier.keras` | **Trained model file** (required by `app.py`). |

---

## 🛠️ Local Setup and Installation

### Prerequisites
* Python 3.8+
* The trained model file (`fruit_classifier.keras`) must be in the same directory as `app.py`.

### Installation Steps

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YourUsername/work_deep_learning_m1.git](https://github.com/YourUsername/work_deep_learning_m1.git)
    cd work_deep_learning_m1
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Running the App Locally

To launch the interactive web application on your machine:

```bash
streamlit run app.py
