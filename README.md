# Anchal-kumari_25SCS1003001312_IILM-GN
# ✍️ Stylodel - Author Verification System

## 📌 Overview

Stylodel is an AI/ML-based project that verifies whether a given text is written by a specific author. It uses machine learning techniques and text similarity analysis to compare writing styles and predict authorship.

---

## 🎯 Objective

The main goal of this project is to:

* Analyze writing style using text features
* Compare author samples with a test document
* Predict whether both texts are written by the same author

---

## ⚙️ Technologies Used

* Python 🐍
* Tkinter (for GUI)
* NumPy
* Scikit-learn
* TF-IDF Vectorizer
* Support Vector Machine (SVM)

---

## 📂 Project Structure

stylodel/
│── training_unknown/      # Unknown author training files (.txt)
│── main.py               # Main application file
│── README.md
│── requirements.txt

---

## 🧠 How It Works

* The system loads multiple unknown text files to train a model
* It extracts:

  * Word patterns
  * Sentence structures
  * Writing style features
* It creates similarity scores between texts
* A machine learning model (SVM) predicts authorship

---

## 🚀 How to Run the Project

1. Install required libraries:
   pip install -r requirements.txt

2. Make sure you have a folder:
   training_unknown/
   (Add at least 3 .txt files of different writing styles)

3. Run the program:
   python main.py

4. Use the GUI:

   * Upload Author A sample files
   * Upload a test file
   * Click "Start verification"

---

## 📊 Output

The system gives one of the following results:

* LIKELY Author A ✅
* DON'T KNOW ⚠️
* LIKELY NOT Author A ❌

Along with a probability score.

---

## 📈 Features

* Simple graphical interface (GUI)
* Uses TF-IDF and cosine similarity
* Style-based analysis (sentence length, punctuation, etc.)
* Machine learning classification using SVM

---

## ⚠️ Requirements

* Minimum 3 unknown author text files
* Text files should not be too short

---

## 👨‍💻 Author

* Anchal Kumari
* IILM UNIVERSITY GN

