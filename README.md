# Brain Tumor MRI Classifier

A Streamlit application for classifying brain MRI images with a trained ResNet50 model. It predicts one of four classes and can display class probabilities and a Grad-CAM heatmap to help visualize image regions that contributed to the prediction.

> **Important:** This project is an educational and research prototype, not a medical diagnostic tool. Do not use its predictions for medical decisions; consult a qualified healthcare professional.

## Features

- Upload an MRI image in JPG, JPEG, or PNG format.
- Classify the image as glioma, meningioma, no tumor, or pituitary tumor.
- View the model's confidence and per-class probabilities.
- Generate a Grad-CAM visualization for the prediction.

## Requirements

- Python 3.11
- The trained model file at `models/resnet50/best_resnet50.keras`
- Git LFS to clone/download the model files from GitHub

## Setup

Open PowerShell in the project root and create a virtual environment if one does not already exist:

```powershell
py -3.11 -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the application dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The model files are stored with [Git LFS](https://git-lfs.com/). Install Git LFS and run `git lfs install` before cloning this repository. The raw MRI dataset under `data/raw/` is intentionally excluded from GitHub; place your dataset there locally if you need it for notebook workflows.

## Run the app

From the project root, run:

```powershell
python -m streamlit run .\app\app.py
```

Streamlit will print a local URL, usually `http://localhost:8501`, which you can open in your browser. Stop the server with `Ctrl+C` in the terminal.

## Project structure

```text
Tumor_Detection/
├── app/
│   └── app.py
├── data/
│   ├── processed/
│   └── raw/
├── models/
│   ├── cnn/
│   ├── mobilenetv2/
│   ├── resnet50/
│   └── vgg16/
├── notebooks/
├── results/
├── .gitignore
└── README.md
```

The Streamlit app loads the ResNet50 model from `models/resnet50/best_resnet50.keras`. Make sure the model file is present before running the app.
