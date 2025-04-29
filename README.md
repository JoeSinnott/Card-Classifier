# 🃏 Card Classifier

**Card Classifier** is a machine learning project that identifies playing cards from images using a custom-trained neural network.

The project consists of:
- A **PyTorch**-trained neural network model for classification.
- A **Flask API** backend that serves model predictions.
- A **web-based demo** built with HTML, TailwindCSS, and JavaScript.

🔗 **Live Demo:** [joesinnott.github.io/Card-Classifier/](https://joesinnott.github.io/Card-Classifier/)

---

## 📷 Project Overview

This application allows users to upload an image of a playing card, which is then classified into one of 53 categories (all 52 standard cards + Joker). The model was trained using PyTorch and served via a lightweight Flask API.

---

## 🚀 Technologies Used

- **PyTorch** — Model training and inference
- **Flask** — API backend
- **Flask-CORS** — Cross-origin request handling
- **PIL (Pillow)** — Image preprocessing
- **TailwindCSS** — Web frontend styling
- **JavaScript (Fetch API)** — Uploading files and displaying results

---

## 📂 Repository Structure

```
/Card-Classifier
├── /api/                   # Flask API backend
│   ├── app.py              # Main server code
│   ├── model_weights.pth   # Trained model weights
│
├── /frontend/              # Static website files
│   ├── index.html          # Web interface
│
├── /training/              # Model training scripts
│   ├── train_model.py      # Model training code
│
├── README.md               # This file
├── requirements.txt        # Python dependencies
├── railway.json            # Deployment configuration file
├── LICENSE                 # Repo license
```

---

## 🛠️ How It Works

1. **Frontend**: The user uploads an image through the web page.
2. **API Call**: The image is sent via a POST request to the Flask server.
3. **Model Inference**: The server loads the image, preprocesses it, feeds it into the PyTorch model, and gets a prediction.
4. **Result**: The predicted card name is returned and displayed on the web page.

---

## 🧠 Model Details

- **Architecture**: Based on EfficientNet-B0 (via [timm](https://github.com/huggingface/pytorch-image-models) library)
- **Input Size**: 128x128 resized images
- **Classes**: 53 (Standard deck of cards + Joker)
- **Training Framework**: PyTorch

---

## 📦 Installation (Local Development)

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/Card-Classifier.git
   cd Card-Classifier
   ```

2. Install Python dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the Flask server:

   ```bash
   cd api
   python card_api.py
   ```

4. Open `frontend/index.html` locally or host it via GitHub Pages for deployment.

---

## 📄 Requirements

- Python 3.8+
- PyTorch
- torchvision
- timm
- Flask
- Flask-CORS
- Pillow

All requirements are listed in `requirements.txt`.

---

## 📜 License

This project is licensed under the MIT License.  
Feel free to use, modify, and share it!

---

## ✨ Acknowledgements

- [PyTorch](https://pytorch.org/)
- [timm](https://github.com/huggingface/pytorch-image-models)
- [TailwindCSS](https://tailwindcss.com/)
- [Playing cards datasets and resources used for training](https://www.kaggle.com/datasets/gpiosenka/cards-image-datasetclassification)
