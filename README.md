# Human-Behaviour-Determination-Using-Deep-Learning-Techniques-CNN-LSTM

## 🚀 Overview  
This project implements a **Human Behaviour Determination** system using **Deep Learning techniques (CNN + LSTM)**. The model extracts spatiotemporal features from video frames to classify different human activities and behaviors accurately. It combines **Convolutional Neural Networks (CNNs)** for feature extraction and **Long Short-Term Memory (LSTM)** networks for sequential learning, making it robust for real-time applications.

## 🔥 Features  
- ✅ **Real-time video processing** using OpenCV/MediaPipe  
- ✅ **CNN for feature extraction** from frames  
- ✅ **LSTM for temporal sequence modeling**  
- ✅ **Pretrained CNN models** (MobileNet, ResNet, VGG16, etc.)  
- ✅ **Dataset preprocessing** for optimal model performance  
- ✅ **Graphical User Interface (GUI) support** (optional)  

## 🛠 Technology Stack  
- **Python** (TensorFlow/Keras, OpenCV, NumPy, Pandas)  
- **Deep Learning** (CNN + LSTM architecture)  
- **Data Processing** (Feature extraction, Augmentation, Normalization)  
- **Visualization** (Matplotlib, Seaborn for performance analysis)  

## 🎯 How It Works  
1. **Frame Extraction**: Extracts keyframes from input video streams.  
2. **Feature Extraction (CNN)**: Uses a pretrained CNN to extract meaningful spatial features.  
3. **Sequence Modeling (LSTM)**: Uses LSTM to learn patterns from feature sequences over time.  
4. **Prediction & Classification**: Determines the human behavior class based on learned patterns.  

## 📌 Installation & Usage  
### 1️⃣ Clone the Repository  
```sh
git clone https://github.com/your-username/Human-Behaviour-Determination-DeepLearning.git
cd Human-Behaviour-Determination-DeepLearning
```
2️⃣ Install Dependencies
```sh
pip install -r requirements.txt
```

3️⃣ Run the Model
```sh
python main.py

```

📂 Dataset
The model can be trained on benchmark datasets like UCF101, HMDB51, or custom video datasets for human activity recognition.
This model is Trained in UFC101 Dataset
📊 Results & Performance
Achieves high accuracy in behavior classification.
Generalizes well across various real-world scenarios.
Can be deployed for surveillance, healthcare monitoring, and behavioral analysis applications.
🔮 Future Improvements
Implementing Transformer-based models for enhanced sequence learning.
Fine-tuning with self-supervised learning techniques.
Deploying as a real-time mobile application.
🤝 Contributing
Feel free to contribute! Fork the repo, create a branch, and submit a pull request.

📜 License
This project is open-source under the MIT License.

🚀 **Let me know if you need any modifications!** 🚀

Steps To Run The Code:::
1️⃣ Step 1 :Create Environment with Specified Name
```sh
python -m venv venv
venv\Scripts\activate
```
2️⃣Step 2: Install Dependencies
```sh
pip install tensorflow opencv-python numpy pandas matplotlib seaborn

```
3️⃣ Step 3: Download or Prepare the Dataset
 ```sh
use the videos in the dataset to view convert .AVI to .MP4 using any online Website

```
4️⃣Step 4:Check For The Versions of Python Use Above (3.10. -3.11. )

 ```sh
Other versions are not Supporting tenserFlow As of 2025 Feb

```
5️⃣Step 5: Run the Streamlit App
Run the following command in your terminal:

```sh

streamlit run app.py
```
This will open a web-based UI where you can upload a video and get behavior predictions. 🚀

📌 Features of This App
✅ Simple UI for video upload and classification
✅ Efficient CNN feature extraction
✅ LSTM for sequential learning
✅ Displays behavior predictions with confidence scores of Top three






