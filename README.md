# 🖐️ Hand Gesture & Sign Recognition (Numbers, Alphabets & Daily Actions)
This project implements a real-time hand gesture recognition system using MediaPipe, OpenCV, and XGBoost. 

It can detect and classify Numbers (0–9), Alphabets (A–Z), and Daily Action Signs (like LOVE, THANK YOU, YES, NO, STOP, FOOD, etc.).

# Features

- Data Collection: Capture hand landmarks from your webcam for numbers, alphabets, and daily actions.

- Training: Train an XGBoost model on collected gestures.

- Real-Time Detection: Predict hand signs live using your webcam.

- Dataset Export: Save dataset as .pkl or structured .npy files for each gesture.

- Daily Actions Support: Recognize 14 common daily life signs (e.g., LOVE, THANK YOU, HAPPY, STOP).

- Menu-Driven CLI: Easy to use console interface for managing data, training, and predictions.


# 📂 Project Structure
├── Code.py                    # Main script (menu-driven system)

├── hand_sign_data_full.pkl      # Saved dataset (generated after collection)

├── hand_sign_model_full.pkl     # Trained model (generated after training)

├── dataset/                     # Exported dataset in folder format

└── README.md                    # Project documentation

# 🛠️ Installation

- Clone the repository:

  git clone https://github.com/yourusername/hand-sign-recognition.git
  cd hand-sign-recognition


- Install dependencies:

  pip install opencv-python mediapipe numpy scikit-learn xgboost


# ▶️ Usage

- Run the program:

  python Code.py

# Menu Options

1. Collect training data (numbers, alphabets, daily actions)

2. Append more data to existing dataset

3. Train model (XGBoost)

4. Real-time detection using webcam

5. Load existing data

6. Show data statistics

7. Clear all data (reset project)

8. Show daily actions list with key mappings

9. Exit

10. Export dataset to structured folders

# 🎮 Controls (During Data Collection)

0–9 → Numbers

A–Z → Alphabets

Special Keys / F1–F12 → Daily Actions (LOVE, THANK YOU, etc.)

SPACE → Stop collecting current sign

S → Save data

Q → Quit

# 📊 Supported Sign Categories
- 🔢 Numbers (0–9)

  0, 1, 2, 3, 4, 5, 6, 7, 8, 9

- 🔤 Alphabets (A–Z)

  A, B, C, ... Z

- 🏷️ Daily Actions (14 Signs)

  LOVE ❤️, THANK YOU 🙏, YOU ARE WELCOME 🤗, HAPPY 😀, SAD 😢, GOOD 👍, BAD 👎, YES ✅, NO ❌, MOVE ↔️, STOP ✋, ALL THE BEST 🎉, FOOD 🍽️, SLEEP 😴

# 📌 Note:

- Ensure good lighting and clear hand visibility during data collection.

- You may need to adjust key mappings for daily actions in newslr.py.

- The model requires enough samples per class (ideally 1500-2000) for stable training.



