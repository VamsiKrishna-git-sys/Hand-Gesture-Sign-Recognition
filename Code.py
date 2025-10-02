import cv2
import mediapipe as mp
import numpy as np
import time
from collections import Counter
import pickle
import os
import shutil


class HandSignDetector:
    def __init__(self):
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils

        # For data collection and training
        self.data = []
        self.labels = []
        self.current_label = None
        self.collecting = False

        # For prediction
        self.model = None
        self.prediction_buffer = []
        self.buffer_size = 10

        # Sign labels for Numbers (0-9), Alphabets (A-Z), and Specific Daily Actions
        self.sign_names = {}
        self.sign_categories = {}  # Track which category each sign belongs to

        # Numbers 0-9 (labels 0-9)
        for i in range(10):
            self.sign_names[i] = str(i)
            self.sign_categories[i] = "Number"

        # Alphabets A-Z (labels 10-35)
        for i in range(26):
            label = 10 + i
            self.sign_names[label] = chr(65 + i)  # A=10, B=11, ..., Z=35
            self.sign_categories[label] = "Alphabet"

        # Specific Daily Actions (labels start from 36)
        # Ordered as requested for consistent label assignment
        specific_daily_actions = [
            "LOVE", "THANK_YOU", "YOU_ARE_WELCOME", "HAPPY", "SAD",
            "GOOD", "BAD", "YES", "NO", "MOVE", "STOP", "ALL_THE_BEST",
            "FOOD", "SLEEP"
        ]

        # Assign labels starting from 36
        self.daily_action_keys = {}  # To map specific keys to these new labels
        current_daily_action_label = 36
        for action in specific_daily_actions:
            # Assign labels sequentially
            self.sign_names[current_daily_action_label] = action
            self.sign_categories[current_daily_action_label] = "Daily Action"
            current_daily_action_label += 1

        # Manually map keys to the specific daily actions
        # You'll need to decide on the keyboard keys for these 14 actions.
        # For demonstration, I'll use a sequential mapping with numbers and F-keys for examples.
        # IMPORTANT: Adjust these key mappings (ord('key') or OpenCV F-key codes) as needed for your preference.
        # Example mapping (you can change these):
        self.daily_action_keys = {
            ord('!'): self._get_label_by_name("LOVE"),
            ord('@'): self._get_label_by_name("THANK_YOU"),
            ord('#'): self._get_label_by_name("YOU_ARE_WELCOME"),
            ord('$'): self._get_label_by_name("HAPPY"),
            ord('%'): self._get_label_by_name("SAD"),
            ord('^'): self._get_label_by_name("GOOD"),
            ord('&'): self._get_label_by_name("BAD"),
            ord('*'): self._get_label_by_name("YES"),  # 'e' for yEs
            ord('('): self._get_label_by_name("NO"),
            ord(')'): self._get_label_by_name("MOVE"),  # 'v' for moVe
            ord('_'): self._get_label_by_name("STOP"),  # 'p' for stoP
            ord('+'): self._get_label_by_name("ALL_THE_BEST"),  # 'a' for All
            ord('-'): self._get_label_by_name("FOOD"),
            ord('/'): self._get_label_by_name("SLEEP"),  # 'z' for sZzz
            # Example using F-keys, assuming OpenCV's waitKey can capture them (might vary by system)
            7340032: self._get_label_by_name("LOVE"),  # F1
            7405568: self._get_label_by_name("THANK_YOU"),  # F2
            7471104: self._get_label_by_name("YOU_ARE_WELCOME"),  # F3
            7536640: self._get_label_by_name("HAPPY"),  # F4
            7602176: self._get_label_by_name("SAD"),  # F5
            7667712: self._get_label_by_name("GOOD"),  # F6
            7733248: self._get_label_by_name("BAD"),  # F7
            7798784: self._get_label_by_name("YES"),  # F8
            7864320: self._get_label_by_name("NO"),  # F9
            7929856: self._get_label_by_name("MOVE"),  # F10
            7995392: self._get_label_by_name("STOP"),  # F11
            8060928: self._get_label_by_name("ALL_THE_BEST"),  # F12
            # Add more specific F-key mappings if you have more than 12 actions and can detect them
            # For the last two (FOOD, SLEEP), if F-keys are used up, you might need other special chars or combinations.
        }
        # Filter out None values in daily_action_keys if a name wasn't found
        self.daily_action_keys = {k: v for k, v in self.daily_action_keys.items() if v is not None}

    def _get_label_by_name(self, name):
        """Helper to get label from sign name"""
        for label, sign_name in self.sign_names.items():
            if sign_name == name:
                return label
        return None

    def extract_landmarks(self, hand_landmarks):
        """Extract normalized landmark coordinates"""
        landmarks = []
        if hand_landmarks:
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
        return landmarks

    def get_label_from_key(self, key):
        """Convert key press to label - SUPPORTS NUMBERS, ALPHABETS, AND DAILY ACTIONS"""
        # Handle numbers 0-9
        if key in [ord(str(i)) for i in range(10)]:
            return int(chr(key))

        # Handle alphabets A-Z (both upper and lower case)
        if key >= ord('a') and key <= ord('z'):
            return 10 + (key - ord('a'))  # A=10, B=11, etc.
        elif key >= ord('A') and key <= ord('Z'):
            return 10 + (key - ord('A'))  # A=10, B=11, etc.

        # Handle daily actions
        if key in self.daily_action_keys:
            return self.daily_action_keys[key]

        return None

    def get_category_from_label(self, label):
        """Get the category of a sign from its label"""
        return self.sign_categories.get(label, "Unknown")

    def collect_data(self):
        """Data collection mode - SUPPORTS NUMBERS, ALPHABETS, AND DAILY ACTIONS"""
        print("=== DATA COLLECTION MODE ===")
        print("Available signs for collection:")

        # Display numbers
        numbers = [self.sign_names[i] for i in range(10)]
        print(f"Numbers (0-9): {', '.join(numbers)}")

        # Display alphabets
        alphabets = [self.sign_names[i] for i in range(10, 36)]
        print(f"Alphabets (A-Z): {', '.join(alphabets)}")

        # Display daily actions and their key mappings
        print("Daily Actions (press corresponding key):")
        for key_code, label in self.daily_action_keys.items():
            key_char = ""
            if key_code < 256:  # Regular character
                key_char = f"'{chr(key_code)}'"
            else:  # F-key or special key code
                # This part is a bit tricky as OpenCV's waitKey doesn't always give distinct F-key codes consistently across systems
                # For common F-keys it might work, but it's not universally robust for all special keys.
                # The best way to identify these is by printing `key` during data collection.
                if key_code == 7340032:
                    key_char = "F1"
                elif key_code == 7405568:
                    key_char = "F2"
                elif key_code == 7471104:
                    key_char = "F3"
                elif key_code == 7536640:
                    key_char = "F4"
                elif key_code == 7602176:
                    key_char = "F5"
                elif key_code == 7667712:
                    key_char = "F6"
                elif key_code == 7733248:
                    key_char = "F7"
                elif key_code == 7798784:
                    key_char = "F8"
                elif key_code == 7864320:
                    key_char = "F9"
                elif key_code == 7929856:
                    key_char = "F10"
                elif key_code == 7995392:
                    key_char = "F11"
                elif key_code == 8060928:
                    key_char = "F12"
                else:
                    key_char = f"Key_{key_code}"  # Fallback for unmapped special keys

            print(f"  {key_char}: {self.sign_names.get(label, 'Unknown')}")

        print("\nControls:")
        print("- Press 0-9 for numbers")
        print("- Press A-Z for alphabets (case insensitive)")
        print("- Press the assigned key for Daily Actions (see list above)")
        print("- Press SPACE to stop collecting current sign")
        print("- Press 's' to save data")
        print("- Press 'q' to quit")
        print("\nYou can now collect ANY sign from all three categories!")

        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)

            # Draw landmarks
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                    # Collect data if in collection mode
                    if self.collecting and self.current_label is not None:
                        landmarks = self.extract_landmarks(hand_landmarks)
                        if landmarks:
                            self.data.append(landmarks)
                            self.labels.append(self.current_label)

            # Display current collection info
            if self.collecting and self.current_label is not None:
                current_sign = self.sign_names.get(self.current_label, 'Unknown')
                current_category = self.get_category_from_label(self.current_label)
                status_text = f"Collecting: {current_sign} ({current_category})"
            else:
                status_text = "Collection: Stopped"

            cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Total Samples: {len(self.data)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0),
                        2)

            # Show instructions on screen (simplified)
            cv2.putText(frame, "0-9: Numbers | A-Z: Alphabets | Custom keys: Daily Actions", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            cv2.putText(frame, "SPACE=Stop | S=Save | Q=Quit", (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

            cv2.imshow('Hand Sign Data Collection - All Categories', frame)

            key = cv2.waitKey(1) & 0xFF

            # Handle key presses
            if key == ord('q'):
                break
            elif key == ord(' '):
                self.collecting = False
                print(f"Stopped collecting. Total samples: {len(self.data)}")
            elif key == ord('s'):
                self.save_data()
            else:
                label = self.get_label_from_key(key)

                if label is not None:
                    self.current_label = label
                    self.collecting = True
                    sign_name = self.sign_names[label]
                    category = self.get_category_from_label(label)
                    print(f"Started collecting for: {sign_name} ({category})")

        cap.release()
        cv2.destroyAllWindows()

    def save_data(self):
        """Save collected data with all three categories"""
        if len(self.data) > 0:
            data_dict = {
                'data': np.array(self.data),
                'labels': np.array(self.labels),
                'sign_names': self.sign_names,
                'sign_categories': self.sign_categories
            }
            with open('hand_sign_data_full.pkl', 'wb') as f:
                pickle.dump(data_dict, f)
            print(f"Saved {len(self.data)} samples to hand_sign_data_full.pkl")

            # Show distribution of collected data by category
            unique_labels, counts = np.unique(self.labels, return_counts=True)
            print("Data distribution:")

            # Separate by categories
            numbers_data = []
            alphabets_data = []
            daily_actions_data = []

            for label, count in zip(unique_labels, counts):
                category = self.get_category_from_label(label)
                sign_name = self.sign_names[label]

                if category == "Number":
                    numbers_data.append(f"{sign_name}: {count}")
                elif category == "Alphabet":
                    alphabets_data.append(f"{sign_name}: {count}")
                elif category == "Daily Action":
                    daily_actions_data.append(f"{sign_name}: {count}")

            if numbers_data:
                print(f"  Numbers ({len(numbers_data)}): {', '.join(numbers_data)}")
            if alphabets_data:
                print(f"  Alphabets ({len(alphabets_data)}): {', '.join(alphabets_data)}")
            if daily_actions_data:
                print(f"  Daily Actions ({len(daily_actions_data)}): {', '.join(daily_actions_data)}")
        else:
            print("No data to save!")

    def load_data(self):
        """Load saved data"""
        try:
            with open('hand_sign_data_full.pkl', 'rb') as f:
                data_dict = pickle.load(f)
            self.data = data_dict['data'].tolist()
            self.labels = data_dict['labels'].tolist()

            if 'sign_names' in data_dict:
                self.sign_names = data_dict['sign_names']
            if 'sign_categories' in data_dict:
                self.sign_categories = data_dict['sign_categories']

            print(f"Loaded {len(self.data)} samples")

            # Show distribution of loaded data by category
            unique_labels, counts = np.unique(self.labels, return_counts=True)
            print("Loaded data distribution:")

            # Separate by categories
            numbers_data = []
            alphabets_data = []
            daily_actions_data = []

            for label, count in zip(unique_labels, counts):
                category = self.get_category_from_label(label)
                sign_name = self.sign_names[label]

                if category == "Number":
                    numbers_data.append(f"{sign_name}: {count}")
                elif category == "Alphabet":
                    alphabets_data.append(f"{sign_name}: {count}")
                elif category == "Daily Action":
                    daily_actions_data.append(f"{sign_name}: {count}")

            if numbers_data:
                print(f"  Numbers ({len(numbers_data)}): {', '.join(numbers_data)}")
            if alphabets_data:
                print(f"  Alphabets ({len(alphabets_data)}): {', '.join(alphabets_data)}")
            if daily_actions_data:
                print(f"  Daily Actions ({len(daily_actions_data)}): {', '.join(daily_actions_data)}")

            return True
        except FileNotFoundError:
            print("No saved data found. Please collect data first.")
            return False

    def train_model(self):
        """Train a classifier for all three categories"""
        print("Starting training process for all categories...")

        if len(self.data) == 0:
            print("No data available for training!")
            return

        try:
            print("Importing scikit-learn and xgboost...")
            from xgboost import XGBClassifier
            from sklearn.model_selection import train_test_split, GridSearchCV
            from sklearn.metrics import accuracy_score, classification_report
            from sklearn.preprocessing import LabelEncoder
            print("scikit-learn and xgboost imported successfully!")

            print("Converting data to numpy arrays...")
            X = np.array(self.data)
            y = np.array(self.labels)

            # Encode labels to 0-based consecutive integers
            self.label_encoder = LabelEncoder()
            y_encoded = self.label_encoder.fit_transform(y)
            print(f"Classes after encoding: {self.label_encoder.classes_}")

            print(f"Training with {len(X)} samples for {len(np.unique(y))} different signs...")
            print(f"Feature shape: {X.shape}")

            # Show what we're training on by category
            unique_labels, counts = np.unique(y, return_counts=True)
            print("Training on these signs:")

            numbers_training = []
            alphabets_training = []
            daily_actions_training = []

            for label, count in zip(unique_labels, counts):
                category = self.get_category_from_label(label)
                sign_name = self.sign_names[label]

                if category == "Number":
                    numbers_training.append(f"{sign_name} ({count})")
                elif category == "Alphabet":
                    alphabets_training.append(f"{sign_name} ({count})")
                elif category == "Daily Action":
                    daily_actions_training.append(f"{sign_name} ({count})")

            if numbers_training:
                print(f"  Numbers: {', '.join(numbers_training)}")
            if alphabets_training:
                print(f"  Alphabets: {', '.join(alphabets_training)}")
            if daily_actions_training:
                print(f"  Daily Actions: {', '.join(daily_actions_training)}")

            min_samples = min(counts)
            print(f"Minimum samples per class: {min_samples}")

            if min_samples < 2:
                print("WARNING: Some classes have less than 2 samples. Training may fail.")
                print("Proceeding without stratification...")
                X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
            else:
                print("Splitting data with stratification...")
                X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

            print(f"Training set size: {len(X_train)}")
            print(f"Test set size: {len(X_test)}")

            # Train model with GridSearchCV
            print("Setting up parameter grid for XGBoost...")
            # Good parameters for 90k+ samples and 40-50 classes (CPU parallelism)
            xgb = XGBClassifier(
                n_estimators=300,
                max_depth=10,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                gamma=0.1,
                random_state=42,
                n_jobs=-1,
                eval_metric='mlogloss',
                tree_method='hist'  # Fastest for CPU
            )
            print("Training XGBoost model with fixed parameters...")
            xgb.fit(X_train, y_train)
            print("Model training completed!")
            self.model = xgb

            # Test accuracy
            print("Evaluating model...")
            y_pred = self.model.predict(X_test)
            # Decode predictions and test labels
            y_pred_decoded = self.label_encoder.inverse_transform(y_pred)
            y_test_decoded = self.label_encoder.inverse_transform(y_test)
            accuracy = accuracy_score(y_test_decoded, y_pred_decoded)

            print(f"Model trained! Overall Accuracy: {accuracy:.3f}")

            # Detailed classification report
            print("\nDetailed Classification Report:")
            target_names = [f"{self.sign_names.get(label, f'Label_{label}')} ({self.get_category_from_label(label)})"
                            for label in sorted(np.unique(y))]
            print(classification_report(y_test_decoded, y_pred_decoded, target_names=target_names, zero_division=0))

            # Save model
            print("Saving model...")
            model_data = {
                'model': self.model,
                'sign_names': self.sign_names,
                'sign_categories': self.sign_categories,
                'label_encoder': self.label_encoder
            }
            with open('hand_sign_model_full.pkl', 'wb') as f:
                pickle.dump(model_data, f)
            print("Model saved to hand_sign_model_full.pkl")

        except ImportError as e:
            print(f"Import Error: {e}")
            print("Please install scikit-learn: pip install scikit-learn")
        except Exception as e:
            print(f"Training Error: {e}")
            print("Please check your data and try again.")

    def load_model(self):
        """Load trained model"""
        try:
            with open('hand_sign_model_full.pkl', 'rb') as f:
                model_data = pickle.load(f)
            self.model = model_data['model']

            if 'sign_names' in model_data:
                self.sign_names = model_data['sign_names']
            if 'sign_categories' in model_data:
                self.sign_categories = model_data['sign_categories']
            if 'label_encoder' in model_data:
                self.label_encoder = model_data['label_encoder']

            print("Model loaded successfully!")

            # Show what the model can recognize by category
            print("Model can recognize these signs:")
            if hasattr(self.model, 'classes_'):
                numbers = []
                alphabets = []
                daily_actions = []

                for class_label in sorted(self.model.classes_):
                    sign_name = self.sign_names.get(class_label, f'Label_{class_label}')
                    category = self.get_category_from_label(class_label)

                    if category == "Number":
                        numbers.append(sign_name)
                    elif category == "Alphabet":
                        alphabets.append(sign_name)
                    elif category == "Daily Action":
                        daily_actions.append(sign_name)

                if numbers:
                    print(f"  Numbers ({len(numbers)}): {', '.join(sorted(numbers))}")
                if alphabets:
                    print(f"  Alphabets ({len(alphabets)}): {', '.join(sorted(alphabets))}")
                if daily_actions:
                    print(f"  Daily Actions ({len(daily_actions)}): {', '.join(sorted(daily_actions))}")
            else:
                print(f"  Total signs: {len(self.sign_names)}")

            return True
        except FileNotFoundError:
            print("No trained model found. Please train a model first.")
            return False

    def predict_realtime(self):
        """Real-time prediction mode for all three categories"""
        if self.model is None:
            print("No model loaded!")
            return
        if not hasattr(self, 'label_encoder'):
            print("No label encoder loaded!")
            return

        print("=== REAL-TIME PREDICTION MODE ===")
        print("Press 'q' to quit")

        # Show what the model can recognize by category
        print("Model can recognize these signs:")
        if hasattr(self.model, 'classes_'):
            numbers = []
            alphabets = []
            daily_actions = []

            for class_label in sorted(self.model.classes_):
                sign_name = self.sign_names.get(class_label, f'Label_{class_label}')
                category = self.get_category_from_label(class_label)

                if category == "Number":
                    numbers.append(sign_name)
                elif category == "Alphabet":
                    alphabets.append(sign_name)
                elif category == "Daily Action":
                    daily_actions.append(sign_name)

            if numbers:
                print(f"  Numbers: {', '.join(sorted(numbers))}")
            if alphabets:
                print(f"  Alphabets: {', '.join(sorted(alphabets))}")
            if daily_actions:
                print(f"  Daily Actions: {', '.join(sorted(daily_actions))}")

        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)

            prediction_text = "No hand detected"
            confidence_text = ""
            category_text = ""

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw landmarks
                    self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                    # Make prediction
                    landmarks = self.extract_landmarks(hand_landmarks)
                    if landmarks:
                        try:
                            prediction = self.model.predict([landmarks])[0]
                            probabilities = self.model.predict_proba([landmarks])[0]
                            confidence = max(probabilities)

                            # Decode prediction
                            stable_prediction = self.label_encoder.inverse_transform([prediction])[0]

                            # Use buffer for stable predictions
                            self.prediction_buffer.append(stable_prediction)
                            if len(self.prediction_buffer) > self.buffer_size:
                                self.prediction_buffer.pop(0)

                            # Get most common prediction
                            if len(self.prediction_buffer) >= 5:
                                stable_prediction = Counter(self.prediction_buffer).most_common(1)[0][0]
                                predicted_sign = self.sign_names.get(stable_prediction, 'Unknown')
                                category = self.get_category_from_label(stable_prediction)

                                prediction_text = f"Sign: {predicted_sign}"
                                confidence_text = f"Confidence: {confidence:.3f}"
                                category_text = f"Category: {category}"

                        except Exception as e:
                            prediction_text = "Prediction error"
                            confidence_text = ""
                            category_text = ""

            # Display prediction with color coding by category
            if "Number" in category_text:
                color = (0, 255, 0)  # Green for numbers
            elif "Alphabet" in category_text:
                color = (255, 0, 0)  # Blue for alphabets
            elif "Daily Action" in category_text:
                color = (0, 255, 255)  # Yellow for daily actions
            else:
                color = (255, 255, 255)  # White for no detection

            cv2.putText(frame, prediction_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
            cv2.putText(frame, confidence_text, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(frame, category_text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            # Add legend
            cv2.putText(frame, "Green=Numbers, Blue=Alphabets, Yellow=Daily Actions", (10, frame.shape[0] - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, "Press 'q' to quit", (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 2)

            cv2.imshow('Hand Sign Detection - Numbers, Alphabets & Daily Actions', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def clear_all_data(self):
        """Clear all data and start fresh"""
        # Clear memory data
        self.data = []
        self.labels = []
        self.model = None
        self.prediction_buffer = []

        # Try to delete saved files
        files_to_delete = [
            'hand_sign_data_full.pkl',
            'hand_sign_model_full.pkl',
            'hand_sign_data_extended.pkl',
            'hand_sign_model_extended.pkl'
        ]

        deleted_files = []
        for filename in files_to_delete:
            try:
                if os.path.exists(filename):
                    os.remove(filename)
                    deleted_files.append(filename)
            except Exception as e:
                print(f"Could not delete {filename}: {e}")

        if deleted_files:
            print(f"✓ Cleared all data! Deleted files: {', '.join(deleted_files)}")
        else:
            print("✓ Memory cleared! No files to delete.")

        print("You can now start collecting new gesture data fresh!")

    def append_to_existing_data(self):
        """Add more data to existing dataset"""
        print("=== APPEND TO EXISTING DATA ===")
        if self.load_data():
            print("Existing data loaded. You can now add more samples!")
            print("Current dataset contains:")

            unique_labels, counts = np.unique(self.labels, return_counts=True)
            numbers_data = []
            alphabets_data = []
            daily_actions_data = []

            for label, count in zip(unique_labels, counts):
                category = self.get_category_from_label(label)
                sign_name = self.sign_names[label]

                if category == "Number":
                    numbers_data.append(f"{sign_name}: {count}")
                elif category == "Alphabet":
                    alphabets_data.append(f"{sign_name}: {count}")
                elif category == "Daily Action":
                    daily_actions_data.append(f"{sign_name}: {count}")

            if numbers_data:
                print(f"  Numbers: {', '.join(numbers_data)}")
            if alphabets_data:
                print(f"  Alphabets: {', '.join(alphabets_data)}")
            if daily_actions_data:
                print(f"  Daily Actions: {', '.join(daily_actions_data)}")

            print("\nNow you can collect additional samples...")
            self.collect_data()
        else:
            print("No existing data found. Starting fresh collection...")
            self.collect_data()

    def export_dataset_to_folders(self):
        """Export collected data to dataset/<sign_name>/sample_#.npy files"""
        if not self.data or not self.labels:
            print("No data to export!")
            return
        base_dir = 'dataset'
        if os.path.exists(base_dir):
            shutil.rmtree(base_dir)
        os.makedirs(base_dir, exist_ok=True)
        counter = {}
        for sample, label in zip(self.data, self.labels):
            sign_name = str(self.sign_names[label])
            sign_dir = os.path.join(base_dir, sign_name)
            os.makedirs(sign_dir, exist_ok=True)
            count = counter.get(sign_name, 0)
            file_path = os.path.join(sign_dir, f'sample_{count}.npy')
            np.save(file_path, np.array(sample))
            counter[sign_name] = count + 1
        print(f"Exported {len(self.data)} samples into '{base_dir}/<sign_name>/' folders as .npy files.")


def main():
    detector = HandSignDetector()

    while True:
        print("\n=== COMPREHENSIVE HAND SIGN DETECTION SYSTEM ===")
        print("Supports: Numbers (0-9), Alphabets (A-Z), and Specific Daily Actions (14 signs)")
        print("1. Collect training data (all three categories)")
        print("2. Add more data to existing dataset")
        print("3. Train model")
        print("4. Real-time detection")
        print("5. Load existing data")
        print("6. Show data statistics")
        print("7. Clear all data (start fresh)")
        print("8. Show daily actions list")
        print("9. Exit")
        print("10. Export dataset to folders (by sign)")

        choice = input("Enter your choice (1-10): ").strip()

        if choice == '1':
            detector.collect_data()
        elif choice == '2':
            detector.append_to_existing_data()
        elif choice == '3':
            if len(detector.data) == 0:
                if not detector.load_data():
                    print("No data available. Please collect data first.")
                    continue
            detector.train_model()
        elif choice == '4':
            if detector.model is None:
                if not detector.load_model():
                    print("No model available. Please train a model first.")
                    continue
            detector.predict_realtime()
        elif choice == '5':
            detector.load_data()
        elif choice == '6':
            if len(detector.data) > 0 or detector.load_data():
                unique_labels, counts = np.unique(detector.labels, return_counts=True)
                print(f"\nTotal samples: {len(detector.data)}")
                print(f"Unique signs: {len(unique_labels)}")
                print("Distribution by category:")

                numbers_data = []
                alphabets_data = []
                daily_actions_data = []

                for label, count in zip(unique_labels, counts):
                    category = detector.get_category_from_label(label)
                    sign_name = detector.sign_names[label]

                    if category == "Number":
                        numbers_data.append(f"{sign_name}: {count}")
                    elif category == "Alphabet":
                        alphabets_data.append(f"{sign_name}: {count}")
                    elif category == "Daily Action":
                        daily_actions_data.append(f"{sign_name}: {count}")

                if numbers_data:
                    print(f"  Numbers ({len(numbers_data)}): {', '.join(numbers_data)}")
                if alphabets_data:
                    print(f"  Alphabets ({len(alphabets_data)}): {', '.join(alphabets_data)}")
                if daily_actions_data:
                    print(f"  Daily Actions ({len(daily_actions_data)}): {', '.join(daily_actions_data)}")
            else:
                print("No data available.")
        elif choice == '7':
            confirm = input(
                "Are you sure you want to clear all data? This will delete saved files. (y/n): ").strip().lower()
            if confirm == 'y':
                detector.clear_all_data()
            else:
                print("Operation cancelled.")
        elif choice == '8':
            print("\n=== DAILY ACTIONS LIST ===")
            print("Daily Action (Key Mapping) - Label")
            # Sort by label for consistent display
            sorted_daily_actions = sorted([(detector.sign_names[label], key_code, label)
                                           for key_code, label in detector.daily_action_keys.items()
                                           if detector.get_category_from_label(label) == "Daily Action"],
                                          key=lambda x: x[2])  # Sort by label

            for sign_name, key_code, label in sorted_daily_actions:
                key_char = ""
                if key_code < 256:
                    key_char = chr(key_code)
                else:
                    if key_code == 7340032:
                        key_char = "F1"
                    elif key_code == 7405568:
                        key_char = "F2"
                    elif key_code == 7471104:
                        key_char = "F3"
                    elif key_code == 7536640:
                        key_char = "F4"
                    elif key_code == 7602176:
                        key_char = "F5"
                    elif key_code == 7667712:
                        key_char = "F6"
                    elif key_code == 7733248:
                        key_char = "F7"
                    elif key_code == 7798784:
                        key_char = "F8"
                    elif key_code == 7864320:
                        key_char = "F9"
                    elif key_code == 7929856:
                        key_char = "F10"
                    elif key_code == 7995392:
                        key_char = "F11"
                    elif key_code == 8060928:
                        key_char = "F12"
                    else:
                        key_char = f"RawKey:{key_code}"
                print(f"  {sign_name} ({key_char})")
            print("-" * 30)
        elif choice == '9':
            print("Exiting program. Goodbye!")
            break
        elif choice == '10':
            detector.export_dataset_to_folders()
        else:
            print("Invalid choice. Please enter a number between 1 and 10.")


if __name__ == "__main__":
    main()