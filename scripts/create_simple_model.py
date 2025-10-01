"""
Create Simple Gesture Model - Create basic gesture model untuk testing
Buat model sederhana dengan beberapa gesture dasar
"""

import pickle
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
from datetime import datetime


def create_synthetic_gesture_data():
    """Create synthetic gesture data untuk testing"""
    
    # Define basic gestures
    gestures = ['tolong', 'halo', 'terima_kasih', 'ya', 'tidak']
    
    # Generate synthetic data (42 features - 21 landmarks x 2 coordinates)
    samples_per_gesture = 50
    feature_count = 42
    
    X = []
    y = []
    
    print("🔧 Generating synthetic gesture data...")
    
    for gesture in gestures:
        print(f"  Creating {samples_per_gesture} samples for '{gesture}'")
        
        for _ in range(samples_per_gesture):
            if gesture == 'tolong':
                # Raised hand pattern - higher Y coordinates
                sample = np.random.normal(0, 0.1, feature_count)
                # Adjust y-coordinates (odd indices) to be more negative (higher)
                for i in range(1, feature_count, 2):
                    sample[i] -= 0.3 + np.random.normal(0, 0.05)
                    
            elif gesture == 'halo':
                # Waving pattern - varying X coordinates
                sample = np.random.normal(0, 0.15, feature_count)
                # Add wave motion to x-coordinates (even indices)
                for i in range(0, feature_count, 2):
                    sample[i] += 0.1 * np.sin(i) + np.random.normal(0, 0.05)
                    
            elif gesture == 'terima_kasih':
                # Hand to chest pattern
                sample = np.random.normal(0, 0.08, feature_count)
                # Bring hand closer to body (lower X variation)
                for i in range(0, feature_count, 2):
                    sample[i] *= 0.7
                    
            elif gesture == 'ya':
                # Thumbs up pattern - specific thumb positioning
                sample = np.random.normal(0, 0.12, feature_count)
                # Thumb landmarks (indices 2-8) - higher position
                for i in range(6, 9, 2):  # Thumb tip area
                    sample[i+1] -= 0.2 + np.random.normal(0, 0.03)
                    
            elif gesture == 'tidak':
                # Index finger pointing/wagging
                sample = np.random.normal(0, 0.10, feature_count)
                # Index finger landmarks - extended position
                for i in range(12, 17, 2):  # Index finger area
                    sample[i] += 0.15 + np.random.normal(0, 0.03)
            
            X.append(sample)
            y.append(gesture)
    
    return np.array(X), np.array(y), gestures


def create_and_save_model():
    """Create dan save gesture recognition model"""
    
    print("🚀 Creating simple gesture recognition model...")
    
    # Generate data
    X, y, gesture_classes = create_synthetic_gesture_data()
    
    print(f"📊 Dataset created:")
    print(f"  Total samples: {len(X)}")
    print(f"  Features per sample: {X.shape[1]}")
    print(f"  Gestures: {gesture_classes}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Create dan train model
    print("🤖 Training RandomForest model...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight='balanced'
    )
    
    # Train model
    model.fit(X_train, y_train)
    
    # Test model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"✅ Model training completed!")
    print(f"  Training accuracy: {model.score(X_train, y_train):.3f}")
    print(f"  Test accuracy: {accuracy:.3f}")
    
    # Create scaler (optional, tapi include untuk compatibility)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Create model package
    model_package = {
        'model': model,
        'scaler': scaler,
        'use_scaling': False,  # Don't use scaling untuk simple model
        'model_type': 'RandomForestClassifier',
        'gesture_classes': list(gesture_classes),
        'training_date': datetime.now().isoformat(),
        'feature_count': X.shape[1],
        'sample_count': len(X),
        'training_results': {
            'train_accuracy': model.score(X_train, y_train),
            'test_accuracy': accuracy,
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
    }
    
    # Save model
    model_path = '/app/models/gesture_model.pkl'
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    print(f"💾 Saving model to {model_path}")
    with open(model_path, 'wb') as f:
        pickle.dump(model_package, f)
    
    # Also save dengan joblib untuk compatibility
    joblib_path = '/app/models/gesture_model_joblib.pkl'
    joblib.dump(model_package, joblib_path)
    
    print(f"✅ Model saved successfully!")
    print(f"  Pickle version: {model_path}")
    print(f"  Joblib version: {joblib_path}")
    
    # Test loading
    print("🔧 Testing model loading...")
    try:
        with open(model_path, 'rb') as f:
            loaded_model = pickle.load(f)
        
        # Test prediction
        test_sample = X[0].reshape(1, -1)
        prediction = loaded_model['model'].predict(test_sample)[0]
        
        print(f"✅ Model loading test successful!")
        print(f"  Test prediction: {prediction}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model loading test failed: {e}")
        return False


def create_sample_dataset():
    """Create sample dataset CSV file"""
    
    print("📊 Creating sample dataset CSV...")
    
    # Generate data
    X, y, _ = create_synthetic_gesture_data()
    
    # Create DataFrame-like structure
    import pandas as pd
    
    # Create feature columns
    feature_columns = []
    for i in range(21):
        feature_columns.append(f'landmark_{i}_x')
        feature_columns.append(f'landmark_{i}_y')
    
    # Create DataFrame
    df = pd.DataFrame(X, columns=feature_columns)
    df['label'] = y
    
    # Save CSV
    csv_path = '/app/data/gestures.csv'
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False)
    
    print(f"✅ Dataset CSV saved to {csv_path}")
    print(f"  Samples: {len(df)}")
    print(f"  Features: {len(feature_columns)}")
    
    return csv_path


if __name__ == "__main__":
    print("🚀 Setting up GestureTalk models dan datasets...")
    
    # Create directories
    os.makedirs('/app/models', exist_ok=True)
    os.makedirs('/app/data', exist_ok=True)
    
    # Create model
    model_success = create_and_save_model()
    
    # Create dataset
    dataset_path = create_sample_dataset()
    
    if model_success:
        print("\n🎉 SUCCESS! GestureTalk setup completed:")
        print("  ✅ Gesture recognition model created")
        print("  ✅ Sample dataset created")
        print("  ✅ Ready untuk testing dengan 'python app.py'")
    else:
        print("\n❌ Setup failed. Please check the errors above.")