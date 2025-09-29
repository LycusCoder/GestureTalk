"""
Gesture Predictor Module - Real-time gesture prediction using trained ML model
Handles model loading, prediction, and confidence scoring
"""

import pickle
import joblib
import numpy as np
import os
import time
import logging
from typing import Optional, Tuple, Dict, List
from sklearn.preprocessing import StandardScaler

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GesturePredictor:
    """
    Real-time gesture predictor untuk assistive communication
    
    Features:
    - Load trained ML models (RandomForest, SVM, etc.)
    - Real-time gesture prediction dengan confidence scoring
    - Model metadata dan validation
    - Prediction history untuk stability
    - Performance monitoring
    - Multiple prediction modes (strict, relaxed)
    """
    
    def __init__(self, 
                 model_path: str = 'home/Nourivex/AIProject/GestureTalk/models/gesture_model.pkl',
                 confidence_threshold: float = 0.6,
                 stability_window: int = 3):
        """
        Initialize GesturePredictor
        
        Args:
            model_path: Path ke trained model file
            confidence_threshold: Minimum confidence untuk valid prediction
            stability_window: Jumlah consecutive predictions untuk stability
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.stability_window = stability_window
        
        # Model components
        self.model = None
        self.scaler = None
        self.model_metadata = {}
        self.gesture_classes = []
        self.use_scaling = False
        
        # Prediction tracking
        self.prediction_history = []
        self.last_prediction = None
        self.last_confidence = 0.0
        self.prediction_count = 0
        
        # Performance monitoring
        self.prediction_times = []
        self.average_prediction_time = 0.0
        
        # Status
        self.is_loaded = False
        self.model_type = "unknown"
        
        # Load model
        self._load_model()
    
    def _load_model(self) -> bool:
        """
        Load trained model dari file dengan comprehensive validation
        
        Returns:
            bool: True jika berhasil load model
        """
        logger.info(f"🔧 Loading gesture model from {self.model_path}")
        
        if not os.path.exists(self.model_path):
            logger.error(f"❌ Model file tidak ditemukan: {self.model_path}")
            return False
        
        try:
            # Try loading dengan joblib first (untuk model package format)
            try:
                model_package = joblib.load(self.model_path)
                
                if isinstance(model_package, dict):
                    # Full model package format
                    self.model = model_package['model']
                    self.scaler = model_package.get('scaler', None)
                    self.use_scaling = model_package.get('use_scaling', False)
                    self.model_metadata = {
                        'model_type': model_package.get('model_type', 'unknown'),
                        'training_date': model_package.get('training_date', 'unknown'),
                        'accuracy': model_package.get('training_results', {}).get('test_accuracy', 0.0),
                        'feature_count': model_package.get('feature_count', 42),
                        'sample_count': model_package.get('sample_count', 0)
                    }
                    self.gesture_classes = model_package.get('gesture_classes', [])
                    self.model_type = model_package.get('model_type', 'unknown')
                    
                    logger.info("✅ Full model package loaded")
                else:
                    # Simple model format
                    self.model = model_package
                    self.gesture_classes = getattr(self.model, 'classes_', [])
                    self.model_type = type(self.model).__name__
                    logger.info("✅ Simple model loaded")
                    
            except Exception as e:
                logger.warning(f"⚠️  Joblib loading failed, trying pickle: {e}")
                
                # Fallback ke pickle
                with open(self.model_path, 'rb') as f:
                    self.model = pickle.load(f)
                    self.gesture_classes = getattr(self.model, 'classes_', [])
                    self.model_type = type(self.model).__name__
                    logger.info("✅ Model loaded dengan pickle")
            
            # Validate model
            if not self._validate_model():
                logger.error("❌ Model validation failed")
                return False
            
            self.is_loaded = True
            self._show_model_info()
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading model: {e}")
            return False
    
    def _validate_model(self) -> bool:
        """
        Validate loaded model untuk ensure compatibility
        
        Returns:
            bool: True jika model valid
        """
        try:
            # Check if model has required methods
            if not hasattr(self.model, 'predict'):
                logger.error("❌ Model tidak punya method predict")
                return False
            
            # Check gesture classes
            if not self.gesture_classes or len(self.gesture_classes) == 0:
                logger.warning("⚠️  No gesture classes found in model")
                # Try to extract from model
                if hasattr(self.model, 'classes_'):
                    self.gesture_classes = list(self.model.classes_)
                else:
                    logger.error("❌ Cannot determine gesture classes")
                    return False
            
            # Test prediction dengan dummy data
            dummy_features = np.zeros((1, 42))  # 42 features expected
            
            try:
                if self.use_scaling and self.scaler:
                    dummy_features_scaled = self.scaler.transform(dummy_features)
                    _ = self.model.predict(dummy_features_scaled)
                else:
                    _ = self.model.predict(dummy_features)
                
                logger.info("✅ Model validation test passed")
                return True
                
            except Exception as e:
                logger.error(f"❌ Model prediction test failed: {e}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Model validation error: {e}")
            return False
    
    def _show_model_info(self):
        """Show informasi detail tentang loaded model"""
        logger.info("\n📊 MODEL INFORMATION:")
        logger.info("-" * 30)
        logger.info(f"Model Type: {self.model_type}")
        logger.info(f"Gesture Classes: {self.gesture_classes}")
        logger.info(f"Total Classes: {len(self.gesture_classes)}")
        logger.info(f"Uses Scaling: {self.use_scaling}")
        
        if self.model_metadata:
            logger.info(f"Training Date: {self.model_metadata.get('training_date', 'unknown')}")
            logger.info(f"Training Accuracy: {self.model_metadata.get('accuracy', 0):.3f}")
            logger.info(f"Training Samples: {self.model_metadata.get('sample_count', 0)}")
        
        logger.info(f"Confidence Threshold: {self.confidence_threshold}")
        logger.info(f"Stability Window: {self.stability_window}")
    
    def predict_gesture(self, landmarks: List[float]) -> Tuple[Optional[str], float]:
        """
        Predict gesture dari hand landmarks dengan confidence scoring
        
        Args:
            landmarks: List of 42 normalized coordinates [x1,y1,x2,y2,...,x21,y21]
            
        Returns:
            Tuple[gesture_name, confidence]: Predicted gesture dan confidence score
        """
        if not self.is_loaded:
            logger.error("❌ Model belum loaded")
            return None, 0.0
        
        if not landmarks or len(landmarks) != 42:
            logger.error(f"❌ Invalid landmarks: expected 42, got {len(landmarks) if landmarks else 0}")
            return None, 0.0
        
        try:
            start_time = time.time()
            
            # Convert ke numpy array
            features = np.array(landmarks).reshape(1, -1)
            
            # Apply scaling jika diperlukan
            if self.use_scaling and self.scaler:
                features = self.scaler.transform(features)
            
            # Get prediction
            prediction = self.model.predict(features)[0]
            
            # Get confidence score (probability if available)
            confidence = self._get_prediction_confidence(features, prediction)
            
            # Track prediction time
            prediction_time = time.time() - start_time
            self._update_performance_metrics(prediction_time)
            
            # Update prediction history
            self._update_prediction_history(prediction, confidence)
            
            # Apply stability filtering
            stable_prediction, stable_confidence = self._apply_stability_filter()
            
            # Store last prediction
            self.last_prediction = stable_prediction
            self.last_confidence = stable_confidence
            self.prediction_count += 1
            
            return stable_prediction, stable_confidence
            
        except Exception as e:
            logger.error(f"❌ Prediction error: {e}")
            return None, 0.0
    
    def _get_prediction_confidence(self, features: np.ndarray, prediction: str) -> float:
        """
        Get confidence score untuk prediction
        
        Args:
            features: Preprocessed features
            prediction: Predicted gesture name
            
        Returns:
            float: Confidence score (0.0 - 1.0)
        """
        try:
            # If model supports probability prediction
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(features)[0]
                
                # Find index of predicted class
                pred_index = list(self.model.classes_).index(prediction)
                confidence = probabilities[pred_index]
                
                return float(confidence)
            
            # If model supports decision function (SVM)
            elif hasattr(self.model, 'decision_function'):
                decision_scores = self.model.decision_function(features)[0]
                
                if len(self.gesture_classes) == 2:
                    # Binary classification
                    confidence = 1.0 / (1.0 + np.exp(-decision_scores))  # Sigmoid
                else:
                    # Multi-class: use softmax on decision scores
                    exp_scores = np.exp(decision_scores - np.max(decision_scores))
                    probabilities = exp_scores / np.sum(exp_scores)
                    
                    pred_index = list(self.model.classes_).index(prediction)
                    confidence = probabilities[pred_index]
                
                return float(confidence)
            
            else:
                # No probability support, return fixed confidence
                return 0.8  # Default confidence untuk deterministic predictions
                
        except Exception as e:
            logger.warning(f"⚠️  Error calculating confidence: {e}")
            return 0.5  # Fallback confidence
    
    def _update_prediction_history(self, prediction: str, confidence: float):
        """Update prediction history untuk stability tracking"""
        self.prediction_history.append({
            'prediction': prediction,
            'confidence': confidence,
            'timestamp': time.time()
        })
        
        # Keep only recent predictions
        max_history = self.stability_window * 2
        if len(self.prediction_history) > max_history:
            self.prediction_history = self.prediction_history[-max_history:]
    
    def _apply_stability_filter(self) -> Tuple[Optional[str], float]:
        """
        Apply stability filter untuk reduce prediction noise
        
        Returns:
            Tuple[stable_prediction, stable_confidence]: Filtered prediction
        """
        if len(self.prediction_history) < self.stability_window:
            # Not enough history, return latest
            if self.prediction_history:
                latest = self.prediction_history[-1]
                if latest['confidence'] >= self.confidence_threshold:
                    return latest['prediction'], latest['confidence']
            return None, 0.0
        
        # Get recent predictions
        recent_predictions = self.prediction_history[-self.stability_window:]
        
        # Count occurrences
        prediction_counts = {}
        total_confidence = 0.0
        
        for pred_data in recent_predictions:
            pred = pred_data['prediction']
            conf = pred_data['confidence']
            
            if pred not in prediction_counts:
                prediction_counts[pred] = {'count': 0, 'total_confidence': 0.0}
            
            prediction_counts[pred]['count'] += 1
            prediction_counts[pred]['total_confidence'] += conf
            total_confidence += conf
        
        # Find most frequent prediction dengan sufficient confidence
        best_pred = None
        best_confidence = 0.0
        
        for pred, data in prediction_counts.items():
            count = data['count']
            avg_confidence = data['total_confidence'] / count
            
            # Require majority dan sufficient confidence
            if (count >= self.stability_window / 2 and 
                avg_confidence >= self.confidence_threshold and
                avg_confidence > best_confidence):
                
                best_pred = pred
                best_confidence = avg_confidence
        
        return best_pred, best_confidence
    
    def _update_performance_metrics(self, prediction_time: float):
        """Update performance monitoring metrics"""
        self.prediction_times.append(prediction_time)
        
        # Keep only recent times
        if len(self.prediction_times) > 100:
            self.prediction_times = self.prediction_times[-100:]
        
        # Update average
        self.average_prediction_time = np.mean(self.prediction_times)
    
    def get_prediction_stats(self) -> Dict:
        """
        Get prediction statistics dan performance metrics
        
        Returns:
            Dict: Comprehensive prediction statistics
        """
        return {
            'model_loaded': self.is_loaded,
            'model_type': self.model_type,
            'gesture_classes': self.gesture_classes,
            'total_predictions': self.prediction_count,
            'last_prediction': self.last_prediction,
            'last_confidence': self.last_confidence,
            'confidence_threshold': self.confidence_threshold,
            'stability_window': self.stability_window,
            'average_prediction_time': self.average_prediction_time,
            'history_length': len(self.prediction_history),
            'model_metadata': self.model_metadata
        }
    
    def update_settings(self, 
                       confidence_threshold: Optional[float] = None,
                       stability_window: Optional[int] = None):
        """
        Update predictor settings secara real-time
        
        Args:
            confidence_threshold: New confidence threshold
            stability_window: New stability window size
        """
        if confidence_threshold is not None:
            if 0.0 <= confidence_threshold <= 1.0:
                self.confidence_threshold = confidence_threshold
                logger.info(f"✅ Confidence threshold updated to {confidence_threshold}")
            else:
                logger.warning("⚠️  Invalid confidence threshold (0.0-1.0)")
        
        if stability_window is not None:
            if stability_window > 0:
                self.stability_window = stability_window
                logger.info(f"✅ Stability window updated to {stability_window}")
            else:
                logger.warning("⚠️  Invalid stability window (must be > 0)")
    
    def clear_prediction_history(self):
        """Clear prediction history untuk reset stability tracking"""
        self.prediction_history = []
        self.last_prediction = None
        self.last_confidence = 0.0
        logger.info("✅ Prediction history cleared")
    
    def reload_model(self, new_model_path: Optional[str] = None) -> bool:
        """
        Reload model (untuk update model tanpa restart application)
        
        Args:
            new_model_path: Path ke model baru (optional)
            
        Returns:
            bool: True jika berhasil reload
        """
        if new_model_path:
            self.model_path = new_model_path
        
        # Reset state
        self.is_loaded = False
        self.clear_prediction_history()
        
        # Reload model
        success = self._load_model()
        
        if success:
            logger.info(f"✅ Model successfully reloaded from {self.model_path}")
        else:
            logger.error(f"❌ Model reload failed from {self.model_path}")
        
        return success
    
    def is_prediction_stable(self) -> bool:
        """
        Check apakah current prediction stable berdasarkan history
        
        Returns:
            bool: True jika prediction stable
        """
        if not self.last_prediction or len(self.prediction_history) < self.stability_window:
            return False
        
        # Check recent predictions
        recent = self.prediction_history[-self.stability_window:]
        same_prediction_count = sum(1 for p in recent if p['prediction'] == self.last_prediction)
        
        return same_prediction_count >= self.stability_window / 2


# Test function untuk development
def test_gesture_predictor():
    """
    Test function untuk verify GesturePredictor berfungsi dengan baik
    """
    print("🚀 Testing GesturePredictor...")
    
    # Initialize predictor
    predictor = GesturePredictor()
    
    if not predictor.is_loaded:
        print("❌ Model tidak berhasil loaded")
        return
    
    print("✅ Model berhasil loaded")
    
    # Show stats
    stats = predictor.get_prediction_stats()
    print(f"📊 Model Stats: {stats}")
    
    # Test predictions dengan dummy data
    print("🧪 Testing predictions...")
    
    for i in range(5):
        # Create dummy landmarks
        dummy_landmarks = np.random.normal(0, 0.3, 42).tolist()
        
        # Predict
        gesture, confidence = predictor.predict_gesture(dummy_landmarks)
        
        print(f"  Test {i+1}: {gesture} (confidence: {confidence:.3f})")
        
        time.sleep(0.1)  # Simulate real-time delay
    
    # Show final stats
    final_stats = predictor.get_prediction_stats()
    print(f"\n📊 Final Stats:")
    print(f"  Total Predictions: {final_stats['total_predictions']}")
    print(f"  Last Prediction: {final_stats['last_prediction']}")
    print(f"  Average Prediction Time: {final_stats['average_prediction_time']:.4f}s")
    print(f"  Prediction Stable: {predictor.is_prediction_stable()}")
    
    print("✅ GesturePredictor test completed")


if __name__ == "__main__":
    test_gesture_predictor()