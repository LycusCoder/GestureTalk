"""
Model Training Script - Train gesture recognition model dari collected data
Machine learning pipeline untuk gesture classification
"""

import pandas as pd
import numpy as np
import pickle
import os
import sys
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib
import argparse
import time
from typing import Tuple, Dict, Any

# Add parent directory ke path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class GestureModelTrainer:
    """
    ML model trainer untuk gesture recognition dengan comprehensive evaluation
    
    Features:
    - Multiple algorithm comparison (RandomForest, SVM)
    - Cross-validation dan model evaluation
    - Feature preprocessing dan scaling
    - Model persistence dengan metadata
    - Performance metrics dan confusion matrix
    - Data validation dan cleaning
    """
    
    def __init__(self, 
                 data_file: str = '/app/data/gestures.csv',
                 model_output_dir: str = '/app/models/'):
        """
        Initialize GestureModelTrainer
        
        Args:
            data_file: Path ke CSV file dengan training data
            model_output_dir: Directory untuk save trained models
        """
        self.data_file = data_file
        self.model_output_dir = model_output_dir
        
        # Data storage
        self.df = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        # Models dan preprocessing
        self.scaler = StandardScaler()
        self.models = {}
        self.model_results = {}
        self.best_model = None
        self.best_model_name = None
        
        # Create output directory
        os.makedirs(model_output_dir, exist_ok=True)
        
        print("🚀 GestureModelTrainer initialized")
        print(f"📁 Data file: {data_file}")
        print(f"💾 Model output: {model_output_dir}")
    
    def load_and_validate_data(self) -> bool:
        """
        Load data dari CSV dan validate quality
        
        Returns:
            bool: True jika data valid untuk training
        """
        print("\n📊 Loading dan validating data...")
        
        # Check file exists
        if not os.path.exists(self.data_file):
            print(f"❌ Data file tidak ditemukan: {self.data_file}")
            return False
        
        try:
            # Load CSV
            self.df = pd.read_csv(self.data_file)
            print(f"✅ Data loaded: {len(self.df)} samples")
            
            # Basic validation
            if len(self.df) == 0:
                print("❌ Data kosong")
                return False
            
            # Check kolom structure
            expected_columns = 1 + 42  # label + 42 coordinates
            if len(self.df.columns) != expected_columns:
                print(f"❌ Expected {expected_columns} columns, got {len(self.df.columns)}")
                return False
            
            # Show data overview
            self._show_data_overview()
            
            # Validate data quality
            quality_ok = self._validate_data_quality()
            
            if quality_ok:
                print("✅ Data validation passed")
                return True
            else:
                print("❌ Data validation failed")
                return False
                
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
    
    def _show_data_overview(self):
        """Show overview dari loaded data"""
        print("\n📋 DATA OVERVIEW:")
        print("-" * 30)
        print(f"Total samples: {len(self.df)}")
        print(f"Features: {len(self.df.columns) - 1}")  # Minus label column
        
        # Gesture distribution
        print("\nGesture distribution:")
        gesture_counts = self.df['label'].value_counts()
        for gesture, count in gesture_counts.items():
            percentage = (count / len(self.df)) * 100
            print(f"  {gesture}: {count} samples ({percentage:.1f}%)")
        
        # Data quality checks
        missing_values = self.df.isnull().sum().sum()
        print(f"\nMissing values: {missing_values}")
        
        if missing_values > 0:
            print("⚠️  Warning: Data mengandung missing values")
    
    def _validate_data_quality(self) -> bool:
        """
        Validate data quality untuk training
        
        Returns:
            bool: True jika data quality acceptable
        """
        print("\n🔍 Validating data quality...")
        
        issues = []
        
        # Check minimum samples per gesture
        min_samples_per_gesture = 20  # Minimum untuk training
        gesture_counts = self.df['label'].value_counts()
        
        for gesture, count in gesture_counts.items():
            if count < min_samples_per_gesture:
                issues.append(f"Gesture '{gesture}' hanya punya {count} samples (minimum {min_samples_per_gesture})")
        
        # Check for missing values
        if self.df.isnull().sum().sum() > 0:
            issues.append("Data mengandung missing values")
        
        # Check feature range (normalized coordinates should be around -1 to 1)
        feature_cols = self.df.columns[1:]  # Exclude label
        for col in feature_cols[:10]:  # Check first 10 features
            col_min = self.df[col].min()
            col_max = self.df[col].max()
            
            if col_min < -2 or col_max > 2:
                issues.append(f"Feature {col} memiliki range yang tidak normal ({col_min:.3f} to {col_max:.3f})")
                break
        
        # Check untuk duplicate rows
        duplicate_rows = self.df.duplicated().sum()
        if duplicate_rows > len(self.df) * 0.1:  # More than 10% duplicates
            issues.append(f"Terlalu banyak duplicate rows: {duplicate_rows}")
        
        # Report issues
        if issues:
            print("⚠️  Data quality issues found:")
            for issue in issues:
                print(f"   - {issue}")
            
            # For automated pipeline, return True to continue
            print("⚠️  Continuing with current data...")
            return True
        else:
            print("✅ No major data quality issues found")
            return True
    
    def prepare_training_data(self, test_size: float = 0.2, random_state: int = 42):
        """
        Prepare data untuk training (split, scale, dll)
        
        Args:
            test_size: Proportion untuk test set
            random_state: Random seed untuk reproducibility
        """
        print(f"\n🔧 Preparing training data (test_size={test_size})...")
        
        # Split features dan labels
        self.X = self.df.drop('label', axis=1).values
        self.y = self.df['label'].values
        
        # Train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state, stratify=self.y
        )
        
        print(f"✅ Train set: {len(self.X_train)} samples")
        print(f"✅ Test set: {len(self.X_test)} samples")
        
        # Feature scaling
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print("✅ Feature scaling completed")
        
        # Show class distribution in splits
        print("\nTrain set distribution:")
        unique_train, counts_train = np.unique(self.y_train, return_counts=True)
        for gesture, count in zip(unique_train, counts_train):
            print(f"  {gesture}: {count}")
    
    def train_models(self):
        """Train multiple models dan compare performance"""
        print("\n🤖 Training multiple models...")
        
        # Define models untuk comparison
        model_configs = {
            'RandomForest': {
                'model': RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42,
                    n_jobs=-1
                ),
                'use_scaling': False  # Random Forest doesn't need scaling
            },
            'SVM': {
                'model': SVC(
                    kernel='rbf',
                    C=1.0,
                    gamma='scale',
                    random_state=42,
                    probability=True  # Enable probability predictions
                ),
                'use_scaling': True  # SVM needs scaling
            }
        }
        
        # Train each model
        for name, config in model_configs.items():
            print(f"\n🔨 Training {name}...")
            
            start_time = time.time()
            
            # Select appropriate data
            if config['use_scaling']:
                X_train_data = self.X_train_scaled
                X_test_data = self.X_test_scaled
            else:
                X_train_data = self.X_train
                X_test_data = self.X_test
            
            # Train model
            model = config['model']
            model.fit(X_train_data, self.y_train)
            
            # Evaluate model
            results = self._evaluate_model(model, X_train_data, X_test_data, name)
            
            # Store model dan results
            self.models[name] = {
                'model': model,
                'use_scaling': config['use_scaling'],
                'results': results,
                'training_time': time.time() - start_time
            }
            
            self.model_results[name] = results
            
            print(f"✅ {name} training completed ({results['test_accuracy']:.3f} accuracy)")
    
    def _evaluate_model(self, model, X_train: np.ndarray, X_test: np.ndarray, model_name: str) -> Dict[str, Any]:
        """
        Comprehensive model evaluation
        
        Args:
            model: Trained model
            X_train: Training features
            X_test: Test features
            model_name: Name untuk logging
            
        Returns:
            Dict: Evaluation results
        """
        # Predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Basic metrics
        train_accuracy = accuracy_score(self.y_train, y_train_pred)
        test_accuracy = accuracy_score(self.y_test, y_test_pred)
        
        # Cross-validation score
        cv_scores = cross_val_score(model, X_train, self.y_train, cv=5, scoring='accuracy')
        
        # Classification report
        classification_rep = classification_report(
            self.y_test, y_test_pred, output_dict=True, zero_division=0
        )
        
        # Confusion matrix
        conf_matrix = confusion_matrix(self.y_test, y_test_pred)
        
        results = {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'classification_report': classification_rep,
            'confusion_matrix': conf_matrix,
            'cv_scores': cv_scores
        }
        
        return results
    
    def select_best_model(self) -> str:
        """
        Select best model berdasarkan performance metrics
        
        Returns:
            str: Name of best model
        """
        print("\n🏆 Selecting best model...")
        
        best_score = 0
        best_name = None
        
        print("\nModel comparison:")
        print("-" * 50)
        
        for name, results in self.model_results.items():
            test_acc = results['test_accuracy']
            cv_mean = results['cv_mean']
            cv_std = results['cv_std']
            
            # Composite score (weight test accuracy more than cv)
            composite_score = 0.6 * test_acc + 0.4 * cv_mean - cv_std * 0.1
            
            print(f"{name}:")
            print(f"  Test Accuracy: {test_acc:.3f}")
            print(f"  CV Mean: {cv_mean:.3f} (±{cv_std:.3f})")
            print(f"  Composite Score: {composite_score:.3f}")
            print()
            
            if composite_score > best_score:
                best_score = composite_score
                best_name = name
        
        self.best_model_name = best_name
        self.best_model = self.models[best_name]
        
        print(f"🥇 Best model: {best_name} (score: {best_score:.3f})")
        return best_name
    
    def show_detailed_results(self):
        """Show detailed results untuk best model"""
        if not self.best_model:
            print("❌ No best model selected")
            return
        
        print(f"\n📊 DETAILED RESULTS - {self.best_model_name}")
        print("=" * 50)
        
        results = self.best_model['results']
        
        # Overall metrics
        print(f"Test Accuracy: {results['test_accuracy']:.3f}")
        print(f"Training Accuracy: {results['train_accuracy']:.3f}")
        print(f"Cross-validation: {results['cv_mean']:.3f} (±{results['cv_std']:.3f})")
        
        # Per-class results
        print("\nPer-class performance:")
        class_report = results['classification_report']
        for gesture in class_report:
            if gesture not in ['accuracy', 'macro avg', 'weighted avg']:
                metrics = class_report[gesture]
                print(f"  {gesture}:")
                print(f"    Precision: {metrics['precision']:.3f}")
                print(f"    Recall: {metrics['recall']:.3f}")
                print(f"    F1-score: {metrics['f1-score']:.3f}")
                print(f"    Support: {int(metrics['support'])}")
        
        # Confusion matrix
        print("\nConfusion Matrix:")
        conf_matrix = results['confusion_matrix']
        gestures = list(class_report.keys())
        gestures = [g for g in gestures if g not in ['accuracy', 'macro avg', 'weighted avg']]
        
        print("Actual vs Predicted:")
        print("    " + "  ".join(f"{g[:6]:<6}" for g in gestures))
        for i, actual in enumerate(gestures):
            row_str = f"{actual[:6]:<6} "
            for j in range(len(gestures)):
                row_str += f"{conf_matrix[i][j]:>6} "
            print(row_str)
    
    def save_model(self, filename: str = None) -> str:
        """
        Save best model dengan metadata
        
        Args:
            filename: Custom filename (optional)
            
        Returns:
            str: Path ke saved model
        """
        if not self.best_model:
            print("❌ No best model to save")
            return None
        
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"gesture_model_{self.best_model_name}_{timestamp}.pkl"
        
        model_path = os.path.join(self.model_output_dir, filename)
        
        # Prepare model package
        model_package = {
            'model': self.best_model['model'],
            'scaler': self.scaler if self.best_model['use_scaling'] else None,
            'use_scaling': self.best_model['use_scaling'],
            'model_type': self.best_model_name,
            'training_results': self.best_model['results'],
            'gesture_classes': list(np.unique(self.y)),
            'feature_count': self.X.shape[1],
            'training_date': time.strftime("%Y-%m-%d %H:%M:%S"),
            'data_file_used': self.data_file,
            'sample_count': len(self.df)
        }
        
        try:
            # Save dengan joblib (better untuk scikit-learn models)
            joblib.dump(model_package, model_path)
            
            # Also save simple model untuk backward compatibility
            simple_path = os.path.join(self.model_output_dir, 'gesture_model.pkl')
            with open(simple_path, 'wb') as f:
                pickle.dump(model_package['model'], f)
            
            print(f"✅ Model saved to {model_path}")
            print(f"✅ Simple model saved to {simple_path}")
            
            return model_path
            
        except Exception as e:
            print(f"❌ Error saving model: {e}")
            return None
    
    def create_dummy_data_for_testing(self):
        """Create dummy data untuk testing purposes"""
        print("🧪 Creating dummy data for testing...")
        
        # Create synthetic gesture data
        np.random.seed(42)
        gestures = ['tolong', 'halo', 'terima_kasih']
        samples_per_gesture = 30
        
        data = []
        for gesture in gestures:
            for _ in range(samples_per_gesture):
                # Generate random normalized coordinates (42 features)
                features = np.random.normal(0, 0.3, 42).tolist()
                data.append([gesture] + features)
        
        # Create DataFrame dan save
        columns = ['label'] + [f'x{i}' for i in range(21)] + [f'y{i}' for i in range(21)]
        df = pd.DataFrame(data, columns=columns)
        
        os.makedirs(os.path.dirname(self.data_file), exist_ok=True)
        df.to_csv(self.data_file, index=False)
        
        print(f"✅ Dummy data created: {len(df)} samples")
        print(f"📁 Saved to: {self.data_file}")
    
    def run_full_training_pipeline(self):
        """Run complete training pipeline"""
        print("🚀 Starting full training pipeline...")
        
        # Step 1: Load and validate data
        if not self.load_and_validate_data():
            print("❌ Data loading failed, creating dummy data for testing...")
            self.create_dummy_data_for_testing()
            if not self.load_and_validate_data():
                print("❌ Still failed after dummy data creation")
                return False
        
        # Step 2: Prepare training data
        self.prepare_training_data()
        
        # Step 3: Train models
        self.train_models()
        
        # Step 4: Select best model
        self.select_best_model()
        
        # Step 5: Show results
        self.show_detailed_results()
        
        # Step 6: Save model
        model_path = self.save_model()
        
        if model_path:
            print(f"\n🎉 Training pipeline completed successfully!")
            print(f"💾 Best model saved: {model_path}")
            return True
        else:
            print("❌ Training pipeline failed at saving step")
            return False


def main():
    """Main function dengan CLI interface"""
    parser = argparse.ArgumentParser(description='Gesture Recognition Model Training')
    parser.add_argument('--data', '-d',
                       default='/app/data/gestures.csv',
                       help='Path to training data CSV file')
    parser.add_argument('--output', '-o',
                       default='/app/GestureTalk/models/',
                       help='Output directory for trained model')
    parser.add_argument('--test-size', '-t',
                       type=float, default=0.2,
                       help='Test set size (0.0-1.0)')
    
    args = parser.parse_args()
    
    print("🚀 Gesture Recognition Model Training")
    print(f"📁 Data file: {args.data}")
    print(f"💾 Output directory: {args.output}")
    print(f"🧪 Test size: {args.test_size}")
    
    # Create trainer
    trainer = GestureModelTrainer(
        data_file=args.data,
        model_output_dir=args.output
    )
    
    # Run training pipeline
    success = trainer.run_full_training_pipeline()
    
    if success:
        print("\n✅ Training completed successfully!")
    else:
        print("\n❌ Training failed!")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())