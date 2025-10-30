'''
=============================================================================================================================
Author  : Daflah Tsany Gusra
Build   : 11 Oktober 2025
Purpose : -
=============================================================================================================================
'''

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier as SKRandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import sys
import json

# Add config to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'config'))
import CONFIG

class RandomForestModel: 
    def __init__(self, n_estimators=CONFIG.RF_ESTIMATORS, random_state=CONFIG.RF_RANDOM_STATE):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.model = None
        self.label_encoder = LabelEncoder()
        self.feature_importance = None
        
    def prepare_features_labels(self, lda_results_df):
        """
        Siapkan features (topic distributions) dan labels untuk training
        
        Parameters:
        lda_results_df: DataFrame dari lda_results.csv
        
        Returns:
        X: features (topic distributions)
        y: labels (encoded categories)
        label_mapping: mapping label encoded ke original
        """
        # Features: semua kolom yang diawali dengan 'Topic_'
        topic_columns = [col for col in lda_results_df.columns if col.startswith('Topic_')]
        X = lda_results_df[topic_columns].values
        
        # Labels: kolom 'label'
        if 'label' not in lda_results_df.columns:
            raise ValueError("Kolom 'label' tidak ditemukan dalam data LDA results")
        
        y_original = lda_results_df['label'].values
        
        # Encode labels ke numerical
        y = self.label_encoder.fit_transform(y_original)
        
        # Create label mapping
        label_mapping = dict(zip(self.label_encoder.classes_, self.label_encoder.transform(self.label_encoder.classes_)))
        
        print(f"Features shape: {X.shape}")
        print(f"Labels shape: {y.shape}")
        print(f"Jumlah kelas: {len(np.unique(y))}")
        print(f"Mapping label: {label_mapping}")
        
        return X, y, label_mapping
    
    def train(self, X, y, test_size=0.2):
        """
        Train Random Forest model
        
        Parameters:
        X: features (topic distributions)
        y: labels (encoded)
        test_size: proporsi data test
        
        Returns:
        model: trained Random Forest model
        """
        print("\n=== TRAINING RANDOM FOREST ===")
        print(f"Jumlah estimators: {self.n_estimators}")
        print(f"Random state: {self.random_state}")
        print(f"Test size: {test_size}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        
        # Inisialisasi dan train model
        self.model = SKRandomForestClassifier(  # ← GUNAKAN SKLearn Random Forest
            n_estimators=self.n_estimators,
            random_state=self.random_state
            # Hapus n_jobs untuk menghindari error
        )
        
        self.model.fit(X_train, y_train)
        
        # Calculate feature importance
        self.feature_importance = self.model.feature_importances_
        
        # Evaluate on test set
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Accuracy on test set: {accuracy:.4f}")
        
        return X_train, X_test, y_train, y_test, y_pred
    
    def evaluate(self, X_test, y_test, y_pred, label_mapping):
        """
        Evaluasi model dengan berbagai metrics
        """
        print("\n=== EVALUASI MODEL ===")
        
        # Accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy:.4f}")
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=self.label_encoder.classes_))
        
        # Cross-validation
        cv_scores = None
        if len(X_test) >= 5:
            try:
                cv_scores = cross_val_score(self.model, X_test, y_test, cv=min(5, len(X_test)))
                print(f"Cross-validation scores: {cv_scores}")
                print(f"CV Mean: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            except Exception as e:
                print(f"Cross-validation error: {e}")
                cv_scores = None
        else:
            print(f"Cross-validation skipped: Test samples ({len(X_test)}) < 5")
            cv_scores = None
        
        return accuracy, cv_scores
    
    def get_feature_importance(self, topic_columns):
        """
        Mendapatkan importance untuk setiap topik (feature)
        
        Parameters:
        topic_columns: list of topic column names
        
        Returns:
        importance_df: DataFrame berisi importance per topik
        """
        if self.feature_importance is None:
            print("Error: Model belum di-training")
            return None
        
        importance_df = pd.DataFrame({
            'Topic': topic_columns,
            'Importance': self.feature_importance
        }).sort_values('Importance', ascending=False)
        
        return importance_df
    
    def predict(self, X):
        """
        Predict labels untuk data baru
        
        Parameters:
        X: features (topic distributions)
        
        Returns:
        predictions: predicted labels (original)
        probabilities: prediction probabilities
        """
        if self.model is None:
            print("Error: Model belum di-training")
            return None, None
        
        predictions_encoded = self.model.predict(X)
        probabilities = self.model.predict_proba(X)
        
        # Convert back to original labels
        predictions = self.label_encoder.inverse_transform(predictions_encoded)
        
        return predictions, probabilities
    
    def save_model(self, filepath):
        """Menyimpan model Random Forest"""
        if self.model is None:
            print("Error: Model RF belum di-training")
            return False
        
        try:
            with open(filepath, 'wb') as f:
                pickle.dump({
                    'model': self.model,
                    'label_encoder': self.label_encoder,
                    'feature_importance': self.feature_importance,
                    'config': {
                        'n_estimators': self.n_estimators,
                        'random_state': self.random_state
                    }
                }, f)
            print(f"Random Forest model disimpan di: {filepath}")
            return True
        except Exception as e:
            print(f"Error menyimpan model: {e}")
            return False
    
    def load_model(self, filepath):
        """Load model Random Forest yang sudah disimpan"""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.label_encoder = model_data['label_encoder']
            self.feature_importance = model_data['feature_importance']
            self.n_estimators = model_data['config']['n_estimators']
            self.random_state = model_data['config']['random_state']
            
            print(f"Random Forest model loaded dari: {filepath}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

def load_lda_results():
    """Load hasil LDA untuk training Random Forest"""
    lda_file = os.path.join(CONFIG.LDA_OUTPUT_DIR, "lda_results.csv")
    
    if not os.path.exists(lda_file):
        print(f"Error: File LDA results tidak ditemukan: {lda_file}")
        print("Pastikan sudah menjalankan CNP_METODE_LDA.py terlebih dahulu")
        return None
    
    df = pd.read_csv(lda_file, encoding=CONFIG.ENCODING)
    print(f"Berhasil load LDA results: {len(df)} dokumen")
    print(f"Kolom: {list(df.columns)}")
    
    return df

def visualize_results(rf_model, X_test, y_test, y_pred, topic_columns, label_mapping):
    """Visualisasi hasil Random Forest"""
    print("\n=== MEMBUAT VISUALISASI ===")
    
    # Buat folder untuk visualisasi
    viz_dir = os.path.join(CONFIG.MODEL_RESULTS_DIR, "visualization")
    if not os.path.exists(viz_dir):
        os.makedirs(viz_dir)
    
    # 1. Confusion Matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=rf_model.label_encoder.classes_,
                yticklabels=rf_model.label_encoder.classes_)
    plt.title('Confusion Matrix - Random Forest')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Feature Importance (Topic Importance)
    importance_df = rf_model.get_feature_importance(topic_columns)
    plt.figure(figsize=(10, 6))
    sns.barplot(data=importance_df, x='Importance', y='Topic', palette='viridis')
    plt.title('Feature Importance - Topik yang Paling Berpengaruh')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'feature_importance.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Accuracy per Class
    from sklearn.metrics import precision_recall_fscore_support
    precision, recall, fscore, support = precision_recall_fscore_support(y_test, y_pred, average=None)
    
    class_metrics = pd.DataFrame({
        'Class': rf_model.label_encoder.classes_,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': fscore,
        'Support': support
    })
    
    plt.figure(figsize=(12, 6))
    x = np.arange(len(class_metrics))
    width = 0.25
    
    plt.bar(x - width, class_metrics['Precision'], width, label='Precision', alpha=0.7)
    plt.bar(x, class_metrics['Recall'], width, label='Recall', alpha=0.7)
    plt.bar(x + width, class_metrics['F1-Score'], width, label='F1-Score', alpha=0.7)
    
    plt.xlabel('Kelas')
    plt.ylabel('Score')
    plt.title('Precision, Recall, dan F1-Score per Kelas')
    plt.xticks(x, class_metrics['Class'], rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'class_metrics.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualisasi disimpan di: {viz_dir}")

def save_rf_results(rf_model, X_test, y_test, y_pred, topic_columns, original_df, filename="rf_results.csv"):
    """Menyimpan hasil Random Forest"""
    # Buat folder results jika belum ada
    if not os.path.exists(CONFIG.MODEL_RESULTS_DIR):
        os.makedirs(CONFIG.MODEL_RESULTS_DIR)
    
    output_file = os.path.join(CONFIG.MODEL_RESULTS_DIR, filename)
    
    # Get predictions and probabilities
    predictions, probabilities = rf_model.predict(X_test)
    
    # Create results DataFrame
    results_df = pd.DataFrame({
        'Actual_Label': rf_model.label_encoder.inverse_transform(y_test),
        'Predicted_Label': predictions,
        'Prediction_Correct': (predictions == rf_model.label_encoder.inverse_transform(y_test))
    })
    
    # Add probabilities for each class
    for i, class_name in enumerate(rf_model.label_encoder.classes_):
        results_df[f'Prob_{class_name}'] = probabilities[:, i]
    
    # Add topic distributions
    topic_columns_test = [col for col in original_df.columns if col.startswith('Topic_')]
    if topic_columns_test:
        results_df[topic_columns_test] = original_df[topic_columns_test].iloc[-len(X_test):].reset_index(drop=True)
    
    # Add metadata jika ada
    if 'filename' in original_df.columns:
        results_df['filename'] = original_df['filename'].iloc[-len(X_test):].reset_index(drop=True)
    if 'title' in original_df.columns:
        results_df['title'] = original_df['title'].iloc[-len(X_test):].reset_index(drop=True)
    
    results_df.to_csv(output_file, index=False, encoding=CONFIG.ENCODING)
    print(f"Hasil Random Forest disimpan di: {output_file}")
    
    return output_file

def save_model_metrics(rf_model, accuracy, cv_scores, topic_columns, filename="model_metrics.json"):
    """Menyimpan metrics model"""
    if not os.path.exists(CONFIG.MODEL_RESULTS_DIR):
        os.makedirs(CONFIG.MODEL_RESULTS_DIR)
    
    output_file = os.path.join(CONFIG.MODEL_RESULTS_DIR, filename)
    
    # Feature importance
    importance_df = rf_model.get_feature_importance(topic_columns)
    
    # Convert numpy types to Python native types for JSON serialization
    metrics = {
        'accuracy': float(accuracy),  # Convert to Python float
        'cross_validation_mean': float(cv_scores.mean()) if cv_scores is not None else 0.0,
        'cross_validation_std': float(cv_scores.std()) if cv_scores is not None else 0.0,
        'cross_validation_scores': cv_scores.tolist() if cv_scores is not None else [],  # Convert to list
        'feature_importance': importance_df.to_dict('records'),
        'label_mapping': {str(k): int(v) for k, v in dict(zip(
            rf_model.label_encoder.classes_, 
            rf_model.label_encoder.transform(rf_model.label_encoder.classes_)
        )).items()},  # Convert to string keys and int values
        'model_parameters': {
            'n_estimators': int(rf_model.n_estimators),  # Convert to int
            'random_state': int(rf_model.random_state)   # Convert to int
        }
    }
    
    try:
        with open(output_file, 'w', encoding=CONFIG.ENCODING) as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        
        print(f"Model metrics disimpan di: {output_file}")
        return output_file
    except Exception as e:
        print(f"Error menyimpan metrics: {e}")
        return None

def save_rf_model(rf_model, filename="random_forest_model.pkl"):
    """Menyimpan model Random Forest"""
    if not os.path.exists(CONFIG.MODEL_RESULTS_DIR):
        os.makedirs(CONFIG.MODEL_RESULTS_DIR)
    
    model_file = os.path.join(CONFIG.MODEL_RESULTS_DIR, filename)
    rf_model.save_model(model_file)
    return model_file

def save_rf_summary(rf_model, accuracy, cv_scores, X_test, y_test, y_pred, topic_columns, label_mapping, filename="rf_summary.txt"):
    """Menyimpan summary statistik Random Forest"""
    if not os.path.exists(CONFIG.MODEL_RESULTS_DIR):
        os.makedirs(CONFIG.MODEL_RESULTS_DIR)
    
    output_file = os.path.join(CONFIG.MODEL_RESULTS_DIR, filename)
    
    with open(output_file, 'w', encoding=CONFIG.ENCODING) as f:
        f.write("RANDOM FOREST CLASSIFICATION SUMMARY REPORT\n")
        f.write("=" * 60 + "\n\n")
        
        # Model Configuration
        f.write("MODEL CONFIGURATION:\n")
        f.write(f"Number of Estimators: {rf_model.n_estimators}\n")
        f.write(f"Random State: {rf_model.random_state}\n")
        f.write(f"Number of Features (Topics): {len(topic_columns)}\n")
        f.write(f"Number of Classes: {len(label_mapping)}\n\n")
        
        # Dataset Information
        f.write("DATASET INFORMATION:\n")
        f.write(f"Total Test Samples: {len(X_test)}\n")
        f.write(f"Number of Features: {X_test.shape[1]}\n")
        f.write(f"Classes: {', '.join([f'{k} ({v})' for k, v in label_mapping.items()])}\n\n")
        
        # Model Performance
        f.write("MODEL PERFORMANCE:\n")
        f.write(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)\n")
        
        if cv_scores is not None:
            f.write(f"Cross-Validation Mean: {cv_scores.mean():.4f}\n")
            f.write(f"Cross-Validation Std: {cv_scores.std():.4f}\n")
            f.write(f"Cross-Validation Scores: {[f'{score:.4f}' for score in cv_scores]}\n")
        else:
            f.write("Cross-Validation: Not available (insufficient test samples)\n")
        
        # Detailed Classification Metrics
        from sklearn.metrics import precision_recall_fscore_support, classification_report
        
        f.write("\nDETAILED CLASSIFICATION METRICS:\n")
        f.write("-" * 50 + "\n")
        
        # Get classification report as string
        report = classification_report(y_test, y_pred, target_names=rf_model.label_encoder.classes_, output_dict=False)
        f.write(report)
        f.write("\n")
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, average=None)
        
        f.write("PER-CLASS METRICS:\n")
        f.write(f"{'Class':<15} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}\n")
        f.write("-" * 60 + "\n")
        
        for i, class_name in enumerate(rf_model.label_encoder.classes_):
            f.write(f"{class_name:<15} {precision[i]:<10.4f} {recall[i]:<10.4f} {f1[i]:<10.4f} {support[i]:<10}\n")
        
        f.write("\n")
        
        # Feature Importance (Topic Importance)
        f.write("FEATURE IMPORTANCE (TOPIC RANKING):\n")
        f.write("-" * 50 + "\n")
        
        importance_df = rf_model.get_feature_importance(topic_columns)
        total_importance = importance_df['Importance'].sum()
        
        f.write(f"{'Rank':<5} {'Topic':<15} {'Importance':<12} {'Percentage':<12}\n")
        f.write("-" * 50 + "\n")
        
        for i, (idx, row) in enumerate(importance_df.iterrows(), 1):
            percentage = (row['Importance'] / total_importance) * 100
            f.write(f"{i:<5} {row['Topic']:<15} {row['Importance']:<12.4f} {percentage:<12.2f}%\n")
        
        f.write(f"\nTotal Importance: {total_importance:.4f}\n")
        
        # Top 5 Most Important Topics
        f.write(f"\nTOP 5 MOST IMPORTANT TOPICS:\n")
        top_5 = importance_df.head(5)
        for i, (idx, row) in enumerate(top_5.iterrows(), 1):
            percentage = (row['Importance'] / total_importance) * 100
            f.write(f"  {i}. {row['Topic']}: {row['Importance']:.4f} ({percentage:.1f}%)\n")
        
        # Confusion Matrix Summary
        from sklearn.metrics import confusion_matrix
        
        f.write("\nCONFUSION MATRIX SUMMARY:\n")
        cm = confusion_matrix(y_test, y_pred)
        
        f.write("Actual \\ Predicted -> ")
        f.write("  ".join([f"{name:<10}" for name in rf_model.label_encoder.classes_]) + "\n")
        
        for i, actual_class in enumerate(rf_model.label_encoder.classes_):
            f.write(f"{actual_class:<18} ")
            for j in range(len(rf_model.label_encoder.classes_)):
                f.write(f"{cm[i,j]:<10} ")
            f.write("\n")
        
        # Error Analysis
        f.write("\nERROR ANALYSIS:\n")
        total_errors = len(y_test) - accuracy_score(y_test, y_pred) * len(y_test)
        error_rate = (total_errors / len(y_test)) * 100
        
        f.write(f"Total Misclassifications: {int(total_errors)}/{len(y_test)} ({error_rate:.2f}%)\n")
        
        # Most confused classes
        if len(rf_model.label_encoder.classes_) > 1:
            f.write("\nMOST CONFUSED CLASS PAIRS (off-diagonal in confusion matrix):\n")
            max_off_diag = 0
            most_confused_pair = ("", "")
            
            for i in range(len(cm)):
                for j in range(len(cm)):
                    if i != j and cm[i,j] > max_off_diag:
                        max_off_diag = cm[i,j]
                        most_confused_pair = (rf_model.label_encoder.classes_[i], rf_model.label_encoder.classes_[j])
            
            if max_off_diag > 0:
                f.write(f"  {most_confused_pair[0]} → {most_confused_pair[1]}: {max_off_diag} misclassifications\n")
        
        # Model Insights
        f.write("\nMODEL INSIGHTS:\n")
        f.write(f"- Top {min(3, len(importance_df))} topics contribute {importance_df.head(3)['Importance'].sum()/total_importance*100:.1f}% of predictive power\n")
        f.write(f"- Model uses {len(topic_columns)} LDA topics as features\n")
        f.write(f"- Random Forest with {rf_model.n_estimators} decision trees\n")
        
        if accuracy > 0.8:
            f.write("- ✅ Excellent performance (accuracy > 80%)\n")
        elif accuracy > 0.6:
            f.write("- ✅ Good performance (accuracy > 60%)\n")
        else:
            f.write("- ⚠️  Model may need improvement (accuracy < 60%)\n")
        
        f.write(f"\nReport generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    print(f"✅ Random Forest summary disimpan di: {output_file}")
    return output_file

def test_random_forest():
    """Function untuk testing Random Forest"""
    print("=== TESTING RANDOM FOREST CLASSIFICATION ===")
    
    # Load LDA results
    lda_df = load_lda_results()
    if lda_df is None:
        return
    
    # Prepare data untuk Random Forest
    rf_model = RandomForestModel()
    
    try:
        X, y, label_mapping = rf_model.prepare_features_labels(lda_df)
    except ValueError as e:
        print(f"Error: {e}")
        return
    
    # Get topic columns untuk feature importance
    topic_columns = [col for col in lda_df.columns if col.startswith('Topic_')]
    
    # Train model
    X_train, X_test, y_train, y_test, y_pred = rf_model.train(X, y)
    
    # Evaluate model
    accuracy, cv_scores = rf_model.evaluate(X_test, y_test, y_pred, label_mapping)
    
    # TAMBAHAN: Simpan RF Summary
    summary_file = save_rf_summary(rf_model, accuracy, cv_scores, X_test, y_test, y_pred, topic_columns, label_mapping)
    
    # Save results
    results_file = save_rf_results(rf_model, X_test, y_test, y_pred, topic_columns, lda_df)
    metrics_file = save_model_metrics(rf_model, accuracy, cv_scores, topic_columns)
    model_file = save_rf_model(rf_model)
    
    # Visualisasi
    try:
        visualize_results(rf_model, X_test, y_test, y_pred, topic_columns, label_mapping)
    except Exception as e:
        print(f"Warning: Gagal membuat visualisasi: {e}")
        print("Pastikan matplotlib dan seaborn terinstall")
    
    print(f"\n RANDOM FOREST SELESAI!")
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   File hasil: {results_file}")
    print(f"   File metrics: {metrics_file}")
    print(f"   Model RF: {model_file}")
    print(f"   Summary RF: {summary_file}")  # TAMBAHAN
    
    return rf_model, accuracy

if __name__ == "__main__":
    rf_model, accuracy = test_random_forest()