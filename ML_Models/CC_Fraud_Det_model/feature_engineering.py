"""
Feature Engineering Module for Fraud Detection
Handles feature extraction, encoding, and scaling
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
import warnings
warnings.filterwarnings('ignore')


class FeatureEngineer:
    """
    Feature engineering pipeline for fraud detection
    """
    
    def __init__(self):
        self.label_encoders = {}
        self.scaler = None
        self.selected_features = None
        self.feature_importance = {}
        self.interaction_pairs = None
        self.scaled_columns = None
        
    def extract_datetime_features(self, df, datetime_cols=None):
        """
        Extract features from datetime columns
        
        Parameters:
        -----------
        df : DataFrame
            Input dataframe
        datetime_cols : list
            List of datetime column names
            
        Returns:
        --------
        DataFrame : Dataframe with extracted datetime features
        """
        print("\nExtracting datetime features...")
        
        if datetime_cols is None:
            datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        
        for col in datetime_cols:
            if col in df.columns:
                print(f"  Processing {col}")
                df[f'{col}_year'] = df[col].dt.year
                df[f'{col}_month'] = df[col].dt.month
                df[f'{col}_day'] = df[col].dt.day
                df[f'{col}_hour'] = df[col].dt.hour
                df[f'{col}_dayofweek'] = df[col].dt.dayofweek
                df[f'{col}_is_weekend'] = (df[col].dt.dayofweek >= 5).astype(int)
                
                # Drop original datetime column
                df = df.drop(columns=[col])
        
        return df
    
    def encode_categorical_features(self, df, target_col=None, method='label'):
        """
        Encode categorical features
        
        Parameters:
        -----------
        df : DataFrame
            Input dataframe
        target_col : str
            Target column name (to exclude from encoding)
        method : str
            'label' or 'onehot'
            
        Returns:
        --------
        DataFrame : Dataframe with encoded features
        """
        print(f"\nEncoding categorical features using {method} encoding...")
        
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Remove target column if present
        if target_col and target_col in categorical_cols:
            categorical_cols.remove(target_col)
        
        for col in categorical_cols:
            if method == 'label':
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
                else:
                    # For test data, replace unseen labels with the most frequent known class
                    known_classes = set(self.label_encoders[col].classes_)
                    most_frequent = self.label_encoders[col].classes_[0]
                    df[col] = df[col].astype(str).apply(
                        lambda x: x if x in known_classes else most_frequent
                    )
                    df[col] = self.label_encoders[col].transform(df[col])
                
                print(f"  Label encoded: {col}")
            
            elif method == 'onehot':
                # One-hot encoding with max 10 categories to avoid explosion
                if df[col].nunique() <= 10:
                    dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                    df = pd.concat([df, dummies], axis=1)
                    df = df.drop(columns=[col])
                    print(f"  One-hot encoded: {col}")
                else:
                    # Too many categories, use label encoding instead
                    if col not in self.label_encoders:
                        self.label_encoders[col] = LabelEncoder()
                        df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
                    else:
                        df[col] = self.label_encoders[col].transform(df[col].astype(str))
                    print(f"  Label encoded (too many categories): {col}")
        
        return df
    
    def scale_features(self, df, target_col=None, method='standard'):
        """
        Scale numeric features
        
        Parameters:
        -----------
        df : DataFrame
            Input dataframe
        target_col : str
            Target column name (to exclude from scaling)
        method : str
            'standard', 'minmax', or 'robust'
            
        Returns:
        --------
        DataFrame : Dataframe with scaled features
        """
        print(f"\nScaling features using {method} scaling...")
        
        # Get numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove target column if present
        if target_col and target_col in numeric_cols:
            numeric_cols.remove(target_col)
        
        if len(numeric_cols) == 0:
            print("  No numeric columns to scale")
            return df
        
        # Initialize scaler
        if self.scaler is None:
            if method == 'standard':
                self.scaler = StandardScaler()
            elif method == 'minmax':
                self.scaler = MinMaxScaler()
            elif method == 'robust':
                self.scaler = RobustScaler()
            
            self.scaled_columns = numeric_cols  # Store columns used during fit
            df[numeric_cols] = self.scaler.fit_transform(df[numeric_cols])
            print(f"  Fitted and transformed {len(numeric_cols)} features")
        else:
            # For test data, use the same columns that were fitted
            cols_to_scale = [c for c in self.scaled_columns if c in df.columns]
            df[cols_to_scale] = self.scaler.transform(df[cols_to_scale])
            print(f"  Transformed {len(cols_to_scale)} features using fitted scaler")
        
        return df
    
    def create_interaction_features(self, df, feature_pairs=None, is_train=True):
        """
        Create interaction features between numeric columns
        
        Parameters:
        -----------
        df : DataFrame
            Input dataframe
        feature_pairs : list of tuples
            Pairs of features to create interactions
        is_train : bool
            Whether this is training data
            
        Returns:
        --------
        DataFrame : Dataframe with interaction features
        """
        print("\nCreating interaction features...")
        
        if is_train:
            # Determine and store pairs during training
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if feature_pairs is None and len(numeric_cols) >= 2:
                feature_pairs = [(numeric_cols[i], numeric_cols[i+1]) 
                               for i in range(min(3, len(numeric_cols)-1))]
            self.interaction_pairs = feature_pairs
        else:
            # Reuse stored pairs for test data
            feature_pairs = self.interaction_pairs
        
        if feature_pairs:
            for feat1, feat2 in feature_pairs:
                if feat1 in df.columns and feat2 in df.columns:
                    df[f'{feat1}_x_{feat2}'] = df[feat1] * df[feat2]
                    print(f"  Created: {feat1}_x_{feat2}")
        
        return df
    
    def select_features(self, X, y, k=20, method='f_classif'):
        """
        Select top K features using statistical tests
        
        Parameters:
        -----------
        X : DataFrame
            Feature dataframe
        y : Series
            Target variable
        k : int
            Number of features to select
        method : str
            'f_classif' or 'mutual_info'
            
        Returns:
        --------
        DataFrame : Dataframe with selected features
        """
        print(f"\nSelecting top {k} features using {method}...")
        
        # Ensure k is not larger than number of features
        k = min(k, X.shape[1])
        
        if method == 'f_classif':
            selector = SelectKBest(score_func=f_classif, k=k)
        else:
            selector = SelectKBest(score_func=mutual_info_classif, k=k)
        
        X_selected = selector.fit_transform(X, y)
        
        # Get selected feature names
        selected_mask = selector.get_support()
        self.selected_features = X.columns[selected_mask].tolist()
        
        # Store feature scores
        scores = selector.scores_
        self.feature_importance = dict(zip(X.columns, scores))
        
        print(f"  Selected features: {self.selected_features[:10]}...")
        
        return pd.DataFrame(X_selected, columns=self.selected_features, index=X.index)
    
    def engineer_features(self, df, target_col=None, is_train=True):
        """
        Complete feature engineering pipeline
        
        Parameters:
        -----------
        df : DataFrame
            Input dataframe
        target_col : str
            Target column name
        is_train : bool
            Whether this is training data
            
        Returns:
        --------
        DataFrame : Engineered dataframe
        """
        print("="*60)
        print(f"FEATURE ENGINEERING ({'TRAIN' if is_train else 'TEST'})")
        print("="*60)
        
        # Extract datetime features
        df = self.extract_datetime_features(df)
        
        # Encode categorical features
        df = self.encode_categorical_features(df, target_col=target_col, method='label')
        
        # Create interaction features (for both train and test using same pairs)
        df = self.create_interaction_features(df, is_train=is_train)
        
        # Scale features
        df = self.scale_features(df, target_col=target_col, method='robust')
        
        print("\n" + "="*60)
        print("FEATURE ENGINEERING COMPLETE")
        print("="*60)
        print(f"Final shape: {df.shape}")
        
        return df


if __name__ == "__main__":
    # Example usage
    engineer = FeatureEngineer()
    
    # Load processed data
    train_df = pd.read_csv('data/processed/train_processed.csv')
    test_df = pd.read_csv('data/processed/test_processed.csv')
    
    # Identify target column
    target_col = 'is_fraud'  # Adjust based on your dataset
    
    # Engineer features
    train_engineered = engineer.engineer_features(train_df, target_col=target_col, is_train=True)
    test_engineered = engineer.engineer_features(test_df, target_col=target_col, is_train=False)
    
    # Save engineered data
    train_engineered.to_csv('data/processed/train_engineered.csv', index=False)
    test_engineered.to_csv('data/processed/test_engineered.csv', index=False)
    
    print("\nFeature engineering complete!")
