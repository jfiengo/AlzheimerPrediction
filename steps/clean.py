import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer

class Cleaner:
    def __init__(self):
        self.imputer = SimpleImputer(strategy='most_frequent', missing_values=np.nan)
        
        
    def clean_data(self, data):
        # Identify column types
        categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
        categorical_cols.remove("Alzheimer’s Diagnosis")  # Remove target from features

        # Define column groups
        numerical_cols = ['Age', 'Education Level', 'BMI', 'Cognitive Test Score']

        binary_cols = ['Gender', 'Diabetes', 'Hypertension', 'Cholesterol Level', 
                    'Family History of Alzheimer’s', 'Genetic Risk Factor (APOE-ε4 allele)', 
                    'Urban vs Rural Living']

        nominal_cols = ['Country', 'Smoking Status', 'Alcohol Consumption', 
                        'Employment Status', 'Marital Status']

        # Define ordinal mappings with correct order for each ordinal variable
        physical_activity_order = ['Low', 'Medium', 'High']
        depression_order = ['Low', 'Medium', 'High']
        sleep_order = ['Poor', 'Average', 'Good']
        diet_order = ['Unhealthy', 'Average', 'Healthy']
        pollution_order = ['Low', 'Medium', 'High']
        social_order = ['Low', 'Medium', 'High']
        income_order = ['Low', 'Medium', 'High']
        stress_order = ['Low', 'Medium', 'High']

        # Create ordinal encoder for each ordinal feature with its specific ordering
        ordinal_transformers = [
            ('physical', OrdinalEncoder(categories=[physical_activity_order]), ['Physical Activity Level']),
            ('depression', OrdinalEncoder(categories=[depression_order]), ['Depression Level']),
            ('sleep', OrdinalEncoder(categories=[sleep_order]), ['Sleep Quality']),
            ('diet', OrdinalEncoder(categories=[diet_order]), ['Dietary Habits']),
            ('pollution', OrdinalEncoder(categories=[pollution_order]), ['Air Pollution Exposure']),
            ('social', OrdinalEncoder(categories=[social_order]), ['Social Engagement Level']),
            ('income', OrdinalEncoder(categories=[income_order]), ['Income Level']),
            ('stress', OrdinalEncoder(categories=[stress_order]), ['Stress Levels'])
        ]

        # Create the preprocessing pipeline
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_cols),
                ('bin', OrdinalEncoder(), binary_cols),
                ('nom', OneHotEncoder(drop='first', handle_unknown='ignore'), nominal_cols),
                *ordinal_transformers
            ],
            remainder='drop'  # Drop any columns not specified
        )
        # Use preprocessing pipeline
        features = data.drop('Alzheimer’s Diagnosis', axis=1)
        features_transformed = preprocessor.fit_transform(X)
        
        return features_transformed