import pandas as pd
import numpy as np
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DriftDataGenerator:
    def __init__(self, base_data_path="data/interim/student_interim_clean.csv"):
        self.base_data_path = base_data_path
        self.df = None
        self.drift_df = None

    def load_base_data(self):
        logger.info(f"Cargando datos base desde {self.base_data_path}")
        self.df = pd.read_csv(self.base_data_path)
        return self.df

    def generate_drift_data(self, drift_percentage=0.3, seed=42):
        np.random.seed(seed)
        if self.df is None:
            self.load_base_data()
        
        self.drift_df = self.df.copy()
        n_samples = len(self.drift_df)
        n_drift = int(n_samples * drift_percentage)
        drift_indices = np.random.choice(n_samples, n_drift, replace=False)
        
        # Drift en coaching
        if 'coaching' in self.drift_df.columns:
            idx = drift_indices[:int(n_drift * 0.4)]
            self.drift_df.loc[idx, 'coaching'] = 'no'
        
        # Drift en tiempo de estudio
        if 'time' in self.drift_df.columns:
            idx = drift_indices[int(n_drift * 0.4):int(n_drift * 0.7)]
            self.drift_df.loc[idx, 'time'] = np.random.choice(
                ['less than 1 hour', '1-2 hours'], len(idx)
            )
        
        # Drift en rendimiento
        if 'Class_ X_Percentage' in self.drift_df.columns:
            idx = drift_indices[:int(n_drift * 0.5)]
            self.drift_df.loc[idx, 'Class_ X_Percentage'] = np.random.choice(
                ['average', 'good'], len(idx)
            )
        
        logger.info(f"Drift generado en {n_drift} muestras")
        return self.drift_df

    def save_drift_data(self, output_path="data/interim/student_drift_data.csv"):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self.drift_df.to_csv(output_path, index=False)
        logger.info(f"Datos guardados en: {output_path}")
        return output_path

if __name__ == "__main__":
    generator = DriftDataGenerator()
    generator.generate_drift_data(drift_percentage=0.3)
    generator.save_drift_data()