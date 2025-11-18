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
        
        # Drift en coaching: forzar a un valor existente (idealmente "no")
        if 'coaching' in self.drift_df.columns:
            idx = drift_indices[: int(n_drift * 0.4)]
            existing_vals = self.df['coaching'].dropna().unique()
            if len(existing_vals) > 0:
                if 'no' in existing_vals:
                    new_value = 'no'
                else:
                    # Si "no" no existe, usamos el valor más frecuente
                    new_value = self.df['coaching'].value_counts().idxmax()
                self.drift_df.loc[idx, 'coaching'] = new_value
        
        # Drift en tiempo de estudio: usar solo valores reales del dataset
        if 'time' in self.drift_df.columns:
            idx = drift_indices[int(n_drift * 0.4) : int(n_drift * 0.7)]
            existing_time_vals = self.df['time'].dropna().unique()
            if len(existing_time_vals) > 0:
                # Intentar sesgar hacia los valores "más bajos" tomando las primeras categorías
                # (en muchos casos serán 'one', 'two', etc.)
                existing_time_vals_sorted = sorted(existing_time_vals)
                if len(existing_time_vals_sorted) >= 2:
                    candidates = existing_time_vals_sorted[:2]
                else:
                    candidates = existing_time_vals_sorted
                self.drift_df.loc[idx, 'time'] = np.random.choice(candidates, len(idx))
        
        # Drift en rendimiento (Class_ X_Percentage): usar solo categorías existentes
        if 'Class_ X_Percentage' in self.drift_df.columns:
            idx = drift_indices[: int(n_drift * 0.5)]
            existing_perf_vals = self.df['Class_ X_Percentage'].dropna().unique()
            if len(existing_perf_vals) > 0:
                # Preferir categorías de rendimiento más bajas si están presentes
                low_perf_candidates = [
                    v for v in existing_perf_vals if v in ['average', 'good']
                ]
                if not low_perf_candidates:
                    low_perf_candidates = list(existing_perf_vals)
                self.drift_df.loc[idx, 'Class_ X_Percentage'] = np.random.choice(
                    low_perf_candidates, len(idx)
                )
        
        for col in self.drift_df.columns:
            if col == 'Performance':
                continue
            if col in ['coaching', 'time', 'Class_ X_Percentage']:
                continue
            existing_vals = self.df[col].dropna().unique()
            if len(existing_vals) == 0:
                continue
            idx_all = drift_indices
            most_freq = self.df[col].value_counts().idxmax()
            self.drift_df.loc[idx_all, col] = most_freq
        
        logger.info(f"Drift generado en {n_drift} muestras")
        return self.drift_df

    def save_drift_data(self, output_path="data/interim/student_drift_data.csv"):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self.drift_df.to_csv(output_path, index=False)
        logger.info(f"Datos guardados en: {output_path}")

if __name__ == "__main__":
    generator = DriftDataGenerator()
    generator.generate_drift_data(drift_percentage=0.3)
    generator.save_drift_data()