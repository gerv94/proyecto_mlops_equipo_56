# -----------------------------------------------------------------------------
# Genera un reporte HTML interactivo de comparación de modelos
# usando Plotly, basado en los resultados de entrenamiento guardados en MLflow
# o en los resultados directos de train_multiple_models.py
# -----------------------------------------------------------------------------

from pathlib import Path
import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import mlflow
from .config import REPORTS

# Carpeta de salida
REPORTS_HTML = REPORTS / "experiments_html"
REPORTS_HTML.mkdir(parents=True, exist_ok=True)

# Paleta de colores moderna
PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", 
          "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]


# -----------------------------------------------------------------------------
# Funciones auxiliares
# -----------------------------------------------------------------------------

def load_results_from_mlflow(experiment_name="student_performance_complete_experiment", tracking_uri=None):
    """
    Carga resultados de experimentos desde MLflow.
    
    Args:
        experiment_name: Nombre del experimento en MLflow
        tracking_uri: URI del tracking server de MLflow (si None usa local)
        
    Returns:
        dict: Diccionario con resultados de modelos
    """
    if tracking_uri is None:
        # Intentar cargar desde variable de entorno o usar local
        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns")
    
    mlflow.set_tracking_uri(tracking_uri)
    
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            print(f"[WARNING] Experiment '{experiment_name}' not found. Returning empty results.")
            return {}
        
        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
        
        if runs.empty:
            print(f"[WARNING] No runs found in experiment '{experiment_name}'.")
            return {}
        
        # Agrupar por modelo (asumiendo que el nombre del modelo está en tags o params)
        results = {}
        for idx, row in runs.iterrows():
            # Intentar identificar el modelo desde el run_name
            run_name = row.get('tags.mlflow.runName', 'unknown')
            model_name = run_name.split('_')[0] if '_' in run_name else 'unknown'
            
            # Si ya existe este modelo, comparar métricas y quedarse con el mejor
            if model_name in results:
                if row.get('metrics.accuracy', 0) > results[model_name].get('metrics.accuracy', 0):
                    results[model_name] = row.to_dict()
            else:
                results[model_name] = row.to_dict()
        
        print(f"[OK] Loaded {len(results)} models from MLflow.")
        return results
        
    except Exception as e:
        print(f"[ERROR] Error loading from MLflow: {str(e)}")
        return {}


def format_metrics_dict(results_dict):
    """
    Formatea resultados de MLflow en un formato estándar.
    
    Args:
        results_dict: Diccionario de resultados (de MLflow o directo)
        
    Returns:
        pd.DataFrame: DataFrame con métricas estandarizadas
    """
    formatted_data = []
    
    for model_name, data in results_dict.items():
        # Extraer métricas según el formato de origen
        if isinstance(data, dict):
            if 'accuracy' in data:
                # Formato directo de train_multiple_models.py
                formatted_data.append({
                    'model': model_name,
                    'accuracy': data.get('accuracy', 0),
                    'f1_weighted': data.get('f1_weighted', 0),
                    'precision_weighted': data.get('precision_weighted', 0),
                    'recall_weighted': data.get('recall_weighted', 0),
                    'cv_mean': data.get('cv_mean', 0),
                    'cv_std': data.get('cv_std', 0)
                })
            elif 'metrics.accuracy' in data:
                # Formato de MLflow
                formatted_data.append({
                    'model': model_name,
                    'accuracy': data.get('metrics.accuracy', 0),
                    'f1_weighted': data.get('metrics.f1_weighted', 0),
                    'precision_weighted': data.get('metrics.precision_weighted', 0),
                    'recall_weighted': data.get('metrics.recall_weighted', 0),
                    'cv_mean': data.get('metrics.cv_mean', 0),
                    'cv_std': data.get('metrics.cv_std', 0)
                })
    
    df = pd.DataFrame(formatted_data)
    
    if df.empty:
        print("[WARNING] No valid metrics found in results.")
        return df
    
    # Ordenar por F1 score descendente
    df = df.sort_values('f1_weighted', ascending=False).reset_index(drop=True)
    df['rank'] = range(1, len(df) + 1)
    
    return df


# -----------------------------------------------------------------------------
# Funciones de visualización
# -----------------------------------------------------------------------------

def create_metrics_comparison_bar(df):
    """
    Crea gráfico de barras horizontal comparando métricas principales.
    """
    fig = go.Figure()
    
    metrics = ['accuracy', 'f1_weighted', 'precision_weighted', 'recall_weighted']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for metric, color in zip(metrics, colors):
        fig.add_trace(go.Bar(
            name=metric.replace('_', ' ').title(),
            x=df['model'],
            y=df[metric],
            marker_color=color,
            text=[f'{val:.3f}' for val in df[metric]],
            textposition='outside'
        ))
    
    fig.update_layout(
        title='Comparación de Métricas por Modelo',
        xaxis_title='Modelo',
        yaxis_title='Score',
        barmode='group',
        height=500,
        yaxis=dict(range=[0, 1.05]),
        hovermode='x unified'
    )
    
    return fig


def create_radar_chart(df, top_n=5):
    """
    Crea gráfico radar para los mejores modelos.
    """
    top_models = df.head(top_n)
    
    metrics_normalized = ['accuracy', 'f1_weighted', 'precision_weighted', 'recall_weighted']
    
    fig = go.Figure()
    
    for idx, row in top_models.iterrows():
        fig.add_trace(go.Scatterpolar(
            r=[row[m] for m in metrics_normalized],
            theta=[m.replace('_', ' ').title() for m in metrics_normalized],
            fill='toself',
            name=row['model']
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(range=[0, 1], visible=True)
        ),
        showlegend=True,
        title=f'Radar Chart - Top {top_n} Modelos',
        height=600
    )
    
    return fig


def create_ranking_table(df):
    """
    Crea tabla HTML con el ranking de modelos.
    """
    html = """
    <div class="ranking-table">
        <h3>📊 Ranking de Modelos</h3>
        <table>
            <thead>
                <tr>
                    <th>Rank</th>
                    <th>Modelo</th>
                    <th>Accuracy</th>
                    <th>F1 (weighted)</th>
                    <th>Precision</th>
                    <th>Recall</th>
                    <th>CV Score</th>
                </tr>
            </thead>
            <tbody>
    """
    
    for _, row in df.iterrows():
        badge = "🥇" if row['rank'] == 1 else "🥈" if row['rank'] == 2 else "🥉" if row['rank'] == 3 else ""
        html += f"""
                <tr>
                    <td><strong>{badge} {int(row['rank'])}</strong></td>
                    <td><strong>{row['model']}</strong></td>
                    <td>{row['accuracy']:.4f}</td>
                    <td>{row['f1_weighted']:.4f}</td>
                    <td>{row['precision_weighted']:.4f}</td>
                    <td>{row['recall_weighted']:.4f}</td>
                    <td>{row['cv_mean']:.4f} ± {row['cv_std']:.4f}</td>
                </tr>
        """
    
    html += """
            </tbody>
        </table>
    </div>
    """
    
    return html


# -----------------------------------------------------------------------------
# Función principal de construcción HTML
# -----------------------------------------------------------------------------

def build_html(results_dict=None, experiment_name="student_performance_complete_experiment", tracking_uri=None):
    """
    Construye el reporte HTML completo de comparación de modelos.
    
    Args:
        results_dict: Diccionario con resultados (opcional, si None carga de MLflow)
        experiment_name: Nombre del experimento en MLflow
        tracking_uri: URI del tracking server de MLflow
        
    Returns:
        Path: Ruta del archivo HTML generado
    """
    
    # Cargar resultados
    if results_dict is None:
        results_dict = load_results_from_mlflow(experiment_name, tracking_uri)
    
    if not results_dict:
        print("[ERROR] No results available to generate report.")
        return None
    
    # Formatear datos
    df = format_metrics_dict(results_dict)
    
    if df.empty:
        print("[ERROR] No valid metrics to display.")
        return None
    
    # Crear visualizaciones
    bar_fig = create_metrics_comparison_bar(df)
    radar_fig = create_radar_chart(df, top_n=5)
    ranking_html = create_ranking_table(df)
    
    # Determinar mejor modelo
    best_model = df.iloc[0]
    
    # Generar insights
    insights = generate_insights(df, best_model)
    
    # Construir HTML
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Reporte de Comparación de Modelos - Student Performance</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            * {{ margin: 0; padding: 0; box-sizing: border-box; }}
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 20px;
                color: #333;
            }}
            .container {{
                max-width: 1400px;
                margin: 0 auto;
                background: white;
                border-radius: 15px;
                box-shadow: 0 10px 40px rgba(0,0,0,0.2);
                overflow: hidden;
            }}
            .header {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 30px;
                text-align: center;
            }}
            .header h1 {{
                font-size: 2.5em;
                margin-bottom: 10px;
            }}
            .header p {{
                font-size: 1.1em;
                opacity: 0.9;
            }}
            .content {{
                padding: 30px;
            }}
            .insights {{
                background: #f8f9fa;
                border-left: 5px solid #667eea;
                padding: 20px;
                margin: 20px 0;
                border-radius: 5px;
            }}
            .insights h3 {{
                color: #667eea;
                margin-bottom: 15px;
            }}
            .insights ul {{
                list-style-type: none;
                padding-left: 0;
            }}
            .insights li {{
                padding: 8px 0;
                border-bottom: 1px solid #e0e0e0;
            }}
            .insights li:last-child {{
                border-bottom: none;
            }}
            .plot-container {{
                margin: 30px 0;
                padding: 20px;
                background: #f8f9fa;
                border-radius: 10px;
            }}
            .ranking-table {{
                margin: 20px 0;
            }}
            .ranking-table table {{
                width: 100%;
                border-collapse: collapse;
                margin-top: 15px;
            }}
            .ranking-table th, .ranking-table td {{
                padding: 12px;
                text-align: left;
                border-bottom: 2px solid #ddd;
            }}
            .ranking-table th {{
                background: #667eea;
                color: white;
                font-weight: bold;
            }}
            .ranking-table tr:hover {{
                background: #f5f5f5;
            }}
            .footer {{
                text-align: center;
                padding: 20px;
                background: #f8f9fa;
                color: #666;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>📊 Reporte de Comparación de Modelos</h1>
                <p>Proyecto: Student Performance on an Entrance Examination</p>
                <p><small>Equipo 56 - MLOps Fase 2 | Generado automáticamente</small></p>
            </div>
            
            <div class="content">
                <div class="insights">
                    <h3>🎯 Mejor Modelo: {best_model['model']}</h3>
                    <p><strong>Métricas:</strong> Accuracy: {best_model['accuracy']:.4f} | 
                       F1 Score: {best_model['f1_weighted']:.4f} | 
                       CV Score: {best_model['cv_mean']:.4f} ± {best_model['cv_std']:.4f}</p>
                </div>
                
                <div class="insights">
                    <h3>🔍 Insights Principales</h3>
                    {insights}
                </div>
                
                <div class="plot-container">
                    <div id="bar-chart" style="width: 100%; height: 500px;"></div>
                </div>
                
                <div class="plot-container">
                    <div id="radar-chart" style="width: 100%; height: 600px;"></div>
                </div>
                
                {ranking_html}
            </div>
            
            <div class="footer">
                <p>📌 Este reporte se actualiza automáticamente con cada nuevo entrenamiento</p>
                <p>💡 Para ver más detalles, consulta MLflow UI: <code>mlflow ui</code></p>
            </div>
        </div>
        
        <script>
            var barData = {bar_fig.to_json()};
            var radarData = {radar_fig.to_json()};
            
            Plotly.newPlot('bar-chart', barData.data, barData.layout);
            Plotly.newPlot('radar-chart', radarData.data, radarData.layout);
        </script>
    </body>
    </html>
    """
    
    # Guardar archivo
    output_path = REPORTS_HTML / "models_comparison_report.html"
    output_path.write_text(html_content, encoding='utf-8')
    
    print(f"[OK] Model comparison report saved to: {output_path}")
    return output_path


def generate_insights(df, best_model):
    """
    Genera insights automáticos basados en los resultados.
    """
    num_models = len(df)
    best_score = best_model['f1_weighted']
    avg_score = df['f1_weighted'].mean()
    
    # Calcular diferencias
    score_range = df['f1_weighted'].max() - df['f1_weighted'].min()
    
    insights_html = "<ul>"
    
    # Insights automáticos
    insights_html += f"<li>✅ Se evaluaron <strong>{num_models} modelos distintos</strong></li>"
    insights_html += f"<li>🏆 El mejor modelo ({best_model['model']}) alcanzó un <strong>F1-score de {best_score:.4f}</strong></li>"
    insights_html += f"<li>📊 El rendimiento promedio del conjunto de modelos es <strong>{avg_score:.4f}</strong></li>"
    insights_html += f"<li>📈 La diferencia entre el mejor y peor modelo es <strong>{score_range:.4f}</strong> puntos</li>"
    
    # Análisis de estabilidad CV
    if best_model['cv_std'] < 0.01:
        insights_html += "<li>✨ El mejor modelo muestra <strong>alta estabilidad</strong> en validación cruzada</li>"
    elif best_model['cv_std'] < 0.05:
        insights_html += "<li>✓ El mejor modelo muestra <strong>estabilidad moderada</strong> en validación cruzada</li>"
    else:
        insights_html += "<li>⚠️ El mejor modelo muestra <strong>cierta variabilidad</strong> en validación cruzada</li>"
    
    # Comparar con promedio
    if best_score > avg_score + 0.1:
        insights_html += "<li>🚀 El mejor modelo <strong>destaca significativamente</strong> sobre el promedio</li>"
    
    insights_html += "</ul>"
    
    return insights_html


if __name__ == "__main__":
    # Ejecutar reporte standalone
    print("Generating model comparison report...")
    build_html()
    print("Done!")

