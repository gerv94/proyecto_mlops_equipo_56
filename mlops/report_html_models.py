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
        # Si no se especifica experimento, combinar todos los experimentos
        if experiment_name is None:
            all_experiments = mlflow.search_experiments()
            if not all_experiments:
                print("[WARNING] No experiments found in MLflow.")
                return {}
            
            print(f"[INFO] Combinando {len(all_experiments)} experimento(s) (no se especificó --experiment):")
            experiment_ids = []
            for exp in all_experiments:
                runs_count = len(mlflow.search_runs([exp.experiment_id]))
                print(f"  - {exp.name} ({runs_count} runs)")
                experiment_ids.append(exp.experiment_id)
            
            runs = mlflow.search_runs(experiment_ids=experiment_ids)
        else:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                print(f"[WARNING] Experiment '{experiment_name}' not found.")
                # Listar experimentos disponibles
                all_experiments = mlflow.search_experiments()
                if all_experiments:
                    print(f"[INFO] Experimentos disponibles en MLflow:")
                    for exp in all_experiments:
                        runs_count = len(mlflow.search_runs([exp.experiment_id]))
                        print(f"  - {exp.name} ({runs_count} runs)")
                    print(f"\n[INFO] Usa --experiment <nombre> para especificar un experimento")
                return {}
            
            runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
        
        if runs.empty:
            exp_name = experiment_name if experiment_name else "experimentos combinados"
            print(f"[WARNING] No runs found in '{exp_name}'.")
            return {}
        
        # Agrupar por modelo (asumiendo que el nombre del modelo está en tags o params)
        results = {}
        # Obtener nombres de experimentos para incluir en el nombre del modelo
        exp_names = {}
        if experiment_name is None:
            all_experiments = mlflow.search_experiments()
            for exp in all_experiments:
                exp_names[exp.experiment_id] = exp.name
        
        for idx, row in runs.iterrows():
            # Intentar identificar el modelo desde el run_name
            run_name = row.get('tags.mlflow.runName', 'unknown')
            exp_id = row.get('experiment_id', 'unknown')
            exp_name = exp_names.get(exp_id, '')
            
            # Usar el run_name completo para GridSearch, o el run_name completo si hay pocos runs
            if 'GridSearch' in run_name or 'grid' in run_name.lower():
                # Para GridSearch, usar el run_name completo como identificación única
                base_model_name = run_name
            else:
                # Para otros casos, usar el run_name completo para mejor diferenciación
                base_model_name = run_name
            
            # Si se combinan múltiples experimentos, incluir el nombre del experimento
            if experiment_name is None and exp_name:
                model_name = f"{exp_name} - {base_model_name}"
            else:
                model_name = base_model_name
            
            # Si ya existe este modelo, comparar métricas y quedarse con el mejor
            if model_name in results:
                # Comparar por F1 o accuracy dependiendo de qué métrica esté disponible
                current_f1 = row.get('metrics.test_f1_weighted', row.get('metrics.f1_weighted', 0))
                best_f1 = results[model_name].get('metrics.test_f1_weighted', results[model_name].get('metrics.f1_weighted', 0))
                if current_f1 > best_f1:
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
            elif any(k.startswith('metrics.') for k in data.keys()):
                # Formato de MLflow - soporta ambos formatos de métricas
                # Formato 1: test_acc, test_f1_weighted, test_f1_macro, cv_f1_weighted_mean, cv_f1_weighted_std
                # Formato 2: acc_test, acc_train, f1_macro, f1_micro, f1_weighted
                formatted_data.append({
                    'model': model_name,
                    # Test accuracy - buscar en ambos formatos
                    'test_acc': data.get('metrics.test_acc', data.get('metrics.acc_test', 0)),
                    # Train accuracy (solo disponible en formato 2)
                    'train_acc': data.get('metrics.acc_train', 0),
                    # F1 weighted - buscar en ambos formatos
                    'test_f1_weighted': data.get('metrics.test_f1_weighted', data.get('metrics.f1_weighted', 0)),
                    # F1 macro - buscar en ambos formatos
                    'test_f1_macro': data.get('metrics.test_f1_macro', data.get('metrics.f1_macro', 0)),
                    # F1 micro (solo disponible en formato 2)
                    'test_f1_micro': data.get('metrics.f1_micro', 0),
                    # CV metrics (solo disponibles en formato 1)
                    'cv_f1_weighted_mean': data.get('metrics.cv_f1_weighted_mean', 0),
                    'cv_f1_weighted_std': data.get('metrics.cv_f1_weighted_std', 0)
                })
    
    df = pd.DataFrame(formatted_data)
    
    if df.empty:
        print("[WARNING] No valid metrics found in results.")
        return df
    
    # Ordenar por F1 score descendente
    sort_col = 'test_f1_weighted' if 'test_f1_weighted' in df.columns else ('f1_weighted' if 'f1_weighted' in df.columns else df.columns[1])
    df = df.sort_values(sort_col, ascending=False).reset_index(drop=True)
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
    
    # Usar solo las métricas que existen en el DataFrame
    available_metrics = []
    metric_mapping = {
        'test_acc': 'Test Accuracy',
        'test_f1_weighted': 'Test F1 Weighted',
        'test_f1_macro': 'Test F1 Macro',
        'cv_f1_weighted_mean': 'CV F1 Mean',
        'accuracy': 'Accuracy',
        'f1_weighted': 'F1 Weighted'
    }
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for i, col in enumerate(df.columns):
        if col in metric_mapping and col != 'model' and col != 'rank':
            available_metrics.append(col)
    
    # Limitar a las primeras métricas disponibles con colores
    metrics_to_plot = available_metrics[:len(colors)]
    
    for metric, color in zip(metrics_to_plot, colors[:len(metrics_to_plot)]):
        fig.add_trace(go.Bar(
            name=metric_mapping.get(metric, metric.replace('_', ' ').title()),
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
    
    # Usar solo las métricas que existen en el DataFrame
    available_metrics = []
    metric_labels = {
        'test_acc': 'Test Acc',
        'test_f1_weighted': 'F1 Weighted',
        'test_f1_macro': 'F1 Macro',
        'cv_f1_weighted_mean': 'CV Mean',
        'accuracy': 'Accuracy',
        'f1_weighted': 'F1 Weighted'
    }
    
    for col in df.columns:
        if col in metric_labels and col != 'model' and col != 'rank':
            available_metrics.append(col)
    
    # Limitar a 4 métricas principales para el radar
    metrics_to_plot = available_metrics[:4]
    
    fig = go.Figure()
    
    for idx, row in top_models.iterrows():
        fig.add_trace(go.Scatterpolar(
            r=[row[m] for m in metrics_to_plot],
            theta=[metric_labels.get(m, m.replace('_', ' ').title()) for m in metrics_to_plot],
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
                    <th>Test Acc</th>
                    <th>Test F1 (weighted)</th>
                    <th>Test F1 (macro)</th>
                    <th>CV F1 Mean</th>
                    <th>CV F1 Std</th>
                </tr>
            </thead>
            <tbody>
    """
    
    for _, row in df.iterrows():
        badge = "🥇" if row['rank'] == 1 else "🥈" if row['rank'] == 2 else "🥉" if row['rank'] == 3 else ""
        # Usar las métricas que existen, con fallback si no están
        test_acc = row.get('test_acc', row.get('accuracy', 0))
        test_f1_w = row.get('test_f1_weighted', row.get('f1_weighted', 0))
        test_f1_m = row.get('test_f1_macro', 0)
        cv_mean = row.get('cv_f1_weighted_mean', row.get('cv_mean', 0))
        cv_std = row.get('cv_f1_weighted_std', row.get('cv_std', 0))
        
        html += f"""
                <tr>
                    <td><strong>{badge} {int(row['rank'])}</strong></td>
                    <td><strong>{row['model']}</strong></td>
                    <td>{test_acc:.4f}</td>
                    <td>{test_f1_w:.4f}</td>
                    <td>{test_f1_m:.4f}</td>
                    <td>{cv_mean:.4f} ± {cv_std:.4f}</td>
                    <td>{cv_std:.4f}</td>
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
                    <p><strong>Métricas:</strong> Test Acc: {best_model.get('test_acc', best_model.get('accuracy', 0)):.4f} | 
                       Test F1 (weighted): {best_model.get('test_f1_weighted', best_model.get('f1_weighted', 0)):.4f} | 
                       Test F1 (macro): {best_model.get('test_f1_macro', 0):.4f} | 
                       CV F1 Mean: {best_model.get('cv_f1_weighted_mean', best_model.get('cv_mean', 0)):.4f} ± {best_model.get('cv_f1_weighted_std', best_model.get('cv_std', 0)):.4f}</p>
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
    
    # Guardar archivo con nombre basado en el experimento
    if experiment_name:
        # Limpiar nombre del experimento para usar como nombre de archivo
        safe_name = experiment_name.replace(" ", "_").replace("/", "_").replace("\\", "_")
        filename = f"models_comparison_{safe_name}.html"
    else:
        filename = "models_comparison_report.html"
    
    output_path = REPORTS_HTML / filename
    output_path.write_text(html_content, encoding='utf-8')
    
    print(f"[OK] Model comparison report saved to: {output_path}")
    return output_path


def generate_insights(df, best_model):
    """
    Genera insights automáticos basados en los resultados.
    """
    num_models = len(df)
    # Usar las métricas correctas con fallback
    best_f1_col = 'test_f1_weighted' if 'test_f1_weighted' in df.columns else 'f1_weighted'
    best_score = best_model.get(best_f1_col, best_model.get('f1_weighted', 0))
    avg_score = df[best_f1_col].mean() if best_f1_col in df.columns else df.get('f1_weighted', pd.Series([0])).mean()
    
    # Calcular diferencias
    score_range = df[best_f1_col].max() - df[best_f1_col].min() if best_f1_col in df.columns else 0
    
    insights_html = "<ul>"
    
    # Insights automáticos
    insights_html += f"<li>✅ Se evaluaron <strong>{num_models} modelos distintos</strong></li>"
    insights_html += f"<li>🏆 El mejor modelo ({best_model['model']}) alcanzó un <strong>F1-score (weighted) de {best_score:.4f}</strong></li>"
    insights_html += f"<li>📊 El rendimiento promedio del conjunto de modelos es <strong>{avg_score:.4f}</strong></li>"
    insights_html += f"<li>📈 La diferencia entre el mejor y peor modelo es <strong>{score_range:.4f}</strong> puntos</li>"
    
    # Análisis de estabilidad CV
    cv_std = best_model.get('cv_f1_weighted_std', best_model.get('cv_std', 0))
    if cv_std < 0.01:
        insights_html += "<li>✨ El mejor modelo muestra <strong>alta estabilidad</strong> en validación cruzada</li>"
    elif cv_std < 0.05:
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

