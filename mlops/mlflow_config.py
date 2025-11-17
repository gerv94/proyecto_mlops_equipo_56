# ============================================================
# MLflow & AWS Configuration Module
# Centralized configuration for experiment tracking and S3 storage
# ============================================================

import os
import logging
from typing import Optional

import mlflow

# ============================================================
# CONFIGURATION CONSTANTS
# ============================================================

# MLflow tracking URI (can be overridden by MLFLOW_TRACKING_URI env var)
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")

# S3 artifact root for MLflow (can be overridden by MLFLOW_S3_ARTIFACT_ROOT env var)
MLFLOW_S3_ARTIFACT_ROOT = os.environ.get(
    "MLFLOW_S3_ARTIFACT_ROOT",
    "s3://itesm-mna/202502-equipo56/mlflow"
)

# AWS configuration
AWS_PROFILE = os.environ.get("AWS_PROFILE", "equipo56")
AWS_REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-2")

# Logging
logger = logging.getLogger(__name__)


# ============================================================
# CONFIGURATION FUNCTIONS
# ============================================================

def configure_aws_credentials() -> None:
    """Configure AWS credentials and region in environment variables.
    
    Sets AWS_PROFILE, AWS_DEFAULT_REGION, and AWS_REGION if not already set.
    This ensures boto3 and MLflow can access S3 without additional configuration.
    """
    if "AWS_PROFILE" not in os.environ:
        os.environ["AWS_PROFILE"] = AWS_PROFILE
        logger.debug(f"Set AWS_PROFILE={AWS_PROFILE}")
    
    if "AWS_DEFAULT_REGION" not in os.environ:
        os.environ["AWS_DEFAULT_REGION"] = AWS_REGION
        logger.debug(f"Set AWS_DEFAULT_REGION={AWS_REGION}")
    
    if "AWS_REGION" not in os.environ:
        os.environ["AWS_REGION"] = AWS_REGION
        logger.debug(f"Set AWS_REGION={AWS_REGION}")


def setup_mlflow(experiment_name: str, run_name: Optional[str] = None) -> None:
    """Configure MLflow tracking URI and experiment.
    
    This function should be called at the beginning of any script that uses MLflow.
    It ensures consistent configuration across all training and reporting scripts.
    
    Args:
        experiment_name: Name of the MLflow experiment to use/create
        run_name: Optional name for the MLflow run (useful for identification)
    
    Example:
        >>> from mlops.mlflow_config import setup_mlflow
        >>> setup_mlflow("student_performance_experiment")
        >>> with mlflow.start_run():
        >>>     mlflow.log_param("model", "RandomForest")
    """
    # Configure AWS credentials first
    configure_aws_credentials()
    
    # Set MLflow tracking URI
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    logger.info(f"MLflow tracking URI: {MLFLOW_TRACKING_URI}")
    
    # Set or create experiment
    mlflow.set_experiment(experiment_name)
    logger.info(f"MLflow experiment: {experiment_name}")
    
    # Set run name if provided (will be used in next mlflow.start_run())
    if run_name:
        mlflow.set_tag("mlflow.runName", run_name)
        logger.info(f"MLflow run name: {run_name}")


def get_mlflow_client():
    """Get configured MLflow client instance.
    
    Returns:
        mlflow.tracking.MlflowClient: Configured client for MLflow operations
    
    Example:
        >>> from mlops.mlflow_config import get_mlflow_client
        >>> client = get_mlflow_client()
        >>> experiments = client.search_experiments()
    """
    configure_aws_credentials()
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    return mlflow.tracking.MlflowClient()
