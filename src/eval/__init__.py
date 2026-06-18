# Evaluation subpackage
from .metrics import compute_detection_metrics, compute_all_metrics
from .robustness import compute_robustness_delta, robustness_report
from .aggregator import aggregate_results, generate_tables
