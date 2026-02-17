"""
C-BERT: Causal BERT for Fine-Grained Causal Relation Extraction

Architecture versions:
    - v2: 14-class relation classification (original)
    - v3: Factorized role + polarity + salience classification

Usage:
    from causalbert import CausalBERTMultiTaskModel, train, sentence_analysis
    from causalbert.model import ROLE2ID, SPAN_LABEL_MAP
"""

__version__ = "3.0.0"

# Core model components
from .model import (
    CausalBERTMultiTaskModel,
    CausalBERTMultiTaskConfig,
    MultiTaskCollator,
)

from .utils import (
    get_compute_dtype,
)

# Label mappings and constants
from .model import (
    ROLE_LABELS,
    ROLE2ID,
    ID2ROLE,
    POLARITY_LABELS,
    POLARITY2ID,
    ID2POLARITY,
    SPAN_LABEL_MAP,
    PARALLEL_SEP_TOKEN,
    V2_TO_V3_MAPPING,
    RELATION_MAP_V2,
    SALIENCE_LABELS, 
    SALIENCE2ID, 
    ID2SALIENCE
)

# Inference functions
from .infer import (
    load_model,
    sentence_analysis,
    extract_tuples,
    merge_bio_entities,
)

# Dataset creation
from .dataset import (
    create_datasets,
)

# Training
from .train import (
    train
)

# Error Analysis
from .error_analysis import (
    error_analysis, 
    token_error_analysis
)

import os
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
INDICATORS_CSV = os.path.join(DATA_DIR, "indicators.csv")

__all__ = [
    # Model classes
    "CausalBERTMultiTaskModel",
    "CausalBERTMultiTaskConfig", 
    "MultiTaskCollator",
    
    # Label maps (v2)
    "RELATION_MAP_V2",
    "SPAN_LABEL_MAP",
    
    # Label maps (v3)
    "ROLE_LABELS",
    "ROLE2ID",
    "ID2ROLE",
    "POLARITY_LABELS",
    "POLARITY2ID",
    "ID2POLARITY",
    "SALIENCE_LABELS", 
    "SALIENCE2ID", 
    "ID2SALIENCE",
    
    # Mappings
    "V2_TO_V3_MAPPING",
    
    # Constants
    "PARALLEL_SEP_TOKEN",
    
    # Functions
    "load_model",
    "sentence_analysis",
    "extract_tuples",
    "merge_bio_entities",
    "create_datasets",
    "train",
    "get_compute_dtype",
    "error_analysis",
    "token_error_analysis"
]

