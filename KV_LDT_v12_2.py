import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")      # deterministic cuBLAS

import gc
import hashlib
import itertools
import json
import logging
import math
import re
import shutil
import sys
import time
import zlib
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from packaging.version import Version
from scipy import stats
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from statsmodels.stats.multitest import multipletests
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

if Version(transformers.__version__) < Version("4.53"):
    raise RuntimeError("transformers>=4.53 is required (AttentionMaskInterface, "
                       f"SmolLM3, Qwen3); found {transformers.__version__}")
from transformers.masking_utils import AttentionMaskInterface, sdpa_mask
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, AttentionInterface

try:
    from datasets import load_dataset
    HF_DATASETS_AVAILABLE = True
except ImportError:
    HF_DATASETS_AVAILABLE = False

torch.set_num_threads(1)

# ════════════════════════════════════════════════════════════════════════════
# GLOBALS: logging, seeds, device
# ════════════════════════════════════════════════════════════════════════════

PROJECT_ROOT = Path(os.environ.get("KV_PROJECT_ROOT", Path(__file__).resolve().parent)).resolve()
SEED = 42

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger("kv_ldt_v12_2")


def add_file_log(output_dir: str):
    """The log file lives next to the results it documents (added once per process)."""
    path = os.path.abspath(os.path.join(output_dir, "kv_ldt_v12_2.log"))
    root = logging.getLogger()
    if not any(isinstance(h, logging.FileHandler) and h.baseFilename == path for h in root.handlers):
        handler = logging.FileHandler(path)
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
        root.addHandler(handler)


def seed_everything(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


os.environ["PYTHONHASHSEED"] = str(SEED)
seed_everything(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=True)

if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
    # bf16 is the training dtype of every model in the sweep and cannot overflow
    # the residual stream the way fp16 does (Qwen2.5 is known to overflow in fp16).
    COMPUTE_DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
else:
    DEVICE = torch.device("cpu")
    COMPUTE_DTYPE = torch.float32
logger.info(f"device={DEVICE} compute_dtype={COMPUTE_DTYPE} "
            f"torch={torch.__version__} transformers={transformers.__version__}")


def safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.@#-]+", "_", s)


def free_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class ModelConfig:
    name: str
    model_id: str
    family: str                    # unit of the leave-one-family-out sensitivity (M1)
    batch_size: int = 128          # upper bound; reduced once at start-up if it does not fit


@dataclass(frozen=True)
class Regime:
    """One KV-sharing regime.  Only the primary regime runs the full policy design."""
    name: str
    reuse_factor: int
    exempt_fraction: float
    role: str                      # "primary" | "dose" | "sensitivity"

    def __post_init__(self):
        if self.role not in ("primary", "dose", "sensitivity"):
            raise ValueError(f"unknown regime role {self.role!r}")


@dataclass
class ExperimentConfig:
    WORDS_PATH: str = str(Path(os.environ.get("KV_WORDS_PATH", PROJECT_ROOT / "Items.csv")))
    NONWORDS_PATH: str = str(Path(os.environ.get("KV_NONWORDS_PATH", PROJECT_ROOT / "NonWord.csv")))
    OUTPUT_DIR: str = str(Path(os.environ.get("KV_OUTPUT_DIR", PROJECT_ROOT / "kv_results_v12_2")))

    MODELS: List[ModelConfig] = field(default_factory=lambda: [
        ModelConfig("SmolLM2-360M", "HuggingFaceTB/SmolLM2-360M", "SmolLM", 512),
        ModelConfig("Qwen2.5-1.5B", "Qwen/Qwen2.5-1.5B", "Qwen", 256),
        ModelConfig("Llama-3.2-1B", "meta-llama/Llama-3.2-1B", "Llama", 256),
        ModelConfig("Qwen2.5-3B", "Qwen/Qwen2.5-3B", "Qwen", 256),
        ModelConfig("Llama-3.2-3B", "meta-llama/Llama-3.2-3B", "Llama", 256),
        ModelConfig("SmolLM3-3B-Base", "HuggingFaceTB/SmolLM3-3B-Base", "SmolLM", 256),
        ModelConfig("Qwen3-4B-Base", "Qwen/Qwen3-4B-Base", "Qwen", 128),
        ModelConfig("Qwen3-8B-Base", "Qwen/Qwen3-8B-Base", "Qwen", 128),
        ModelConfig("Llama-3.1-8B", "meta-llama/Llama-3.1-8B", "Llama", 128),
    ])

    # ── Stimuli ───────────────────────────────────────────────────────
    # Representational readout: must end with "{stimulus}", so the readout token
    # is the stimulus' final subword (the locus where multi-token words are assembled).
    STIMULUS_TEMPLATE: str = "Here is a letter string: {stimulus}"
    # Behavioural readout: few-shot lexical decision; demonstration strings are
    # removed from the item pool.
    TASK_INSTRUCTION: str = "Decide whether each letter string is a real English word."
    TASK_DEMOS: Tuple[Tuple[str, int], ...] = (("garden", 1), ("flirp", 0), ("river", 1),
                                               ("trelk", 0), ("snorb", 0), ("table", 1))
    TASK_ITEM: str = "String: {stimulus}\nAnswer:"
    TASK_ANSWERS: Tuple[str, str] = (" Yes", " No")
    BEHAVIOR_MIN_BASE_MASS: float = 0.5            # H8a: no_reuse must put >= this mass on Yes/No
    BEHAVIOR_MASS_MIN_RATIO: float = 0.5           # H8b/H8c: policy keeps >= this share of that mass
    RELATIVE_DAMAGE_MIN_BASE: float = 0.5          # relative damage only where baseline evidence >= 0.5 SD
    # The item table is filtered by the tokenizers of ALL these models, whatever subset
    # a job runs (default: every model in MODELS at construction).
    ITEM_FILTER_MODEL_IDS: List[str] = field(default_factory=list)
    MAX_ITEMS_PER_CLASS: Optional[int] = None      # None = all balanced items
    HIGH_FREQ_PERCENTILE: float = 66.0             # extreme tertiles for HF / LF
    LOW_FREQ_PERCENTILE: float = 33.0
    TOKEN_STRATA: Tuple[int, ...] = (1, 2, 3, 4)   # last stratum = ">= 4"

    # ── Splits / probes ───────────────────────────────────────────────
    TEST_SIZE: float = 0.15
    VAL_SIZE: float = 0.15
    PROBE_SEEDS: List[int] = field(default_factory=lambda: [42, 123, 2024])
    PROBE_L2_GRID: List[float] = field(default_factory=lambda: [1e-4, 1e-3, 1e-2, 1e-1])
    PROBE_MAX_ITER: int = 500
    NONFINITE_MAX_FRACTION: float = 0.01           # layer is invalid above this

    # ── KV sharing ────────────────────────────────────────────────────
    REGIMES: List[Regime] = field(default_factory=lambda: [
        Regime("rf2_ex20", 2, 0.20, "primary"),     # CLA-2, first 20% of layers exempt
        Regime("rf4_ex20", 4, 0.20, "dose"),        # stronger compression (C5)
        Regime("rf2_ex00", 2, 0.00, "sensitivity"),  # detokenisation layers included (C2)
    ])
    GATE_FRACTIONS: List[float] = field(default_factory=lambda: [1.0 / 6.0, 1.0 / 3.0, 0.5])
    PRIMARY_GATE_FRACTION: float = 1.0 / 3.0
    GATED_FAMILIES: List[str] = field(default_factory=lambda: [
        "cka_high", "cka_low", "kvdist_low", "kvdist_high", "fidelity_high"])
    N_NULL_DRAWS: int = 50                         
    N_AUX_DRAWS: int = 3                           
    DEPTH_BINS: int = 3
    COUNTERFACTUAL_ENABLED: bool = True
    CALIB_N_ITEMS: int = 1024                     
    CALIB_MAX_ROWS: int = 4096                    
    
    KVSHARER_COS_THRESHOLD: float = float(os.environ.get("KV_KVSHARER_THRESHOLD", 0.90))
    KVSHARER_THRESHOLD_VERIFIED: bool = os.environ.get("KV_KVSHARER_THRESHOLD_VERIFIED", "").lower() in (
        "1", "true", "yes")
    KVSHARER_CHECK_ITEMS: int = 256

    # ── Inference ─────────────────────────────────────────────────────
    N_BANDS: int = 3                               
    N_BOOTSTRAP: int = 2000
    N_BOOTSTRAP_FRAGILITY: int = 1000
    N_PERMUTATIONS: int = 10000
    MIN_STRATUM_SIZE: int = 20                     
    MIN_ALIGNMENT_RUNS: int = 6                    
    ALPHA: float = 0.05
    DECISION_MIN_MODEL_FRACTION: float = 7.0 / 9.0
    MIN_MODELS_WILCOXON: int = 6
    H5_EARLY_DEPTH: float = 0.60
    CEILING_AUC: float = 0.98                      
    FLOOR_AUC: float = 0.60                       
    NEG_CONTROL_ATOL: float = 1e-3                
    LINEAR_ADEQUACY_TOL: float = 0.02              

    # ── Matching ──────────────────────────────────────────────────────
    MATCH_COVARIATES: List[str] = field(default_factory=lambda: ["length", "ortho_n", "bg_mean"])
    MATCH_CALIPER_SD: float = 0.25                 

    # ── Probe reliability ─────────────────────────────────────────────
    RELIABILITY_ENABLED: bool = True
    RELIABILITY_POLICIES: List[str] = field(default_factory=lambda: ["no_reuse", "full"])
    MLP_ARCHITECTURES: Dict[str, Tuple[int, ...]] = field(
        default_factory=lambda: {"mlp_1x256": (256,), "mlp_512x256": (512, 256)})
    MLP_EPOCHS: int = 30
    MLP_PATIENCE: int = 5
    MLP_LR: float = 1e-3
    MLP_WEIGHT_DECAY: float = 1e-2
    MLP_DROPOUT: float = 0.3
    MLP_BATCH: int = 256
    MDL_FRACTIONS: List[float] = field(default_factory=lambda: [
        0.001, 0.002, 0.004, 0.008, 0.016, 0.032, 0.0625, 0.125, 0.25, 0.5, 1.0])

    # ── Mechanistic diagnostics ───────────────────────────────────────
    MECH_N_ITEMS: int = 2048

    # ── Fragility scan ────────────────────────────────────────────────
    FRAGILITY_ENABLED: bool = True

    # ── Generative evaluation ─────────────────────────────────────────
    DOWNSTREAM_ENABLED: bool = True
    DOWNSTREAM_MODELS: List[str] = field(default_factory=list)   # empty = all
    PPL_WINDOW: int = 1024
    PPL_STRIDE: int = 512
    PPL_MAX_WINDOWS: int = 200
    LAMBADA_N: int = 500
    GENERATIVE_FLOOR_PPL_RATIO: float = 10.0       # flag: policy PPL >= 10x no_reuse PPL

    RESUME: bool = True
    DPI: int = 300
    # Full-tier runs hold every layer's state for every item (~18-21 GB for an 8B
    # model with ~70k items in bf16).  "disk" keeps them in bit-exact memory maps.
    STATE_STORE: str = os.environ.get("KV_STATE_STORE", "memory")
    STATE_DIR: str = os.environ.get("KV_STATE_DIR", "")

    def __post_init__(self):
        roles = [r.role for r in self.REGIMES]
        if roles.count("primary") != 1 or self.REGIMES[0].role != "primary":
            raise ValueError("exactly one primary regime is required and it must come first")
        if len({r.name for r in self.REGIMES}) != len(self.REGIMES):
            raise ValueError("regime names must be unique")
        unknown = set(self.GATED_FAMILIES) - {"cka_high", "cka_low", "kvdist_low", "kvdist_high", "fidelity_high"}
        if unknown:
            raise ValueError(f"unknown gated families {sorted(unknown)}")
        if not any(abs(f - self.PRIMARY_GATE_FRACTION) < 1e-12 for f in self.GATE_FRACTIONS):
            raise ValueError("PRIMARY_GATE_FRACTION must be one of GATE_FRACTIONS")
        if self.STIMULUS_TEMPLATE.count("{stimulus}") != 1 or not self.STIMULUS_TEMPLATE.endswith("{stimulus}"):
            raise ValueError("STIMULUS_TEMPLATE must contain '{stimulus}' once, at the end")
        if self.STATE_STORE not in ("memory", "disk"):
            raise ValueError("STATE_STORE must be 'memory' or 'disk'")
        if not self.ITEM_FILTER_MODEL_IDS:
            self.ITEM_FILTER_MODEL_IDS = [m.model_id for m in self.MODELS]
        self.RESULTS_DIR = os.path.join(self.OUTPUT_DIR, "results")
        self.PAPER_DIR = os.path.join(self.OUTPUT_DIR, "paper_artifacts")
        for d in (self.OUTPUT_DIR, self.RESULTS_DIR, self.PAPER_DIR):
            os.makedirs(d, exist_ok=True)

    @property
    def primary_regime(self) -> Regime:
        return self.REGIMES[0]

    def task_template(self) -> str:
        yes, no = self.TASK_ANSWERS
        shots = "".join(self.TASK_ITEM.format(stimulus=s) + (yes if lab else no) + "\n\n"
                        for s, lab in self.TASK_DEMOS)
        return f"{self.TASK_INSTRUCTION}\n\n{shots}{self.TASK_ITEM}"

    def model_dir(self, model_name: str) -> str:
        d = os.path.join(self.RESULTS_DIR, safe_name(model_name))
        os.makedirs(d, exist_ok=True)
        return d


# ════════════════════════════════════════════════════════════════════════════
# KV SUBSTITUTION ENGINE (attention-interface level)
# ════════════════════════════════════════════════════════════════════════════
#
# Every attention call of a model loaded with attn_implementation=KV_ATTN_IMPL is
# routed through `kv_substitution_attention`.  With no active controller it is
# exactly SDPA.  The (key, value) tensors it receives are post-norm, post-RoPE and
# pre-GQA-repeat, i.e. what a KV cache stores, so substituting them is the
# faithful model of cross-layer KV-cache sharing.

KV_ATTN_IMPL = "kv_substitution"
_BASE_ATTENTION = ALL_ATTENTION_FUNCTIONS["sdpa"]


class KVRouter:
    active = None


def kv_substitution_attention(module, query, key, value, attention_mask, **kwargs):
    ctl = KVRouter.active
    if ctl is None:
        return _BASE_ATTENTION(module, query, key, value, attention_mask, **kwargs)
    return ctl.attend(module, query, key, value, attention_mask, kwargs)


AttentionInterface.register(KV_ATTN_IMPL, kv_substitution_attention)
# Without a mask registration, transformers builds NO mask for a custom
# implementation; registering the SDPA mask keeps causal masks identical.
AttentionMaskInterface.register(KV_ATTN_IMPL, sdpa_mask)


@contextmanager
def routed(controller):
    KVRouter.active = controller
    try:
        yield controller
    finally:
        KVRouter.active = None


class LiveSubstitution:
    """Target reads the K/V its (earlier) source produced in the same pass."""

    def __init__(self, source_map: Dict[int, int]):
        self.source_map = source_map
        self.sources = frozenset(source_map.values())
        self.cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}

    def attend(self, module, query, key, value, mask, kwargs):
        li = module.layer_idx
        src = self.source_map.get(li)
        if src is not None:
            key, value = self.cache[src]
        if li in self.sources:
            self.cache[li] = (key, value)
        return _BASE_ATTENTION(module, query, key, value, mask, **kwargs)


class CleanCapture:
    """Pass 1 of CLEAN semantics: record source K/V of an unperturbed pass."""

    def __init__(self, sources):
        self.sources = frozenset(sources)
        self.cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}

    def attend(self, module, query, key, value, mask, kwargs):
        if module.layer_idx in self.sources:
            self.cache[module.layer_idx] = (key, value)
        return _BASE_ATTENTION(module, query, key, value, mask, **kwargs)


class CleanInjection:
    """Pass 2 of CLEAN semantics: targets read the recorded clean source K/V."""

    def __init__(self, source_map: Dict[int, int], cache):
        self.source_map = source_map
        self.cache = cache

    def attend(self, module, query, key, value, mask, kwargs):
        src = self.source_map.get(module.layer_idx)
        if src is not None:
            key, value = self.cache[src]
        return _BASE_ATTENTION(module, query, key, value, mask, **kwargs)


class CriterionMeter:
    """
    Calibration pass (no substitution).  The substitution acts on EVERY position (BOS /
    attention sink, template prefix and stimulus), so every criterion is measured on
    all positions of `row_mask`:
      * kv_sum   per layer and split half, the sum of flattened [K; V] rows: the
                 token-occurrence weighted layer mean that KVSharer ranks pairs by.
      * err/own  per row, for every target t with source s, the query-aware
                 residual-stream error of reading the source's K/V:
                     err = || W_O (A(q_t, K_s, V_s) - A(q_t, K_t, V_t)) ||^2
                     own = || o_proj(A(q_t, K_t, V_t)) ||^2
                 (W_O without its bias, which cancels in a difference).
      * k/v      K and V rows of the pair layers at `keep_mask` positions only (a
                 deterministic subsample used for CKA and the row-wise distance).
    Results of a batch are committed only after its forward pass completes, so a batch
    that is retried after an out-of-memory error is never counted twice.
    """

    def __init__(self, pairs: Dict[int, int]):
        self.pairs = pairs
        self.sources = frozenset(pairs.values())
        self.layers = self.sources | frozenset(pairs)
        self.row_mask: Optional[torch.Tensor] = None
        self.keep_mask: Optional[torch.Tensor] = None
        self.row_half: Optional[torch.Tensor] = None
        self.batch_kv: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        self.rows: Dict[str, Dict[int, List[torch.Tensor]]] = {
            name: defaultdict(list) for name in ("k", "v", "err", "own", "kv_sum")}
        self._pending: Dict[str, Dict[int, List[torch.Tensor]]] = {}

    def begin_batch(self, row_mask: torch.Tensor, keep_mask: torch.Tensor, row_half: torch.Tensor):
        """row_mask (b, t): rows entering the criteria; keep_mask (b, t): subset whose K/V
        rows are stored; row_half (b, t) in {0, 1}: split half of the row's item."""
        self.row_mask, self.keep_mask, self.row_half = row_mask, keep_mask, row_half
        self.batch_kv.clear()
        self._pending = {name: defaultdict(list) for name in self.rows}

    def commit(self):
        for name, per_layer in self._pending.items():
            for li, chunks in per_layer.items():
                self.rows[name][li] += chunks
        self._pending = {}

    def _flat(self, x: torch.Tensor) -> torch.Tensor:
        b, h, t, d = x.shape
        return x.permute(0, 2, 1, 3).reshape(b, t, h * d)[self.row_mask].float()

    def attend(self, module, query, key, value, mask, kwargs):
        li = module.layer_idx
        out = _BASE_ATTENTION(module, query, key, value, mask, **kwargs)
        kv = torch.cat([self._flat(key), self._flat(value)], 1)
        half = self.row_half[self.row_mask]
        self._pending["kv_sum"][li].append(
            torch.stack([kv[half == 0].sum(0), kv[half == 1].sum(0)]).double().cpu())
        if li in self.layers:
            keep = self.keep_mask[self.row_mask]
            d = kv.shape[1] // 2
            self._pending["k"][li].append(kv[keep, :d].cpu())
            self._pending["v"][li].append(kv[keep, d:].cpu())
        if li in self.sources:
            self.batch_kv[li] = (key, value)
        src = self.pairs.get(li)
        if src is not None:
            ks, vs = self.batch_kv[src]
            o_sub = _BASE_ATTENTION(module, query, ks, vs, mask, **kwargs)[0]
            o_own = out[0]
            n = int(self.row_mask.sum())
            diff = (o_sub - o_own)[self.row_mask].reshape(n, -1).float()
            own = o_own[self.row_mask].reshape(n, -1)
            self._pending["err"][li].append(
                F.linear(diff, module.o_proj.weight.float()).pow(2).sum(-1).cpu())
            self._pending["own"][li].append(module.o_proj(own).float().pow(2).sum(-1).cpu())
        return out


# ════════════════════════════════════════════════════════════════════════════
# POLICIES
# ════════════════════════════════════════════════════════════════════════════

NO_REUSE = "no_reuse"
FULL = "full"
UNIFORM = "random_same_ratio"
RANDOM_SOURCE = "random_source"
RANDOM_PAIRS = "random_pairs"
KVSHARER_FAMILIES = ("kvsharer_dissimilar", "kvsharer_similar")
DEPTH_PREFIX = "rdm_"
LIVE_PREV = "live_prev_head"
CF_PREV = "cf_prev_head"
CF_NEXT = "cf_next_head"
CONTROL = "control"                 # regime label of no_reuse in every output table


# Gated family -> (calibration column, sign); the k highest signed scores are selected.
GATE_CRITERIA = {"cka_high": ("cka", 1.0), "cka_low": ("cka", -1.0),
                 "kvdist_low": ("kvsharer_distance", -1.0), "kvdist_high": ("kvsharer_distance", 1.0),
                 "fidelity_high": ("attn_out_rel_err", -1.0)}


def exempt_cutoff(num_layers: int, fraction: float) -> int:
    return min(int(math.ceil(num_layers * fraction)), num_layers - 1)


def group_heads(num_layers: int, reuse_factor: int, ec: int) -> Dict[int, int]:
    """CLA-style grouping of eligible layers [ec, L): each layer -> its group head."""
    return {li: ec + ((li - ec) // reuse_factor) * reuse_factor for li in range(ec, num_layers)}


def eligible_bands(num_layers: int, ec: int, n_bands: int) -> Dict[str, List[int]]:
    """Pre-specified contiguous equal-count bands over eligible layers + 'all'."""
    names = ["early", "middle", "late"] if n_bands == 3 else [f"band{i + 1}" for i in range(n_bands)]
    eligible = np.arange(ec, num_layers)
    bands = {n: [int(x) for x in part] for n, part in zip(names, np.array_split(eligible, n_bands))}
    bands["all"] = [int(x) for x in eligible]
    return bands


def gate_sizes(n_targets: int, cfg: ExperimentConfig) -> Tuple[List[int], int]:
    """Distinct gated target counts k (1 <= k < n_targets) and the primary k."""
    def size(f):
        return min(n_targets - 1, max(1, int(round(f * n_targets))))
    return sorted({size(f) for f in cfg.GATE_FRACTIONS}), size(cfg.PRIMARY_GATE_FRACTION)


@dataclass
class PolicySpec:
    name: str                       # unique run name, e.g. "rdm_cka_high@k2#7"
    family: str                     # policy family, e.g. "rdm_cka_high"
    source_map: Dict[int, int]      # target layer -> source layer
    semantics: str                  # "live" | "clean"
    deployable: bool                # realisable at inference with a shared cache
    num_layers: int
    regime: Optional[str] = None
    draw: int = 0                   # 0 = deterministic policy; >= 1 = null draw
    tier: str = "full"              # "full": frozen + retrained readouts | "frozen": frozen on test items only
    behavioral: bool = True
    downstream: bool = False
    null_space_size: int = 0
    exhaustive: bool = False

    def __post_init__(self):
        targets, sources = set(self.source_map), set(self.source_map.values())
        if targets & sources:
            raise ValueError(f"{self.name}: a source is also a target (reuse chain)")
        if self.semantics == "live" and any(s >= t for t, s in self.source_map.items()):
            raise ValueError(f"{self.name}: live semantics needs every source < target")
        if any(not (0 <= x < self.num_layers) for x in targets | sources):
            raise ValueError(f"{self.name}: layer index out of range")
        if self.tier not in ("full", "frozen"):
            raise ValueError(f"{self.name}: unknown tier {self.tier!r}")

    @property
    def k(self) -> int:
        """Number of substituted layers."""
        return len(self.source_map)

    @property
    def targets(self) -> List[int]:
        return sorted(self.source_map)

    @property
    def kv_memory_fraction(self) -> float:
        """Exact fraction of layers whose K/V must be cached."""
        return 1.0 - len(self.source_map) / self.num_layers

    def to_json(self) -> Dict:
        dist = [t - s for t, s in self.source_map.items()]
        return {"name": self.name, "family": self.family, "semantics": self.semantics,
                "deployable": self.deployable, "regime": self.regime, "k": self.k,
                "draw": self.draw, "tier": self.tier, "behavioral": self.behavioral,
                "downstream": self.downstream, "null_space_size": self.null_space_size,
                "exhaustive": self.exhaustive, "num_layers": self.num_layers,
                "kv_memory_fraction": self.kv_memory_fraction,
                "mean_source_distance": float(np.mean(dist)) if dist else 0.0,
                "source_map": {str(t): int(s) for t, s in sorted(self.source_map.items())}}


def _rng(*keys) -> np.random.Generator:
    return np.random.default_rng([SEED] + [zlib.crc32(str(k).encode()) for k in keys])


def no_reuse_spec(num_layers: int) -> PolicySpec:
    return PolicySpec(NO_REUSE, NO_REUSE, {}, "live", True, num_layers, downstream=True)


def null_draws(blocks: List[Tuple[List[int], int]], n_draws: int,
               rng: np.random.Generator) -> Tuple[List[List[int]], int, bool]:
    """
    Null space = product over blocks of the size-`need` subsets of `pool`.  A space
    with <= n_draws elements is enumerated exhaustively (exact randomisation test,
    order shuffled); otherwise n_draws elements are drawn uniformly with replacement
    (Monte-Carlo test).  Each element is the concatenation of the chosen members of
    every block, in block order.
    """
    size = math.prod(math.comb(len(pool), need) for pool, need in blocks)
    if size <= n_draws:
        space = [[int(x) for combo in choice for x in combo] for choice in
                 itertools.product(*[itertools.combinations(pool, need) for pool, need in blocks])]
        return [space[i] for i in rng.permutation(len(space))], size, True
    draws = [[int(x) for pool, need in blocks
              for x in sorted(rng.choice(pool, size=need, replace=False))] for _ in range(n_draws)]
    return draws, size, False


def random_pair_maps(eligible: Sequence[int], k: int, n_draws: int,
                     rng: np.random.Generator) -> Tuple[List[Dict[int, int]], int, bool]:
    """
    Null space of the KVSharer-search families: every chain-free live map with k
    targets in `eligible` (all >= 1) whose sources are earlier non-target layers.
    For targets t_1 < ... < t_k there are prod_j (t_j - j) source assignments
    (layers below t_j minus the j targets below it), so target sets are drawn with
    that weight and sources uniformly: every map is equally likely, and a searched
    map is a member of the space (exchangeable under H0).
    """
    if math.comb(len(eligible), k) > 5_000_000:
        raise ValueError(f"random-pair null space too large to weight exactly (C({len(eligible)},{k}))")
    combos = np.array(list(itertools.combinations(sorted(eligible), k)), dtype=np.int64)
    weights = np.prod(combos - np.arange(k)[None, :], axis=1)
    size = int(weights.sum())

    def sources(tset):
        return {t: [x for x in range(t) if x not in tset] for t in tset}

    if size <= n_draws:
        maps = [dict(zip(row, choice)) for row in combos.tolist()
                for choice in itertools.product(*sources(set(row)).values())]
        return [maps[i] for i in rng.permutation(len(maps))], size, True
    rows = rng.choice(len(combos), size=n_draws, p=weights / weights.sum())
    maps = []
    for r in rows:
        tset = [int(t) for t in combos[r]]
        maps.append({t: int(rng.choice(src)) for t, src in sources(set(tset)).items()})
    return maps, size, False


def build_policy_specs(num_layers: int, regime: Regime, criteria: pd.DataFrame, cfg: ExperimentConfig,
                       searched: Optional[Dict[str, Dict[int, int]]] = None) -> List[PolicySpec]:
    """
    All runs of one (model, regime).  Non-primary regimes run `full` only.  In the
    primary regime, `criteria` has one row per target of the `full` map with
    columns target, source, cka, kvsharer_distance, attn_out_rel_err, and
    `searched` holds the KVSharer-search maps (family -> source map).
    Tiers: deterministic policies at the primary k (and `full`, counterfactuals,
    KVSharer-search maps) get frozen + retrained readouts; gated policies at other
    k and all null draws get the frozen readout on test items only.  Null families
    used by decision rules (depth-matched, uniform, random-pair) are drawn at the
    primary k; the uniform family is also drawn at the other k (dose-response).
    Deterministic runs and the first N_AUX_DRAWS draws of every null family get the
    behavioural readout; those of the uniform and random-source families also get
    the generative evaluation (points for H6 / H8c).
    """
    L, R = num_layers, cfg.N_NULL_DRAWS
    ec = exempt_cutoff(L, regime.exempt_fraction)
    gm = group_heads(L, regime.reuse_factor, ec)
    full_map = {t: s for t, s in gm.items() if t != s}
    targets = sorted(full_map)

    def spec(name, family, smap, draw=0, tier="full", behavioral=True, downstream=True, semantics="live",
             size=0, exhaustive=False):
        return PolicySpec(name, family, smap, semantics, semantics == "live", L, regime.name,
                          draw=draw, tier=tier, behavioral=behavioral, downstream=downstream,
                          null_space_size=size, exhaustive=exhaustive)

    specs = [spec(FULL, FULL, dict(full_map))]
    if regime.role != "primary":
        return specs
    if len(targets) < 3:
        logger.warning(f"  only {len(targets)} targets: gated and null policies skipped")
        return specs

    ks, k0 = gate_sizes(len(targets), cfg)
    crit = criteria.set_index("target").reindex(targets)
    score = {fam: sign * crit[col] for fam, (col, sign) in GATE_CRITERIA.items()}
    bins = [set(int(x) for x in b) for b in np.array_split(np.arange(ec, L), cfg.DEPTH_BINS)]

    def add_nulls(family, k, draws, size, exhaustive):
        for d, smap in enumerate(draws, 1):
            aux = d <= cfg.N_AUX_DRAWS
            specs.append(spec(f"{family}@k{k}#{d}", family, smap, draw=d, tier="frozen",
                              behavioral=aux, downstream=aux and family in (UNIFORM, RANDOM_SOURCE),
                              size=size, exhaustive=exhaustive))

    def restricted_nulls(family, blocks, k, seed_key):
        draws, size, exhaustive = null_draws(blocks, R, _rng(seed_key, L, regime.name, k))
        add_nulls(family, k, [{t: full_map[t] for t in sorted(c)} for c in draws], size, exhaustive)

    for k in ks:
        for fam in cfg.GATED_FAMILIES:
            ranked = score[fam].sort_values(ascending=False, kind="mergesort")
            sel = sorted(int(t) for t in ranked.index[:k])
            specs.append(spec(f"{fam}@k{k}", fam, {t: full_map[t] for t in sel},
                              tier="full" if k == k0 else "frozen"))
            if k == k0:
                blocks = [(sorted(set(targets) & b), len(set(sel) & b)) for b in bins]
                restricted_nulls(DEPTH_PREFIX + fam, [bk for bk in blocks if bk[1]], k, ("depth_matched", fam))
        restricted_nulls(UNIFORM, [(targets, k)], k, (UNIFORM,))

    # H4 null (sanity check): every target reads a uniformly random cached layer of the
    # eligible range below it (own head included, so `full` is a member of the null).
    # Exempt layers are excluded: reading layer 0..ec-1 is a different, near-certainly
    # catastrophic perturbation that would make the test trivially significant.
    cached = [li for li in range(ec, L) if li not in full_map]
    draws, size, exhaustive = null_draws([([c for c in cached if c < t], 1) for t in targets], R,
                                         _rng(RANDOM_SOURCE, L, regime.name))
    add_nulls(RANDOM_SOURCE, len(targets), [dict(zip(targets, c)) for c in draws], size, exhaustive)

    # KVSharer search maps (all layer pairs, greedy, output-similarity acceptance) and
    # their null: uniformly random chain-free maps with the same number of targets.
    for fam, smap in (searched or {}).items():
        if smap:
            specs.append(spec(f"{fam}@k{len(smap)}", fam, smap))
    eligible = list(range(max(ec, 1), L))
    for k in sorted({len(m) for m in (searched or {}).values() if m}):
        draws, size, exhaustive = random_pair_maps(eligible, k, R, _rng(RANDOM_PAIRS, L, regime.name, k))
        add_nulls(RANDOM_PAIRS, k, draws, size, exhaustive)

    if cfg.COUNTERFACTUAL_ENABLED:
        heads = sorted(set(gm.values()))
        nxt = {t: next((h for h in heads if h > t), None) for t in targets}
        cf_targets = [t for t in targets if nxt[t] is not None]
        if cf_targets:
            if cf_targets != targets:
                specs.append(spec(LIVE_PREV, LIVE_PREV, {t: full_map[t] for t in cf_targets}, downstream=False))
            specs.append(spec(CF_PREV, CF_PREV, {t: full_map[t] for t in cf_targets}, semantics="clean"))
            specs.append(spec(CF_NEXT, CF_NEXT, {t: nxt[t] for t in cf_targets}, semantics="clean"))
    return specs


def single_layer_spec(num_layers: int, target: int) -> PolicySpec:
    """Fragility scan: substitute K/V at one layer only, from the previous layer."""
    return PolicySpec(f"single_{target}", "single_layer", {target: target - 1}, "live", True, num_layers)


# ════════════════════════════════════════════════════════════════════════════
# STIMULI, TOKENISATION AND SPLITS
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class Encoding:
    ids: List[List[int]]
    n_tokens: np.ndarray            # stimulus subwords; -1 = no clean stimulus boundary
    seq_len: np.ndarray


def bos_prefix(tokenizer) -> List[int]:
    """[bos] if the tokenizer prepends BOS by default, else []."""
    bos = tokenizer.bos_token_id
    return [bos] if bos is not None and tokenizer("a").input_ids[:1] == [bos] else []


def encode(tokenizer, template: str, stimuli: Sequence[str], bos: List[int]) -> Encoding:
    """
    Tokenise template-filled stimuli without automatic special tokens (so no EOS
    can follow the stimulus) and prepend BOS when the model expects it.  A stimulus
    has a clean boundary when every token overlapping it contains only stimulus
    characters plus leading whitespace; n_tokens is the number of such tokens.
    For a template ending in the stimulus, its final token must end the string.
    """
    prefix, suffix = template.split("{stimulus}")
    start = len(prefix)
    texts = [prefix + s + suffix for s in stimuli]
    enc = tokenizer(texts, add_special_tokens=False, return_offsets_mapping=True)
    ids_out, n_tok = [], []
    for ids, offs, s, text in zip(enc["input_ids"], enc["offset_mapping"], stimuli, texts):
        end = start + len(s)
        n, clean = 0, True
        for a, b in offs:
            if b <= start or a >= end:
                continue
            n += 1
            if text[a:start].strip() or b > end:
                clean = False
        if not suffix and (not offs or offs[-1][1] != end):
            clean = False
        ids_out.append(bos + list(ids))
        n_tok.append(n if (n >= 1 and clean) else -1)
    return Encoding(ids_out, np.asarray(n_tok), np.asarray([len(x) for x in ids_out]))


def read_pools(cfg: ExperimentConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Word and nonword pools.  Only lowercase alphabetic strings are kept (nonwords
    are letter strings); duplicates are removed within a class; strings present in
    both classes (ambiguous label, train/test leakage) and the few-shot
    demonstration strings are removed.
    """
    def read(path: str, is_word: int) -> pd.DataFrame:
        df = pd.read_csv(path)

        def num(col):
            if col not in df.columns:
                return np.nan
            return pd.to_numeric(df[col].replace("#", np.nan), errors="coerce").to_numpy()

        out = pd.DataFrame({"stimulus": df["Word"].astype(str).str.strip().str.lower()})
        out["is_word"] = is_word
        out["log_freq"] = num("Log_Freq_HAL") if is_word else np.nan
        out["ortho_n"] = num("Ortho_N")
        out["bg_mean"] = num("BG_Mean")
        out = out[out["stimulus"].str.fullmatch(r"[a-z]+")]
        return out.drop_duplicates("stimulus")

    words, nonwords = read(cfg.WORDS_PATH, 1), read(cfg.NONWORDS_PATH, 0)
    drop = (set(words["stimulus"]) & set(nonwords["stimulus"])) | {s for s, _ in cfg.TASK_DEMOS}
    logger.info(f"Pools: {len(words)} words, {len(nonwords)} nonwords; removing "
                f"{len(drop)} cross-class duplicates / demonstration strings")
    return words[~words["stimulus"].isin(drop)], nonwords[~nonwords["stimulus"].isin(drop)]


def tokenizer_validity(cfg: ExperimentConfig, stimuli: Sequence[str]) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    M6: a string is kept only if it has a clean stimulus boundary in BOTH templates
    for EVERY model's tokenizer, so all models are evaluated on identical items.
    """
    valid = np.ones(len(stimuli), bool)
    report = []
    task = cfg.task_template()
    for model_id in cfg.ITEM_FILTER_MODEL_IDS:
        tok = AutoTokenizer.from_pretrained(model_id)
        bos = bos_prefix(tok)
        ok = ((encode(tok, cfg.STIMULUS_TEMPLATE, stimuli, bos).n_tokens >= 1)
              & (encode(tok, task, stimuli, bos).n_tokens >= 1))
        report.append({"model_id": model_id, "n_strings": len(stimuli), "n_excluded": int((~ok).sum())})
        valid &= ok
    return valid, pd.DataFrame(report)


def build_items(cfg: ExperimentConfig, words: pd.DataFrame, nonwords: pd.DataFrame) -> pd.DataFrame:
    """Balanced item table with frequency groups and a wordlikeness flag for nonwords."""
    n = min(len(words), len(nonwords))
    if cfg.MAX_ITEMS_PER_CLASS:
        n = min(n, int(cfg.MAX_ITEMS_PER_CLASS))
    words = words.sample(n=n, random_state=SEED)
    nonwords = nonwords.sample(n=n, random_state=SEED)
    items = pd.concat([words, nonwords], ignore_index=True)
    items["length"] = items["stimulus"].str.len().astype(float)

    items["freq_group"] = np.where(items["is_word"] == 1, "unknown", "nonword")
    fm = (items["is_word"] == 1) & items["log_freq"].notna()
    hi = np.percentile(items.loc[fm, "log_freq"], cfg.HIGH_FREQ_PERCENTILE)
    lo = np.percentile(items.loc[fm, "log_freq"], cfg.LOW_FREQ_PERCENTILE)
    items.loc[fm, "freq_group"] = "mid"
    items.loc[fm & (items["log_freq"] >= hi), "freq_group"] = "high"
    items.loc[fm & (items["log_freq"] <= lo), "freq_group"] = "low"

    # Wordlike nonwords (C1, hardest contrast): above the nonword median of the mean
    # z-score of OrthoN and bigram frequency (whichever are present).
    nw = items["is_word"] == 0
    zs = [(items.loc[nw, c] - items.loc[nw, c].mean()) / items.loc[nw, c].std()
          for c in ("ortho_n", "bg_mean") if items.loc[nw, c].notna().mean() >= 0.9]
    items["wordlike"] = False
    if zs:
        score = pd.concat(zs, axis=1).mean(axis=1)
        items.loc[score.index, "wordlike"] = (score > score.median()).to_numpy()

    items = items.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    items.insert(0, "item_id", np.arange(len(items)))
    logger.info(f"Items: {len(items)} ({n} words / {n} nonwords); "
                f"HF>={hi:.3f}: {(items.freq_group == 'high').sum()}, "
                f"LF<={lo:.3f}: {(items.freq_group == 'low').sum()}, "
                f"wordlike nonwords: {int(items.wordlike.sum())}")
    return items


@dataclass
class Splits:
    test: np.ndarray
    train: Dict[int, np.ndarray]
    val: Dict[int, np.ndarray]


def make_splits(items: pd.DataFrame, cfg: ExperimentConfig) -> Splits:
    """Test split fixed by SEED (same items for every seed, policy and model)."""
    strat = np.where(items["is_word"] == 0,
                     np.where(items["wordlike"], "nonword_wordlike", "nonword"),
                     "word_" + items["freq_group"].astype(str))
    idx = np.arange(len(items))
    rest, test = train_test_split(idx, test_size=cfg.TEST_SIZE, stratify=strat, random_state=SEED)
    vsz = cfg.VAL_SIZE / (1.0 - cfg.TEST_SIZE)
    train, val = {}, {}
    for s in cfg.PROBE_SEEDS:
        tr, va = train_test_split(rest, test_size=vsz, stratify=strat[rest], random_state=s)
        train[s], val[s] = np.sort(tr), np.sort(va)
    return Splits(np.sort(test), train, val)


# ════════════════════════════════════════════════════════════════════════════
# MODEL RUNNER
# ════════════════════════════════════════════════════════════════════════════

def unbiased_linear_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    """Linear CKA with the unbiased HSIC estimator (Song et al., 2012). Rows = samples."""
    X = X.to(DEVICE, torch.float64)
    Y = Y.to(DEVICE, torch.float64)
    n = X.shape[0]
    if n < 8:
        return float("nan")
    X = X - X.mean(0, keepdim=True)
    Y = Y - Y.mean(0, keepdim=True)

    def hsic(K, M):
        K = K.clone()
        M = M.clone()
        K.fill_diagonal_(0.0)
        M.fill_diagonal_(0.0)
        t1 = (K * M).sum()
        t2 = K.sum() * M.sum() / ((n - 1) * (n - 2))
        t3 = 2.0 * (K.sum(0) @ M.sum(1)) / (n - 2)
        return (t1 + t2 - t3) / (n * (n - 3))

    K, M = X @ X.T, Y @ Y.T
    den = torch.sqrt(hsic(K, K) * hsic(M, M))
    return float(hsic(K, M) / den) if den > 0 else float("nan")


class FrozenReadout:
    """
    Seed-averaged no_reuse linear probes collapsed exactly into one affine map per
    layer:  mean_s [((x - mu_s) / sd_s) . w_s + b_s] = x . W + c.  Layers without a
    valid control probe return NaN.
    """

    def __init__(self, bank: "ProbeBank"):
        ws = bank.w.astype(np.float64) / bank.sd.astype(np.float64)
        W = ws.mean(0)
        c = (bank.b.astype(np.float64) - (bank.mu.astype(np.float64) * ws).sum(-1)).mean(0)
        invalid = np.isnan(bank.lam)
        W[invalid] = np.nan
        c[invalid] = np.nan
        self.W = torch.as_tensor(W, dtype=torch.float32, device=DEVICE)
        self.c = torch.as_tensor(c, dtype=torch.float32, device=DEVICE)

    def logits(self, h: torch.Tensor, layer: int) -> torch.Tensor:
        return h @ self.W[layer] + self.c[layer]


class ModelRunner:
    """One frozen CausalLM with length-bucketed, padding-free batched forwards."""

    def __init__(self, mc: ModelConfig, cfg: ExperimentConfig, items: pd.DataFrame):
        self.mc, self.cfg = mc, cfg
        logger.info(f"Loading {mc.name} ({mc.model_id})")
        self.tokenizer = AutoTokenizer.from_pretrained(mc.model_id)
        dtype_kw = ({"dtype": COMPUTE_DTYPE} if Version(transformers.__version__) >= Version("4.56")
                    else {"torch_dtype": COMPUTE_DTYPE})
        self.model = AutoModelForCausalLM.from_pretrained(
            mc.model_id, attn_implementation=KV_ATTN_IMPL, low_cpu_mem_usage=True, **dtype_kw)
        self.model.to(DEVICE).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.layers = self.model.get_decoder().layers
        self.L = len(self.layers)
        self.H = int(self.model.config.hidden_size)
        for i, layer in enumerate(self.layers):
            if getattr(layer.self_attn, "layer_idx", None) != i:
                raise RuntimeError(f"{mc.name}: layer {i} has no matching self_attn.layer_idx")
        # M3: SmolLM3 skips RoPE in some layers (NoPE); substitution across a RoPE/NoPE
        # pair is a qualitatively different perturbation and is flagged.
        self.rope = [bool(getattr(layer.self_attn, "use_rope", True)) for layer in self.layers]
        self.store_dtype = COMPUTE_DTYPE
        self.bos = bos_prefix(self.tokenizer)
        stimuli = items["stimulus"].tolist()
        self.carrier = encode(self.tokenizer, cfg.STIMULUS_TEMPLATE, stimuli, self.bos)
        self.task = encode(self.tokenizer, cfg.task_template(), stimuli, self.bos)
        if (self.carrier.n_tokens < 1).any() or (self.task.n_tokens < 1).any():
            raise RuntimeError(f"{mc.name}: items without a clean stimulus boundary reached the "
                               f"model (the global tokenizer filter was bypassed)")
        self.n_tokens = self.carrier.n_tokens
        self.answer_ids = [self._first_token(a) for a in cfg.TASK_ANSWERS]
        if self.answer_ids[0] == self.answer_ids[1]:
            raise RuntimeError(f"{mc.name}: answers {cfg.TASK_ANSWERS} share their first token "
                               f"{self.answer_ids[0]}; the behavioural score would be identically 0")
        self.oom_splits = 0
        self.batch_size = self._fit_batch_size(mc.batch_size)
        self.self_test()
        logger.info(f"  layers={self.L} hidden={self.H} bos={bool(self.bos)} "
                    f"nope_layers={[i for i, r in enumerate(self.rope) if not r]} "
                    f"batch={self.batch_size} n_tokens: " + ", ".join(
                        f"{k}:{v}" for k, v in sorted(pd.Series(self.n_tokens).value_counts().items())))

    def _first_token(self, text: str) -> int:
        ids = self.tokenizer(text, add_special_tokens=False).input_ids
        if len(ids) != 1:
            logger.warning(f"  answer {text!r} is {len(ids)} tokens; its first token is scored")
        return int(ids[0])

    def rope_mismatch(self, source_map: Dict[int, int]) -> int:
        return int(sum(self.rope[t] != self.rope[s] for t, s in source_map.items()))

    # ── Batching ──────────────────────────────────────────────────────
    def _fit_batch_size(self, upper: int) -> int:
        """
        Fix the batch size once, on the longest sequence, with every layer recorded
        and a substitution active.  A fixed batch plan makes frozen readouts of
        different policies bitwise comparable (exact negative controls).
        """
        if DEVICE.type != "cuda":
            return upper
        longest = int(np.argmax(self.task.seq_len))
        smap = {t: t - 1 for t in range(1, self.L, 2)}
        bs = upper
        while bs >= 1:
            try:
                ids = torch.tensor([self.task.ids[longest]] * bs, dtype=torch.long, device=DEVICE)
                with torch.no_grad(), self.recording(range(self.L)):
                    with routed(LiveSubstitution(smap)):
                        self.model(input_ids=ids, use_cache=False, logits_to_keep=1)
                free_memory()
                return bs
            except torch.cuda.OutOfMemoryError:
                free_memory()
                bs //= 2
        raise RuntimeError(f"{self.mc.name}: a single sequence does not fit in memory")

    def run_batches(self, enc: Encoding, item_idx: np.ndarray,
                    step: Callable[[np.ndarray, torch.Tensor], None]):
        """
        Equal-length batches (no padding) in a plan that depends only on
        (encoding, item_idx, batch size).  An out-of-memory batch is split in two
        (logged and counted); `step` must commit its results only after its forward.
        """
        lengths = enc.seq_len[item_idx]
        queue = []
        for length in np.unique(lengths):
            pos = np.flatnonzero(lengths == length)
            queue += [pos[i:i + self.batch_size] for i in range(0, len(pos), self.batch_size)]
        while queue:
            pos = queue.pop(0)
            ids = torch.tensor([enc.ids[item_idx[p]] for p in pos], dtype=torch.long, device=DEVICE)
            try:
                step(pos, ids)
            except torch.cuda.OutOfMemoryError:
                free_memory()
                if len(pos) == 1:
                    raise
                self.oom_splits += 1
                logger.warning(f"  OOM: batch of {len(pos)} split (batch plan deviates)")
                half = len(pos) // 2
                queue[:0] = [pos[:half], pos[half:]]

    def forward(self, ids: torch.Tensor, spec: Optional[PolicySpec], logits_to_keep: int = 1):
        kw = dict(input_ids=ids, use_cache=False, logits_to_keep=logits_to_keep)
        if spec is None or not spec.source_map:
            return self.model(**kw)
        if spec.semantics == "live":
            with routed(LiveSubstitution(spec.source_map)):
                return self.model(**kw)
        with routed(CleanCapture(spec.source_map.values())) as cap:
            self.model(**{**kw, "logits_to_keep": 1})
        with routed(CleanInjection(spec.source_map, cap.cache)):
            return self.model(**kw)

    @contextmanager
    def recording(self, layer_ids: Sequence[int]):
        """Forward hooks storing each decoder layer's output at the last position."""
        rec: Dict[int, torch.Tensor] = {}

        def make(i):
            def hook(_m, _inp, out):
                h = out[0] if isinstance(out, tuple) else out
                rec[i] = h[:, -1, :].detach()
            return hook

        handles = [self.layers[i].register_forward_hook(make(i)) for i in layer_ids]
        try:
            yield rec
        finally:
            for h in handles:
                h.remove()

    # ── Start-up self-tests of the substitution machinery ─────────────
    @torch.no_grad()
    def self_test(self):
        """
        (1) an empty live map reproduces the unrouted model exactly; (2) clean
        injection of a layer's own captured K/V reproduces it exactly (the capture
        point is the point of use); (3) substituting a layer's K/V changes the
        output (the custom attention implementation is actually in use).
        """
        lengths = self.carrier.seq_len
        length = np.bincount(lengths).argmax()
        idx = np.flatnonzero(lengths == length)[:4]
        ids = torch.tensor([self.carrier.ids[i] for i in idx], dtype=torch.long, device=DEVICE)
        kw = dict(input_ids=ids, use_cache=False, logits_to_keep=1)
        base = self.model(**kw).logits
        t = max(1, self.L // 2)
        with routed(LiveSubstitution({})):
            same = self.model(**kw).logits
        with routed(CleanCapture([t])) as cap:
            self.model(**kw)
        with routed(CleanInjection({t: t}, cap.cache)):
            injected = self.model(**kw).logits
        with routed(LiveSubstitution({t: t - 1})):
            changed = self.model(**kw).logits
        if not torch.equal(base, same) or not torch.equal(base, injected):
            raise RuntimeError(f"{self.mc.name}: substitution engine is not an exact identity")
        if torch.equal(base, changed):
            raise RuntimeError(f"{self.mc.name}: KV substitution had no effect; the custom "
                               f"attention implementation is not in use")

    # ── Readout passes ────────────────────────────────────────────────
    @torch.no_grad()
    def states_pass(self, item_idx: np.ndarray, spec: Optional[PolicySpec], layer_ids: Sequence[int],
                    store: Optional[Dict[int, torch.Tensor]] = None,
                    frozen: Optional[FrozenReadout] = None) -> Tuple[Dict[int, np.ndarray], Optional[np.ndarray]]:
        """
        Carrier forward over `item_idx`.  Writes last-token states into
        store[layer][global item index] (if `store`) and returns, per layer, the global
        indices of items whose state is non-finite, and frozen-probe logits
        (len(layer_ids), n) aligned with item_idx (if `frozen`).  A non-finite state is
        written to the store as zeros (the store must stay finite) but its item is
        excluded from every probe fit and evaluation, and its frozen logit is NaN.
        """
        layer_ids = list(layer_ids)
        bad: Dict[int, List[int]] = {li: [] for li in layer_ids}
        logits = np.full((len(layer_ids), len(item_idx)), np.nan, np.float32) if frozen else None

        def step(pos, ids):
            with self.recording(layer_ids) as rec:
                self.forward(ids, spec)
            rows = torch.as_tensor(item_idx[pos])
            for j, li in enumerate(layer_ids):
                h = rec[li].float()
                bad_rows = ~torch.isfinite(h).all(-1)
                if bad_rows.any():
                    bad[li] += item_idx[pos][bad_rows.cpu().numpy()].tolist()
                    h[bad_rows] = 0.0
                if store is not None:
                    store[li][rows] = h.to(self.store_dtype).cpu()
                if frozen is not None:
                    z = frozen.logits(h, li)
                    z[bad_rows] = float("nan")
                    logits[j, pos] = z.cpu().numpy()

        self.run_batches(self.carrier, item_idx, step)
        return {li: np.asarray(sorted(v), dtype=np.int64) for li, v in bad.items()}, logits

    @torch.no_grad()
    def behavior_pass(self, item_idx: np.ndarray, spec: Optional[PolicySpec]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Behavioural lexical decision per item (task prompt): the score
        logit(Yes) - logit(No) and the probability mass on the two answer tokens.
        The mass shows whether the policy preserved the answer format at all; a
        score change without retained mass is loss of the task format, not of
        lexical knowledge.
        """
        out = np.full(len(item_idx), np.nan, np.float32)
        mass = np.full(len(item_idx), np.nan, np.float32)
        yes, no = self.answer_ids

        def step(pos, ids):
            lg = self.forward(ids, spec).logits[:, -1].float()
            lp = lg.log_softmax(-1)
            out[pos] = (lg[:, yes] - lg[:, no]).cpu().numpy()
            mass[pos] = (lp[:, yes].exp() + lp[:, no].exp()).cpu().numpy()

        self.run_batches(self.task, item_idx, step)
        return out, mass

    @torch.no_grad()
    def stimulus_logprob(self, item_idx: np.ndarray) -> np.ndarray:
        """M4: model-internal familiarity, log P(stimulus subwords | carrier prefix)."""
        out = np.full(len(item_idx), np.nan)

        def step(pos, ids):
            nt = self.n_tokens[item_idx[pos]]
            k = int(nt.max()) + 1
            lp = self.forward(ids, None, logits_to_keep=k).logits.float().log_softmax(-1)
            T = ids.shape[1]
            vals = []
            for b, n in enumerate(nt):
                # kept positions T-k..T-1; position p predicts token p+1
                pred = lp[b, k - 1 - n: k - 1]
                vals.append(float(pred.gather(1, ids[b, T - n:, None]).sum()))
            out[pos] = vals

        self.run_batches(self.carrier, item_idx, step)
        return out

    @torch.no_grad()
    def static_features(self, lengths: np.ndarray) -> torch.Tensor:
        """
        RQ8a non-contextual baseline per item: [input embedding of the final
        subword; mean input embedding of the stimulus subwords; n_tokens; length].
        """
        E = self.model.get_input_embeddings().weight
        out = torch.empty((len(self.carrier.ids), 2 * self.H + 2), dtype=torch.float32)
        for i, (ids, n) in enumerate(zip(self.carrier.ids, self.n_tokens)):
            stim = E[torch.tensor(ids[-n:], device=E.device)].float()
            out[i, :self.H] = stim[-1].cpu()
            out[i, self.H:2 * self.H] = stim.mean(0).cpu()
            out[i, -2], out[i, -1] = float(n), float(lengths[i])
        return out

    # ── Calibration of partner criteria ───────────────────────────────
    @torch.no_grad()
    def calibrate(self, item_idx: np.ndarray, pairs: Dict[int, int]) -> Tuple[pd.DataFrame, Dict, np.ndarray]:
        """
        Per (target, source) pair of the primary `full` map, on a clean pass over ALL
        positions (BOS / sink, template prefix, stimulus) of calibration (train) items,
        because the substitution replaces K/V at every position:
          cka                 mean of unbiased linear CKA of K and of V over distinct
                              states: the item-independent prefix rows once (they are
                              identical across items under causal attention) plus a
                              deterministic subsample of stimulus rows
          kvsharer_distance   ||mean[K_s;V_s] - mean[K_t;V_t]||, token-occurrence weighted
                              layer means over all positions (the KVSharer ranking quantity)
          rel_kv_distance     row-wise ||[K_s;V_s]-[K_t;V_t]|| / ||[K_t;V_t]|| on the CKA rows
          attn_out_rel_err    sqrt(sum err / sum own) over all positions: the query-aware
                              residual-stream error, which weights positions by attention
        plus item-level split-half reliability of every criterion's pair ranking and rank
        agreement between criteria.  Also returns the layer-averaged flattened [K; V] of
        every layer (L, 2 * kv_dim) for the KVSharer search.
        """
        meter = CriterionMeter(pairs)
        rng = np.random.default_rng(SEED)
        uniq = rng.permutation(np.unique(item_idx))
        half_of = {int(i): (0 if r < len(uniq) // 2 else 1) for r, i in enumerate(uniq)}
        prefix_owner = int(item_idx[0])           # the prefix rows are stored for this item only
        n_stim_rows = int(self.n_tokens[item_idx].sum())
        q = min(1.0, self.cfg.CALIB_MAX_ROWS / max(n_stim_rows, 1))
        row_item: List[int] = []                  # all positions, in meter row order
        keep_item: List[int] = []                 # stored K/V rows
        n_rows_half = np.zeros(2)

        def step(pos, ids):
            b_, T = ids.shape
            keep = torch.zeros((b_, T), dtype=torch.bool)
            half = torch.zeros((b_, T), dtype=torch.long)
            batch_rows, batch_keep, batch_half = [], [], np.zeros(2)
            for b, p in enumerate(pos):
                it = int(item_idx[p])
                n = int(self.n_tokens[it])
                # deterministic per item, independent of the batch plan
                u = np.random.default_rng([SEED, it]).random(n) < q
                keep[b, T - n:] = torch.as_tensor(u)
                if it == prefix_owner:
                    keep[b, :T - n] = True
                half[b] = half_of[it]
                batch_rows += [it] * T
                batch_keep += [it] * int(keep[b].sum())
                batch_half[half_of[it]] += T
            meter.begin_batch(torch.ones((b_, T), dtype=torch.bool, device=DEVICE),
                              keep.to(DEVICE), half.to(DEVICE))
            with routed(meter):
                self.model(input_ids=ids, use_cache=False, logits_to_keep=1)
            meter.commit()
            row_item.extend(batch_rows)
            keep_item.extend(batch_keep)
            n_rows_half[:] += batch_half

        self.run_batches(self.carrier, item_idx, step)
        cat = {name: {li: (torch.stack(v).sum(0) if name == "kv_sum" else torch.cat(v)) for li, v in d.items()}
               for name, d in meter.rows.items()}
        row_item, keep_item = np.asarray(row_item), np.asarray(keep_item)
        layer_means = torch.stack([cat["kv_sum"][li].sum(0) for li in range(self.L)]) / n_rows_half.sum()
        row_half = np.array([half_of[int(i)] for i in row_item])
        keep_half = np.array([half_of[int(i)] for i in keep_item])
        halves = (np.flatnonzero(row_half == 0), np.flatnonzero(row_half == 1))
        keep_halves = (np.flatnonzero(keep_half == 0), np.flatnonzero(keep_half == 1))
        every_kept = np.arange(len(keep_item))
        everything = np.arange(len(row_item))

        def kvdist(t, s, h=None):
            if h is None:
                mt, ms = cat["kv_sum"][t].sum(0) / n_rows_half.sum(), cat["kv_sum"][s].sum(0) / n_rows_half.sum()
            else:
                mt, ms = cat["kv_sum"][t][h] / n_rows_half[h], cat["kv_sum"][s][h] / n_rows_half[h]
            return float((ms - mt).norm())

        def cka(t, s, rows):
            return 0.5 * (unbiased_linear_cka(cat["k"][t][rows], cat["k"][s][rows])
                          + unbiased_linear_cka(cat["v"][t][rows], cat["v"][s][rows]))

        def fidelity(t, rows):
            return math.sqrt(float(cat["err"][t][rows].sum()) / float(cat["own"][t][rows].sum()))

        out = []
        for t, s in sorted(pairs.items()):
            Kt, Vt, Ks, Vs = (cat[kv][li] for li in (t, s) for kv in ("k", "v"))
            num = (Ks - Kt).pow(2).sum() + (Vs - Vt).pow(2).sum()
            den = Kt.pow(2).sum() + Vt.pow(2).sum()
            out.append({"target": t, "source": s,
                        "cka_k": unbiased_linear_cka(Kt, Ks), "cka_v": unbiased_linear_cka(Vt, Vs),
                        "cka": cka(t, s, every_kept),
                        "cka_half1": cka(t, s, keep_halves[0]), "cka_half2": cka(t, s, keep_halves[1]),
                        "kvsharer_distance": kvdist(t, s),
                        "kvsharer_distance_half1": kvdist(t, s, 0),
                        "kvsharer_distance_half2": kvdist(t, s, 1),
                        "rel_kv_distance": float(torch.sqrt(num / den)),
                        "attn_out_rel_err": fidelity(t, everything),
                        "attn_out_rel_err_half1": fidelity(t, halves[0]),
                        "attn_out_rel_err_half2": fidelity(t, halves[1]),
                        "rope_mismatch": self.rope[t] != self.rope[s]})
        df = pd.DataFrame(out)
        summary = {"n_rows_all_positions": int(len(row_item)), "n_rows_cka": int(len(keep_item)),
                   "stimulus_row_sampling_rate": q, "n_pairs": int(len(df))}
        if len(df) >= 3:
            for c in ("cka", "kvsharer_distance", "attn_out_rel_err"):
                summary[f"split_half_spearman_{c}"] = float(stats.spearmanr(df[f"{c}_half1"], df[f"{c}_half2"])[0])
            crit = {"cka": df.cka, "kvsharer_similarity": -df.kvsharer_distance,
                    "attn_fidelity": -df.attn_out_rel_err, "rel_kv_similarity": -df.rel_kv_distance}
            names = list(crit)
            for i, a in enumerate(names):
                for b in names[i + 1:]:
                    summary[f"kendall_{a}_vs_{b}"] = float(stats.kendalltau(crit[a], crit[b])[0])
        return df, summary, layer_means.numpy()

    @torch.no_grad()
    def final_states(self, item_idx: np.ndarray, source_map: Dict[int, int]) -> torch.Tensor:
        """
        Final-norm hidden states at ALL positions (rows, H) under a live substitution map,
        in a batch-plan order that depends only on item_idx (the KVSharer acceptance
        quantity compares whole-sequence output states).
        """
        spec = PolicySpec("trial", "trial", source_map, "live", True, self.L) if source_map else None
        norm = getattr(self.model.get_decoder(), "norm", None)
        norm = self.layers[-1] if norm is None else norm
        per_item: Dict[int, torch.Tensor] = {}

        def step(pos, ids):
            got = {}

            def hook(_m, _inp, out):
                got["h"] = (out[0] if isinstance(out, tuple) else out).detach()

            handle = norm.register_forward_hook(hook)
            try:
                self.forward(ids, spec)
            finally:
                handle.remove()
            h = got["h"].float().cpu()
            for b, p in enumerate(pos):
                per_item[int(p)] = h[b]

        self.run_batches(self.carrier, item_idx, step)
        return torch.cat([per_item[p] for p in range(len(item_idx))])

    @torch.no_grad()
    def kvsharer_search(self, item_idx: np.ndarray, layer_means: np.ndarray, ec: int, k: int,
                        dissimilar: bool, threshold: float) -> Tuple[Dict[int, int], pd.DataFrame]:
        """
        KVSharer-style search (Yang et al., 2024) over ALL layer pairs s < t with t
        eligible: pairs are ranked by the Euclidean distance between token-averaged
        flattened K/V (descending = KVSharer's dissimilar preference; ascending = the
        similar control); a pair is tried if it keeps the map chain-free (t not yet a
        target or source, s not a target) and accepted if the mean cosine similarity of
        final-norm states at all positions with vs without the tentative map, on
        calibration items, is >= threshold.  The later layer always reads the earlier
        layer's cache, the only direction realisable in a causal prefill with a shared
        cache.  The search stops at k targets (rate-matched to the gated policies); a
        shortfall is logged and reported.
        """
        base = self.final_states(item_idx, {})
        dist = {(t, s): float(np.linalg.norm(layer_means[t] - layer_means[s]))
                for t in range(max(ec, 1), self.L) for s in range(t)}
        order = sorted(dist, key=lambda p: ((-dist[p] if dissimilar else dist[p]), p))
        smap, log = {}, []
        for t, s in order:
            if len(smap) == k:
                break
            if t in smap or t in smap.values() or s in smap:
                continue
            trial = {**smap, t: s}
            cos = float(F.cosine_similarity(self.final_states(item_idx, trial), base, dim=-1).mean())
            accepted = cos >= threshold
            log.append({"target": t, "source": s, "distance": dist[(t, s)], "final_cosine": cos,
                        "accepted": accepted, "n_targets_after": len(trial) if accepted else len(smap)})
            if accepted:
                smap = trial
        if len(smap) < k:
            logger.warning(f"  KVSharer search ({'dissimilar' if dissimilar else 'similar'}) accepted only "
                           f"{len(smap)}/{k} targets at threshold {threshold}")
        return smap, pd.DataFrame(log)

    # ── Mechanistic diagnostics ───────────────────────────────────────
    @torch.no_grad()
    def mechanistic(self, item_idx: np.ndarray, spec: PolicySpec) -> pd.DataFrame:
        """Clean vs policy states (all layers) and next-token distributions."""
        n = len(item_idx)
        clean = {li: torch.empty((n, self.H), dtype=self.store_dtype) for li in range(self.L)}
        pol = {li: torch.empty((n, self.H), dtype=self.store_dtype) for li in range(self.L)}
        kl = torch.empty(n)
        top1 = torch.empty(n)

        def step(pos, ids):
            tpos = torch.as_tensor(pos)
            with self.recording(range(self.L)) as rc:
                lc = self.forward(ids, None).logits[:, -1].float().log_softmax(-1)
            with self.recording(range(self.L)) as rp:
                lp = self.forward(ids, spec).logits[:, -1].float().log_softmax(-1)
            for li in range(self.L):
                clean[li][tpos] = rc[li].to(self.store_dtype).cpu()
                pol[li][tpos] = rp[li].to(self.store_dtype).cpu()
            kl[tpos] = (lc.exp() * (lc - lp)).sum(-1).cpu()
            top1[tpos] = (lc.argmax(-1) == lp.argmax(-1)).float().cpu()

        self.run_batches(self.carrier, item_idx, step)
        rows = []
        for li in range(self.L):
            c, p = clean[li].float(), pol[li].float()
            cos = F.cosine_similarity(c, p, dim=-1)
            rel = (p - c).norm(dim=-1) / c.norm(dim=-1).clamp_min(1e-8)
            rows.append({"layer": li, "hidden_cka_unbiased": unbiased_linear_cka(c, p),
                         "cosine_mean": float(cos.mean()), "relative_shift_mean": float(rel.mean()),
                         "next_token_kl_mean": float(kl.mean()), "next_token_top1_agreement": float(top1.mean())})
        return pd.DataFrame(rows)

    def release(self):
        del self.model
        free_memory()


# ════════════════════════════════════════════════════════════════════════════
# PROBES
# ════════════════════════════════════════════════════════════════════════════

def standardizer(X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    return X.mean(0), X.std(0).clamp_min(1e-6)


def fit_logistic(X: torch.Tensor, y: torch.Tensor, lam: float, max_iter: int):
    """L2-regularised logistic regression; strictly convex, zero init, full-batch L-BFGS."""
    with torch.enable_grad():
        w = torch.zeros(X.shape[1], device=X.device, requires_grad=True)
        b = torch.zeros(1, device=X.device, requires_grad=True)
        opt = torch.optim.LBFGS([w, b], lr=1.0, max_iter=max_iter, history_size=20,
                                line_search_fn="strong_wolfe",
                                tolerance_grad=1e-7, tolerance_change=1e-12)

        def closure():
            opt.zero_grad()
            loss = F.binary_cross_entropy_with_logits(X @ w + b, y) + 0.5 * lam * w.dot(w)
            loss.backward()
            return loss

        opt.step(closure)
    return w.detach(), b.detach()


@dataclass
class ProbeBank:
    """no_reuse linear probes: frozen readout for every other policy."""
    mu: np.ndarray      # (S, L, H)
    sd: np.ndarray      # (S, L, H)
    w: np.ndarray       # (S, L, H)
    b: np.ndarray       # (S, L)
    lam: np.ndarray     # (L,)  NaN = layer without a valid control probe

    @classmethod
    def empty(cls, S: int, L: int, H: int) -> "ProbeBank":
        return cls(np.zeros((S, L, H), np.float32), np.ones((S, L, H), np.float32),
                   np.zeros((S, L, H), np.float32), np.zeros((S, L), np.float32), np.full(L, np.nan))

    def save(self, path: str):
        np.savez_compressed(path, mu=self.mu, sd=self.sd, w=self.w, b=self.b, lam=self.lam)

    @classmethod
    def load(cls, path: str) -> "ProbeBank":
        z = np.load(path)
        return cls(z["mu"], z["sd"], z["w"], z["b"], z["lam"])


class LinearProbeRunner:
    """Primary readout: multi-seed linear probes on a fixed test split."""

    def __init__(self, cfg: ExperimentConfig, splits: Splits, y: np.ndarray):
        self.cfg, self.splits = cfg, splits
        self.y = torch.tensor(np.asarray(y, dtype=np.float32), device=DEVICE)
        self.y_test = np.asarray(y)[splits.test]

    @staticmethod
    def _good(idx: np.ndarray, good: Optional[np.ndarray]) -> np.ndarray:
        return idx if good is None else idx[good[idx]]

    def select_lambda(self, X: torch.Tensor, good: Optional[np.ndarray] = None) -> float:
        """Validation-loss lambda selection on the first seed's split of THESE states
        (items with non-finite states, good == False, are excluded)."""
        s0 = self.cfg.PROBE_SEEDS[0]
        tr, va = self._good(self.splits.train[s0], good), self._good(self.splits.val[s0], good)
        mu, sd = standardizer(X[tr])
        Ztr, Zva = (X[tr] - mu) / sd, (X[va] - mu) / sd
        losses = []
        for lam in self.cfg.PROBE_L2_GRID:
            w, b = fit_logistic(Ztr, self.y[tr], lam, self.cfg.PROBE_MAX_ITER)
            losses.append(float(F.binary_cross_entropy_with_logits(Zva @ w + b, self.y[va])))
        return float(self.cfg.PROBE_L2_GRID[int(np.argmin(losses))])

    def fit_layer(self, X: torch.Tensor, lam: float, good: Optional[np.ndarray] = None):
        """
        Returns test logits per seed (S, n_test; NaN for excluded items), the fitted
        parameters and each seed's VALIDATION AUC (S,).  Validation AUC is what any
        later choice of a readout layer is based on, so that the test items that
        evaluate a layer never also select it.
        """
        te = self.splits.test
        te_ok = np.ones(len(te), bool) if good is None else good[te]
        logits, params, auc_val = [], [], []
        for s in self.cfg.PROBE_SEEDS:
            tr, va = self._good(self.splits.train[s], good), self._good(self.splits.val[s], good)
            mu, sd = standardizer(X[tr])
            w, b = fit_logistic((X[tr] - mu) / sd, self.y[tr], lam, self.cfg.PROBE_MAX_ITER)
            z = (((X[te] - mu) / sd) @ w + b).cpu().numpy()
            z[~te_ok] = np.nan
            logits.append(z)
            zv = (((X[va] - mu) / sd) @ w + b).cpu().numpy()
            auc_val.append(roc_auc_score(self.y[va].cpu().numpy(), zv))
            params.append((mu.cpu().numpy(), sd.cpu().numpy(), w.cpu().numpy(), float(b)))
        return np.stack(logits), params, np.asarray(auc_val)

    def test_auc(self, z: np.ndarray) -> float:
        ok = np.isfinite(z)
        return float(roc_auc_score(self.y_test[ok], z[ok]))


class MLPProbe(nn.Module):
    def __init__(self, d_in: int, hidden: Tuple[int, ...], dropout: float):
        super().__init__()
        layers, d = [], d_in
        for h in hidden:
            lin = nn.Linear(d, h)
            nn.init.kaiming_normal_(lin.weight, nonlinearity="relu")
            nn.init.zeros_(lin.bias)
            layers += [lin, nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def fit_mlp(Ztr, ytr, Zva, yva, hidden, cfg: ExperimentConfig, seed: int) -> MLPProbe:
    """AdamW with early stopping on validation loss; returns the best-validation state."""
    seed_everything(seed)
    model = MLPProbe(Ztr.shape[1], hidden, cfg.MLP_DROPOUT).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.MLP_LR, weight_decay=cfg.MLP_WEIGHT_DECAY)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    best, patience = float("inf"), 0
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    with torch.enable_grad():
        for _ in range(cfg.MLP_EPOCHS):
            model.train()
            for idx in torch.randperm(Ztr.shape[0], generator=gen).split(cfg.MLP_BATCH):
                if len(idx) < 2:
                    continue
                idx = idx.to(DEVICE)
                opt.zero_grad()
                F.binary_cross_entropy_with_logits(model(Ztr[idx]), ytr[idx]).backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                vl = float(F.binary_cross_entropy_with_logits(model(Zva), yva))
            if vl < best - 1e-6:
                best, patience = vl, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                patience += 1
                if patience >= cfg.MLP_PATIENCE:
                    break
    model.load_state_dict(best_state)
    return model.eval()


def online_codelength(X: torch.Tensor, y: torch.Tensor, lam: float,
                      fractions: Sequence[float], max_iter: int, seed: int) -> Tuple[float, float]:
    """
    Online (prequential) code length in bits of the labels given the
    representations (Voita & Titov, 2020).  Returns (online_bits, compression),
    compression = uniform_bits / online_bits with uniform_bits = n * log2(2).
    """
    n = X.shape[0]
    order = torch.as_tensor(np.random.default_rng(seed).permutation(n), device=X.device)
    sizes = sorted({max(2, int(round(f * n))) for f in fractions} | {n})
    bits = float(sizes[0])
    for prev, cur in zip(sizes[:-1], sizes[1:]):
        tr, nxt = order[:prev], order[prev:cur]
        mu, sd = standardizer(X[tr])
        w, b = fit_logistic((X[tr] - mu) / sd, y[tr], lam, max_iter)
        z = ((X[nxt] - mu) / sd) @ w + b
        bits += float(F.binary_cross_entropy_with_logits(z, y[nxt], reduction="sum")) / math.log(2)
    return bits, n / bits


# ════════════════════════════════════════════════════════════════════════════
# PER-MODEL PIPELINE
# ════════════════════════════════════════════════════════════════════════════

class MemmapLayer:
    """One layer's states for all items on disk; 16-bit floats are stored bit-exactly as int16."""

    def __init__(self, path: str, n: int, H: int, dtype: torch.dtype):
        self.dtype = dtype
        self.half = torch.empty((), dtype=dtype).element_size() == 2
        self.arr = np.lib.format.open_memmap(path, mode="w+", dtype=np.int16 if self.half else np.float32,
                                             shape=(n, H))

    def __setitem__(self, rows, h: torch.Tensor):
        h = h.contiguous()
        self.arr[np.asarray(rows)] = (h.view(torch.int16) if self.half else h.float()).numpy()

    def to(self, device, dtype) -> torch.Tensor:
        t = torch.from_numpy(np.array(self.arr))
        return (t.view(self.dtype) if self.half else t).to(device, dtype)


class ModelPipeline:
    """
    Order per model:
      0. item metadata (model log-probability of every string)
      1. no_reuse control: retrained probes (= probe bank), frozen pass, behaviour,
         non-contextual baseline, reliability
      2. per regime: criterion calibration (primary) -> policy specs -> one run each
      3. single-layer fragility scan (frozen + behavioural readouts)
      4. generative evaluation of the downstream policy set
    Every run is saved to disk and skipped on resume.
    """

    def __init__(self, cfg: ExperimentConfig, mc: ModelConfig, items: pd.DataFrame, splits: Splits):
        self.cfg, self.mc, self.items, self.splits = cfg, mc, items, splits
        self.dir = cfg.model_dir(mc.name)
        self.runner = ModelRunner(mc, cfg, items)
        self.L, self.H = self.runner.L, self.runner.H
        self.y = items["is_word"].to_numpy()
        self.probes = LinearProbeRunner(cfg, splits, self.y)
        self.test = splits.test
        self.nontest = np.setdiff1d(np.arange(len(items)), splits.test)
        self.reliability_layers = sorted({self.L // 4, self.L // 2, (3 * self.L) // 4, self.L - 1})
        self._write_item_metadata()
        with open(os.path.join(self.dir, "model_info.json"), "w") as f:
            json.dump({"model": mc.name, "model_id": mc.model_id, "family": mc.family,
                       "num_layers": self.L, "hidden_size": self.H, "adds_bos": bool(self.runner.bos),
                       "rope_layers": self.runner.rope, "mixed_rope": len(set(self.runner.rope)) > 1,
                       "template": cfg.STIMULUS_TEMPLATE, "task_template": cfg.task_template(),
                       "answer_token_ids": self.runner.answer_ids, "compute_dtype": str(COMPUTE_DTYPE),
                       "batch_size": self.runner.batch_size, "test_items": self.test.tolist(),
                       "regimes": {r.name: {"reuse_factor": r.reuse_factor,
                                            "exempt_fraction": r.exempt_fraction, "role": r.role,
                                            "exempt_cutoff": exempt_cutoff(self.L, r.exempt_fraction)}
                                   for r in cfg.REGIMES}}, f, indent=2)

    def _write_item_metadata(self):
        path = os.path.join(self.dir, "items.csv")
        if self.cfg.RESUME and os.path.exists(path):
            return
        meta = self.items.copy()
        meta["n_tokens"] = self.runner.n_tokens
        meta["n_tokens_task"] = self.runner.task.n_tokens
        meta["lm_logprob"] = self.runner.stimulus_logprob(np.arange(len(self.items)))
        meta.to_csv(path, index=False)

    # ── paths ─────────────────────────────────────────────────────────
    def run_path(self, regime: Optional[str], name: str, ext: str) -> str:
        sub = self.dir if regime is None else os.path.join(self.dir, regime)
        os.makedirs(sub, exist_ok=True)
        return os.path.join(sub, f"{safe_name(name)}.{ext}")

    # ── orchestration ─────────────────────────────────────────────────
    def run(self):
        bank = self._no_reuse()
        frozen = FrozenReadout(bank)
        downstream = [(None, no_reuse_spec(self.L))]
        for regime in self.cfg.REGIMES:
            for spec in self._policy_specs(regime):
                self._run_policy(regime, spec, frozen)
                if spec.downstream:
                    downstream.append((regime.name, spec))
        if self.cfg.FRAGILITY_ENABLED:
            self._fragility_scan(frozen)
        wanted = self.cfg.DOWNSTREAM_MODELS
        if self.cfg.DOWNSTREAM_ENABLED and (not wanted or self.mc.name in wanted):
            DownstreamEvaluator(self.cfg, self.runner).run(downstream, self.run_path(None, "downstream", "csv"))
        self.runner.release()

    def _alloc(self, run: str) -> Dict[int, object]:
        """All-item states of every layer, in RAM or in memory-mapped files (STATE_STORE)."""
        n, dt = len(self.items), self.runner.store_dtype
        if self.cfg.STATE_STORE == "memory":
            return {li: torch.empty((n, self.H), dtype=dt) for li in range(self.L)}
        d = self._state_dir(run)
        os.makedirs(d, exist_ok=True)
        return {li: MemmapLayer(os.path.join(d, f"layer{li}.npy"), n, self.H, dt) for li in range(self.L)}

    def _state_dir(self, run: str) -> str:
        root = self.cfg.STATE_DIR or os.path.join(self.cfg.OUTPUT_DIR, "state_cache")
        return os.path.join(root, safe_name(self.mc.name), safe_name(run))

    def _release_states(self, run: str):
        if self.cfg.STATE_STORE == "disk":
            shutil.rmtree(self._state_dir(run), ignore_errors=True)

    def _nonfinite_mask(self, bad: Dict[int, np.ndarray], n: int) -> np.ndarray:
        return np.array([len(bad[li]) / n > self.cfg.NONFINITE_MAX_FRACTION for li in range(self.L)])

    @staticmethod
    def _merge_bad(*parts: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        return {li: np.unique(np.concatenate([p[li] for p in parts])).astype(np.int64) for li in parts[0]}

    # ── step 1: control ───────────────────────────────────────────────
    def _no_reuse(self) -> ProbeBank:
        npz, bank_path = self.run_path(None, NO_REUSE, "npz"), self.run_path(None, "probe_bank", "npz")
        if self.cfg.RESUME and os.path.exists(npz) and os.path.exists(bank_path):
            if "auc_val_seed" not in np.load(npz).files:
                raise RuntimeError(f"{npz} predates validation-based layer selection; rerun it "
                                   f"(delete it or use a fresh KV_OUTPUT_DIR)")
            logger.info(f"  RESUME {self.mc.name}/{NO_REUSE}")
            return ProbeBank.load(bank_path)
        spec = no_reuse_spec(self.L)
        t0 = time.time()
        store = self._alloc(NO_REUSE)
        bad_test, _ = self.runner.states_pass(self.test, spec, range(self.L), store)
        bad_rest, _ = self.runner.states_pass(self.nontest, spec, range(self.L), store)
        bad = self._merge_bad(bad_test, bad_rest)
        bank = ProbeBank.empty(len(self.cfg.PROBE_SEEDS), self.L, self.H)
        rt, auc_seed, auc_val = self._retrained(store, bad, spec, None, bank)
        self._release_states(NO_REUSE)
        frozen = FrozenReadout(bank)
        _, fz = self.runner.states_pass(self.test, spec, range(self.L), frozen=frozen)
        beh, mass = self.runner.behavior_pass(self.test, spec)
        self._static_baseline()
        bank.save(bank_path)
        self._save_run(None, spec, time.time() - t0, bad, frozen=fz, retrained=rt,
                       auc_seed=auc_seed, auc_val_seed=auc_val, behavior=beh, behavior_mass=mass)
        return bank

    def _retrained(self, store, bad: Dict[int, np.ndarray], spec: PolicySpec, regime: Optional[Regime],
                   bank: Optional[ProbeBank] = None):
        """
        Retrained readout: lambda selected on THIS policy's validation split, then
        multi-seed fits; items with non-finite states are excluded from fitting and
        evaluation.  With `bank` (no_reuse only) the fits become the probe bank.
        Returns test logits (L, n_test), test AUC (S, L) and validation AUC (S, L).
        """
        n_test, S = len(self.test), len(self.cfg.PROBE_SEEDS)
        rt = np.full((self.L, n_test), np.nan, np.float32)
        auc_seed = np.full((S, self.L), np.nan)
        auc_val = np.full((S, self.L), np.nan)
        excluded = self._nonfinite_mask(bad, len(self.items))
        reliability = (self.cfg.RELIABILITY_ENABLED and spec.family in self.cfg.RELIABILITY_POLICIES
                       and (regime is None or regime.role == "primary"))
        keep = {}
        for li in tqdm(range(self.L), desc=f"probe {self.mc.name}/{spec.name}", leave=False):
            if excluded[li]:
                logger.warning(f"  layer {li}: non-finite rows above threshold -> layer excluded")
                store[li] = None
                continue
            good = np.ones(len(self.items), bool)
            good[bad[li]] = False
            X = store[li].to(DEVICE, torch.float32)
            lam = self.probes.select_lambda(X, good)
            seed_logits, params, val = self.probes.fit_layer(X, lam, good)
            rt[li] = seed_logits.mean(0)
            auc_seed[:, li] = [self.probes.test_auc(z) for z in seed_logits]
            auc_val[:, li] = val
            if bank is not None:
                bank.lam[li] = lam
                for s, (mu, sd, w, b) in enumerate(params):
                    bank.mu[s, li], bank.sd[s, li], bank.w[s, li], bank.b[s, li] = mu, sd, w, b
            if reliability and li in self.reliability_layers:
                keep[li] = (store[li], lam, good)
            store[li] = None
            del X
            free_memory()
        if keep:
            self._reliability(keep, spec, regime)
        return rt, auc_seed, auc_val

    def _static_baseline(self):
        """RQ8a: linear probe on non-contextual features (+ its online code length)."""
        X = self.runner.static_features(self.items["length"].to_numpy()).to(DEVICE)
        lam = self.probes.select_lambda(X)
        seed_logits, _, val = self.probes.fit_layer(X, lam)
        tr = torch.as_tensor(self.splits.train[self.cfg.PROBE_SEEDS[0]], device=DEVICE)
        bits, comp = online_codelength(X[tr], self.probes.y[tr], lam, self.cfg.MDL_FRACTIONS,
                                       self.cfg.PROBE_MAX_ITER, self.cfg.PROBE_SEEDS[0])
        np.savez_compressed(self.run_path(None, "static_baseline", "npz"), test_items=self.test,
                            logits=seed_logits.mean(0), auc_seed=[self.probes.test_auc(z) for z in seed_logits],
                            auc_val_seed=val, lam=lam, mdl_bits=bits, mdl_compression=comp)
        del X
        free_memory()

    # ── step 2: policies ──────────────────────────────────────────────
    def _policy_specs(self, regime: Regime) -> List[PolicySpec]:
        ec = exempt_cutoff(self.L, regime.exempt_fraction)
        full_map = {t: s for t, s in group_heads(self.L, regime.reuse_factor, ec).items() if t != s}
        crit = pd.DataFrame(columns=["target", "source", "cka", "kvsharer_distance", "attn_out_rel_err"])
        searched = {}
        if regime.role == "primary" and full_map:
            s0 = self.cfg.PROBE_SEEDS[0]
            calib_idx = np.sort(np.random.default_rng(SEED).choice(
                self.splits.train[s0], size=min(self.cfg.CALIB_N_ITEMS, len(self.splits.train[s0])),
                replace=False))
            crit_path = self.run_path(regime.name, "calibration", "csv")
            means_path = self.run_path(regime.name, "layer_kv_means", "npy")
            if self.cfg.RESUME and os.path.exists(crit_path) and os.path.exists(means_path):
                crit, means = pd.read_csv(crit_path), np.load(means_path)
            else:
                crit, summary, means = self.runner.calibrate(calib_idx, full_map)
                crit.to_csv(crit_path, index=False)
                np.save(means_path, means)
                with open(self.run_path(regime.name, "calibration_summary", "json"), "w") as f:
                    json.dump(summary, f, indent=2)
                logger.info(f"  [calib {regime.name}] {summary}")
            searched = self._kvsharer_maps(regime, ec, len(full_map), calib_idx, means)
        specs = build_policy_specs(self.L, regime, crit, self.cfg, searched)
        self._check_distinct(regime, specs)
        return specs

    def _kvsharer_maps(self, regime: Regime, ec: int, n_targets: int, calib_idx: np.ndarray,
                       means: np.ndarray) -> Dict[str, Dict[int, int]]:
        """KVSharer-style search maps (dissimilar = KVSharer, similar = control), rate-matched to k0."""
        if n_targets < 3:
            return {}
        if not self.cfg.KVSHARER_THRESHOLD_VERIFIED:
            logger.warning(f"  KVSharer acceptance threshold {self.cfg.KVSHARER_COS_THRESHOLD} is NOT verified "
                           f"against the released KVSharer configuration: results are 'KVSharer-style'")
        path = self.run_path(regime.name, "kvsharer_search", "json")
        if self.cfg.RESUME and os.path.exists(path):
            with open(path) as f:
                return {fam: {int(t): int(v) for t, v in m.items()} for fam, m in json.load(f).items()}
        _, k0 = gate_sizes(n_targets, self.cfg)
        check_idx = calib_idx[: self.cfg.KVSHARER_CHECK_ITEMS]
        maps, logs = {}, []
        for fam in KVSHARER_FAMILIES:
            smap, log = self.runner.kvsharer_search(check_idx, means, ec, k0, fam == "kvsharer_dissimilar",
                                                    self.cfg.KVSHARER_COS_THRESHOLD)
            maps[fam] = smap
            logs.append(log.assign(family=fam, requested_k=k0))
        pd.concat(logs, ignore_index=True).to_csv(self.run_path(regime.name, "kvsharer_search_log", "csv"),
                                                  index=False)
        with open(path, "w") as f:
            json.dump({fam: {str(t): v for t, v in m.items()} for fam, m in maps.items()}, f, indent=2)
        with open(self.run_path(regime.name, "kvsharer_search_meta", "json"), "w") as f:
            json.dump({"threshold": self.cfg.KVSHARER_COS_THRESHOLD,
                       "threshold_verified": self.cfg.KVSHARER_THRESHOLD_VERIFIED,
                       "acceptance": "mean cosine of final-norm states over all positions",
                       "direction": "later layer reads the earlier layer's cache", "requested_k": k0,
                       "accepted_k": {fam: len(m) for fam, m in maps.items()}}, f, indent=2)
        return maps

    @staticmethod
    def _check_distinct(regime: Regime, specs: List[PolicySpec]):
        """
        Two deterministic policies with identical source maps are one experiment,
        not two.  A null draw that coincides with another set is a legitimate draw
        from its null distribution and is only logged.
        """
        seen = {}
        for s in specs:
            key = (s.semantics, tuple(sorted(s.source_map.items())))
            if key in seen:
                other = seen[key]
                if s.draw or other.draw:
                    logger.debug(f"  [{regime.name}] {s.name} has the same source map as {other.name}")
                else:
                    logger.warning(f"  [{regime.name}] {s.name} has the same source map as {other.name}; "
                                   f"they must not be reported as distinct conditions")
            seen.setdefault(key, s)

    def _run_policy(self, regime: Regime, spec: PolicySpec, frozen: FrozenReadout):
        npz = self.run_path(regime.name, spec.name, "npz")
        if self.cfg.RESUME and os.path.exists(npz):
            return
        logger.info(f"  RUN {self.mc.name}/{regime.name}/{spec.name}: k={spec.k} tier={spec.tier} "
                    f"semantics={spec.semantics} kv_mem={spec.kv_memory_fraction:.3f}")
        t0 = time.time()
        arrays = {}
        if spec.tier == "full":
            run = f"{regime.name}_{spec.name}"
            store = self._alloc(run)
            bad_test, fz = self.runner.states_pass(self.test, spec, range(self.L), store, frozen)
            bad_rest, _ = self.runner.states_pass(self.nontest, spec, range(self.L), store)
            bad = self._merge_bad(bad_test, bad_rest)
            arrays["retrained"], arrays["auc_seed"], arrays["auc_val_seed"] = self._retrained(
                store, bad, spec, regime)
            self._release_states(run)
        else:
            bad_test, fz = self.runner.states_pass(self.test, spec, range(self.L), frozen=frozen)
            bad = bad_test
        fz[self._nonfinite_mask(bad_test, len(self.test))] = np.nan
        arrays["frozen"] = fz
        if spec.behavioral:
            arrays["behavior"], arrays["behavior_mass"] = self.runner.behavior_pass(self.test, spec)
        self._save_run(regime.name, spec, time.time() - t0, bad, **arrays)
        if spec.tier == "full" and spec.draw == 0 and self.cfg.MECH_N_ITEMS > 0:
            self.runner.mechanistic(self.test[: self.cfg.MECH_N_ITEMS], spec).assign(
                model=self.mc.name, regime=regime.name, policy=spec.name, family=spec.family
            ).to_csv(self.run_path(regime.name, f"{spec.name}_mechanistic", "csv"), index=False)

    def _save_run(self, regime: Optional[str], spec: PolicySpec, seconds: float,
                  bad: Dict[int, np.ndarray], **arrays):
        np.savez_compressed(self.run_path(regime, spec.name, "npz"), test_items=self.test,
                            nonfinite=np.array([len(bad[i]) for i in range(self.L)]), **arrays)
        meta = spec.to_json() | {"model": self.mc.name, "seconds": seconds,
                                 "rope_mismatch": self.runner.rope_mismatch(spec.source_map),
                                 "oom_splits": self.runner.oom_splits}
        with open(self.run_path(regime, spec.name, "json"), "w") as f:
            json.dump(meta, f, indent=2)

    # ── probe reliability ─────────────────────────────────────────────
    def _reliability(self, layers: Dict[int, Tuple[torch.Tensor, float, np.ndarray]], spec: PolicySpec,
                     regime: Optional[Regime]):
        """
        RQ8b capacity check (linear vs MLP test AUC on the real task), memorisation
        capacity (training accuracy on per-item permuted labels under the identical
        training procedure) and the online code length of the real labels.  Items with
        non-finite states at the layer are excluded throughout.
        """
        cfg, sp = self.cfg, self.splits
        y = self.probes.y
        rows = []
        for li, (Xcpu, lam, good) in layers.items():
            X = Xcpu.to(DEVICE, torch.float32)
            te_np = sp.test[good[sp.test]]
            te = torch.as_tensor(te_np, device=DEVICE)
            yt = y[te].cpu().numpy().astype(int)
            for s in cfg.PROBE_SEEDS:
                tr = torch.as_tensor(sp.train[s][good[sp.train[s]]], device=DEVICE)
                va = torch.as_tensor(sp.val[s][good[sp.val[s]]], device=DEVICE)
                g = torch.Generator(device="cpu").manual_seed(s)
                y_ctrl = y.clone()
                y_ctrl[tr] = y[tr][torch.randperm(len(tr), generator=g).to(DEVICE)]
                y_ctrl[va] = y[va][torch.randperm(len(va), generator=g).to(DEVICE)]
                mu, sd = standardizer(X[tr])
                Z = (X - mu) / sd
                probes = {"linear": None} | dict(cfg.MLP_ARCHITECTURES)
                for probe, hidden in probes.items():
                    real, ctrl = [], []
                    for yy, sink in ((y, real), (y_ctrl, ctrl)):
                        if hidden is None:
                            w, b = fit_logistic(Z[tr], yy[tr], lam, cfg.PROBE_MAX_ITER)
                            sink += [Z[te] @ w + b, Z[tr] @ w + b]
                        else:
                            m = fit_mlp(Z[tr], yy[tr], Z[va], yy[va], hidden, cfg, s)
                            with torch.no_grad():
                                sink += [m(Z[te]), m(Z[tr])]
                            del m
                    z_te = real[0].detach().cpu().numpy()
                    rows.append({"layer": li, "probe": probe, "seed": s, "n_test": len(yt),
                                 "test_auc": float(roc_auc_score(yt, z_te)),
                                 "test_accuracy": float(((z_te > 0).astype(int) == yt).mean()),
                                 "control_train_accuracy": float(
                                     ((ctrl[1] > 0).float() == y_ctrl[tr]).float().mean())})
                if s == cfg.PROBE_SEEDS[0]:
                    bits, comp = online_codelength(X[tr], y[tr], lam, cfg.MDL_FRACTIONS, cfg.PROBE_MAX_ITER, s)
                    rows.append({"layer": li, "probe": "linear_mdl", "seed": s,
                                 "mdl_bits": bits, "mdl_compression": comp})
            del X
            free_memory()
        pd.DataFrame(rows).assign(model=self.mc.name, policy=spec.name,
                                  regime=None if regime is None else regime.name).to_csv(
            self.run_path(None if regime is None else regime.name, f"{spec.name}_reliability", "csv"),
            index=False)

    # ── step 3: single-layer fragility scan ───────────────────────────
    def _fragility_scan(self, frozen: FrozenReadout):
        path = self.run_path(None, "fragility", "npz")
        if self.cfg.RESUME and os.path.exists(path):
            logger.info(f"  RESUME {self.mc.name}/fragility")
            return
        base = np.load(self.run_path(None, NO_REUSE, "npz"))
        # readout layer chosen on VALIDATION AUC: the test items evaluate it, never select it
        peak = int(np.nanargmax(np.nanmean(base["auc_val_seed"], 0)))
        readouts = sorted({peak, self.L - 1})
        out = np.full((self.L, len(readouts), len(self.test)), np.nan, np.float32)
        beh = np.full((self.L, len(self.test)), np.nan, np.float32)
        mass = np.full((self.L, len(self.test)), np.nan, np.float32)
        mismatch = np.zeros(self.L, bool)
        for t in tqdm(range(1, self.L), desc=f"fragility {self.mc.name}"):
            spec = single_layer_spec(self.L, t)
            _, out[t] = self.runner.states_pass(self.test, spec, readouts, frozen=frozen)
            beh[t], mass[t] = self.runner.behavior_pass(self.test, spec)
            mismatch[t] = self.runner.rope_mismatch(spec.source_map) > 0
        np.savez_compressed(path, test_items=self.test, readouts=np.array(readouts),
                            base_logits=base["frozen"][readouts], logits=out,
                            base_behavior=base["behavior"], behavior=beh, base_behavior_mass=base["behavior_mass"],
                            behavior_mass=mass, rope_mismatch=mismatch)


# ════════════════════════════════════════════════════════════════════════════
# GENERATIVE EVALUATION
# ════════════════════════════════════════════════════════════════════════════

class DownstreamEvaluator:
    """WikiText-2 perplexity and LAMBADA (exact-match + target NLL) with the policy active."""

    def __init__(self, cfg: ExperimentConfig, runner: ModelRunner):
        self.cfg, self.runner, self.tok = cfg, runner, runner.tokenizer

    def _data(self):
        wiki = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
        ids = self.tok("\n\n".join(wiki["text"]), add_special_tokens=False).input_ids
        n_keep = self.cfg.PPL_WINDOW + self.cfg.PPL_STRIDE * (self.cfg.PPL_MAX_WINDOWS - 1)
        lam = load_dataset("EleutherAI/lambada_openai", "default", split="test")
        return ids[:n_keep], list(lam["text"][: self.cfg.LAMBADA_N])

    @torch.no_grad()
    def _perplexity(self, ids: List[int], spec: PolicySpec) -> float:
        """Sliding window; each token is scored once, with up to W-S tokens of context."""
        bos = self.runner.bos
        nll, ntok, prev_end = 0.0, 0, 0
        for begin in range(0, len(ids), self.cfg.PPL_STRIDE):
            end = min(begin + self.cfg.PPL_WINDOW, len(ids))
            x = torch.tensor([bos + ids[begin:end]], device=DEVICE)
            start = max(1, x.shape[1] - (end - prev_end))
            logits = self.runner.forward(x, spec, logits_to_keep=0).logits[0].float()
            nll += float(F.cross_entropy(logits[start - 1:-1], x[0, start:], reduction="sum"))
            ntok += x.shape[1] - start
            prev_end = end
            if end == len(ids):
                break
        return math.exp(nll / ntok)

    @torch.no_grad()
    def _lambada(self, texts: List[str], spec: PolicySpec) -> Tuple[float, float]:
        correct, nll, n = 0, 0.0, 0
        for text in texts:
            ctx, last = text.rsplit(" ", 1)
            c = self.runner.bos + self.tok(ctx, add_special_tokens=False).input_ids
            t = self.tok(" " + last, add_special_tokens=False).input_ids
            x = torch.tensor([c + t], device=DEVICE)
            lp = self.runner.forward(x, spec, logits_to_keep=len(t) + 1).logits[0, :-1].float().log_softmax(-1)
            tgt = torch.tensor(t, device=DEVICE)
            correct += int(bool((lp.argmax(-1) == tgt).all()))
            nll += float(-lp.gather(1, tgt[:, None]).sum())
            n += 1
        return correct / n, nll / n

    def run(self, specs: List[Tuple[Optional[str], PolicySpec]], path: str):
        if self.cfg.RESUME and os.path.exists(path):
            logger.info(f"  RESUME downstream {self.runner.mc.name}")
            return
        if not HF_DATASETS_AVAILABLE:
            logger.warning("  [generative] `datasets` not installed: downstream skipped")
            return
        try:
            ids, texts = self._data()
        except Exception as e:
            logger.warning(f"  [generative] dataset loading failed ({e}): downstream skipped")
            return
        rows = []
        for regime, spec in tqdm(specs, desc=f"downstream {self.runner.mc.name}"):
            acc, lnll = self._lambada(texts, spec)
            rows.append({"model": self.runner.mc.name, "regime": regime, "policy": spec.name,
                         "family": spec.family, "k": spec.k, "draw": spec.draw,
                         "semantics": spec.semantics, "kv_memory_fraction": spec.kv_memory_fraction,
                         "wikitext2_ppl": self._perplexity(ids, spec),
                         "lambada_acc": acc, "lambada_target_nll": lnll})
        pd.DataFrame(rows).to_csv(path, index=False)


# ════════════════════════════════════════════════════════════════════════════
# STATISTICS: primitives
# ════════════════════════════════════════════════════════════════════════════

def weighted_auc_parts(scores: np.ndarray, pos: np.ndarray, W: np.ndarray,
                       chunk: int = 256) -> Tuple[np.ndarray, np.ndarray]:
    """
    Mann-Whitney AUC under integer resampling weights, exact with ties (=1/2).
    scores (n,), pos (n,) bool, W (B, n).  Returns (num, den), AUC = num / den.
    One sort serves all B replicates: O(B n).
    """
    order = np.argsort(scores, kind="mergesort")
    s, p = scores[order], pos[order]
    starts = np.flatnonzero(np.r_[True, s[1:] != s[:-1]])
    num, den = np.empty(W.shape[0]), np.empty(W.shape[0])
    for i in range(0, W.shape[0], chunk):
        Wo = W[i:i + chunk][:, order].astype(np.float64)
        wn = np.add.reduceat(Wo * ~p, starts, axis=1)
        wp = np.add.reduceat(Wo * p, starts, axis=1)
        below = np.cumsum(wn, axis=1) - wn
        num[i:i + chunk] = (wp * (below + 0.5 * wn)).sum(1)
        den[i:i + chunk] = wp.sum(1) * wn.sum(1)
    return num, den


def weighted_cohens_d(scores: np.ndarray, pos: np.ndarray, W: np.ndarray) -> np.ndarray:
    """
    Cohen's d of scores (positives - negatives, pooled SD) under resampling weights
    W (B, n).  Unbounded counterpart of AUC: AUC = Phi(d / sqrt(2)) under the
    equal-variance binormal model, so d keeps resolving differences at AUC ceiling.
    """
    z = scores.astype(np.float64) - scores.mean()
    p = pos.astype(np.float64)
    q = 1.0 - p
    n1, n0 = W @ p, W @ q
    m1, m0 = (W @ (z * p)) / n1, (W @ (z * q)) / n0
    ss = (W @ (z * z * p)) - n1 * m1 ** 2 + (W @ (z * z * q)) - n0 * m0 ** 2
    with np.errstate(invalid="ignore", divide="ignore"):
        return (m1 - m0) / np.sqrt(ss / (n1 + n0 - 2))


def bootstrap_weights(strata: np.ndarray, n_boot: int, seed: int) -> np.ndarray:
    """(n_boot + 1, n): row 0 = observed sample, rows 1.. = stratified resamples."""
    rng = np.random.default_rng(seed)
    W = np.zeros((n_boot + 1, len(strata)), np.float64)
    W[0] = 1.0
    for g in np.unique(strata):
        idx = np.flatnonzero(strata == g)
        W[1:, idx] = rng.multinomial(len(idx), np.full(len(idx), 1.0 / len(idx)), size=n_boot)
    return W


def hierarchical_strata(levels: Sequence[np.ndarray], min_size: int) -> np.ndarray:
    """
    Cross-classified resampling strata with hierarchical back-off: when a cell has
    fewer than `min_size` items, its whole parent cell (last level dropped) becomes one
    stratum; repeated up to the first level.  A one- or two-item cell would contribute (almost) no
    resampling variability and bias bootstrap standard errors downward.
    """
    levels = [np.asarray(x).astype(str) for x in levels]
    keys = ["|".join(t) for t in zip(*levels)]
    lab = np.asarray(keys, dtype=object)
    for depth in range(len(levels) - 1, 0, -1):
        counts = pd.Series(lab).value_counts()
        small = np.isin(lab, counts.index[counts < min_size])
        if not small.any():
            break
        parent = np.asarray(["|".join(t) + "|*" for t in zip(*levels[:depth])], dtype=object)
        # the whole parent cell is merged, so the small cell joins its siblings
        lab = np.where(np.isin(parent, np.unique(parent[small])), parent, lab)
    return lab.astype(str)


def within_k_spearman(x: np.ndarray, y: np.ndarray, k: np.ndarray, n_perm: int,
                      seed: int) -> Tuple[float, float, int, int]:
    """
    Partial Spearman correlation of x and y given the dose k (number of substituted
    layers): ranks are centred within each k stratum (equivalent to partialling out k
    as a categorical covariate), and the p-value comes from permuting y WITHIN strata,
    which keeps every run's dose fixed.  A pooled correlation over runs with different
    k is positive by dose alone and does not show that lexical damage tracks the
    outcome.  Strata with a single run carry no within-k information and are dropped.
    Returns (rho, two-sided p, runs used, strata used).
    """
    x, y, k = (np.asarray(v, dtype=float) for v in (x, y, k))
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(k)
    x, y, k = x[ok], y[ok], k[ok]
    groups = [np.flatnonzero(k == g) for g in np.unique(k)]
    groups = [g for g in groups if g.size >= 2]
    if not groups:
        return np.nan, np.nan, 0, 0
    idx = np.concatenate(groups)
    rx, ry = stats.rankdata(x[idx]), stats.rankdata(y[idx])
    pos = np.cumsum([0] + [g.size for g in groups])
    for a, b in zip(pos[:-1], pos[1:]):
        rx[a:b] -= rx[a:b].mean()
        ry[a:b] -= ry[a:b].mean()

    def corr(u, v):
        den = np.sqrt((u * u).sum(-1) * (v * v).sum(-1))
        with np.errstate(invalid="ignore", divide="ignore"):
            return (u * v).sum(-1) / den

    rho = float(corr(rx, ry))
    if not np.isfinite(rho):
        return np.nan, np.nan, int(idx.size), len(groups)
    rng = np.random.default_rng(seed)
    perm = np.empty((n_perm, ry.size))
    for a, b in zip(pos[:-1], pos[1:]):
        perm[:, a:b] = ry[a:b][np.argsort(rng.random((n_perm, b - a)), axis=1)]
    null = corr(rx[None, :], perm)
    p = float((1 + np.sum(np.abs(null) >= abs(rho) - 1e-12)) / (1 + n_perm))
    return rho, p, int(idx.size), len(groups)


def summarize(v: np.ndarray) -> Dict[str, float]:
    """
    Point estimate (row 0), bootstrap SE, percentile 95% CI, a two-sided Wald
    p-value with the bootstrap SE (Efron & Tibshirani, 1993; used for Holm because
    a percentile-count p-value cannot go below 2/(B+1)), and whether the percentile
    CI excludes 0 (M2: decisions require both).
    """
    est, boot = float(v[0]), v[1:][np.isfinite(v[1:])]
    if not np.isfinite(est) or boot.size < 2:
        return {"estimate": np.nan, "se": np.nan, "ci_low": np.nan, "ci_high": np.nan,
                "p_value": np.nan, "ci_excludes_zero": False}
    lo, hi = np.percentile(boot, [2.5, 97.5])
    se = float(boot.std(ddof=1))
    if se > 0:
        p = float(2 * stats.norm.sf(abs(est) / se))
    else:                      # degenerate bootstrap: no sampling variability at all
        p = 1.0 if est == 0 else 0.0
    return {"estimate": est, "se": se, "ci_low": float(lo), "ci_high": float(hi), "p_value": p,
            "ci_excludes_zero": bool(lo > 0 or hi < 0)}


def sign_flip_test(d: np.ndarray, n_perm: int, seed: int, chunk: int = 1000) -> float:
    """Exact-form paired permutation p-value, (1 + #{|null| >= |obs|}) / (1 + n_perm)."""
    rng = np.random.default_rng(seed)
    obs, hits = abs(d.mean()), 0
    for i in range(0, n_perm, chunk):
        m = min(chunk, n_perm - i)
        signs = rng.integers(0, 2, size=(m, d.size)) * 2 - 1
        hits += int(np.sum(np.abs((signs * d).mean(1)) >= obs - 1e-12))
    return (1 + hits) / (1 + n_perm)


def randomization_test(obs: float, null: np.ndarray, exhaustive: bool) -> Dict[str, float]:
    """
    Percentile-rank (reference-distribution) test of a deterministically selected
    policy's effect against the effects of its null target sets.  Exhaustive null (obs
    is a member): p_up = #{null >= obs} / N.  Monte-Carlo null: p_up = (1 + #{null >=
    obs}) / (1 + N) (Phipson & Smyth, 2010).  The mid-p (Lancaster, 1961) counts ties
    with weight 1/2 and is used for combination.
    This is NOT a Fisher randomisation test: the policy was not assigned at random.
    The p-value is exact only under the assumption that, under H0 (the selection
    criterion carries no information about damage), the selected map is exchangeable
    with the null draws.  Deterministic selection does not guarantee that, so results
    read as "the selected map ranks at this percentile of random maps".
    """
    null = null[np.isfinite(null)]
    N = null.size
    if not np.isfinite(obs) or N == 0:
        return {"n_null": N, "p_upper": np.nan, "p_lower": np.nan, "p_two_sided": np.nan,
                "mid_p_upper": np.nan, "null_mean": np.nan, "null_sd": np.nan, "z_vs_null": np.nan}
    eps = 1e-12 * max(1.0, abs(obs))
    gt, lt = int((null > obs + eps).sum()), int((null < obs - eps).sum())
    eq = N - gt - lt
    if exhaustive:
        p_up, p_lo, mid = (gt + eq) / N, (lt + eq) / N, (gt + 0.5 * eq) / N
    else:
        p_up, p_lo = (1 + gt + eq) / (1 + N), (1 + lt + eq) / (1 + N)
        mid = (gt + 0.5 * (eq + 1)) / (1 + N)
    sd = float(null.std(ddof=1)) if N > 1 else np.nan
    return {"n_null": N, "p_upper": p_up, "p_lower": p_lo, "p_two_sided": min(1.0, 2 * min(p_up, p_lo)),
            "mid_p_upper": mid, "null_mean": float(null.mean()), "null_sd": sd,
            "z_vs_null": (obs - null.mean()) / sd if sd and sd > 0 else np.nan}


def mid_p_reference(obs: float, null: np.ndarray, exhaustive: bool) -> np.ndarray:
    """
    Mid-p values (same definition as `randomization_test`) of every member of the
    reference set: the null space itself (exhaustive) or null + observed
    (Monte-Carlo).  Under H0 the observed mid-p is a uniform draw from this set.
    """
    ref = null[np.isfinite(null)]
    ref = ref if exhaustive else np.append(ref, obs)
    eps = 1e-12 * np.maximum(1.0, np.abs(ref))
    gt = (ref[None, :] > ref[:, None] + eps[:, None]).sum(1)
    eq = (np.abs(ref[None, :] - ref[:, None]) <= eps[:, None]).sum(1)
    return (gt + 0.5 * eq) / ref.size


def stouffer_randomization(z_obs: np.ndarray, supports: List[np.ndarray], n_draws: int,
                           seed: int) -> Tuple[float, float]:
    """
    Stouffer statistic Z = sum_m z_m / sqrt(M), z_m = Phi^-1(1 - mid-p_m), with its
    exact randomisation null: every model's z is drawn independently and uniformly
    from its own reference support.  Unlike the N(0,1) approximation this keeps the
    nominal size when null spaces are tiny.  Returns (Z, two-sided p).
    """
    M = len(z_obs)
    if M == 0:
        return np.nan, np.nan
    Z = float(np.sum(z_obs) / math.sqrt(M))
    rng = np.random.default_rng(seed)
    null = sum(rng.choice(sup, size=n_draws) for sup in supports) / math.sqrt(M)
    return Z, float((1 + np.sum(np.abs(null) >= abs(Z) - 1e-12)) / (1 + n_draws))


def exact_spearman_upper(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Spearman rho and its exact one-sided (rho > 0) permutation p-value (n <= 8)."""
    rho = float(stats.spearmanr(x, y)[0])
    n = len(x)
    if n < 3 or n > 8 or not np.isfinite(rho):
        return rho, np.nan
    null = [stats.spearmanr(x, np.asarray(perm))[0] for perm in itertools.permutations(y)]
    return rho, float(np.mean(np.asarray(null) >= rho - 1e-12))


def dprime(hits: np.ndarray, false_alarms: np.ndarray) -> float:
    """d' with the log-linear correction (Hautus, 1995)."""
    h = (hits.sum() + 0.5) / (hits.size + 1)
    f = (false_alarms.sum() + 0.5) / (false_alarms.size + 1)
    return float(stats.norm.ppf(h) - stats.norm.ppf(f))


def adjust(df: pd.DataFrame, pcol: str, by: List[str], method: str, out: str) -> pd.DataFrame:
    df[out] = np.nan
    for _, g in df.groupby(by, dropna=False):
        g = g[g[pcol].notna()]
        if len(g):
            df.loc[g.index, out] = multipletests(g[pcol].to_numpy(), method=method)[1]
    return df


def ols_hc3(y: np.ndarray, X: pd.DataFrame):
    return sm.OLS(y, sm.add_constant(X, has_constant="add")).fit(cov_type="HC3")


def zscore(x: np.ndarray) -> np.ndarray:
    sd = np.nanstd(x)
    return (x - np.nanmean(x)) / (sd if sd > 0 else 1.0)


DETERMINISTIC_CONTRASTS = [
    # (label, policy A, policy B); A and B are deterministic runs at the primary k
    ("H3a_cka_similar_vs_dissimilar", "cka_high", "cka_low"),
    ("H3a_kvdist_within_cla_similar_vs_dissimilar", "kvdist_low", "kvdist_high"),
    ("H3a_kvsharer_search_similar_vs_dissimilar", "kvsharer_similar", "kvsharer_dissimilar"),
    ("H3c_fidelity_vs_cka", "fidelity_high", "cka_high"),
    ("H3c_fidelity_vs_kvdist_within_cla", "fidelity_high", "kvdist_high"),
    ("C1_live_vs_clean_semantics", LIVE_PREV, CF_PREV),
    ("C2_previous_vs_next_head_source", CF_PREV, CF_NEXT),
]
READOUTS = ("frozen", "retrained")
BAND_METRICS = ("auc_all", "d_all", "auc_hf", "auc_lf", "auc_lf_hard", "auc_tok", "sel_auc", "sel_d")
# sel_auc is bounded: high-frequency words sit near AUC = 1, so their change is compressed and
# sel_auc is biased toward "LF damaged more" with no true effect.  It is descriptive only;
# sel_d (unbounded) is the selectivity metric reported alongside it.
DESCRIPTIVE_METRICS = ("sel_auc",)


# ════════════════════════════════════════════════════════════════════════════
# STATISTICS: per model
# ════════════════════════════════════════════════════════════════════════════

class ModelAnalysis:
    """
    All per-model inference.  Items are the resampling unit; layers never are.
    Inference is conditional on the fitted (seed-averaged) probes (M5).
    """

    def __init__(self, cfg: ExperimentConfig, model_name: str):
        self.cfg, self.model = cfg, model_name
        self.dir = cfg.model_dir(model_name)
        self.out = os.path.join(self.dir, "analysis")
        os.makedirs(self.out, exist_ok=True)
        with open(os.path.join(self.dir, "model_info.json")) as f:
            self.info = json.load(f)
        self.L = int(self.info["num_layers"])
        self.family = self.info["family"]
        self.test = np.asarray(self.info["test_items"])
        items = pd.read_csv(os.path.join(self.dir, "items.csv"))
        self.t = items.iloc[self.test].reset_index(drop=True)
        self.y = self.t["is_word"].to_numpy() == 1
        self.fg = self.t["freq_group"].to_numpy()
        self.ntok = self.t["n_tokens"].to_numpy()
        self.wordlike = self.t["wordlike"].to_numpy(bool) & ~self.y
        self.tok_stratum = np.minimum(self.ntok, max(cfg.TOKEN_STRATA))
        # Resampling strata = frequency group x token stratum x wordlikeness, so every
        # replicate keeps the composition of every subgroup a metric is computed on.
        # Cells smaller than MIN_STRATUM_SIZE are merged into their parent cell.
        self.strata = hierarchical_strata([self.fg, self.tok_stratum, self.wordlike], cfg.MIN_STRATUM_SIZE)
        self.W = bootstrap_weights(self.strata, cfg.N_BOOTSTRAP, SEED)
        self.W0 = self.W[:1]
        self.covariates = [c for c in cfg.MATCH_COVARIATES if self.t.loc[self.y, c].notna().mean() >= 0.9]
        self.base = (None, NO_REUSE)
        self._discover_runs()
        self._arrays: Dict[Tuple, Dict[str, np.ndarray]] = {}
        self._cache: Dict[Tuple, np.ndarray] = {}
        base = self.arrays(self.base)
        # Common ruler for item evidence: pooled within-class SD of the no_reuse logits.
        self.scale = {"frozen": np.array([self._pooled_sd(base["frozen"][li]) for li in range(self.L)]),
                      "behavior": np.array([self._pooled_sd(base["behavior"])])}
        self.k0 = self._primary_k()

    # ── loading ───────────────────────────────────────────────────────
    def _discover_runs(self):
        self.meta: Dict[Tuple, Dict] = {}
        self.paths: Dict[Tuple, str] = {}
        dirs = [(None, self.dir)] + [(r.name, os.path.join(self.dir, r.name)) for r in self.cfg.REGIMES]
        for regime, d in dirs:
            if not os.path.isdir(d):
                continue
            for fn in sorted(os.listdir(d)):
                stem = fn[:-5]
                npz = os.path.join(d, stem + ".npz")
                if not fn.endswith(".json") or not os.path.exists(npz):
                    continue
                with open(os.path.join(d, fn)) as f:
                    meta = json.load(f)
                if "family" not in meta or (regime is None and meta["name"] != NO_REUSE):
                    continue
                self.meta[(regime, meta["name"])] = meta
                self.paths[(regime, meta["name"])] = npz
        if self.base not in self.meta:
            raise RuntimeError(f"{self.model}: no_reuse control missing")

    def arrays(self, key: Tuple) -> Dict[str, np.ndarray]:
        """Arrays of one run; cached for bootstrapped (deterministic full-tier) runs only."""
        if key in self._arrays:
            return self._arrays[key]
        z = np.load(self.paths[key])
        if not np.array_equal(z["test_items"], self.test):
            raise RuntimeError(f"{self.model}/{key}: test items differ from model_info")
        arr = {k: z[k] for k in z.files}
        if self.is_boot(key):
            self._arrays[key] = arr
        return arr

    def is_boot(self, key: Tuple) -> bool:
        m = self.meta[key]
        return m["draw"] == 0 and m["tier"] == "full"

    def keys(self, regime: Optional[str] = None, family: Optional[str] = None,
             k: Optional[int] = None, deterministic: Optional[bool] = None) -> List[Tuple]:
        out = []
        for key, m in self.meta.items():
            if key == self.base:
                continue
            if ((regime is None or key[0] == regime) and (family is None or m["family"] == family)
                    and (k is None or m["k"] == k) and (deterministic is None or (m["draw"] == 0) == deterministic)):
                out.append(key)
        return sorted(out, key=lambda x: (str(x[0]), x[1]))

    def _primary_k(self) -> Optional[int]:
        ks = {m["k"] for key, m in self.meta.items()
              if m["family"] in self.cfg.GATED_FAMILIES and m["tier"] == "full" and m["draw"] == 0}
        return ks.pop() if len(ks) == 1 else None

    def ec(self, regime: str) -> int:
        return int(self.info["regimes"][regime]["exempt_cutoff"])

    def bands(self, regime: str) -> Dict[str, List[int]]:
        return eligible_bands(self.L, self.ec(regime), self.cfg.N_BANDS)

    def _finite_ok(self, z: np.ndarray) -> Optional[np.ndarray]:
        """Finite-item mask, or None if more than NONFINITE_MAX_FRACTION items are non-finite."""
        f = np.isfinite(z)
        return f if f.mean() >= 1.0 - self.cfg.NONFINITE_MAX_FRACTION else None

    def _pooled_sd(self, z: np.ndarray) -> float:
        f = self._finite_ok(z)
        if f is None:
            return np.nan
        y, z = self.y[f], z[f]
        v = (((y.sum() - 1) * z[y].var(ddof=1) + ((~y).sum() - 1) * z[~y].var(ddof=1)) / (len(z) - 2))
        return float(np.sqrt(v))

    # ── metric replicates ─────────────────────────────────────────────
    def _subsets(self, group: str) -> List[np.ndarray]:
        nw = ~self.y
        if group == "all":
            return [np.ones_like(self.y)]
        if group == "hf":
            return [nw | (self.fg == "high")]
        if group == "lf":
            return [nw | (self.fg == "low")]
        if group == "lf_hard":
            return [self.wordlike | (self.fg == "low")]
        if group == "tok":           # token-count-matched: pairs only within a stratum
            return [self.tok_stratum == k for k in self.cfg.TOKEN_STRATA]
        if group.startswith("tok"):
            return [self.tok_stratum == int(group[3:])]
        raise ValueError(group)

    def metric_rep(self, scores: np.ndarray, metric: str, W: np.ndarray) -> np.ndarray:
        kind, group = metric.split("_", 1)
        fin = self._finite_ok(scores)
        if fin is None:
            return np.full(W.shape[0], np.nan)
        if kind == "d":
            (m,) = self._subsets(group)
            m = m & fin
            if not (self.y[m].any() and (~self.y[m]).any()):
                return np.full(W.shape[0], np.nan)
            return weighted_cohens_d(scores[m], self.y[m], W[:, m])
        num, den = np.zeros(W.shape[0]), np.zeros(W.shape[0])
        for m in self._subsets(group):
            m = m & fin
            if self.y[m].any() and (~self.y[m]).any():
                a, b = weighted_auc_parts(scores[m], self.y[m], W[:, m])
                num, den = num + a, den + b
        with np.errstate(invalid="ignore", divide="ignore"):
            return num / den

    def curve(self, key: Tuple, readout: str, metric: str, boot: bool = True) -> np.ndarray:
        """(n_layers, B+1) metric replicates of one run; behaviour has a single 'layer'."""
        W = self.W if boot else self.W0
        ck = (key, readout, metric, boot)
        if ck in self._cache:
            return self._cache[ck]
        scores = self.arrays(key)[readout]
        scores = scores[None, :] if scores.ndim == 1 else scores
        out = np.stack([self.metric_rep(s, metric, W) for s in scores])
        self._cache[ck] = out
        return out

    def delta(self, key: Tuple, readout: str, metric: str, layers: Sequence[int],
              boot: bool = True, other: Optional[Tuple] = None) -> np.ndarray:
        """
        Band mean over `layers` of metric(key) - metric(other) (default no_reuse);
        sel_<m> = delta(<m>_hf) - delta(<m>_lf): positive = LF damaged more.
        """
        other = self.base if other is None else other
        if metric.startswith("sel_"):
            m = metric[4:]
            return (self.delta(key, readout, f"{m}_hf", layers, boot, other)
                    - self.delta(key, readout, f"{m}_lf", layers, boot, other))
        layers = [0] if readout == "behavior" else list(layers)
        d = self.curve(key, readout, metric, boot)[layers] - self.curve(other, readout, metric, boot)[layers]
        with np.errstate(invalid="ignore"):
            return np.nanmean(d, axis=0) if np.isfinite(d).any() else np.full(d.shape[1], np.nan)

    def base_auc(self, readout: str, layers: Sequence[int]) -> float:
        layers = [0] if readout == "behavior" else list(layers)
        return float(np.nanmean(self.curve(self.base, readout, "auc_all", boot=False)[layers, 0]))

    def primary_metric(self, readout: str, layers: Sequence[int]) -> str:
        """AUC, or Cohen's d when no_reuse is at ceiling on these layers (C1)."""
        return "d_all" if self.base_auc(readout, layers) >= self.cfg.CEILING_AUC else "auc_all"

    def mass_ratio(self, key: Tuple) -> float:
        """Median answer-token mass under the policy / under no_reuse (task-format retention)."""
        arr = self.arrays(key)
        if "behavior_mass" not in arr:
            return np.nan
        return float(np.nanmedian(arr["behavior_mass"]) / np.nanmedian(self.arrays(self.base)["behavior_mass"]))

    def readouts_of(self, key: Tuple) -> List[str]:
        return [r for r in READOUTS if r in self.arrays(key)]

    # ── item evidence (frequency analyses) ────────────────────────────
    def evidence(self, keys: List[Tuple], readout: str, layers: Sequence[int]) -> np.ndarray:
        """
        Word evidence per item: probe logit / no_reuse pooled SD, averaged over band
        layers (each layer weighted equally) and over the runs in `keys`.
        """
        layers = [0] if readout == "behavior" else list(layers)
        scale = self.scale[readout][layers][:, None]
        vals = []
        for key in keys:
            z = self.arrays(key)[readout]
            z = z[None, :] if z.ndim == 1 else z
            vals.append(np.nanmean(z[layers] / scale, axis=0))
        return np.mean(vals, axis=0)

    def word_mask(self) -> np.ndarray:
        m = self.y & self.t["log_freq"].notna().to_numpy()
        for c in self.covariates:
            m &= self.t[c].notna().to_numpy()
        return m

    # ── 1. layer table ────────────────────────────────────────────────
    def layer_table(self) -> pd.DataFrame:
        rows = []
        for key in [self.base] + [k for k in self.keys() if self.is_boot(k)]:
            meta = self.meta[key]
            targets = {int(t) for t in meta["source_map"]}
            for ro in self.readouts_of(key):
                c = {m: self.curve(key, ro, m) for m in ("auc_all", "auc_hf", "auc_lf", "auc_lf_hard",
                                                          "auc_tok", "d_all")}
                logits = self.arrays(key)[ro]
                for li in range(self.L):
                    f = self._finite_ok(logits[li])
                    finite = f is not None
                    f = f if finite else np.ones(len(self.y), bool)
                    pred = logits[li] > 0
                    row = {"model": self.model, "regime": key[0] or CONTROL, "policy": key[1],
                           "family": meta["family"], "readout": ro, "layer": li, "depth": li / max(self.L - 1, 1),
                           "is_target": li in targets,
                           **{m: v[li, 0] for m, v in c.items()},
                           "accuracy": float((pred == self.y)[f].mean()) if finite else np.nan,
                           "dprime_hf": (dprime(pred[f & self.y & (self.fg == "high")], pred[f & ~self.y])
                                         if finite else np.nan),
                           "dprime_lf": (dprime(pred[f & self.y & (self.fg == "low")], pred[f & ~self.y])
                                         if finite else np.nan)}
                    s = summarize(c["auc_all"][li])
                    row["auc_all_ci_low"], row["auc_all_ci_high"] = s["ci_low"], s["ci_high"]
                    if key != self.base:
                        for m in ("auc_all", "d_all"):
                            row.update({f"delta_{m}_{k}": v for k, v in
                                        summarize(self.delta(key, ro, m, [li])).items()})
                    rows.append(row)
        df = pd.DataFrame(rows)
        if "delta_auc_all_p_value" in df:
            df = adjust(df, "delta_auc_all_p_value", ["regime", "policy", "readout"], "fdr_bh",
                        "delta_auc_all_p_fdr")
        idx = ["regime", "policy", "layer"]
        rt = df[df.readout == "retrained"].set_index(idx)["auc_all"]
        fz = df[df.readout == "frozen"].set_index(idx)["auc_all"]
        gap = (rt - fz).rename("recovery_gap").reset_index()
        return df.merge(gap, on=idx, how="left")

    # ── 2. band contrasts vs no_reuse (H1, H1x) ──────────────────────
    def band_contrasts(self) -> pd.DataFrame:
        rows = []
        for key in [k for k in self.keys() if self.is_boot(k)]:
            meta = self.meta[key]
            for ro in self.readouts_of(key):
                for band, layers in self.bands(key[0]).items():
                    prim = self.primary_metric(ro, layers)
                    pol_auc = float(np.nanmean(self.curve(key, ro, "auc_all")[layers, 0]))
                    for metric in BAND_METRICS:
                        if metric.endswith("lf_hard") and not self.wordlike.any():
                            continue
                        rows.append({"model": self.model, "family_model": self.family, "regime": key[0],
                                     "policy": key[1], "family": meta["family"], "k": meta["k"],
                                     "kv_memory_fraction": meta["kv_memory_fraction"],
                                     "rope_mismatch": meta["rope_mismatch"], "readout": ro, "band": band,
                                     "metric": metric, "is_primary_metric": metric == prim,
                                     "descriptive_only": metric in DESCRIPTIVE_METRICS,
                                     "base_auc_band": self.base_auc(ro, layers), "policy_auc_band": pol_auc,
                                     "at_ceiling": self.base_auc(ro, layers) >= self.cfg.CEILING_AUC,
                                     "at_floor": pol_auc <= self.cfg.FLOOR_AUC,
                                     **summarize(self.delta(key, ro, metric, layers))})
        df = pd.DataFrame(rows)
        return adjust(df, "p_value", ["regime", "readout", "metric"], "holm", "p_holm") if len(df) else df

    # ── 3. behavioural lexical decision (H8a, H8b) ───────────────────
    def behavior_table(self) -> pd.DataFrame:
        if "behavior" not in self.arrays(self.base):
            return pd.DataFrame()
        base_mass = float(np.nanmedian(self.arrays(self.base)["behavior_mass"]))
        rows = [{"model": self.model, "regime": CONTROL, "policy": NO_REUSE, "family": NO_REUSE,
                 "metric": "abs_auc_all", "answer_mass_median": base_mass, "answer_mass_ratio": 1.0,
                 "format_retained": True, **summarize(self.curve(self.base, "behavior", "auc_all")[0])}]
        for key in [k for k in self.keys() if self.is_boot(k) and "behavior" in self.arrays(k)]:
            ratio = self.mass_ratio(key)
            for metric in ("auc_all", "d_all", "auc_hf", "auc_lf", "auc_lf_hard", "sel_auc", "sel_d"):
                if metric.endswith("lf_hard") and not self.wordlike.any():
                    continue
                rows.append({"model": self.model, "regime": key[0], "policy": key[1],
                             "family": self.meta[key]["family"], "metric": metric,
                             "answer_mass_median": float(np.nanmedian(self.arrays(key)["behavior_mass"])),
                             "answer_mass_ratio": ratio,
                             "format_retained": ratio >= self.cfg.BEHAVIOR_MASS_MIN_RATIO,
                             **summarize(self.delta(key, "behavior", metric, [0]))})
        df = pd.DataFrame(rows)
        pol = df[df.metric != "abs_auc_all"].copy()
        pol = adjust(pol, "p_value", ["regime", "metric"], "holm", "p_holm")
        return pd.concat([df[df.metric == "abs_auc_all"], pol], ignore_index=True)

    # ── 4. run-level effects of every run (randomisation, dose, H6, H8c) ──
    def run_effects(self) -> pd.DataFrame:
        """
        Point estimates for every run, including all null draws, plus the exact
        negative control: frozen logits on layers below the first target must equal
        no_reuse (same code path and batch plan).
        """
        base_fz = self.arrays(self.base)["frozen"]
        rows = []
        for key in self.keys():
            meta = self.meta[key]
            arr = self.arrays(key)
            targets = sorted(int(t) for t in meta["source_map"])
            first = targets[0] if targets else self.L
            neg = (float(np.nanmax(np.abs(arr["frozen"][:first] - base_fz[:first]))) if first > 0 else np.nan)
            beh = "behavior" in arr
            for band, layers in self.bands(key[0]).items():
                prim = self.primary_metric("frozen", layers)
                row = {"model": self.model, "family_model": self.family, "regime": key[0], "policy": key[1],
                       "family": meta["family"], "k": meta["k"], "draw": meta["draw"], "tier": meta["tier"],
                       "semantics": meta["semantics"], "deployable": meta["deployable"],
                       "kv_memory_fraction": meta["kv_memory_fraction"],
                       "mean_source_distance": meta["mean_source_distance"],
                       "null_space_size": meta["null_space_size"], "exhaustive": meta["exhaustive"],
                       "rope_mismatch": meta["rope_mismatch"], "seconds": meta["seconds"],
                       "band": band, "primary_metric": prim,
                       "delta_auc_all": self.delta(key, "frozen", "auc_all", layers, boot=False)[0],
                       "delta_d_all": self.delta(key, "frozen", "d_all", layers, boot=False)[0],
                       "policy_auc_all": float(np.nanmean(self.curve(key, "frozen", "auc_all", False)[layers, 0])),
                       "delta_behavior_auc": (self.delta(key, "behavior", "auc_all", [0], boot=False)[0]
                                              if beh else np.nan),
                       "answer_mass_ratio": self.mass_ratio(key) if beh else np.nan,
                       "neg_control_layers": first, "neg_control_max_abs_logit": neg,
                       "neg_control_pass": bool(not np.isfinite(neg) or neg <= self.cfg.NEG_CONTROL_ATOL)}
                row["delta_primary"] = row["delta_d_all"] if prim == "d_all" else row["delta_auc_all"]
                rows.append(row)
        df = pd.DataFrame(rows)
        bad = df[~df.neg_control_pass].drop_duplicates("policy") if len(df) else df
        if len(bad):
            logger.warning(f"  {self.model}: {len(bad)} runs FAIL the pre-target negative control "
                           f"(max |logit change| up to {bad.neg_control_max_abs_logit.max():.2e})")
        return df

    # ── 5. randomisation tests against null target sets (H3b, H3d, H4) ──
    def randomization(self, effects: pd.DataFrame) -> pd.DataFrame:
        P = self.cfg.primary_regime.name
        e = effects[effects.regime == P]
        if e.empty or self.k0 is None:
            return pd.DataFrame(), pd.DataFrame()
        gated = self.cfg.GATED_FAMILIES
        ks = sorted(e.k[e.family.isin(gated)].unique())
        tests = [(f"H3b_{fam}_vs_depth_matched", f"{fam}@k{k}", DEPTH_PREFIX + fam, k) for k in ks for fam in gated]
        tests += [(f"H3d_{fam}_vs_uniform", f"{fam}@k{k}", UNIFORM, k) for k in ks for fam in gated]
        n_full = int(e.k[e.family == FULL].max()) if (e.family == FULL).any() else None
        tests += [("H4_own_head_vs_random_source", FULL, RANDOM_SOURCE, n_full)]
        for fam in KVSHARER_FAMILIES:
            det = e[(e.family == fam) & (e.draw == 0)]
            if len(det):
                tests.append((f"H3e_{fam}_vs_random_pairs", det.policy.iloc[0], RANDOM_PAIRS, int(det.k.iloc[0])))
        rows, support = [], []
        for label, obs_name, null_family, k in tests:
            primary = k == self.k0 or label.startswith(("H4", "H3e"))
            for band, g in e.groupby("band"):
                obs = g[g.policy == obs_name]
                null = g[(g.family == null_family) & (g.k == k)]
                if obs.empty or null.empty:
                    continue
                exhaustive = bool(null.exhaustive.iloc[0])
                for stat in ("delta_primary", "delta_auc_all", "delta_d_all"):
                    o, nv = float(obs[stat].iloc[0]), null[stat].to_numpy()
                    rows.append({"model": self.model, "family_model": self.family, "contrast": label,
                                 "policy": obs_name, "null_family": null_family, "k": k, "band": band,
                                 "statistic": stat, "observed": o, "is_primary_k": primary,
                                 "exhaustive": exhaustive, "null_space_size": int(null.null_space_size.iloc[0]),
                                 **randomization_test(o, nv, exhaustive)})
                    if primary and np.isfinite(o):
                        support += [{"model": self.model, "contrast": label, "band": band, "statistic": stat,
                                     "z": z} for z in stats.norm.isf(mid_p_reference(o, nv, exhaustive))]
        return pd.DataFrame(rows), pd.DataFrame(support)

    # ── 6. deterministic policy-vs-policy contrasts (H3a, H3c, C1, C2) ──
    def policy_contrasts(self) -> pd.DataFrame:
        P = self.cfg.primary_regime.name
        if self.k0 is None:
            return pd.DataFrame()
        names = {fam: (P, f"{fam}@k{self.k0}") for fam in self.cfg.GATED_FAMILIES}
        names |= {fam: keys[0] for fam in KVSHARER_FAMILIES if (keys := self.keys(P, fam, deterministic=True))}
        names |= {LIVE_PREV: (P, LIVE_PREV) if (P, LIVE_PREV) in self.meta else (P, FULL),
                  CF_PREV: (P, CF_PREV), CF_NEXT: (P, CF_NEXT)}
        rows = []
        for label, a, b in DETERMINISTIC_CONTRASTS:
            ka, kb = names.get(a), names.get(b)
            if ka not in self.meta or kb not in self.meta:
                continue
            if self.meta[ka]["k"] != self.meta[kb]["k"]:
                logger.warning(f"  {self.model}: {label} skipped, {ka[1]} and {kb[1]} substitute "
                               f"different numbers of layers (not rate-matched)")
                continue
            for ro in sorted(set(self.readouts_of(ka)) & set(self.readouts_of(kb))) + ["behavior"]:
                if ro == "behavior" and not ("behavior" in self.arrays(ka) and "behavior" in self.arrays(kb)):
                    continue
                for band, layers in (self.bands(P).items() if ro != "behavior" else [("behavior", [0])]):
                    prim = self.primary_metric(ro, layers)
                    for metric in ("auc_all", "d_all", "sel_auc", "sel_d"):
                        rows.append({"model": self.model, "family_model": self.family, "contrast": label,
                                     "policy_a": ka[1], "policy_b": kb[1], "readout": ro, "band": band,
                                     "metric": metric, "is_primary_metric": metric == prim,
                                     "descriptive_only": metric in DESCRIPTIVE_METRICS,
                                     **summarize(self.delta(ka, ro, metric, layers, other=kb))})
        df = pd.DataFrame(rows)
        return adjust(df, "p_value", ["readout", "metric", "band"], "holm", "p_holm") if len(df) else df

    # ── 7. item-level frequency regressions (H2, H5x) ─────────────────
    def _frequency_families(self) -> List[Tuple[str, str, List[Tuple]]]:
        """(regime, label, run keys): `full` of every regime, gated at k0, uniform mean at k0."""
        P = self.cfg.primary_regime.name
        fams = [(r.name, FULL, [(r.name, FULL)]) for r in self.cfg.REGIMES if (r.name, FULL) in self.meta]
        if self.k0 is not None:
            fams += [(P, fam, [(P, f"{fam}@k{self.k0}")]) for fam in self.cfg.GATED_FAMILIES
                     if (P, f"{fam}@k{self.k0}") in self.meta]
            uni = self.keys(P, UNIFORM, self.k0)
            if uni:
                fams.append((P, f"{UNIFORM}_mean", uni))
        return fams

    def item_regressions(self) -> pd.DataFrame:
        """
        Word damage = baseline evidence - policy evidence (no_reuse pooled-SD units).
        Estimands (Lord, 1967; Pearl, 2001), beta_freq < 0  <=>  rarer words lose more:
          primary   change score  damage ~ freq + length/OrthoN/BG + n_tokens (HC3).
                    A CONTROLLED DIRECT effect: n_tokens is itself downstream of
                    frequency (BPE merges frequent strings), so holding it fixed
                    removes the tokenisation path; this is the frequency effect
                    not routed through tokenisation, not the total effect; NO
                    baseline-evidence term, because frequency causes baseline
                    evidence (a mediator, not a confounder: frequency is not
                    randomised).  Assumes no unmeasured common cause of n_tokens
                    and damage beyond the orthographic covariates.
          total     same without n_tokens (also the attenuation reference).
          relative  damage / baseline evidence (scale-free, proportional loss) for
                    words with baseline evidence >= RELATIVE_DAMAGE_MIN_BASE; the
                    restriction selects on a mediator, so it is descriptive only.
          direct    ANCOVA adding e_base and e_base^2: the effect at equal baseline
                    evidence.  Valid only if no item property (e.g. embedding norm,
                    anisotropy) affects both baseline evidence and damage.
        """
        words = self.word_mask()
        T = self.t[words]
        covs = {f"{c}_z": zscore(T[c].to_numpy()) for c in self.covariates}
        freq = {"hal_log_freq": zscore(T["log_freq"].to_numpy())}
        if T["lm_logprob"].notna().all():
            freq["lm_logprob"] = zscore(T["lm_logprob"].to_numpy())
        ntok_c = self.ntok[words] - self.ntok[words].mean()
        rows = []
        for regime, label, keys in self._frequency_families():
            readouts = ["frozen"] + (["behavior"] if all("behavior" in self.arrays(k) for k in keys) else [])
            band_sets = {ro: (self.bands(regime).items() if ro == "frozen" else [("behavior", [0])])
                         for ro in readouts}
            all_layers = self.bands(regime)["all"]
            pol_auc = float(np.mean([np.nanmean(self.curve(k, "frozen", "auc_all", False)[all_layers, 0])
                                     for k in keys]))
            for ro in readouts:
                for band, layers in band_sets[ro]:
                    e_base = self.evidence([self.base], ro, layers)[words]
                    dmg = e_base - self.evidence(keys, ro, layers)[words]
                    f = np.isfinite(dmg) & np.isfinite(e_base)
                    if f.mean() < 1.0 - self.cfg.NONFINITE_MAX_FRACTION or not np.any(dmg[f]):
                        continue          # too many invalid items, or band before the first target (damage 0)
                    e_base, dmg = e_base[f], dmg[f]
                    eb = e_base - e_base.mean()
                    strong = e_base >= self.cfg.RELATIVE_DAMAGE_MIN_BASE
                    covs_f = {c: v[f] for c, v in covs.items()}
                    for fname, fz_all in freq.items():
                        fz = fz_all[f]
                        base_X = pd.DataFrame({"freq_z": fz} | covs_f)
                        X_cs = base_X.assign(n_tokens_c=ntok_c[f])
                        m_cs = ols_hc3(dmg, X_cs)
                        m_tot = ols_hc3(dmg, base_X)
                        m_dir = ols_hc3(dmg, X_cs.assign(e_base_c=eb, e_base_c2=eb ** 2))
                        vif = max(variance_inflation_factor(sm.add_constant(X_cs).to_numpy(), i)
                                  for i in range(1, X_cs.shape[1] + 1))
                        b_cs, b_tot = m_cs.params["freq_z"], m_tot.params["freq_z"]
                        ci = m_cs.conf_int().loc["freq_z"]
                        row = {"model": self.model, "family_model": self.family, "regime": regime,
                               "family": label, "readout": ro, "band": band, "freq_measure": fname,
                               "n_words": int(f.sum()), "mean_damage": float(dmg.mean()),
                               "policy_auc_all": pol_auc, "at_floor": pol_auc <= self.cfg.FLOOR_AUC,
                               "beta_freq": b_cs, "se_freq": m_cs.bse["freq_z"],
                               "ci_freq_low": ci.iloc[0], "ci_freq_high": ci.iloc[1], "p_freq": m_cs.pvalues["freq_z"],
                               "beta_ntok": m_cs.params["n_tokens_c"], "p_ntok": m_cs.pvalues["n_tokens_c"],
                               "beta_freq_total": b_tot, "p_freq_total": m_tot.pvalues["freq_z"],
                               "attenuation_by_ntok": (1 - b_cs / b_tot) if b_tot != 0 else np.nan,
                               "beta_freq_direct": m_dir.params["freq_z"], "p_freq_direct": m_dir.pvalues["freq_z"],
                               "max_vif": float(vif), "covariates": ";".join(self.covariates),
                               "n_words_relative": int(strong.sum())}
                        if strong.sum() > X_cs.shape[1] + 2:
                            m_rel = ols_hc3(dmg[strong] / e_base[strong],
                                            X_cs[strong].assign(freq_z=zscore(fz[strong])))
                            row |= {"beta_freq_relative": m_rel.params["freq_z"],
                                    "p_freq_relative": m_rel.pvalues["freq_z"]}
                        rows.append(row)
        return pd.DataFrame(rows)

    # ── 8. token-count strata ─────────────────────────────────────────
    def token_strata(self) -> pd.DataFrame:
        rows = []
        for key in [k for k in self.keys() if self.is_boot(k)]:
            layers = self.bands(key[0])["all"]
            for k in self.cfg.TOKEN_STRATA:
                m = self.tok_stratum == k
                row = {"model": self.model, "regime": key[0], "policy": key[1],
                       "family": self.meta[key]["family"],
                       "token_stratum": f">={k}" if k == max(self.cfg.TOKEN_STRATA) else str(k),
                       "n_words": int((m & self.y).sum()), "n_nonwords": int((m & ~self.y).sum())}
                if row["n_words"] and row["n_nonwords"]:
                    row["base_auc"] = float(np.nanmean(self.curve(self.base, "frozen", f"auc_tok{k}")[layers, 0]))
                    row.update(summarize(self.delta(key, "frozen", f"auc_tok{k}", layers)))
                rows.append(row)
        return pd.DataFrame(rows)

    # ── 9. matched HF/LF pairs (H7a, H7b) ─────────────────────────────
    def matched_pairs(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        HF/LF words optimally matched (Hungarian) within n_token strata on length,
        OrthoN and bigram frequency (caliper per covariate), never on human accuracy.
        H7a: base evidence HF - LF.  H7b (total effect, primary): selectivity =
        damage(LF) - damage(HF) (> 0: LF damaged more), sign-flip permutation test.
        The direct effect at equal baseline evidence (OLS intercept on the pair's
        baseline gap, HC3) is reported with the same causal caveat as H2; it
        extrapolates when the baseline gap is far from 0 (mean_base_gap).
        """
        covs = self.covariates
        ok = self.y & np.isin(self.fg, ["high", "low"])
        for c in covs:
            ok &= self.t[c].notna().to_numpy()
        Z = np.column_stack([self.t[c].to_numpy(float) for c in covs]) if covs else np.zeros((len(self.t), 0))
        if covs:      # standardised on the HF+LF pool that is being matched
            sd = Z[ok].std(0)
            Z = (Z - Z[ok].mean(0)) / np.where(sd > 0, sd, 1.0)
        pairs = []
        for k in np.unique(self.ntok[ok]):
            hi = np.flatnonzero(ok & (self.fg == "high") & (self.ntok == k))
            lo = np.flatnonzero(ok & (self.fg == "low") & (self.ntok == k))
            if not len(hi) or not len(lo):
                continue
            cost = np.sqrt(((Z[hi][:, None, :] - Z[lo][None, :, :]) ** 2).sum(-1))
            r, c = linear_sum_assignment(cost)
            for i, j in zip(r, c):
                if np.all(np.abs(Z[hi[i]] - Z[lo[j]]) <= self.cfg.MATCH_CALIPER_SD):
                    pairs.append((hi[i], lo[j]))
        if not pairs:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        P = np.array(pairs)
        hf, lf = P[:, 0], P[:, 1]
        pairs_df = pd.DataFrame({"pair": np.arange(len(P)), "hf_stimulus": self.t.stimulus.to_numpy()[hf],
                                 "lf_stimulus": self.t.stimulus.to_numpy()[lf], "n_tokens": self.ntok[hf]})
        bal = []
        allh, alll = ok & (self.fg == "high"), ok & (self.fg == "low")
        for c in covs + ["log_freq", "n_tokens"]:
            x = self.t[c].to_numpy(float) if c != "n_tokens" else self.ntok.astype(float)
            sd = np.sqrt((np.nanvar(x[allh]) + np.nanvar(x[alll])) / 2) or 1.0
            bal.append({"model": self.model, "covariate": c,
                        "smd_before": (np.nanmean(x[allh]) - np.nanmean(x[alll])) / sd,
                        "smd_after": (np.nanmean(x[hf]) - np.nanmean(x[lf])) / sd, "n_pairs": len(P)})

        res = []
        for regime, label, keys in [(CONTROL, NO_REUSE, [self.base])] + self._frequency_families():
            readouts = ["frozen"] + (["behavior"] if all("behavior" in self.arrays(k) for k in keys) else [])
            band_regime = self.cfg.primary_regime.name if regime == CONTROL else regime
            for ro in readouts:
                for band, layers in (self.bands(band_regime).items() if ro == "frozen" else [("behavior", [0])]):
                    eb = self.evidence([self.base], ro, layers)
                    if label == NO_REUSE:
                        d, stat, adj = eb[hf] - eb[lf], "freq_effect_hf_minus_lf", {}
                    else:
                        dmg = eb - self.evidence(keys, ro, layers)
                        d, stat = dmg[lf] - dmg[hf], "selectivity_lf_minus_hf"
                        gap = eb[lf] - eb[hf]
                        ok = np.isfinite(d) & np.isfinite(gap)
                        if ok.mean() >= 1.0 - self.cfg.NONFINITE_MAX_FRACTION and ok.sum() > 3:
                            m = ols_hc3(d[ok], pd.DataFrame({"base_gap": gap[ok]}))
                            adj = {"selectivity_direct": m.params["const"], "p_direct": m.pvalues["const"],
                                   "mean_base_gap": float(gap.mean())}
                        else:
                            adj = {}
                    fin = np.isfinite(d)
                    if fin.mean() < 1.0 - self.cfg.NONFINITE_MAX_FRACTION or fin.sum() < 4:
                        continue
                    d = d[fin]
                    res.append({"model": self.model, "family_model": self.family, "regime": regime,
                                "family": label, "readout": ro, "band": band, "statistic": stat,
                                "n_pairs": len(d), "mean": float(d.mean()), "sd": float(d.std(ddof=1)),
                                "p_sign_flip": sign_flip_test(d, self.cfg.N_PERMUTATIONS, SEED), **adj})
        res = pd.DataFrame(res)
        if len(res):
            res = adjust(res, "p_sign_flip", ["regime", "readout", "band", "statistic"], "holm", "p_holm")
        return pairs_df, pd.DataFrame(bal), res

    # ── 10. single-layer fragility scan (H5, H5m) ─────────────────────
    def fragility(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        path = os.path.join(self.dir, "fragility.npz")
        if not os.path.exists(path):
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        z = np.load(path)
        readouts, base, logits = z["readouts"], z["base_logits"], z["logits"]
        W = self.W[: self.cfg.N_BOOTSTRAP_FRAGILITY + 1]
        metrics = ["auc_all", "d_all", "auc_hf", "auc_lf", "d_hf", "d_lf"] + [f"auc_tok{k}" for k in self.cfg.TOKEN_STRATA]
        channels = [(f"layer{int(r)}", base[ri], logits[:, ri], int(r)) for ri, r in enumerate(readouts)]
        channels.append(("behavior", z["base_behavior"], z["behavior"], self.L))
        rows, peaks = [], []
        for name, b_scores, p_scores, r in channels:
            b = {m: self.metric_rep(b_scores, m, W) for m in metrics}
            prim = "d_all" if b["auc_all"][0] >= self.cfg.CEILING_AUC else "auc_all"
            dmg_rep = np.full((self.L, W.shape[0]), np.nan)
            for t in range(1, self.L):
                d = {m: self.metric_rep(p_scores[t], m, W) - b[m] for m in metrics}
                d["sel_auc"], d["sel_d"] = d["auc_hf"] - d["auc_lf"], d["d_hf"] - d["d_lf"]
                dmg_rep[t] = -d[prim]
                row = {"model": self.model, "family_model": self.family, "readout": name,
                       "readout_layer": r, "substituted_layer": t, "depth": t / max(self.L - 1, 1),
                       "answer_mass_ratio": (float(np.nanmedian(z["behavior_mass"][t])
                                                   / np.nanmedian(z["base_behavior_mass"]))
                                             if name == "behavior" else np.nan),
                       "rope_mismatch": bool(z["rope_mismatch"][t]), "primary_metric": prim,
                       "after_readout": t > r,
                       "max_abs_logit_change": float(np.nanmax(np.abs(p_scores[t] - b_scores)))}
                for m, v in d.items():
                    s = summarize(v)
                    row.update({f"delta_{m}": s["estimate"], f"delta_{m}_ci_low": s["ci_low"],
                                f"delta_{m}_ci_high": s["ci_high"], f"delta_{m}_p": s["p_value"]})
                row["delta_primary"] = row[f"delta_{prim}"]
                row["neg_control_pass"] = (not row["after_readout"]
                                           or row["max_abs_logit_change"] <= self.cfg.NEG_CONTROL_ATOL)
                rows.append(row)
            # Peak fragility depth over substitutions at or before the readout, with an
            # item-bootstrap percentile CI (argmax recomputed in every replicate).  The
            # primary peak (H5) is taken over the ELIGIBLE layers the sharing policies act
            # on: over all layers the peak is all but guaranteed at layer 1 (reading
            # layer 0's K/V, the largest perturbation in the network), so that peak is
            # reported as descriptive only.
            ec = max(self.ec(self.cfg.primary_regime.name), 1)
            peak_row = {"model": self.model, "family_model": self.family, "readout": name,
                        "readout_layer": r, "primary_metric": prim, "first_eligible_layer": ec}
            for tag, lo in (("", ec), ("_all_layers", 1)):
                ts = np.arange(lo, min(r, self.L - 1) + 1)
                if not len(ts):
                    continue
                arg = ts[np.argmax(np.where(np.isfinite(dmg_rep[ts]), dmg_rep[ts], -np.inf), axis=0)]
                depth = arg / max(self.L - 1, 1)
                peak_row |= {f"peak_layer{tag}": int(arg[0]), f"peak_depth{tag}": float(depth[0]),
                             f"peak_depth{tag}_ci_low": float(np.percentile(depth[1:], 2.5)),
                             f"peak_depth{tag}_ci_high": float(np.percentile(depth[1:], 97.5)),
                             f"peak_damage{tag}": float(dmg_rep[arg[0], 0])}
            if "peak_depth" in peak_row:
                peaks.append(peak_row)
        df = pd.DataFrame(rows)
        # H5m (G4): item damage from substitution in the first half of depth, read out
        # at the final layer, regressed on token count with frequency and covariates.
        mech = []
        ri = int(np.flatnonzero(readouts == self.L - 1)[0])
        early = [t for t in range(1, self.L) if t / (self.L - 1) <= 0.5]
        words = self.word_mask()
        X = pd.DataFrame({"n_tokens_c": self.ntok[words] - self.ntok[words].mean(),
                          "log_freq_z": zscore(self.t.loc[words, "log_freq"].to_numpy())}
                         | {f"{c}_z": zscore(self.t.loc[words, c].to_numpy()) for c in self.covariates})
        for name, b_scores, p_scores, sc in (("final_layer", base[ri], logits[:, ri], self.scale["frozen"][self.L - 1]),
                                             ("behavior", z["base_behavior"], z["behavior"], self.scale["behavior"][0])):
            dmg = np.nanmean((b_scores[None, :] - p_scores[early]) / sc, axis=0)[words]
            f = np.isfinite(dmg)
            if f.mean() < 1.0 - self.cfg.NONFINITE_MAX_FRACTION:
                continue
            m = ols_hc3(dmg[f], X[f])
            mech.append({"model": self.model, "family_model": self.family, "readout": name,
                         "substituted_layers": f"1-{max(early)}", "n_words": int(f.sum()),
                         "beta_ntok": m.params["n_tokens_c"], "p_ntok": m.pvalues["n_tokens_c"],
                         "beta_freq": m.params["log_freq_z"], "p_freq": m.pvalues["log_freq_z"]})
        return df, pd.DataFrame(peaks), pd.DataFrame(mech)

    # ── 11. probe validity (RQ8a, RQ8b) ───────────────────────────────
    def reliability(self) -> pd.DataFrame:
        frames = []
        for sub in [self.dir] + [os.path.join(self.dir, r.name) for r in self.cfg.REGIMES]:
            if os.path.isdir(sub):
                frames += [pd.read_csv(os.path.join(sub, f)) for f in sorted(os.listdir(sub))
                           if f.endswith("_reliability.csv")]
        if not frames:
            return pd.DataFrame()
        df = pd.concat(frames, ignore_index=True)
        df["regime"] = df["regime"].fillna(CONTROL)
        idx = ["model", "regime", "policy", "layer"]
        acc = (df[df.probe != "linear_mdl"].groupby(idx + ["probe"], dropna=False)
               [["test_auc", "test_accuracy", "control_train_accuracy"]].mean().reset_index())
        lin = acc[acc.probe == "linear"].set_index(idx)["test_auc"]
        best_mlp = acc[acc.probe != "linear"].groupby(idx, dropna=False)["test_auc"].max()
        gap = (best_mlp - lin).rename("mlp_minus_linear_auc").reset_index()
        mdl = (df[df.probe == "linear_mdl"].groupby(idx, dropna=False)[["mdl_bits", "mdl_compression"]]
               .mean().reset_index())
        return acc.merge(gap, on=idx, how="left").merge(mdl, on=idx, how="left")

    def static_gain(self) -> pd.DataFrame:
        """Contextual gain = AUC(no_reuse layer, retrained) - AUC(non-contextual baseline), test items."""
        path = os.path.join(self.dir, "static_baseline.npz")
        if not os.path.exists(path):
            return pd.DataFrame()
        z = np.load(path)
        static = self.metric_rep(z["logits"], "auc_all", self.W)
        base = self.curve(self.base, "retrained", "auc_all")
        # peak chosen on VALIDATION AUC, so the test-set gain at the peak is not inflated
        peak = int(np.nanargmax(np.nanmean(self.arrays(self.base)["auc_val_seed"], 0)))
        rows = []
        for li in range(self.L):
            rows.append({"model": self.model, "family_model": self.family, "layer": li,
                         "is_peak_layer": li == peak, "static_auc": float(static[0]),
                         "static_mdl_compression": float(z["mdl_compression"]),
                         "layer_auc": float(base[li, 0]), **summarize(base[li] - static)})
        return pd.DataFrame(rows)

    # ── 12. generative alignment (H6) and behavioural alignment (H8c) ──
    def downstream(self, effects: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        path = os.path.join(self.dir, "downstream.csv")
        e_all = effects[effects.band == "all"]
        rows = []
        df = pd.DataFrame()
        if os.path.exists(path):
            df = pd.read_csv(path)
            df["regime"] = df["regime"].fillna(CONTROL)
            base = df[df.policy == NO_REUSE].iloc[0]
            df["delta_log_ppl"] = np.log(df.wikitext2_ppl) - np.log(base.wikitext2_ppl)
            df["delta_lambada_acc"] = df.lambada_acc - base.lambada_acc
            df["delta_lambada_nll"] = df.lambada_target_nll - base.lambada_target_nll
            df["generative_floor"] = df.wikitext2_ppl >= self.cfg.GENERATIVE_FLOOR_PPL_RATIO * base.wikitext2_ppl
            j = df[df.policy != NO_REUSE].merge(e_all[["regime", "policy", "delta_primary"]],
                                                on=["regime", "policy"], how="inner")
            for col in ("delta_log_ppl", "delta_lambada_acc", "delta_lambada_nll"):
                row = self._alignment("H6", col, -j["delta_primary"].to_numpy(), j[col].to_numpy(), j["k"].to_numpy())
                if row:
                    rows.append(row)
        b = e_all[e_all.delta_behavior_auc.notna() & (e_all.answer_mass_ratio >= self.cfg.BEHAVIOR_MASS_MIN_RATIO)]
        row = self._alignment("H8c", "delta_behavior_auc", b.delta_primary.to_numpy(),
                              b.delta_behavior_auc.to_numpy(), b.k.to_numpy())
        if row:
            rows.append(row)
        return df, pd.DataFrame(rows)

    def _alignment(self, label: str, outcome: str, x: np.ndarray, y: np.ndarray, k: np.ndarray) -> Dict:
        """
        Run-level alignment of representational damage with an outcome.  Runs differ in
        the number of substituted layers k, and both quantities grow with k, so the
        pooled Spearman correlation is positive by dose alone (descriptive only).  The
        primary statistic is the within-k partial rank correlation with a stratified
        permutation p-value: does a run that damages lexical representations MORE THAN
        OTHER RUNS OF THE SAME DOSE also damage the outcome more?
        """
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < self.cfg.MIN_ALIGNMENT_RUNS or len(np.unique(x[ok])) < 2 or len(np.unique(y[ok])) < 2:
            return {}
        rho_pool, p_pool = stats.spearmanr(x[ok], y[ok])
        rho, p, n_used, n_strata = within_k_spearman(x, y, k, self.cfg.N_PERMUTATIONS, SEED)
        if n_used < self.cfg.MIN_ALIGNMENT_RUNS:
            rho, p = np.nan, np.nan
        return {"model": self.model, "family_model": self.family, "alignment": label, "outcome": outcome,
                "n_runs": int(ok.sum()), "rho_pooled_descriptive": float(rho_pool), "p_pooled_descriptive": float(p_pool),
                "rho_within_k": rho, "p_within_k": p, "n_runs_within_k": n_used, "n_k_strata": n_strata}

    # ── 13. dose-response (H1d) ───────────────────────────────────────
    def dose_response(self, effects: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        H1d test points share source distance 1 (CLA-2 heads): no_reuse (0 layers),
        the uniform-random family mean at every k, and primary `full`.  `full` at
        reuse factor 4 (sources 1-3 layers back) confounds dose with distance and is
        reported as a point but excluded from the test.
        """
        P = self.cfg.primary_regime.name
        e = effects[effects.band == "all"]
        pts = [{"source": NO_REUSE, "n_substituted": 0, "kv_memory_fraction": 1.0, "mean_source_distance": 0.0,
                "delta_primary": 0.0, "delta_auc_all": 0.0, "in_test": True}]
        uni = e[(e.regime == P) & (e.family == UNIFORM)]
        for k, g in uni.groupby("k"):
            pts.append({"source": f"{UNIFORM}@k{k}", "n_substituted": int(k),
                        "kv_memory_fraction": float(g.kv_memory_fraction.mean()),
                        "mean_source_distance": float(g.mean_source_distance.mean()),
                        "delta_primary": float(g.delta_primary.mean()),
                        "delta_auc_all": float(g.delta_auc_all.mean()), "in_test": True})
        for r in self.cfg.REGIMES:
            g = e[(e.regime == r.name) & (e.policy == FULL)]
            if len(g) and self.ec(r.name) == self.ec(P):
                pts.append({"source": f"{FULL}[{r.name}]", "n_substituted": int(g.k.iloc[0]),
                            "kv_memory_fraction": float(g.kv_memory_fraction.iloc[0]),
                            "mean_source_distance": float(g.mean_source_distance.iloc[0]),
                            "delta_primary": float(g.delta_primary.iloc[0]),
                            "delta_auc_all": float(g.delta_auc_all.iloc[0]),
                            "in_test": bool(g.mean_source_distance.iloc[0] == 1.0)})
        pts = pd.DataFrame(pts).assign(model=self.model, family_model=self.family)
        t = pts[pts.in_test]
        rho, p = exact_spearman_upper(t.n_substituted.to_numpy(float), -t.delta_primary.to_numpy())
        test = pd.DataFrame([{"model": self.model, "family_model": self.family, "n_points": len(t),
                              "spearman_rho": rho, "p_exact_one_sided": p}])
        return pts, test

    # ── driver ────────────────────────────────────────────────────────
    def run(self) -> Dict[str, pd.DataFrame]:
        logger.info(f"[analysis] {self.model}")
        res = {"layers": self.layer_table(), "bands": self.band_contrasts(),
               "behavior": self.behavior_table(), "run_effects": self.run_effects()}
        res["randomization"], res["randomization_support"] = self.randomization(res["run_effects"])
        res["contrasts"] = self.policy_contrasts()
        res["item_regression"] = self.item_regressions()
        res["token_strata"] = self.token_strata()
        res["matched_pairs"], res["matching_balance"], res["matched_tests"] = self.matched_pairs()
        res["fragility"], res["fragility_peaks"], res["fragility_mechanism"] = self.fragility()
        res["reliability"], res["static_gain"] = self.reliability(), self.static_gain()
        res["downstream"], res["alignment"] = self.downstream(res["run_effects"])
        res["dose_points"], res["dose_test"] = self.dose_response(res["run_effects"])
        P = self.cfg.primary_regime.name
        p = os.path.join(self.dir, P, "calibration.csv")
        res["calibration"] = pd.read_csv(p).assign(model=self.model) if os.path.exists(p) else pd.DataFrame()
        neg = res["run_effects"].drop_duplicates(["regime", "policy"])
        res["negative_controls"] = neg[["model", "regime", "policy", "neg_control_layers",
                                        "neg_control_max_abs_logit", "neg_control_pass"]] if len(neg) else neg
        for name, df in res.items():
            if len(df):
                df.to_csv(os.path.join(self.out, f"{name}.csv"), index=False)
        res["info"] = pd.DataFrame([{"model": self.model, "family_model": self.family,
                                     "mixed_rope": bool(self.info["mixed_rope"]), "num_layers": self.L,
                                     "k0": self.k0}])
        return res


# ════════════════════════════════════════════════════════════════════════════
# STATISTICS: across models (models are the units) + decision rules
# ════════════════════════════════════════════════════════════════════════════

class CrossModelAnalysis:
    def __init__(self, cfg: ExperimentConfig, per_model: Dict[str, Dict[str, pd.DataFrame]]):
        self.cfg, self.per_model = cfg, per_model
        self.P = cfg.primary_regime.name
        self.out = cfg.PAPER_DIR
        info = self.cat("info").set_index("model")
        self.models = list(info.index)
        self.family = info["family_model"]
        self.mixed_rope = info["mixed_rope"].astype(bool)

    def cat(self, name: str) -> pd.DataFrame:
        frames = [r[name] for r in self.per_model.values() if name in r and len(r[name])]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def need(self, n_pool: int) -> int:
        return int(math.ceil(self.cfg.DECISION_MIN_MODEL_FRACTION * n_pool))

    # ── models-as-units tests ─────────────────────────────────────────
    def models_as_units(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        bands = self.cat("bands")
        if bands.empty:
            return pd.DataFrame(), pd.DataFrame()
        rows, fam_rows = [], []
        for (regime, fam, ro, band, metric), g in bands.groupby(["regime", "family", "readout", "band", "metric"]):
            v = g["estimate"].dropna().to_numpy()
            # n = 5 cannot reach p < .05 (minimum two-sided p = 0.0625): require MIN_MODELS_WILCOXON
            p = (stats.wilcoxon(v).pvalue if v.size >= self.cfg.MIN_MODELS_WILCOXON and np.any(v != 0)
                 else np.nan)
            rows.append({"regime": regime, "family": fam, "readout": ro, "band": band, "metric": metric,
                         "descriptive_only": metric in DESCRIPTIVE_METRICS, "n_models": v.size, "median_estimate": float(np.median(v)) if v.size else np.nan,
                         "n_negative": int((v < 0).sum()), "n_positive": int((v > 0).sum()),
                         "p_wilcoxon_models": p})
            # M1: one value per model family (mean over its models)
            fm = g.groupby("family_model")["estimate"].mean()
            fam_rows.append({"regime": regime, "family": fam, "readout": ro, "band": band, "metric": metric,
                             **{f"mean_{f}": float(v) for f, v in fm.items()},
                             "n_families_negative": int((fm < 0).sum()), "n_families": int(fm.size)})
        df = adjust(pd.DataFrame(rows), "p_wilcoxon_models", ["regime", "readout", "metric"], "holm", "p_holm")
        return df, pd.DataFrame(fam_rows)

    def combined_randomization(self) -> pd.DataFrame:
        """
        Cross-model percentile-rank test per contrast: Stouffer statistic of the
        per-model mid-p values with its exact randomisation null over the models'
        reference sets (+ LOFO and excluding mixed-RoPE models).  Valid under
        exchangeability of each selected map with its null draws under H0 (see
        `randomization_test`).
        """
        r, sup = self.cat("randomization"), self.cat("randomization_support")
        if r.empty or sup.empty:
            return pd.DataFrame()
        supports = {k: g.z.to_numpy() for k, g in sup.groupby(["contrast", "band", "statistic", "model"])}
        rows = []
        for (contrast, band, stat), g in r[r.is_primary_k].groupby(["contrast", "band", "statistic"]):
            g = g[(g.null_space_size >= 2) & g.mid_p_upper.notna()]   # a 1-set null is the policy itself
            if g.empty:
                continue

            def stouffer(models: List[str]) -> Tuple[float, float]:
                h = g[g.model.isin(models)]
                return stouffer_randomization(stats.norm.isf(h.mid_p_upper.to_numpy()),
                                              [supports[(contrast, band, stat, m)] for m in h.model],
                                              self.cfg.N_PERMUTATIONS, SEED)

            Z, p = stouffer(list(g.model))
            row = {"contrast": contrast, "band": band, "statistic": stat, "n_models": len(g),
                   "stouffer_z": Z, "p_combined": p,
                   "n_models_upper_sig": int((g.p_upper < self.cfg.ALPHA).sum()),
                   "n_models_lower_sig": int((g.p_lower < self.cfg.ALPHA).sum()),
                   "median_null_space": float(g.null_space_size.median()),
                   "n_exhaustive": int(g.exhaustive.sum())}
            for fam in sorted(self.family.unique()):
                row[f"lofo_{fam}_z"], row[f"lofo_{fam}_p"] = stouffer(list(g.model[g.family_model != fam]))
            row["excl_mixed_rope_z"], row["excl_mixed_rope_p"] = stouffer(
                [m for m in g.model if not self.mixed_rope[m]])
            rows.append(row)
        df = pd.DataFrame(rows)
        return adjust(df, "p_combined", ["band", "statistic"], "holm", "p_holm")

    def fragility_profiles(self) -> float:
        """Mean pairwise Spearman correlation of final-layer fragility profiles over depth."""
        fr = self.cat("fragility")
        if fr.empty:
            return np.nan
        grid = np.linspace(0, 1, 21)
        profiles = {}
        for m, g in fr[fr.readout != "behavior"].groupby("model"):
            g = g[g.readout_layer == g.readout_layer.max()].sort_values("depth")
            profiles[m] = np.interp(grid, g["depth"].to_numpy(), -g["delta_primary"].to_numpy())
        names = list(profiles)
        rhos = [stats.spearmanr(profiles[a], profiles[b])[0] for i, a in enumerate(names) for b in names[i + 1:]]
        return float(np.nanmean(rhos)) if rhos else np.nan

    # ── pre-specified decision rules ─────────────────────────────────
    def _subsets(self) -> Dict[str, List[str]]:
        subsets = {f"lofo_{f}": [m for m in self.models if self.family[m] != f] for f in sorted(self.family.unique())}
        subsets["excl_mixed_rope"] = [m for m in self.models if not self.mixed_rope[m]]
        return subsets

    def _count_verdict(self, hits: pd.Series, models: List[str], directional: Optional[pd.Series] = None) -> str:
        h = hits[hits.index.isin(models)]
        need = self.need(len(models))
        if len(h) < need:
            return "insufficient data"
        if directional is None:
            return "supported" if int(h.sum()) >= need else "not supported"
        d = directional[directional.index.isin(models)]
        if int(h.sum()) >= need:
            return "supported (A > B)"
        if int(d.sum()) >= need:
            return "supported (B > A)"
        return "not supported"

    def _wilcoxon_verdict(self, v: pd.Series, models: List[str]) -> Tuple[str, float]:
        x = v[v.index.isin(models)].dropna().to_numpy()
        if x.size < self.cfg.MIN_MODELS_WILCOXON:
            return "insufficient data", np.nan
        p = float(stats.wilcoxon(x).pvalue) if np.any(x != 0) else 1.0
        return ("supported" if np.median(x) > 0 and p < self.cfg.ALPHA else "not supported"), p

    def decision_rules(self, combined: pd.DataFrame, alignment_rho: float) -> pd.DataFrame:
        a, rows = self.cfg.ALPHA, []
        subsets = self._subsets()

        def add_count(h, rule, hits, detail="", lower=None):
            hits = hits.astype(bool)
            verdict = self._count_verdict(hits, self.models, lower)
            sens = {k: self._count_verdict(hits, v, lower) for k, v in subsets.items()}
            rows.append({"hypothesis": h, "rule": rule, "models_meeting_rule": int(hits.sum()),
                         "models_evaluated": len(hits), "models_required": self.need(len(self.models)),
                         "verdict": verdict, "robust_lofo": all(v == verdict for k, v in sens.items()
                                                                if k.startswith("lofo")),
                         "verdict_excl_mixed_rope": sens["excl_mixed_rope"], "detail": detail})

        def add_wilcoxon(h, rule, v, detail=""):
            verdict, p = self._wilcoxon_verdict(v, self.models)
            sens = {k: self._wilcoxon_verdict(v, m)[0] for k, m in subsets.items()}
            rows.append({"hypothesis": h, "rule": rule, "models_meeting_rule": int((v > 0).sum()),
                         "models_evaluated": int(v.notna().sum()), "models_required": self.cfg.MIN_MODELS_WILCOXON,
                         "verdict": verdict, "robust_lofo": all(x == verdict for k, x in sens.items()
                                                                if k.startswith("lofo")),
                         "verdict_excl_mixed_rope": sens["excl_mixed_rope"],
                         "detail": f"median rho = {v.median():.3f}, Wilcoxon p = {p:.4g}. {detail}"})

        def per_model(df, fn) -> pd.Series:
            return df.groupby("model").apply(fn, include_groups=False) if len(df) else pd.Series(dtype=bool)

        bands, beh = self.cat("bands"), self.cat("behavior")
        reg, mt, peaks = self.cat("item_regression"), self.cat("matched_tests"), self.cat("fragility_peaks")
        mech, sg, rel = self.cat("fragility_mechanism"), self.cat("static_gain"), self.cat("reliability")
        al, dose, con = self.cat("alignment"), self.cat("dose_test"), self.cat("contrasts")

        def lowers(g):
            return bool(((g.estimate < 0) & (g.p_holm < a) & g.ci_excludes_zero).any())

        # H1 / H1x
        for r in self.cfg.REGIMES:
            for ro in READOUTS:
                b = bands[(bands.regime == r.name) & (bands.family == FULL) & (bands.readout == ro)
                          & (bands.band != "all") & bands.is_primary_metric] if len(bands) else bands
                if b.empty:
                    continue
                tag = "H1" if r.role == "primary" else "H1x"
                add_count(f"{tag}[{r.name}] ({ro})", "full lowers the primary metric (AUC, or d at ceiling) "
                          "vs no_reuse in >=1 band (Holm, item bootstrap, CI excludes 0)", per_model(b, lowers),
                          f"bands at ceiling: {int(b.at_ceiling.sum())}/{len(b)}")

        # H1d
        if len(dose):
            d = dose.set_index("model")
            add_count("H1d", "damage increases with the number of substituted layers "
                      "(exact one-sided Spearman p < alpha)", (d.spearman_rho > 0) & (d.p_exact_one_sided < a),
                      f"median rho = {d.spearman_rho.median():.3f}")

        # H2 (change score, tokenisation-controlled, primary), with floor exclusion
        if len(reg):
            def h2(regime, family="full"):
                r = reg[(reg.regime == regime) & (reg.family == family) & (reg.readout == "frozen")
                        & (reg.band == "all") & (reg.freq_measure == "hal_log_freq")]
                return r[~r.at_floor].set_index("model"), r
            r, r_all = h2(self.P)
            if len(r_all):
                lm = reg[(reg.regime == self.P) & (reg.family == FULL) & (reg.readout == "frozen")
                         & (reg.band == "all") & (reg.freq_measure == "lm_logprob") & ~reg.at_floor]
                def n_sig(beta, p):
                    if beta not in r.columns:
                        return "n/a"
                    return f"{int(((r[beta] < 0) & (r[p] < a)).sum())}/{int(r[beta].notna().sum())}"
                add_count("H2", "lower-frequency words lose more evidence (change score, n_tokens and orthographic "
                          "covariates controlled, no baseline term; HC3, frozen, band 'all'; floor models not "
                          "evaluable)", (r.beta_freq < 0) & (r.p_freq < a),
                          f"at floor: {int(r_all.at_floor.sum())}; total effect (no n_tokens): "
                          f"{n_sig('beta_freq_total', 'p_freq_total')}; relative damage: "
                          f"{n_sig('beta_freq_relative', 'p_freq_relative')}; direct effect at equal baseline "
                          f"(ANCOVA): {n_sig('beta_freq_direct', 'p_freq_direct')}; "
                          f"LM log-prob measure: {int(((lm.beta_freq < 0) & (lm.p_freq < a)).sum())}/{len(lm)}; "
                          f"median attenuation by n_tokens {np.nanmedian(r.attenuation_by_ntok):.2f}")
            for regime in [x.name for x in self.cfg.REGIMES if x.role == "sensitivity"]:
                rs, rs_all = h2(regime)
                if len(rs_all):
                    add_count(f"H5x[{regime}]", "exemption-free full: n_token slope of word damage > 0 "
                              "(change score, HC3)", (rs.beta_ntok > 0) & (rs.p_ntok < a),
                              f"frequency slope < 0 in {int(((rs.beta_freq < 0) & (rs.p_freq < a)).sum())}")

        # H3a, H3c, C1, C2: directional, item bootstrap
        if len(con):
            c = con[(con.readout == "frozen") & (con.band == "all") & con.is_primary_metric]
            for label, g in c.groupby("contrast"):
                g = g.set_index("model")
                sig = (g.p_holm < a) & g.ci_excludes_zero
                add_count(label, "band 'all' primary-metric difference A - B (Holm, item bootstrap, CI excludes 0); "
                          "directional", sig & (g.estimate > 0),
                          f"A = {g.policy_a.iloc[0]}, B = {g.policy_b.iloc[0]}", lower=sig & (g.estimate < 0))

        # H3b, H3d, H4: combined randomisation tests
        if len(combined):
            cr = combined[(combined.band == "all") & (combined.statistic == "delta_primary")]
            for _, g in cr.iterrows():
                if g.p_holm < a:
                    verdict = "supported (better than null)" if g.stouffer_z > 0 else "supported (worse than null)"
                else:
                    verdict = "not supported"
                lofo = [(g[c], g[c[:-2] + "_z"]) for c in cr.columns if c.startswith("lofo_") and c.endswith("_p")]
                if verdict == "not supported":
                    robust = all(not (p < a) for p, _ in lofo)
                else:
                    robust = all(p < a and np.sign(z) == np.sign(g.stouffer_z) for p, z in lofo)
                if not np.isfinite(g.excl_mixed_rope_p):
                    excl = "insufficient data"
                elif g.excl_mixed_rope_p < a:
                    excl = "supported (better than null)" if g.excl_mixed_rope_z > 0 else "supported (worse than null)"
                else:
                    excl = "not supported"
                note = (" SANITY CHECK, not a finding: the null replaces the adjacent head by any eligible "
                        "head, so `full` is expected to win." if g.contrast.startswith("H4") else
                        " SANITY CHECK, not a finding: the searched map passed an output-similarity acceptance "
                        "check that the random-pair null draws did not." + (
                            "" if self.cfg.KVSHARER_THRESHOLD_VERIFIED else
                            " KVSharer-STYLE search: the acceptance threshold was not verified against the "
                            "released KVSharer configuration.") if g.contrast.startswith("H3e") else "")
                rows.append({"hypothesis": g.contrast, "rule": "Stouffer-combined percentile-rank test of the "
                             "selected map against random null target sets (mid-p, exact null over the "
                             "models' reference sets, Holm over contrasts; sensitivity columns unadjusted). "
                             "Valid under exchangeability of the deterministically selected map with the "
                             "null draws under H0, which the design does not guarantee; conditional on the "
                             "test items." + note,
                             "models_meeting_rule": int(g.n_models_upper_sig + g.n_models_lower_sig),
                             "models_evaluated": int(g.n_models), "models_required": np.nan,
                             "verdict": verdict, "robust_lofo": bool(robust), "verdict_excl_mixed_rope": excl,
                             "detail": f"Z = {g.stouffer_z:.2f}, p_holm = {g.p_holm:.4g}; per-model "
                                       f"upper/lower sig = {int(g.n_models_upper_sig)}/{int(g.n_models_lower_sig)}; "
                                       f"median null space = {g.median_null_space:.0f}"})

        # H5, H5m
        if len(peaks):
            pk = peaks[peaks.readout != "behavior"]
            pk = pk[pk.readout_layer == pk.groupby("model").readout_layer.transform("max")].set_index("model")
            add_count("H5", f"single-layer fragility peaks at normalised depth <= {self.cfg.H5_EARLY_DEPTH} "
                      "among the ELIGIBLE layers the policies act on (final-layer frozen readout)",
                      pk.peak_depth <= self.cfg.H5_EARLY_DEPTH,
                      f"mean pairwise Spearman of profiles = {alignment_rho:.3f}; median eligible-range peak "
                      f"depth {pk.peak_depth.median():.2f}; median all-layer peak depth (descriptive, "
                      f"dominated by layer 1 reading layer 0) {pk.peak_depth_all_layers.median():.2f}")
        if len(mech):
            m = mech[mech.readout == "final_layer"].set_index("model")
            add_count("H5m", "early-substitution word damage increases with n_tokens (HC3)",
                      (m.beta_ntok > 0) & (m.p_ntok < a))

        # H6, H8c
        if len(al):
            h6r = al[(al.alignment == "H6") & (al.outcome == "delta_log_ppl")].set_index("model")
            add_wilcoxon("H6", "per-model WITHIN-k partial Spearman(representational damage, delta log PPL) "
                         "over runs > 0 (Wilcoxon over models); the dose k is held fixed because both "
                         "quantities grow with the number of substituted layers", h6r.rho_within_k,
                         f"pooled (dose-confounded, descriptive) median rho = "
                         f"{h6r.rho_pooled_descriptive.median():.3f}")
            h8r = al[al.alignment == "H8c"].set_index("model")
            add_wilcoxon("H8c", "per-model WITHIN-k partial Spearman(delta representational, delta behavioural "
                         "AUC) over runs > 0 (Wilcoxon over models)", h8r.rho_within_k,
                         f"pooled (dose-confounded, descriptive) median rho = "
                         f"{h8r.rho_pooled_descriptive.median():.3f}")

        # H7a, H7b
        if len(mt):
            for ro in ("frozen", "behavior"):
                band = "all" if ro == "frozen" else "behavior"
                base = mt[(mt.family == NO_REUSE) & (mt.readout == ro) & (mt.band == band)].set_index("model")
                if len(base):
                    add_count(f"H7a ({ro})", "matched HF > LF evidence in no_reuse (sign-flip permutation)",
                              (base["mean"] > 0) & (base.p_sign_flip < a),
                              f"median pairs = {base.n_pairs.median():.0f}")
            full = mt[(mt.regime == self.P) & (mt.family == FULL) & (mt.readout == "frozen") & (mt.band == "all")]
            floor = set(reg.loc[(reg.regime == self.P) & (reg.family == FULL) & reg.at_floor, "model"]) if len(reg) else set()
            full = full[~full.model.isin(floor)].set_index("model")
            full = full.reindex(columns=full.columns.union(["selectivity_direct", "p_direct", "mean_base_gap"]))
            if len(full):
                add_count("H7b", "on matched pairs, full damages LF more than HF (total-effect selectivity > 0, "
                          "sign-flip; floor models not evaluable)", (full["mean"] > 0) & (full.p_sign_flip < a),
                          f"direct effect at equal baseline > 0 & p < alpha in "
                          f"{int(((full.selectivity_direct > 0) & (full.p_direct < a)).sum())}/{len(full)}; "
                          f"median baseline gap = {full.mean_base_gap.median():.2f}")

        # H8a, H8b
        if len(beh):
            b0 = beh[beh.metric == "abs_auc_all"].set_index("model")
            valid = (b0.ci_low > 0.5) & (b0.answer_mass_median >= self.cfg.BEHAVIOR_MIN_BASE_MASS)
            add_count("H8a", f"few-shot behavioural lexical-decision AUC > 0.5 (bootstrap CI) and no_reuse puts "
                      f">= {self.cfg.BEHAVIOR_MIN_BASE_MASS} probability on the answer tokens", valid,
                      f"median AUC = {b0.estimate.median():.3f}, median answer mass = "
                      f"{b0.answer_mass_median.median():.3f}")
            b1 = beh[(beh.regime == self.P) & (beh.family == FULL) & (beh.metric == "auc_all")].set_index("model")
            lost = b1[~b1.format_retained.astype(bool)].index
            b1 = b1[b1.index.isin(valid[valid].index) & ~b1.index.isin(lost)]
            add_count("H8b", f"full lowers behavioural AUC (Holm, CI excludes 0); evaluable only in models passing "
                      f"H8a whose answer mass under full is >= {self.cfg.BEHAVIOR_MASS_MIN_RATIO} x no_reuse",
                      (b1.estimate < 0) & (b1.p_holm < a) & b1.ci_excludes_zero,
                      f"answer format lost under full in: {', '.join(lost) or 'none'}")

        # RQ8a, RQ8b
        if len(sg):
            pk = sg[sg.is_peak_layer].set_index("model")
            add_count("RQ8a", "no_reuse peak layer exceeds the non-contextual baseline (bootstrap CI > 0)",
                      pk.ci_low > 0, f"median static AUC = {pk.static_auc.median():.3f}, median gain = "
                                     f"{pk.estimate.median():.3f}")
        if len(rel):
            lin = rel[(rel.policy == NO_REUSE) & (rel.probe == "linear")]
            ok = per_model(lin, lambda g: bool((g.mlp_minus_linear_auc.median() <= self.cfg.LINEAR_ADEQUACY_TOL)))
            add_count("RQ8b", f"median (best MLP - linear) test AUC <= {self.cfg.LINEAR_ADEQUACY_TOL} at the "
                      "reliability layers", ok,
                      f"median control-label training accuracy (linear) = {lin.control_train_accuracy.median():.3f}")
        return pd.DataFrame(rows)

    # ── figures ───────────────────────────────────────────────────────
    def _save(self, fig, name):
        fig.tight_layout()
        fig.savefig(os.path.join(self.out, f"{name}.png"), dpi=self.cfg.DPI, bbox_inches="tight")
        fig.savefig(os.path.join(self.out, f"{name}.pdf"), bbox_inches="tight")
        plt.close(fig)

    def fig_layerwise(self):
        lt = self.cat("layers")
        for m, g in lt.groupby("model"):
            g = g[(g.regime == self.P) | (g.policy == NO_REUSE)]
            fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
            for ax, ro in zip(axes, READOUTS):
                for pol, h in g[g.readout == ro].groupby("policy"):
                    ax.plot(h.layer, h.auc_all, lw=2.5 if pol in (NO_REUSE, FULL) else 1.2, label=pol)
                info = self.per_model[m]["info"].iloc[0]
                ec = exempt_cutoff(int(info.num_layers), self.cfg.primary_regime.exempt_fraction)
                ax.axvspan(-0.5, ec - 0.5, color="0.9", zorder=0)
                ax.set(xlabel="layer", ylabel="AUC (words vs nonwords)", title=f"{m}: {ro} readout")
                ax.grid(alpha=0.3)
            axes[1].legend(fontsize=7, ncol=2)
            self._save(fig, f"fig_layerwise_{safe_name(m)}")

    def fig_randomization(self):
        e = self.cat("run_effects")
        if e.empty:
            return
        e = e[(e.regime == self.P) & (e.band == "all")]
        fams = [f for f in self.cfg.GATED_FAMILIES if (e.family == f).any()]
        if not fams:
            return
        fig, axes = plt.subplots(1, len(fams), figsize=(4 * len(fams), 0.5 * len(self.models) + 2), sharey=True)
        axes = np.atleast_1d(axes)
        for ax, fam in zip(axes, fams):
            for i, m in enumerate(self.models):
                k0 = self.per_model[m]["info"].k0.iloc[0]
                g = e[(e.model == m) & (e.k == k0)]
                null = g[g.family == DEPTH_PREFIX + fam].delta_primary
                ax.scatter(null, np.full(len(null), i), s=8, color="0.6", alpha=0.6)
                obs = g[g.family == fam].delta_primary
                ax.scatter(obs, np.full(len(obs), i), s=40, color="C3", marker="D")
            ax.axvline(0, color="k", lw=0.8)
            ax.set(title=f"{fam} vs depth-matched null", xlabel="delta primary metric (band all, frozen)")
        axes[0].set(yticks=range(len(self.models)), yticklabels=self.models)
        self._save(fig, "fig_randomization")

    def fig_fragility(self):
        fr = self.cat("fragility")
        if fr.empty:
            return
        fig, axes = plt.subplots(1, 3, figsize=(20, 5))
        for m, g in fr.groupby("model"):
            rep = g[g.readout != "behavior"]
            rep = rep[rep.readout_layer == rep.readout_layer.max()].sort_values("depth")
            axes[0].plot(rep.depth, rep.delta_primary, label=m)
            axes[1].plot(rep.depth, rep.delta_sel_d, label=m)
            beh = g[g.readout == "behavior"].sort_values("depth")
            axes[2].plot(beh.depth, beh.delta_auc_all, label=m)
        axes[0].set(ylabel="delta primary metric at final layer", title="Single-layer KV substitution fragility")
        axes[1].set(ylabel="delta d(HF) - delta d(LF)",
                    title="Frequency selectivity of fragility (Cohen's d; AUC version is ceiling-biased)")
        axes[2].set(ylabel="delta behavioural AUC", title="Behavioural lexical-decision fragility")
        for ax in axes:
            ax.set_xlabel("normalised depth of the substituted layer")
            ax.axhline(0, color="k", lw=0.8)
            ax.grid(alpha=0.3)
        axes[0].legend(fontsize=7)
        self._save(fig, "fig_fragility")

    def fig_dose(self):
        d = self.cat("dose_points")
        if d.empty:
            return
        fig, ax = plt.subplots(figsize=(8, 6))
        for m, g in d.groupby("model"):
            g = g.sort_values("kv_memory_fraction")
            ax.plot(g.kv_memory_fraction, g.delta_primary, marker="o", label=m)
        ax.axhline(0, color="k", lw=0.8)
        ax.set(xlabel="analytic KV-cache memory fraction (cached layers / layers)",
               ylabel="delta primary metric (band all, frozen)", title="Dose-response of lexical damage")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
        self._save(fig, "fig_dose_response")

    def fig_alignment(self):
        e = self.cat("run_effects")
        ds = self.cat("downstream")
        if e.empty:
            return
        e = e[e.band == "all"]
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        if len(ds):
            j = ds.merge(e[["model", "regime", "policy", "delta_primary"]], on=["model", "regime", "policy"])
            for m, g in j.groupby("model"):
                axes[0].scatter(-g.delta_primary, g.delta_log_ppl, s=14, label=m)
        axes[0].set(xlabel="representational damage (-delta primary)", ylabel="delta log PPL",
                    title="H6: representation vs generation (pooled over k; test is within k)")
        b = e[e.delta_behavior_auc.notna()]
        for m, g in b.groupby("model"):
            axes[1].scatter(g.delta_primary, g.delta_behavior_auc, s=14, label=m)
        axes[1].set(xlabel="delta representational primary metric", ylabel="delta behavioural AUC",
                    title="H8c: representation vs behaviour (pooled over k; test is within k)")
        for ax in axes:
            ax.grid(alpha=0.3)
        axes[1].legend(fontsize=7)
        self._save(fig, "fig_alignment")

    def fig_token_strata(self):
        ts = self.cat("token_strata")
        if ts.empty or "estimate" not in ts:
            return
        ts = ts[(ts.regime == self.P) & (ts.family == FULL)]
        strata = list(dict.fromkeys(ts.token_stratum))
        fig, ax = plt.subplots(figsize=(9, 5))
        width = 0.8 / max(ts.model.nunique(), 1)
        for i, (m, g) in enumerate(ts.groupby("model")):
            g = g.set_index("token_stratum").reindex(strata)
            ax.bar(np.arange(len(strata)) + i * width, g.estimate, width, label=m)
        ax.axhline(0, color="k", lw=0.8)
        ax.set(xticks=np.arange(len(strata)) + 0.4, xticklabels=strata, xlabel="subword tokens",
               ylabel="delta token-matched AUC (full vs no_reuse)",
               title="Damage within token-count strata (words vs nonwords of equal length in tokens)")
        ax.legend(fontsize=7)
        self._save(fig, "fig_token_strata")

    def _latex(self, df: pd.DataFrame, name: str, caption: str):
        with open(os.path.join(self.out, f"{name}.tex"), "w") as f:
            f.write("\\begin{table}[t]\n\\centering\n\\small\n"
                    + df.to_latex(index=False, escape=True, float_format="%.3g")
                    + f"\\caption{{{caption}}}\n\\label{{tab:{name}}}\n\\end{{table}}\n")

    def run(self):
        for name in ["layers", "bands", "behavior", "run_effects", "randomization", "randomization_support",
                     "contrasts",
                     "item_regression", "token_strata", "matched_tests", "matching_balance", "fragility",
                     "fragility_peaks", "fragility_mechanism", "reliability", "static_gain", "downstream",
                     "alignment", "dose_points", "dose_test", "calibration", "negative_controls"]:
            df = self.cat(name)
            if len(df):
                df.to_csv(os.path.join(self.out, f"all_{name}.csv"), index=False)
        mu, fam = self.models_as_units()
        if len(mu):
            mu.to_csv(os.path.join(self.out, "models_as_units.csv"), index=False)
            fam.to_csv(os.path.join(self.out, "family_level.csv"), index=False)
        combined = self.combined_randomization()
        if len(combined):
            combined.to_csv(os.path.join(self.out, "randomization_combined.csv"), index=False)
        r = self.cat("randomization")
        if len(r):        # disclosure: per-model resolution of every randomisation test (min mid-p = 0.5 / N)
            r = r[r.is_primary_k & (r.band == "all") & (r.statistic == "delta_primary")]
            r.pivot_table(index="model", columns="contrast", values="null_space_size", aggfunc="first").to_csv(
                os.path.join(self.out, "null_space_sizes.csv"))
        e = self.cat("run_effects")
        if len(e):        # pilot costing: wall-clock per model and tier
            e = e[e.band == "all"]
            e.groupby(["model", "tier"]).seconds.agg(["count", "sum", "mean"]).rename(
                columns={"count": "n_runs", "sum": "total_seconds", "mean": "mean_seconds"}).to_csv(
                os.path.join(self.out, "timing_summary.csv"))
        rules = self.decision_rules(combined, self.fragility_profiles())
        rules.to_csv(os.path.join(self.out, "decision_rules.csv"), index=False)
        if len(rules):
            self._latex(rules[["hypothesis", "models_meeting_rule", "models_evaluated", "verdict", "robust_lofo"]],
                        "decision_rules", "Pre-specified decision rules evaluated on all models "
                        "(robust\\_lofo: verdict unchanged when any one model family is left out).")
        if len(mu):
            prim = mu[(mu.regime == self.P) & (mu.readout == "frozen") & mu.metric.isin(["auc_all", "d_all"])]
            self._latex(prim[["family", "band", "metric", "n_models", "median_estimate", "n_negative", "p_holm"]],
                        "models_as_units", "Change vs no-reuse with models as units "
                        "(Wilcoxon signed-rank, Holm over policy x band).")
        self.fig_layerwise()
        self.fig_randomization()
        self.fig_fragility()
        self.fig_dose()
        self.fig_alignment()
        self.fig_token_strata()
        if len(rules):
            logger.info("\n" + rules[["hypothesis", "models_meeting_rule", "models_evaluated",
                                      "verdict", "robust_lofo"]].to_string(index=False))


# ════════════════════════════════════════════════════════════════════════════
# EXPERIMENT
# ════════════════════════════════════════════════════════════════════════════

REFERENCES = [
    "Brandon et al. (2024) Reducing Transformer KV Cache Size with Cross-Layer Attention. NeurIPS.",
    "Liu et al. (2024) MiniCache: KV Cache Compression in Depth Dimension for LLMs. NeurIPS.",
    "Yang et al. (2024) KVSharer: Efficient Inference via Layer-Wise Dissimilar KV Cache Sharing. arXiv:2410.18517.",
    "Wu & Tu (2024) Layer-Condensed KV Cache for Efficient Inference of LLMs. ACL.",
    "Wu, Wu & Tu (2025) A Systematic Study of Cross-Layer KV Sharing for Efficient LLM Inference. NAACL (short).",
    "Sun et al. (2024) You Only Cache Once: Decoder-Decoder Architectures for Language Models. NeurIPS.",
    "Kaplan et al. (2025) From Tokens to Words: On the Inner Lexicon of LLMs. ICLR.",
    "Feucht et al. (2024) Token Erasure as a Footprint of Implicit Vocabulary Items in LLMs. EMNLP.",
    "Brown et al. (2020) Language Models are Few-Shot Learners. NeurIPS.",
    "Balota et al. (2007) The English Lexicon Project. Behavior Research Methods.",
    "Kornblith et al. (2019) Similarity of Neural Network Representations Revisited. ICML.",
    "Song et al. (2012) Feature Selection via Dependence Maximization (unbiased HSIC). JMLR.",
    "Davari et al. (2023) Reliability of CKA as a Similarity Measure in Deep Learning. ICLR.",
    "Hewitt & Liang (2019) Designing and Interpreting Probes with Control Tasks. EMNLP.",
    "Voita & Titov (2020) Information-Theoretic Probing with Minimum Description Length. EMNLP.",
    "Pimentel et al. (2020) Information-Theoretic Probing for Linguistic Structure. ACL.",
    "Belinkov (2022) Probing Classifiers: Promises, Shortcomings, and Advances. Computational Linguistics.",
    "Hautus (1995) Corrections for extreme proportions in d'. Behavior Research Methods.",
    "Efron & Tibshirani (1993) An Introduction to the Bootstrap. Chapman & Hall.",
    "Phipson & Smyth (2010) Permutation P-values Should Never Be Zero. Stat. Appl. Genet. Mol. Biol.",
    "Lancaster (1961) Significance Tests in Discrete Distributions. JASA.",
    "Stouffer et al. (1949) The American Soldier, Vol. 1. Princeton University Press.",
    "Lord (1967) A Paradox in the Interpretation of Group Comparisons. Psychological Bulletin.",
    "Pearl (2001) Direct and Indirect Effects. UAI.",
    "Holm (1979) A Simple Sequentially Rejective Multiple Test Procedure. Scand. J. Statistics.",
    "Benjamini & Hochberg (1995) Controlling the False Discovery Rate. JRSS-B.",
    "Wilcoxon (1945) Individual Comparisons by Ranking Methods. Biometrics Bulletin.",
    "MacKinnon & White (1985) Heteroskedasticity-consistent covariance matrix estimators (HC3). J. Econometrics.",
    "Stuart (2010) Matching Methods for Causal Inference. Statistical Science.",
    "Xiao et al. (2024) Efficient Streaming Language Models with Attention Sinks. ICLR.",
]


class Experiment:
    def __init__(self, cfg: ExperimentConfig, analysis_only: bool = False):
        self.cfg, self.analysis_only = cfg, analysis_only

    def _items_manifest(self) -> Dict:
        def sha(path):
            with open(path, "rb") as f:
                return hashlib.sha256(f.read()).hexdigest()
        return {"filter_model_ids": sorted(self.cfg.ITEM_FILTER_MODEL_IDS),
                "max_items_per_class": self.cfg.MAX_ITEMS_PER_CLASS,
                "stimulus_template": self.cfg.STIMULUS_TEMPLATE, "task_template": self.cfg.task_template(),
                "words_sha256": sha(self.cfg.WORDS_PATH), "nonwords_sha256": sha(self.cfg.NONWORDS_PATH)}

    def _items(self, build: bool = False) -> pd.DataFrame:
        """
        The item table is a design constant shared by every model: it is built once,
        filtered by the tokenizers of ALL models in ITEM_FILTER_MODEL_IDS, and stored
        with a manifest.  A cached table whose manifest differs (other models, an item
        cap from a pilot, other templates or inputs) is refused, never reused.  A job
        that runs a subset of the models never builds the table itself, so parallel
        per-model jobs cannot race or diverge: build it first (KV_BUILD_ITEMS_ONLY=1).
        """
        path = os.path.join(self.cfg.OUTPUT_DIR, "items.csv")
        meta_path = os.path.join(self.cfg.OUTPUT_DIR, "items_manifest.json")
        expected = self._items_manifest()
        if os.path.exists(path) and not build:
            if not os.path.exists(meta_path):
                raise RuntimeError(f"{path} has no items_manifest.json; rebuild it with KV_BUILD_ITEMS_ONLY=1")
            with open(meta_path) as f:
                built = json.load(f)
            if built != expected:
                raise RuntimeError(f"{path} was built for {built}, not {expected}; use a separate KV_OUTPUT_DIR")
            return pd.read_csv(path)
        if os.path.exists(path):
            raise RuntimeError(f"{path} already exists; delete it deliberately before rebuilding")
        if sorted(m.model_id for m in self.cfg.MODELS) != expected["filter_model_ids"] and not build:
            raise RuntimeError("this job runs a subset of the models and no item table exists: build it first "
                               "with KV_BUILD_ITEMS_ONLY=1 and the full model list")
        words, nonwords = read_pools(self.cfg)
        pool = pd.concat([words, nonwords], ignore_index=True)
        valid, report = tokenizer_validity(self.cfg, pool["stimulus"].tolist())
        report.to_csv(os.path.join(self.cfg.OUTPUT_DIR, "item_exclusions.csv"), index=False)
        logger.info(f"Tokenizer filter (all models, both templates): kept {int(valid.sum())}/{len(valid)} strings\n"
                    + report.to_string(index=False))
        keep = set(pool.loc[valid, "stimulus"])
        items = build_items(self.cfg, words[words.stimulus.isin(keep)], nonwords[nonwords.stimulus.isin(keep)])
        items.to_csv(path + ".tmp", index=False)
        with open(meta_path + ".tmp", "w") as f:
            json.dump(expected, f, indent=2)
        os.replace(meta_path + ".tmp", meta_path)
        os.replace(path + ".tmp", path)
        return items

    def _prespecification(self) -> Dict:
        """
        SHA-256 of this script and of the analysis-relevant configuration, written
        once (first run) with a timestamp and checked on every later run.  Register
        these hashes (with the decision rules) on OSF before the full run: code-defined
        rules alone are not a pre-registration.
        """
        with open(os.path.abspath(__file__), "rb") as f:
            script_sha = hashlib.sha256(f.read()).hexdigest()
        run_scoped = {"OUTPUT_DIR", "RESULTS_DIR", "PAPER_DIR", "MODELS", "RESUME", "STATE_STORE", "STATE_DIR",
                      "WORDS_PATH", "NONWORDS_PATH", "DOWNSTREAM_MODELS"}
        config = json.dumps({k: str(v) for k, v in sorted(vars(self.cfg).items()) if k not in run_scoped})
        current = {"script_sha256": script_sha, "config_sha256": hashlib.sha256(config.encode()).hexdigest()}
        path = os.path.join(self.cfg.OUTPUT_DIR, "prespecification_manifest.json")
        if not os.path.exists(path):
            with open(path, "w") as f:
                json.dump(current | {"frozen_at": f"{datetime.now():%Y-%m-%d %H:%M:%S}"}, f, indent=2)
            return current | {"matches_frozen": True}
        with open(path) as f:
            frozen = json.load(f)
        ok = all(frozen[k] == v for k, v in current.items())
        if not ok:
            logger.warning(f"script or configuration differs from the manifest frozen at {frozen['frozen_at']}: "
                           f"results are NOT from the pre-specified analysis")
        return current | {"matches_frozen": ok, "frozen_at": frozen["frozen_at"]}

    def run(self):
        t0 = datetime.now()
        prespec = self._prespecification()
        items = self._items()
        splits = make_splits(items, self.cfg)
        per_model, failed = {}, {}
        for mc in self.cfg.MODELS:
            if not self.analysis_only:
                try:
                    ModelPipeline(self.cfg, mc, items, splits).run()
                except Exception as e:
                    logger.exception(f"pipeline failed for {mc.name}")
                    failed[mc.name] = f"pipeline: {e!r}"
                    free_memory()
                    continue
                free_memory()
            if not os.path.exists(os.path.join(self.cfg.model_dir(mc.name), "model_info.json")):
                continue
            try:
                per_model[mc.name] = ModelAnalysis(self.cfg, mc.name).run()
            except Exception as e:
                logger.exception(f"analysis failed for {mc.name}")
                failed[mc.name] = f"analysis: {e!r}"
        if per_model:
            try:
                CrossModelAnalysis(self.cfg, per_model).run()
            except Exception as e:
                logger.exception("cross-model analysis failed")
                failed["cross_model"] = repr(e)
        with open(os.path.join(self.cfg.OUTPUT_DIR, "metadata.json"), "w") as f:
            json.dump({"script": "KV_LDT_v12_2", "started": f"{t0:%Y-%m-%d %H:%M:%S}",
                       "finished": f"{datetime.now():%Y-%m-%d %H:%M:%S}",
                       "torch": torch.__version__, "transformers": transformers.__version__,
                       "compute_dtype": str(COMPUTE_DTYPE), "models_analysed": list(per_model),
                       "failures": failed,
                       "prespecification": prespec,
                       "inference_note": "all CIs and p-values are conditional on the fitted, seed-averaged probes; "
                                         "randomisation tests are conditional on the test items (item-level "
                                         "generalisation rests on the item-bootstrap contrasts); null-set "
                                         "tests are percentile-rank tests that assume exchangeability of "
                                         "the selected map with the null draws under H0; sel_auc is "
                                         "ceiling-biased and descriptive only; H6/H8c are tested within "
                                         "dose k",
                       "kvsharer_threshold_verified": self.cfg.KVSHARER_THRESHOLD_VERIFIED,
                       "config": {k: (v if isinstance(v, (int, float, str, bool, list, dict, type(None)))
                                      else str(v)) for k, v in vars(self.cfg).items() if k != "MODELS"},
                       "models": [vars(m) for m in self.cfg.MODELS],
                       "references": REFERENCES}, f, indent=2, default=str)
        logger.info(f"DONE in {datetime.now() - t0} -> {self.cfg.OUTPUT_DIR}")
        return per_model


def main():
    cfg = ExperimentConfig()
    add_file_log(cfg.OUTPUT_DIR)
    if os.environ.get("KV_MODELS"):
        wanted = {m.strip() for m in os.environ["KV_MODELS"].split(",") if m.strip()}
        cfg.MODELS = [m for m in cfg.MODELS if m.name in wanted]
        if not cfg.MODELS:
            raise SystemExit(f"No model matched KV_MODELS={os.environ['KV_MODELS']!r}")
    if os.environ.get("KV_MAX_ITEMS_PER_CLASS"):
        cfg.MAX_ITEMS_PER_CLASS = int(os.environ["KV_MAX_ITEMS_PER_CLASS"])
    if os.environ.get("KV_BUILD_ITEMS_ONLY", "").lower() in ("1", "true", "yes"):
        if os.environ.get("KV_MODELS"):
            raise SystemExit("build the item table with the full model list (unset KV_MODELS)")
        Experiment(cfg)._items(build=True)
        return None
    analysis_only = os.environ.get("KV_ANALYSIS_ONLY", "").lower() in ("1", "true", "yes")
    for path in (cfg.WORDS_PATH, cfg.NONWORDS_PATH):
        if not os.path.exists(path):
            raise SystemExit(f"Missing input file: {path}")
    return Experiment(cfg, analysis_only=analysis_only).run()


if __name__ == "__main__":
    main()
