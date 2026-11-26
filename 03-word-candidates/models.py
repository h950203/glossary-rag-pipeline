"""
Data Transfer Objects (DTOs) and Schemas for Term_Mapping pipeline.
Follows Architecture/CODING_STANDARDS.md: strict dataclasses, no raw untyped dicts.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class SentenceRecord:
    """Represents a single sentence in a language pool."""
    sent_id: str
    lang: str
    text: str
    expected_partner_id: Optional[str] = None
    char_len: int = 0
    token_est: int = 0

    def __post_init__(self) -> None:
        if self.char_len == 0:
            self.char_len = len(self.text)
        if self.token_est == 0:
            self.token_est = len(self.text.split())


@dataclass
class CandidatePair:
    """A candidate sentence pair retrieved by Stage 2."""
    b_sent_id: str
    b_text: str
    score: float
    rank: int


@dataclass
class CandidateBundle:
    """Bundle containing one source A sentence and its Top-K candidate B sentences."""
    a_sent_id: str
    a_text: str
    expected_partner_id: Optional[str] = None
    top1_score: float = 0.0
    alignment_tier: str = "MATCHED"  # "MATCHED" | "GRAY_ZONE" | "UNMATCHED"
    is_aligned_candidate: bool = True
    candidates: List[CandidatePair] = field(default_factory=list)
    is_m_to_1_suspected: bool = False
    context_window_a_ids: List[str] = field(default_factory=list)
    context_window_a_texts: List[str] = field(default_factory=list)


@dataclass
class Stage2AlignmentMetrics:
    """Evaluation metrics for Stage 2 sentence alignment benchmark (with 3-tier partitioning)."""
    total_a: int = 0
    matched_ground_truth: int = 0
    unmatched_ground_truth: int = 0
    matched_tier_count: int = 0
    gray_zone_tier_count: int = 0
    unmatched_tier_count: int = 0
    gray_zone_true_positive_count: int = 0
    gray_zone_distractor_count: int = 0
    top1_correct: int = 0
    top5_correct: int = 0
    top1_recall: float = 0.0
    top5_recall: float = 0.0
    distractor_rejected: int = 0
    distractor_rejection_rate: float = 0.0
    avg_matched_top1_sim: float = 0.0
    avg_distractor_top1_sim: float = 0.0
    m_to_1_flagged_count: int = 0


@dataclass
class RawTermPair:
    """A single term pair extracted by Stage 3 LLM."""
    surface_a: str
    canonical_a: str
    surface_b: str
    canonical_b: str
    source_a_id: str
    source_b_id: str
    is_nested_subterm: bool = False


@dataclass
class Stage3ExtractionResult:
    """Output schema of Stage 3 span extraction for one A sentence."""
    source_a_id: str
    matched_b_ids: List[str] = field(default_factory=list)
    extracted_pairs: List[RawTermPair] = field(default_factory=list)


@dataclass
class MappingEdge:
    """An edge between term_a and term_b in the bipartite graph."""
    term_a: str
    term_b: str
    count: int
    term_sim: float
    sample_sentence_a: str
    sample_sentence_b: str
    raw_term_a: Optional[str] = None
    raw_term_b: Optional[str] = None


@dataclass
class ConceptSynset:
    """A clustered concept synset in the final bilingual glossary."""
    concept_id: str
    terms_a: List[str] = field(default_factory=list)
    terms_b: List[str] = field(default_factory=list)
    mapping_edges: List[MappingEdge] = field(default_factory=list)
    is_polysemous_sense: bool = False


@dataclass
class PolysemySense:
    """A distinct sense of a polysemous term."""
    sense_id: str
    concept_id: str
    mapped_terms: List[str] = field(default_factory=list)
    sense_description: str = ""
    sample_sentence_a: str = ""
    sample_sentence_b: str = ""


@dataclass
class PolysemyEntry:
    """A polysemous term and its decomposed senses in the polysemy glossary."""
    raw_term: str
    lang: str
    senses: List[PolysemySense] = field(default_factory=list)


@dataclass
class UnmappedSentence:
    """Record of a defective/unmapped sentence retaining uncovered multi-word compounds."""
    source_a_id: str
    source_b_id: Optional[str]
    a_text: str
    b_text: str
    labse_score: float
    uncovered_compounds: List[str] = field(default_factory=list)

