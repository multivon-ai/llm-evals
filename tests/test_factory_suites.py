from pydantic import BaseModel

from multivon_eval import EvalSuite
from multivon_eval.evaluators.agent import (
    PlanQuality,
    TaskCompletion,
    ToolCallAccuracy,
    ToolCallNecessity,
    TrajectoryEfficiency,
)
from multivon_eval.evaluators.compliance import PIIEvaluator, SchemaEvaluator
from multivon_eval.evaluators.conversation import (
    ConversationCompleteness,
    ConversationRelevance,
    KnowledgeRetention,
    TurnConsistency,
)
from multivon_eval.evaluators.deterministic import ExactMatch, NotEmpty, ROUGE
from multivon_eval.evaluators.llm_judge import (
    AnswerAccuracy,
    Bias,
    Coherence,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
    Hallucination,
    Relevance,
    Summarization,
    Toxicity,
)


class DocSchema(BaseModel):
    vendor: str
    amount: float


def evaluator_types(suite):
    return {type(evaluator) for evaluator in suite._evaluators}


class TestEvalSuiteFactories:
    def test_for_rag(self):
        suite = EvalSuite.for_rag()
        assert evaluator_types(suite) == {
            NotEmpty,
            Faithfulness,
            Hallucination,
            ContextPrecision,
            ContextRecall,
            Relevance,
        }

    def test_for_agents(self):
        suite = EvalSuite.for_agents()
        assert evaluator_types(suite) == {
            ToolCallAccuracy,
            ToolCallNecessity,
            TrajectoryEfficiency,
            PlanQuality,
            TaskCompletion,
        }

    def test_for_support_bot(self):
        suite = EvalSuite.for_support_bot()
        assert evaluator_types(suite) == {
            NotEmpty,
            Faithfulness,
            Relevance,
            Coherence,
            Toxicity,
        }

    def test_for_summarization(self):
        suite = EvalSuite.for_summarization()
        assert evaluator_types(suite) == {
            NotEmpty,
            Faithfulness,
            Coherence,
            Relevance,
            Summarization,
        }

    def test_for_document_intelligence(self):
        suite = EvalSuite.for_document_intelligence(schema=DocSchema)
        assert evaluator_types(suite) == {
            NotEmpty,
            Faithfulness,
            AnswerAccuracy,
            SchemaEvaluator,
        }

    def test_for_regulated(self):
        suite = EvalSuite.for_regulated(schema=DocSchema)
        assert evaluator_types(suite) == {
            PIIEvaluator,
            NotEmpty,
            Faithfulness,
            Relevance,
            SchemaEvaluator,
        }

    def test_for_chatbot(self):
        suite = EvalSuite.for_chatbot()
        assert evaluator_types(suite) == {
            ConversationRelevance,
            KnowledgeRetention,
            TurnConsistency,
            ConversationCompleteness,
        }

    def test_for_classification(self):
        suite = EvalSuite.for_classification()
        assert evaluator_types(suite) == {
            NotEmpty,
            ExactMatch,
            AnswerAccuracy,
        }

    def test_for_coding(self):
        suite = EvalSuite.for_coding()
        assert isinstance(suite, EvalSuite)
        assert evaluator_types(suite) == {
            NotEmpty,
            ExactMatch,
            AnswerAccuracy,
            ROUGE,
        }

    def test_for_coding_custom_language(self):
        suite = EvalSuite.for_coding("TypeScript Eval", language="typescript")
        assert isinstance(suite, EvalSuite)
        assert suite.name == "TypeScript Eval"
        # Evaluator set is language-agnostic for now
        assert evaluator_types(suite) == {
            NotEmpty,
            ExactMatch,
            AnswerAccuracy,
            ROUGE,
        }

    def test_for_medical_default_jurisdiction(self):
        suite = EvalSuite.for_medical()
        assert isinstance(suite, EvalSuite)
        assert evaluator_types(suite) == {
            PIIEvaluator,
            NotEmpty,
            Faithfulness,
            AnswerAccuracy,
            Hallucination,
        }
        # PII evaluator should use hipaa by default
        pii_ev = next(e for e in suite._evaluators if isinstance(e, PIIEvaluator))
        assert "medical_record_number" in pii_ev._compiled

    def test_for_medical_gdpr_jurisdiction(self):
        suite = EvalSuite.for_medical("Clinical GDPR Eval", jurisdiction="gdpr")
        assert isinstance(suite, EvalSuite)
        pii_ev = next(e for e in suite._evaluators if isinstance(e, PIIEvaluator))
        # GDPR jurisdiction adds eu_vat; HIPAA-only keys like medical_record_number absent
        assert "eu_vat" in pii_ev._compiled
        assert "medical_record_number" not in pii_ev._compiled

    def test_for_legal(self):
        suite = EvalSuite.for_legal()
        assert isinstance(suite, EvalSuite)
        assert evaluator_types(suite) == {
            NotEmpty,
            Faithfulness,
            Hallucination,
            AnswerAccuracy,
            Bias,
        }

    def test_for_financial(self):
        suite = EvalSuite.for_financial()
        assert isinstance(suite, EvalSuite)
        assert evaluator_types(suite) == {
            NotEmpty,
            Faithfulness,
            Hallucination,
            AnswerAccuracy,
            PIIEvaluator,
        }
        # PII evaluator should use "all" jurisdiction
        pii_ev = next(e for e in suite._evaluators if isinstance(e, PIIEvaluator))
        # "all" includes both HIPAA and GDPR extras
        assert "medical_record_number" in pii_ev._compiled
        assert "eu_vat" in pii_ev._compiled

    def test_for_regulated_default_jurisdiction(self):
        suite = EvalSuite.for_regulated()
        assert isinstance(suite, EvalSuite)
        # Without schema: PIIEvaluator + NotEmpty + Faithfulness + Relevance
        assert evaluator_types(suite) == {
            PIIEvaluator,
            NotEmpty,
            Faithfulness,
            Relevance,
        }

    def test_for_regulated_gdpr_jurisdiction(self):
        suite = EvalSuite.for_regulated(jurisdiction="gdpr")
        assert isinstance(suite, EvalSuite)
        pii_ev = next(e for e in suite._evaluators if isinstance(e, PIIEvaluator))
        assert "eu_vat" in pii_ev._compiled
        assert "medical_record_number" not in pii_ev._compiled

    def test_factory_returns_eval_suite_type(self):
        """All factory methods must return EvalSuite instances."""
        factories = [
            EvalSuite.for_rag,
            EvalSuite.for_agents,
            EvalSuite.for_support_bot,
            EvalSuite.for_summarization,
            EvalSuite.for_chatbot,
            EvalSuite.for_classification,
            EvalSuite.for_coding,
            EvalSuite.for_legal,
            EvalSuite.for_financial,
            EvalSuite.for_medical,
        ]
        for factory in factories:
            result = factory()
            assert isinstance(result, EvalSuite), f"{factory.__name__} did not return EvalSuite"

    def test_factory_suites_have_evaluators(self):
        """All factory methods must produce at least one evaluator."""
        factories = [
            EvalSuite.for_rag,
            EvalSuite.for_agents,
            EvalSuite.for_support_bot,
            EvalSuite.for_summarization,
            EvalSuite.for_chatbot,
            EvalSuite.for_classification,
            EvalSuite.for_coding,
            EvalSuite.for_legal,
            EvalSuite.for_financial,
            EvalSuite.for_medical,
        ]
        for factory in factories:
            suite = factory()
            assert len(suite._evaluators) > 0, f"{factory.__name__} returned empty evaluator list"
