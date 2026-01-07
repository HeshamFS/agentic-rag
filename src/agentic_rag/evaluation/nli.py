"""
NLI (Natural Language Inference) verifier for fact checking.

Uses textual entailment to verify if claims in responses
are supported by the retrieved context.

Categories:
- ENTAILMENT: Claim is supported by context
- CONTRADICTION: Claim contradicts context
- NEUTRAL: Context doesn't address the claim
"""

import contextlib
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from agentic_rag.core.models import Chunk
from agentic_rag.core.protocols import Generator
from agentic_rag.evaluation.base import BaseEvaluator, EvaluationResult


class NLILabel(str, Enum):
    """NLI classification labels."""

    ENTAILMENT = "entailment"
    CONTRADICTION = "contradiction"
    NEUTRAL = "neutral"


class ClaimVerification(BaseModel):
    """Verification result for a single claim."""

    claim: str = Field(description="The claim being verified")
    label: NLILabel = Field(description="NLI label")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in the label")
    supporting_evidence: str | None = Field(default=None, description="Evidence if entailment")
    contradiction_evidence: str | None = Field(
        default=None, description="Evidence if contradiction"
    )


class NLIVerifierOutput(BaseModel):
    """Complete NLI verification output."""

    claims: list[ClaimVerification] = Field(
        default_factory=list, description="Individual claim results"
    )
    entailment_count: int = Field(default=0, description="Number of supported claims")
    contradiction_count: int = Field(default=0, description="Number of contradicted claims")
    neutral_count: int = Field(default=0, description="Number of neutral claims")
    overall_score: float = Field(ge=0.0, le=1.0, description="Overall verification score")
    has_hallucinations: bool = Field(default=False, description="Whether any contradictions found")


class NLIVerifier(BaseEvaluator):
    """
    NLI-based fact verification.

    Uses natural language inference to classify each claim
    in the response as entailed, contradicted, or neutral
    with respect to the context.
    """

    def __init__(
        self,
        generator: Generator,
        strict_mode: bool = False,
    ):
        """
        Initialize NLI verifier.

        Args:
            generator: LLM for NLI classification.
            strict_mode: If True, neutral counts as unsupported.
        """
        super().__init__("nli_verification")
        self._generator = generator
        self.strict_mode = strict_mode

    async def evaluate(
        self,
        query: str,
        response: str,
        contexts: list[Chunk],
        ground_truth: str | None = None,
    ) -> EvaluationResult:
        """
        Verify response claims using NLI.

        Args:
            query: User query (not used directly).
            response: Response to verify.
            contexts: Context to verify against.
            ground_truth: Not used.

        Returns:
            NLI verification result.
        """
        if not response:
            return EvaluationResult(
                metric_name=self.name,
                score=0.0,
                reasoning="Empty response",
            )

        if not contexts:
            return EvaluationResult(
                metric_name=self.name,
                score=0.0,
                reasoning="No context to verify against",
            )

        # Extract claims from response
        claims = await self._extract_claims(response)

        if not claims:
            return EvaluationResult(
                metric_name=self.name,
                score=1.0,
                reasoning="No verifiable claims in response",
            )

        # Verify each claim
        context_text = "\n\n".join(c.content for c in contexts)
        verifications = []

        for claim in claims:
            verification = await self._verify_claim(claim, context_text)
            verifications.append(verification)

        # Calculate statistics
        entailment_count = sum(1 for v in verifications if v.label == NLILabel.ENTAILMENT)
        contradiction_count = sum(1 for v in verifications if v.label == NLILabel.CONTRADICTION)
        neutral_count = sum(1 for v in verifications if v.label == NLILabel.NEUTRAL)

        # Calculate score
        if self.strict_mode:
            # Strict: only entailments count as positive
            score = entailment_count / len(claims)
        else:
            # Lenient: penalize contradictions, neutral is okay
            score = (entailment_count + 0.5 * neutral_count) / len(claims)

        return EvaluationResult(
            metric_name=self.name,
            score=score,
            details={
                "claims": [v.model_dump() for v in verifications],
                "entailment_count": entailment_count,
                "contradiction_count": contradiction_count,
                "neutral_count": neutral_count,
                "has_hallucinations": contradiction_count > 0,
            },
            reasoning=f"Verified {len(claims)} claims: {entailment_count} supported, "
            f"{contradiction_count} contradicted, {neutral_count} neutral",
        )

    async def _extract_claims(self, text: str) -> list[str]:
        """
        Extract verifiable claims from text.

        Args:
            text: Text to extract claims from.

        Returns:
            List of claims.
        """
        prompt = f"""Extract all factual claims from this text that can be verified.
Focus on specific facts, not opinions or hedged statements.
Output one claim per line, without numbering.

Text:
{text[:2000]}

Factual claims:"""

        response = await self._generator.generate_text(
            prompt=prompt,
            temperature=0.1,
            max_tokens=512,
        )

        claims = []
        for line in response.strip().split("\n"):
            line = line.strip().lstrip("0123456789.-•) ")
            if line and len(line) > 10:  # Filter very short lines
                claims.append(line)

        return claims[:20]  # Limit to 20 claims

    async def _verify_claim(
        self,
        claim: str,
        context: str,
    ) -> ClaimVerification:
        """
        Verify a single claim using NLI.

        Args:
            claim: Claim to verify.
            context: Context to verify against.

        Returns:
            Claim verification result.
        """
        prompt = f"""Determine if the claim is supported by the context using NLI.

Context:
{context[:3000]}

Claim: "{claim}"

Classify as:
- ENTAILMENT: The context provides evidence that supports this claim
- CONTRADICTION: The context provides evidence that contradicts this claim
- NEUTRAL: The context doesn't provide enough information to verify or refute

Output format:
LABEL: [ENTAILMENT/CONTRADICTION/NEUTRAL]
CONFIDENCE: [0.0-1.0]
EVIDENCE: [quote or paraphrase the relevant evidence, or "none"]"""

        response = await self._generator.generate_text(
            prompt=prompt,
            temperature=0.0,
            max_tokens=200,
        )

        # Parse response
        label = NLILabel.NEUTRAL
        confidence = 0.5
        evidence = None

        for line in response.split("\n"):
            line = line.strip()
            if line.startswith("LABEL:"):
                val = line.split(":", 1)[1].strip().lower()
                if "entailment" in val:
                    label = NLILabel.ENTAILMENT
                elif "contradiction" in val:
                    label = NLILabel.CONTRADICTION
                else:
                    label = NLILabel.NEUTRAL
            elif line.startswith("CONFIDENCE:"):
                try:
                    confidence = float(line.split(":", 1)[1].strip())
                    confidence = min(max(confidence, 0.0), 1.0)
                except ValueError:
                    pass
            elif line.startswith("EVIDENCE:"):
                evidence = line.split(":", 1)[1].strip()
                if evidence.lower() == "none":
                    evidence = None

        return ClaimVerification(
            claim=claim,
            label=label,
            confidence=confidence,
            supporting_evidence=evidence if label == NLILabel.ENTAILMENT else None,
            contradiction_evidence=evidence if label == NLILabel.CONTRADICTION else None,
        )

    async def verify_response(
        self,
        response: str,
        contexts: list[Chunk],
    ) -> NLIVerifierOutput:
        """
        Full NLI verification of a response.

        Args:
            response: Response to verify.
            contexts: Context to verify against.

        Returns:
            Complete NLI verification output.
        """
        result = await self.evaluate(
            query="",
            response=response,
            contexts=contexts,
        )

        details = result.details
        verifications = [ClaimVerification(**v) for v in details.get("claims", [])]

        return NLIVerifierOutput(
            claims=verifications,
            entailment_count=details.get("entailment_count", 0),
            contradiction_count=details.get("contradiction_count", 0),
            neutral_count=details.get("neutral_count", 0),
            overall_score=result.score,
            has_hallucinations=details.get("has_hallucinations", False),
        )


class BatchNLIVerifier(NLIVerifier):
    """
    Optimized NLI verifier that processes multiple claims in one call.
    """

    async def _verify_claims_batch(
        self,
        claims: list[str],
        context: str,
    ) -> list[ClaimVerification]:
        """
        Verify multiple claims in a single LLM call.

        Args:
            claims: Claims to verify.
            context: Context to verify against.

        Returns:
            List of claim verifications.
        """
        claims_text = "\n".join(f"{i + 1}. {claim}" for i, claim in enumerate(claims))

        prompt = f"""Verify each claim against the context using NLI.

Context:
{context[:3000]}

Claims:
{claims_text}

For each claim, output:
CLAIM_N: [ENTAILMENT/CONTRADICTION/NEUTRAL] | [0.0-1.0] | [brief evidence]

Example:
CLAIM_1: ENTAILMENT | 0.9 | "The context states that..."
CLAIM_2: CONTRADICTION | 0.8 | "The context says the opposite..."
CLAIM_3: NEUTRAL | 0.5 | none"""

        response = await self._generator.generate_text(
            prompt=prompt,
            temperature=0.0,
            max_tokens=100 * len(claims),
        )

        # Parse batch response
        verifications = []
        for i, claim in enumerate(claims):
            # Find corresponding line in response
            label = NLILabel.NEUTRAL
            confidence = 0.5
            evidence = None

            pattern = f"CLAIM_{i + 1}:"
            for line in response.split("\n"):
                if pattern in line:
                    parts = line.split("|")
                    if len(parts) >= 2:
                        label_part = parts[0].split(":", 1)[1].strip().lower()
                        if "entailment" in label_part:
                            label = NLILabel.ENTAILMENT
                        elif "contradiction" in label_part:
                            label = NLILabel.CONTRADICTION

                        with contextlib.suppress(ValueError):
                            confidence = float(parts[1].strip())

                        if len(parts) >= 3:
                            evidence = parts[2].strip()
                            if evidence.lower() == "none":
                                evidence = None
                    break

            verifications.append(
                ClaimVerification(
                    claim=claim,
                    label=label,
                    confidence=confidence,
                    supporting_evidence=evidence if label == NLILabel.ENTAILMENT else None,
                    contradiction_evidence=evidence if label == NLILabel.CONTRADICTION else None,
                )
            )

        return verifications


class HallucinationDetector:
    """
    Specialized detector for hallucinations in RAG responses.

    Focuses on identifying content that contradicts or
    is not grounded in the retrieved context.
    """

    def __init__(self, generator: Generator):
        """
        Initialize hallucination detector.

        Args:
            generator: LLM for detection.
        """
        self._generator = generator
        self._nli = NLIVerifier(generator, strict_mode=True)

    async def detect(
        self,
        response: str,
        contexts: list[Chunk],
    ) -> dict[str, Any]:
        """
        Detect hallucinations in response.

        Args:
            response: Response to check.
            contexts: Grounding context.

        Returns:
            Detection results with identified hallucinations.
        """
        result = await self._nli.evaluate(
            query="",
            response=response,
            contexts=contexts,
        )

        details = result.details
        hallucinations = []

        for claim_data in details.get("claims", []):
            if claim_data.get("label") == "contradiction":
                hallucinations.append(
                    {
                        "claim": claim_data.get("claim"),
                        "evidence": claim_data.get("contradiction_evidence"),
                    }
                )

        return {
            "has_hallucinations": len(hallucinations) > 0,
            "hallucination_count": len(hallucinations),
            "hallucinations": hallucinations,
            "overall_score": result.score,
            "total_claims_checked": len(details.get("claims", [])),
        }
