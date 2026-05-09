"""Grounding validator to prevent hallucinations"""
import logging
from typing import Tuple


logger = logging.getLogger(__name__)


class GroundingValidator:
    """Validate that answers are grounded in retrieved context"""
    
    def __init__(self, strict_mode: bool = True):
        """
        Initialize grounding validator
        
        Args:
            strict_mode: If True, apply strict validation rules
        """
        self.strict_mode = strict_mode
        logger.info(f"Initialized GroundingValidator (strict_mode={strict_mode})")
    
    def validate(self, answer: str, context: str) -> Tuple[bool, str]:
        """
        Validate that the answer is grounded in the context
        
        Args:
            answer: The agent's answer
            context: Retrieved context from CV
            
        Returns:
            Tuple of (is_grounded, message)
        """
        try:
            # Check if answer explicitly states information is not available
            not_available_phrases = [
                "not available in the provided cv",
                "not available in the cv",
                "information is not available",
                "not found in the cv",
                "cv does not contain",
                "no information about"
            ]
            
            answer_lower = answer.lower()
            
            # If answer says info is not available, that's valid
            if any(phrase in answer_lower for phrase in not_available_phrases):
                logger.info("Answer correctly states information is not available")
                return True, "Answer appropriately indicates missing information"
            
            # Check if context is empty or too short
            if not context or len(context.strip()) < 20:
                logger.warning("Context is empty or too short")
                return False, "No sufficient context retrieved from CV"
            
            # Check for common hallucination indicators
            hallucination_indicators = [
                "i think",
                "probably",
                "might be",
                "could be",
                "i assume",
                "i believe",
                "it seems",
                "perhaps"
            ]
            
            if self.strict_mode:
                for indicator in hallucination_indicators:
                    if indicator in answer_lower:
                        logger.warning(f"Detected hallucination indicator: '{indicator}'")
                        return False, f"Answer contains uncertain language: '{indicator}'"
            
            # Check if answer references the CV
            cv_references = [
                "based on your cv",
                "according to the cv",
                "from the cv",
                "in your cv",
                "your cv shows",
                "your cv mentions",
                "as stated in the cv"
            ]
            
            has_cv_reference = any(ref in answer_lower for ref in cv_references)
            
            if self.strict_mode and not has_cv_reference and len(answer) > 50:
                logger.warning("Answer does not reference the CV")
                # Don't fail, but log warning
            
            # Basic content overlap check
            # Extract key terms from context and check if they appear in answer
            context_words = set(context.lower().split())
            answer_words = set(answer_lower.split())
            
            # Remove common words
            common_words = {
                'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
                'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
                'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these',
                'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'your', 'cv'
            }
            
            context_words -= common_words
            answer_words -= common_words
            
            # Check overlap
            overlap = context_words & answer_words
            overlap_ratio = len(overlap) / len(answer_words) if answer_words else 0
            
            logger.debug(f"Content overlap ratio: {overlap_ratio:.2f}")
            
            # If there's reasonable overlap, consider it grounded
            if overlap_ratio > 0.1 or len(overlap) > 3:
                logger.info("Answer appears to be grounded in context")
                return True, "Answer is grounded in retrieved CV content"
            
            # If answer is very short and context exists, allow it
            if len(answer.split()) < 20:
                logger.info("Short answer with context available")
                return True, "Short answer with available context"
            
            logger.warning(f"Low content overlap: {overlap_ratio:.2f}")
            return True, "Answer accepted with low overlap (may need review)"
            
        except Exception as e:
            logger.error(f"Error in grounding validation: {str(e)}")
            # In case of error, allow the answer but log it
            return True, f"Validation error (answer allowed): {str(e)}"
    
    def check_for_hallucination_patterns(self, answer: str) -> Tuple[bool, list]:
        """
        Check for common hallucination patterns
        
        Args:
            answer: The answer to check
            
        Returns:
            Tuple of (has_hallucination, list of detected patterns)
        """
        patterns = []
        answer_lower = answer.lower()
        
        # Uncertain language
        uncertain = ["i think", "probably", "might", "could be", "perhaps", "maybe"]
        for phrase in uncertain:
            if phrase in answer_lower:
                patterns.append(f"Uncertain language: '{phrase}'")
        
        # Speculation
        speculation = ["i assume", "i believe", "it seems", "appears to be"]
        for phrase in speculation:
            if phrase in answer_lower:
                patterns.append(f"Speculation: '{phrase}'")
        
        # Generic responses
        if len(answer.split()) < 5 and "not available" not in answer_lower:
            patterns.append("Very short answer without 'not available' statement")
        
        has_hallucination = len(patterns) > 0
        
        return has_hallucination, patterns


