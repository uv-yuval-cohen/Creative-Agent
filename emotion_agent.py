import logging
import time
from typing import Dict, Any, Optional, Tuple
from config import Config
from utils import extract_text_from_response, retry_api_call, save_error_log
import anthropic


class EmotionAgent:
    """
    Emotion Agent for Context Enhancement
    Adds appropriate emotional depth and passion to context while maintaining accuracy
    """

    def __init__(self, session_dir: str):
        self.session_dir = session_dir
        self.logger = logging.getLogger(__name__)

        # Initialize Claude client
        try:
            self.client = anthropic.Anthropic(api_key=Config.ANTHROPIC_API_KEY)
            self.logger.info("Emotion agent initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize emotion agent: {e}")
            raise

        # Performance tracking
        self.enhancement_stats = {
            "total_enhancements": 0,
            "successful_enhancements": 0,
            "average_enhancement_time": 0.0,
            "enhancement_types": {
                "passion_boost": 0,
                "motivation_clarity": 0,
                "human_connection": 0,
                "emotional_depth": 0
            }
        }

    def enhance_context(self, raw_context: str, subject: str, enhancement_type: str = "balanced") -> Tuple[
        str, bool, Dict[str, Any]]:
        """Enhance context with appropriate emotional resonance"""
        start_time = time.time()

        try:
            # Use the config emotion prompt directly
            system_prompt = Config.AGENT_PROMPTS["emotion"].format(
                subject=subject,
                raw_context=raw_context
            )

            # Simple user message
            user_message = "Please enhance this context with minimal modifications."

            # Execute enhancement
            response, success = retry_api_call(
                self.client.messages.create,
                model=Config.MODEL_NAME,
                max_tokens=Config.MAX_TOKENS,
                temperature=0.6,
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}]
            )

            processing_time = time.time() - start_time

            if success and response:
                enhanced_context = extract_text_from_response(response)
                return enhanced_context, True, {"processing_time": processing_time}
            else:
                return raw_context, False, {"error": "Enhancement failed"}

        except Exception as e:
            return raw_context, False, {"error": str(e)}

    def _validate_enhancement_quality(self, original: str, enhanced: str, subject: str) -> float:
        """
        Validate the quality of emotional enhancement
        Returns quality score (0.0 to 1.0)
        """
        try:
            # Basic validation metrics
            length_ratio = len(enhanced) / len(original) if original else 1.0

            # Quality indicators
            quality_score = 0.5  # Base score

            # Length should be reasonably similar (minimal modifications)
            if 0.8 <= length_ratio <= 1.5:
                quality_score += 0.2

            # Should contain some emotional words
            emotional_indicators = [
                "passion", "exciting", "meaningful", "impact", "inspire",
                "potential", "transform", "valuable", "important", "connect",
                "achieve", "create", "develop", "build", "realize"
            ]

            emotional_count = sum(1 for word in emotional_indicators if word in enhanced.lower())
            if emotional_count > 0:
                quality_score += min(0.3, emotional_count * 0.1)

            return min(1.0, quality_score)

        except Exception as e:
            self.logger.error(f"Quality validation failed: {e}")
            return 0.5

    def minimal_modifications(self, raw_context: str, subject: str) -> Tuple[str, bool, Dict[str, Any]]:
        """
        Apply minimal emotional modifications to context
        Focus on subtle enhancements that preserve original meaning
        """
        return self.enhance_context(raw_context, subject, enhancement_type="balanced")

    def subject_appropriate_passion(self, raw_context: str, subject: str) -> Tuple[str, bool, Dict[str, Any]]:
        """
        Add subject-appropriate passionate elements to context
        Adapts emotional tone based on the type of project
        """
        try:
            # Determine appropriate passion level based on subject
            subject_lower = subject.lower()

            if any(keyword in subject_lower for keyword in ["business", "startup", "company", "market"]):
                enhancement_type = "motivation"
            elif any(keyword in subject_lower for keyword in ["art", "creative", "design", "music", "story"]):
                enhancement_type = "passion"
            elif any(keyword in subject_lower for keyword in ["social", "community", "help", "impact", "change"]):
                enhancement_type = "connection"
            elif any(keyword in subject_lower for keyword in ["tech", "software", "app", "platform", "system"]):
                enhancement_type = "balanced"
            else:
                enhancement_type = "depth"

            return self.enhance_context(raw_context, subject, enhancement_type)

        except Exception as e:
            self.logger.error(f"Subject-appropriate enhancement failed: {e}")
            return self.enhance_context(raw_context, subject, "balanced")

    def adaptive_emotional_enhancement(self, raw_context: str, subject: str,
                                       context_history: list = None) -> Tuple[str, bool, Dict[str, Any]]:
        """
        Adaptive enhancement that considers the emotional arc of the session
        Adjusts emotional tone based on progress and context history
        """
        try:
            # Analyze context history to determine emotional trajectory
            if context_history and len(context_history) > 0:
                # Later iterations might need more motivation/energy
                iteration_count = len(context_history)

                if iteration_count <= 2:
                    # Early iterations - build excitement
                    enhancement_type = "passion"
                elif iteration_count <= 5:
                    # Middle iterations - maintain motivation
                    enhancement_type = "motivation"
                else:
                    # Later iterations - focus on connection and depth
                    enhancement_type = "connection"
            else:
                # No history - use subject-appropriate enhancement
                return self.subject_appropriate_passion(raw_context, subject)

            return self.enhance_context(raw_context, subject, enhancement_type)

        except Exception as e:
            self.logger.error(f"Adaptive enhancement failed: {e}")
            return self.enhance_context(raw_context, subject, "balanced")

    def _update_enhancement_stats(self, enhancement_type: str, processing_time: float, success: bool):
        """Update performance statistics for enhancements"""
        try:
            self.enhancement_stats["total_enhancements"] += 1

            if success:
                self.enhancement_stats["successful_enhancements"] += 1

                # Update average processing time
                total_successful = self.enhancement_stats["successful_enhancements"]
                current_avg = self.enhancement_stats["average_enhancement_time"]
                self.enhancement_stats["average_enhancement_time"] = (
                                                                             (current_avg * (
                                                                                         total_successful - 1)) + processing_time
                                                                     ) / total_successful

                # Update enhancement type counts
                if enhancement_type == "passion":
                    self.enhancement_stats["enhancement_types"]["passion_boost"] += 1
                elif enhancement_type == "motivation":
                    self.enhancement_stats["enhancement_types"]["motivation_clarity"] += 1
                elif enhancement_type == "connection":
                    self.enhancement_stats["enhancement_types"]["human_connection"] += 1
                elif enhancement_type in ["depth", "balanced"]:
                    self.enhancement_stats["enhancement_types"]["emotional_depth"] += 1

        except Exception as e:
            self.logger.error(f"Failed to update enhancement stats: {e}")

    def get_enhancement_analytics(self) -> Dict[str, Any]:
        """Get analytics about emotion enhancement performance"""
        try:
            total = self.enhancement_stats["total_enhancements"]
            successful = self.enhancement_stats["successful_enhancements"]

            return {
                "total_enhancements": total,
                "successful_enhancements": successful,
                "success_rate": successful / total if total > 0 else 0.0,
                "average_processing_time": self.enhancement_stats["average_enhancement_time"],
                "enhancement_distribution": self.enhancement_stats["enhancement_types"],
                "most_used_type": max(
                    self.enhancement_stats["enhancement_types"].items(),
                    key=lambda x: x[1]
                )[0] if any(self.enhancement_stats["enhancement_types"].values()) else "none"
            }

        except Exception as e:
            self.logger.error(f"Failed to get enhancement analytics: {e}")
            return {"error": str(e)}

    def test_enhancement_quality(self, test_context: str, subject: str) -> Dict[str, Any]:
        """
        Test enhancement quality with a sample context
        Useful for debugging and optimization
        """
        try:
            self.logger.info("Running enhancement quality test")

            # Test all enhancement types
            test_results = {}

            for enhancement_type in ["balanced", "passion", "motivation", "connection", "depth"]:
                enhanced, success, metadata = self.enhance_context(test_context, subject, enhancement_type)

                test_results[enhancement_type] = {
                    "success": success,
                    "enhanced_length": len(enhanced) if enhanced else 0,
                    "original_length": len(test_context),
                    "length_ratio": len(enhanced) / len(test_context) if test_context and enhanced else 0,
                    "processing_time": metadata.get("processing_time", 0),
                    "quality_score": metadata.get("quality_score", 0)
                }

            return {
                "test_context_length": len(test_context),
                "subject": subject,
                "enhancement_results": test_results,
                "test_timestamp": time.time()
            }

        except Exception as e:
            self.logger.error(f"Enhancement quality test failed: {e}")
            return {"error": str(e)}

    def calibrate_emotional_intensity(self, context: str, subject: str,
                                      desired_intensity: float = 0.5) -> Tuple[str, bool, Dict[str, Any]]:
        """
        Calibrate emotional intensity based on desired level

        Args:
            context: Raw context to enhance
            subject: Project subject
            desired_intensity: 0.0 (minimal) to 1.0 (maximum emotional enhancement)

        Returns:
            (enhanced_context, success, metadata)
        """
        try:
            # Map intensity to enhancement approach
            if desired_intensity <= 0.2:
                # Very subtle enhancement
                enhancement_type = "balanced"
                temperature = 0.3
            elif desired_intensity <= 0.5:
                # Moderate enhancement
                enhancement_type = "motivation"
                temperature = 0.5
            elif desired_intensity <= 0.8:
                # Strong enhancement
                enhancement_type = "passion"
                temperature = 0.7
            else:
                # Maximum emotional enhancement
                enhancement_type = "connection"
                temperature = 0.8

            # Custom prompt for intensity calibration
            system_prompt = f"""You are enhancing context with a specific emotional intensity level.

            **Subject**: {subject}
            **Desired Emotional Intensity**: {desired_intensity} (0.0 = minimal, 1.0 = maximum)
            **Enhancement Focus**: {enhancement_type}

            **Instructions**:
            - Calibrate your emotional enhancement to match the desired intensity level
            - At low intensity: Make very subtle, minimal changes
            - At high intensity: Add more passionate, energetic language
            - Always preserve factual accuracy and core meaning
            - Adjust the emotional "volume" to match the specified level

            Please enhance the following context with the appropriate emotional intensity."""

            response, success = retry_api_call(
                self.client.messages.create,
                model=Config.MODEL_NAME,
                max_tokens=Config.MAX_TOKENS,
                temperature=temperature,
                system=system_prompt,
                messages=[{"role": "user", "content": context}]
            )

            if success and response:
                enhanced_context = extract_text_from_response(response)

                metadata = {
                    "desired_intensity": desired_intensity,
                    "enhancement_type": enhancement_type,
                    "temperature_used": temperature,
                    "intensity_category": "minimal" if desired_intensity <= 0.2 else
                    "moderate" if desired_intensity <= 0.5 else
                    "strong" if desired_intensity <= 0.8 else "maximum"
                }

                return enhanced_context, True, metadata
            else:
                return context, False, {"error": "Intensity calibration failed"}

        except Exception as e:
            self.logger.error(f"Emotional intensity calibration failed: {e}")
            return context, False, {"error": str(e)}

    def analyze_emotional_trajectory(self, context_history: list) -> Dict[str, Any]:
        """
        Analyze the emotional trajectory of a session
        Helps understand how emotional tone evolves through iterations
        """
        try:
            if not context_history:
                return {"error": "No context history provided"}

            trajectory_analysis = {
                "total_iterations": len(context_history),
                "emotional_indicators": [],
                "intensity_progression": [],
                "dominant_emotions": {}
            }

            # Analyze each iteration for emotional content
            for i, iteration in enumerate(context_history):
                context_text = str(iteration)

                # Count emotional indicators
                emotional_words = {
                    "excitement": ["exciting", "amazing", "incredible", "fantastic", "thrilling"],
                    "motivation": ["achieve", "accomplish", "succeed", "realize", "fulfill"],
                    "passion": ["passionate", "love", "devoted", "committed", "dedicated"],
                    "optimism": ["positive", "hopeful", "confident", "promising", "bright"],
                    "determination": ["determined", "persistent", "focused", "driven", "resolved"]
                }

                iteration_emotions = {}
                for emotion_type, words in emotional_words.items():
                    count = sum(1 for word in words if word in context_text.lower())
                    iteration_emotions[emotion_type] = count

                trajectory_analysis["emotional_indicators"].append(iteration_emotions)

                # Calculate overall intensity for this iteration
                total_emotional_words = sum(iteration_emotions.values())
                intensity = min(1.0, total_emotional_words / 10.0)  # Normalize to 0-1
                trajectory_analysis["intensity_progression"].append(intensity)

            # Calculate dominant emotions across all iterations
            all_emotions = {"excitement": 0, "motivation": 0, "passion": 0, "optimism": 0, "determination": 0}
            for iteration_emotions in trajectory_analysis["emotional_indicators"]:
                for emotion, count in iteration_emotions.items():
                    all_emotions[emotion] += count

            trajectory_analysis["dominant_emotions"] = all_emotions

            # Calculate trends
            intensities = trajectory_analysis["intensity_progression"]
            if len(intensities) > 1:
                trajectory_analysis["intensity_trend"] = intensities[-1] - intensities[0]
                trajectory_analysis["average_intensity"] = sum(intensities) / len(intensities)
            else:
                trajectory_analysis["intensity_trend"] = 0.0
                trajectory_analysis["average_intensity"] = intensities[0] if intensities else 0.0

            return trajectory_analysis

        except Exception as e:
            self.logger.error(f"Emotional trajectory analysis failed: {e}")
            return {"error": str(e)}