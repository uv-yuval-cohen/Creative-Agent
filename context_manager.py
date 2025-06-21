import json
import time
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from config import Config
from utils import save_json, load_json, get_session_filepath, extract_text_from_response, retry_api_call, \
    validate_agent_response_format, save_error_log
import anthropic
import os


class ContextManager:
    """
    Manages context and collects data
    """

    def __init__(self, session_dir: str, subject: str):
        self.session_dir = session_dir
        self.subject = subject
        self.logger = logging.getLogger(__name__)
        self.plan = "No plan defined yet"

        # Context management
        self.context_history = []
        self.context = "This is  agentic ideation process for achieving the subject: " + subject
        self.compression_count = 0

        # Analytics data
        self.session_metrics = self._initialize_metrics()
        self.iteration_data = []

        # Claude client for context processing
        self.client = anthropic.Anthropic(api_key=Config.ANTHROPIC_API_KEY)

        # Load existing data if available
        self._load_existing_data()

        self.logger.info(f"ContextManager initialized for session: {session_dir}")

    def _initialize_metrics(self) -> Dict[str, Any]:
        """Initialize the enhanced metrics structure"""
        return {
            "session_start": datetime.now().isoformat(),
            "subject": self.subject,
            "iteration_metrics": [],
            "progress_indicators": {
                "ideas_count": 0,
                "practical_steps": 0,
                "creative_depth_score": 0.0,
                "implementation_readiness": 0.0,
                "context_evolution_score": 0.0
            },
            "agent_utilization": {
                "researcher_calls": 0,
                "engineer_calls": 0,
                "conclude_calls": 0,
                "tool_usage": {}
            },
            "session_analytics": {
                "total_duration": 0.0,
                "context_compressions": 0,
                "error_count": 0,
                "recovery_events": 0
            },
            "convergence_metrics": {
                "conceptual_drift": [],
                "direction_stability": [],
                "refinement_vs_pivot_ratio": [],
                "goal_alignment_score": [],
                "focus_coherence": []
            },
            "strategic_metrics": {
                "agent_selection_appropriateness": [],
                "plan_adherence_score": [],
                "plan_adaptation_intelligence": [],
                "resource_allocation_efficiency": [],
                "critical_path_recognition": [],
                "anticipatory_thinking": []
            },
            "synthesis_metrics": {
                "cross_domain_connections": [],
                "perspective_diversity_score": [],
                "novel_insight_generation": [],
                "information_density": [],
                "redundancy_score": [],
                "knowledge_building_momentum": []
            },
            "implementation_metrics": {
                "feasibility_realism": [],
                "stakeholder_awareness": [],
                "scalability_consideration": [],
                "risk_identification": [],
                "actionability_specificity": [],
                "resource_requirement_clarity": []
            },
            "emotional_metrics": {
                "passion_maintenance": [],
                "motivation_coherence": [],
                "engagement_depth": [],
                "human_connection_score": [],
                "emotional_integration": []
            },
            "meta_process_metrics": {
                "orchestrator_self_awareness": [],
                "learning_from_iterations": [],
                "strategic_thinking_evolution": [],
                "context_utilization_depth": [],
                "emergence_vs_planning": [],
                "process_adaptation": []
            },
            "domain_metrics": {
                "domain_expertise_depth": [],
                "market_validation": [],
                "technical_innovation": [],
                "creative_originality": [],
                "user_value_proposition": [],
                "competitive_differentiation": []
            },
            "research_quality_metrics": {
                "source_diversity": [],
                "information_credibility": [],
                "research_synthesis_quality": [],
                "perspective_integration": [],
                "depth_vs_breadth_balance": [],
                "research_to_insight_conversion": []
            },
            "engineering_quality_metrics": {
                "technical_feasibility_assessment": [],
                "implementation_pathway_clarity": [],
                "constraint_identification": [],
                "solution_elegance": [],
                "maintenance_consideration": [],
                "scalability_architecture": []
            }
        }

    def _load_existing_data(self):
        """Load existing context and metrics data"""
        try:
            # Load context history
            context_file = get_session_filepath(self.session_dir, "context")
            existing_context = load_json(context_file)
            if existing_context:
                self.context_history = existing_context.get("context_history", [])
                self.context = existing_context.get("context", "")
                self.compression_count = existing_context.get("compression_count", 0)
                self.logger.info("Loaded existing context history")

            # Load analytics data
            analytics_file = get_session_filepath(self.session_dir, "analytics")
            existing_metrics = load_json(analytics_file)
            if existing_metrics:
                self.session_metrics.update(existing_metrics)
                self.logger.info("Loaded existing analytics data")

        except Exception as e:
            self.logger.warning(f"Could not load existing data: {e}")

    def add_iteration_result(self, iteration_num: int, response: str,
                             agent_data: Optional[Dict[str, Any]] = None,
                             processing_time: float = 0.0,
                             retry_count: int = 0, agent_type: str = 'orchestrator') -> bool:
        """
        Add new iteration data and update context intelligently
        Returns: Success boolean
        """
        try:
            start_time = time.time()

            # Create iteration data structure
            iteration_data = {
                "iteration_number": iteration_num,
                "timestamp": datetime.now().isoformat(),
                "response": response,
                "agent_data": agent_data,
                "processing_time": processing_time,
                "retry_count": retry_count,
                "agent_type": agent_type
            }

            # Add to history
            self.context_history.append(iteration_data)
            self.iteration_data.append(iteration_data)

            # Update basic iteration metrics
            self._update_iteration_metrics(iteration_data)

            # Intelligent context update
            self._update_context(iteration_data)

            # Collect advanced analytics
            #self._collect_advanced_metrics(iteration_data, iteration_num)

            # Save updated data
            self._save_context_data()
            self._save_analytics_data()

            processing_duration = time.time() - start_time
            self.logger.info(f"Added iteration {iteration_num} data in {processing_duration:.2f}s")

            # Save detailed snapshot for this iteration
            self.save_context_snapshot(iteration_num, "iteration")

            return True

        except Exception as e:
            self.logger.error(f"Failed to add iteration result: {e}")
            return False

    def process_orchestrator_reply(self, iteration_num: int, orchestrator_response: str, parsed_data):
        """
        Complete processing pipeline for orchestrator replies

        1. Save reply in readable format
        2. Parse orchestrator format
        3. Update context
        4. Execute according to reply

        Returns: (execution_result, success, metadata)
        """
        start_time = time.time()

        try:
            self.logger.info(f"Processing orchestrator reply for iteration {iteration_num}")

            # Step 1: Save reply in nice readable format
            save_success = self._save_orchestrator_reply_formatted(
                iteration_num, orchestrator_response
            )
            if not save_success:
                self.logger.warning(f"Failed to save formatted reply for iteration {iteration_num}")


            # Step 3: Update context with this iteration
            context_update_success = self.add_iteration_result(
                iteration_num=iteration_num,
                response=orchestrator_response,
                agent_data=parsed_data,
                processing_time=time.time() - start_time,
                retry_count=0
            )

            if not context_update_success:
                self.logger.error(f"Failed to update context for iteration {iteration_num}")
                return None, False, {
                    "error": "Context update failed",
                    "processing_time": time.time() - start_time,
                    "iteration": iteration_num
                }


        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Orchestrator reply processing failed: {e}"
            self.logger.error(error_msg)

            return None, False, {
                "error": error_msg,
                "processing_time": processing_time,
                "iteration": iteration_num
            }





    def _update_iteration_metrics(self, iteration_data: Dict[str, Any]):
        """Update basic iteration metrics"""
        try:
            metric = {
                "iteration_number": iteration_data["iteration_number"],
                "timestamp": iteration_data["timestamp"],
                "agent_used": self._extract_agent_used(iteration_data["response"]),
                "response_length": len(iteration_data["response"]),
                "processing_time": iteration_data["processing_time"],
                "format_valid": iteration_data["retry_count"] == 0,
                "retry_count": iteration_data["retry_count"]
            }

            self.session_metrics["iteration_metrics"].append(metric)

            # Update agent utilization
            agent_used = metric["agent_used"]
            if agent_used in ["researcher", "engineer"]:
                self.session_metrics["agent_utilization"][f"{agent_used}_calls"] += 1
            elif agent_used == "conclude":
                self.session_metrics["agent_utilization"]["conclude_calls"] += 1

        except Exception as e:
            self.logger.error(f"Failed to update iteration metrics: {e}")

    def _extract_agent_used(self, response: str) -> str:
        """Extract which agent was used from orchestrator response"""
        try:
            response_lower = response.lower()
            if "using agent: researcher" in response_lower:
                return "researcher"
            elif "using agent: engineer" in response_lower:
                return "engineer"
            elif "concluding ideas:" in response_lower:
                return "conclude"
            else:
                return "unknown"
        except:
            return "unknown"

    def _update_context(self, new_iteration_data: Dict[str, Any]):
        """Update context summary using LLM intelligence"""
        try:
            # Check if we need compression
            current_length = len(self.context) + len(str(new_iteration_data))

            if current_length > Config.CONTEXT_COMPRESSION_THRESHOLD:
                self._trigger_full_summarization(new_iteration_data['response'])
            else:
                self._incremental_context_update(new_iteration_data)

        except Exception as e:
            self.logger.error(f"Failed to update context summary: {e}")

    def _incremental_context_update(self, new_iteration_data: Dict[str, Any]):
        """Incrementally update context with new iteration data"""
        try:
            # Step 1: Summarize the last output
            summary_prompt = f"""Summarize this iteration output to keep an agent context updated.
            Start you response with '**New {new_iteration_data['agent_type']} Iteration, number {new_iteration_data['iteration_number']}**:'
            and then your summarization of this last response the agent got.
            You need to speak as if you were the agent, so use 'I' and 'my' in your summary.
            Note that you are also given his context up to now, just to help you understand the context
            and to better summarize the last response.
            **Context (not to summarize, just for you)**: {self.context}
            **Response - to summarize: {new_iteration_data['response']}

            """

            response, success = retry_api_call(
                self.client.messages.create,
                model=Config.MODEL_NAME,
                max_tokens=500,  # Keep summary concise
                temperature=0.3,  # Lower temperature for consistent summarization
                messages=[{"role": "user", "content": summary_prompt}]
            )

            if success and response:
                # Step 2: Add tiny emotion to the summary
                summary = extract_text_from_response(response, not_context=False, agent_name='inc_context', session_dir=self.session_dir)
                enhanced_summary = self._apply_minimal_emotion_to_new_content(summary)

                # Step 3: Append to whole context
                self.context = self.context + "\n\n" + enhanced_summary
                self.logger.debug("Context updated: summarized → emotionally enhanced → appended")
            else:
                # Fallback: simple summary with emotion enhancement
                fallback_summary = f"Iteration {new_iteration_data['iteration_number']}: {new_iteration_data['response'][:200]}..."

                self.context += "\n\n" + fallback_summary
                self.logger.debug("Used fallback summary without emotion enhancement")

        except Exception as e:
            self.logger.error(f"Failed to update context summary: {e}")

    def _trigger_full_summarization(self, new_response: str):
        """Perform full context summarization when length exceeds threshold"""
        try:
            self.compression_count += 1
            self.session_metrics["session_analytics"]["context_compressions"] = self.compression_count

            prompt = f"""Summarize this context for an agent, as it got too long for him.
            This summarization is the most important step in multi agent process,
            as this is all of their context for communicating and working together.
            You have to intelligently summarize the context to keep the most important insights - 
            Planning, and situations and progress are very fundamental to keep, as well as the goal of the process of course.
            try to summarize in the most effective way, that the agents could understand what happened and observe the important insights, 
            while keeping it all in very concise  and efficient way. (maybe if you have to, u cn spik lk dis, or find other ways to be efficient and effective)
            You may see different iteration numbers for different agents. try to follow that, 
            knowing that for the researcher for example can be a different iteration number than the orchestrator..
            KEY THING TO KNOW - you are provided with the full context, and also the most recent iteration. The difference is - the recent
            iteration is not SUMMARIZED AT ALL. meaning it will require much more summarization then the full context that has already been summarized 
            (though you of course can summarize it as well if needed)
            Note that you need to speak as if you were the agent, so use 'I' and 'my' in your summary.
            You are given his full context and need to summarize it to keep the most important insights
            Return only your summary, do not include any thoughts of yours or things like that.
            
            **Full Session Context**: {self.context} @!@ **END OF CONTEXT** @!@
            **Most Recent Iteration (not summarized)**: {new_response} END OF MOST RECENT ITERATION
            

            """

            response, success = retry_api_call(
                self.client.messages.create,
                model=Config.MODEL_NAME,
                max_tokens=Config.MAX_TOKENS // 4,
                temperature=0.3,  # Lower temperature for summarization
                messages=[{"role": "user", "content": prompt}]
            )

            if success and response:
                self.context = extract_text_from_response(response, not_context=False, agent_name='full__context_summarization', session_dir=self.session_dir)
                self.logger.info(f"Full context summarization completed (compression #{self.compression_count})")
            else:
                self.logger.error("Full summarization failed, keeping current context")

        except Exception as e:
            self.logger.error(f"Full summarization failed: {e}")

    def _build_full_narrative(self) -> str:
        """Build complete narrative from all iterations"""
        try:
            narrative_parts = [f"Subject: {self.subject}\n"]

            for iteration in self.context_history:
                narrative_parts.append(f"Iteration {iteration['iteration_number']}:")
                narrative_parts.append(iteration['response'])

                if iteration.get('agent_data'):
                    narrative_parts.append(f"Agent Result: {str(iteration['agent_data'])[:500]}...")

                narrative_parts.append("---")

            return "\n".join(narrative_parts)

        except Exception as e:
            self.logger.error(f"Failed to build full narrative: {e}")
            return self.context

    def _collect_advanced_metrics(self, iteration_data: Dict[str, Any], iteration_num: int):
        """Collect advanced analytics metrics using LLM analysis"""
        try:
            # This is where we'll implement sophisticated metric collection
            # For now, implementing a subset - will expand in subsequent iterations

            context_summary = self.build_orchestrator_context(iteration_num)

            # Analyze using LLM for intelligent metrics
            metrics_prompt = f"""Analyze this iteration for advanced metrics:

            **Subject**: {self.subject}
            **Iteration**: {iteration_num}/8
            **Context**: {context_summary[:1000]}
            **Current Iteration Data**: {iteration_data['response']}

            Rate the following on a scale of 0.0 to 1.0:

            1. **Goal Alignment**: How well does this iteration serve the original subject?
            2. **Creative Depth**: Level of creative insight and originality
            3. **Implementation Readiness**: How actionable and practical is the content?
            4. **Focus Coherence**: Is the work focused or scattered?
            5. **Strategic Thinking**: Quality of strategic decision making

            Return ONLY a JSON object with these keys: goal_alignment, creative_depth, implementation_readiness, focus_coherence, strategic_thinking"""

            response, success = retry_api_call(
                self.client.messages.create,
                model=Config.MODEL_NAME,
                max_tokens=500,
                temperature=0.1,  # Low temperature for consistent scoring
                messages=[{"role": "user", "content": metrics_prompt}]
            )

            if success and response:
                try:
                    metrics_text = extract_text_from_response(response, not_context=False, session_dir=self.session_dir)
                    # Extract JSON from response
                    import re
                    json_match = re.search(r'\{[^}]+\}', metrics_text)
                    if json_match:
                        metrics = json.loads(json_match.group())

                        # Store in appropriate metric categories
                        self.session_metrics["convergence_metrics"]["goal_alignment_score"].append(
                            metrics.get("goal_alignment", 0.5))
                        self.session_metrics["convergence_metrics"]["focus_coherence"].append(
                            metrics.get("focus_coherence", 0.5))
                        self.session_metrics["synthesis_metrics"]["novel_insight_generation"].append(
                            metrics.get("creative_depth", 0.5))
                        self.session_metrics["implementation_metrics"]["feasibility_realism"].append(
                            metrics.get("implementation_readiness", 0.5))
                        self.session_metrics["strategic_metrics"]["agent_selection_appropriateness"].append(
                            metrics.get("strategic_thinking", 0.5))

                except Exception as e:
                    self.logger.error(f"Failed to parse advanced metrics: {e}")

        except Exception as e:
            self.logger.error(f"Advanced metrics collection failed: {e}")



    def export_full_history(self) -> Dict[str, Any]:
        """Export complete session narrative and analytics"""
        try:
            return {
                "session_metadata": {
                    "subject": self.subject,
                    "session_dir": self.session_dir,
                    "total_iterations": len(self.context_history),
                    "compression_count": self.compression_count,
                    "export_timestamp": datetime.now().isoformat()
                },
                "context_history": self.context_history,
                "context": self.context,
                "session_metrics": self.session_metrics,
                "full_narrative": self._build_full_narrative()
            }

        except Exception as e:
            self.logger.error(f"Failed to export full history: {e}")
            return {}

    def get_analytics_summary(self) -> Dict[str, Any]:
        """Get summary of key analytics for visualization"""
        try:
            total_iterations = len(self.session_metrics["iteration_metrics"])

            if total_iterations == 0:
                return {"error": "No iterations completed yet"}

            # Calculate averages for key metrics
            summary = {
                "session_overview": {
                    "subject": self.subject,
                    "total_iterations": total_iterations,
                    "session_duration": time.time() - datetime.fromisoformat(
                        self.session_metrics["session_start"]).timestamp(),
                    "compression_events": self.compression_count
                },
                "agent_distribution": self.session_metrics["agent_utilization"],
                "quality_trends": {
                    "goal_alignment": self._calculate_trend("convergence_metrics", "goal_alignment_score"),
                    "creative_depth": self._calculate_trend("synthesis_metrics", "novel_insight_generation"),
                    "implementation_readiness": self._calculate_trend("implementation_metrics", "feasibility_realism"),
                    "focus_coherence": self._calculate_trend("convergence_metrics", "focus_coherence")
                }
            }

            return summary

        except Exception as e:
            self.logger.error(f"Failed to get analytics summary: {e}")
            return {"error": str(e)}

    def _calculate_trend(self, category: str, metric: str) -> Dict[str, float]:
        """Calculate trend for a specific metric"""
        try:
            values = self.session_metrics.get(category, {}).get(metric, [])
            if not values:
                return {"current": 0.0, "trend": 0.0, "min": 0.0, "max": 0.0}

            return {
                "current": values[-1] if values else 0.0,
                "trend": (values[-1] - values[0]) if len(values) > 1 else 0.0,
                "min": min(values),
                "max": max(values),
                "average": sum(values) / len(values)
            }
        except:
            return {"current": 0.0, "trend": 0.0, "min": 0.0, "max": 0.0}

    def _apply_minimal_emotion_to_new_content(self, new_content: str) -> str:
        """Apply tiny emotional enhancement to only new content"""
        try:
            # Only enhance if content is substantial enough
            if len(new_content.strip()) < 50:
                return new_content

            # Use existing emotion agent prompt from config
            emotion_prompt = Config.AGENT_PROMPTS["emotion"].format(
                raw_context=new_content,
                context=self.context
            )

            response, success = retry_api_call(
                self.client.messages.create,
                model=Config.MODEL_NAME,
                max_tokens=len(new_content) + 100,  # Limit to prevent expansion
                temperature=0.3,  # Low temperature for minimal changes
                system=emotion_prompt,
                messages=[{"role": "user", "content": "Please enhance this context with minimal modifications."}]
            )

            if success and response:
                enhanced = extract_text_from_response(response, not_context=False, agent_name='emotion_enhancer', session_dir=self.session_dir)
                # Safety check - if enhanced version is much longer, use original
                if len(enhanced) > len(new_content) * 1.2:
                    self.logger.warning("Emotional enhancement expanded content too much, using original")
                    return new_content
                return enhanced
            else:
                return new_content

        except Exception as e:
            self.logger.error(f"Minimal emotion enhancement failed: {e}")
            return new_content

    def _save_context_data(self):
        """Save context data to file"""
        try:
            context_data = {
                "context_history": self.context_history,
                "context": self.context,
                "compression_count": self.compression_count,
                "last_updated": datetime.now().isoformat()
            }

            context_file = get_session_filepath(self.session_dir, "context")
            save_json(context_data, context_file)

        except Exception as e:
            self.logger.error(f"Failed to save context data: {e}")

    def _save_analytics_data(self):
        """Save analytics data to file"""
        try:
            analytics_file = get_session_filepath(self.session_dir, "analytics")
            save_json(self.session_metrics, analytics_file)

        except Exception as e:
            self.logger.error(f"Failed to save analytics data: {e}")

    def save_context_snapshot(self, iteration_num: int, snapshot_type: str = "iteration") -> bool:
        """
        Save a context snapshot at specific points for detailed analysis

        Args:
            iteration_num: Current iteration number
            snapshot_type: Type of snapshot (iteration, evaluation, planning, final)
        """
        try:
            snapshot_data = {
                "iteration_number": iteration_num,
                "snapshot_type": snapshot_type,
                "timestamp": datetime.now().isoformat(),
                "raw_context_history": self.context_history,
                "context": self.context,
                "metrics_at_snapshot": self.session_metrics,
                "compression_count": self.compression_count
            }

            snapshot_file = os.path.join(
                self.session_dir,
                "context",
                f"context_snapshot_iter_{iteration_num}_{snapshot_type}.json"
            )

            return save_json(snapshot_data, snapshot_file)

        except Exception as e:
            self.logger.error(f"Failed to save context snapshot: {e}")
            return False

    def __del__(self):
        """Cleanup when context manager is destroyed"""
        try:
            self._save_context_data()
            self._save_analytics_data()
        except:
            pass  # Ignore errors during cleanup

    def _save_orchestrator_reply_formatted(self, iteration_num: int, response: str) -> bool:
        """
        Save orchestrator reply in clean, readable markdown format
        Preserves exact words but improves formatting for readability
        """
        try:
            # Create file path
            filename = f"iteration_{iteration_num}_orchestrator_response.md"
            filepath = os.path.join(self.session_dir, "iterations", filename)

            # Ensure directory exists
            os.makedirs(os.path.dirname(filepath), exist_ok=True)

            # Save markdown file
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(response)

            self.logger.debug(f"Saved formatted orchestrator reply to {filepath}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to save formatted orchestrator reply: {e}")
            return False


