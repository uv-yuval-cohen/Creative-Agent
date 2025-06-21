import logging
import time
from typing import Dict, Any, Optional, Tuple, List

import context_manager
from config import Config
from utils import extract_text_from_response, retry_api_call, validate_agent_response_format, save_error_log, \
    format_error_message
import anthropic
from context_manager import ContextManager


class ClaudeAgentManager:
    """
    Dynamic Agent Creation & Management System
    Handles all Claude agent instances with comprehensive error handling
    """

    def __init__(self, session_dir: str, context_manager: ContextManager):
        self.session_dir = session_dir
        self.logger = logging.getLogger(__name__)
        self.context_manager = context_manager
        self.plan = "no plan yet"

        # Initialize Claude client
        try:
            self.client = anthropic.Anthropic(api_key=Config.ANTHROPIC_API_KEY)
            self.logger.info("Claude client initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize Claude client: {e}")
            raise

        # Agent performance tracking
        self.agent_stats = {
            "orchestrator": {"calls": 0, "failures": 0, "avg_response_time": 0.0},
            "researcher": {"calls": 0, "failures": 0, "avg_response_time": 0.0},
            "engineer": {"calls": 0, "failures": 0, "avg_response_time": 0.0},
            "evaluator": {"calls": 0, "failures": 0, "avg_response_time": 0.0},
            "context_manager": {"calls": 0, "failures": 0, "avg_response_time": 0.0}
        }

    def create_orchestrator(self, current_iteration: int) -> Tuple[
        Optional[str], bool, Dict[str, Any]]:
        """
        Create and execute orchestrator agent
        Returns: (response_text, success, metadata)
        """
        agent_name = "orchestrator"
        start_time = time.time()
        if current_iteration == 8:
            current_iteration = "8 THIS IS FINAL ITERATION!!!! TIME TO FINALIZE THE PROJECT!!!"

        try:
            self.logger.info(f"Creating orchestrator for iteration {current_iteration}")

            # Build system prompt
            system_prompt = Config.ORCHESTRATOR_SYSTEM_PROMPT.format(
                subject=self.context_manager.subject,
                context=self.context_manager.context,
                current_iteration=current_iteration,
                current_plan=self.context_manager.plan
            )

            # Execute with retry logic
            response, success = retry_api_call(
                self.client.messages.create,
                model=Config.MODEL_NAME,
                max_tokens=Config.MAX_TOKENS,
                temperature=Config.TEMPERATURE,
                system=system_prompt,
                messages=[{"role": "user",
                           "content": f"Please proceed with iteration {current_iteration} of the orchestration process."}]
            )

            processing_time = time.time() - start_time

            if success and response:
                response_text = extract_text_from_response(response, agent_name=agent_name, iteration=current_iteration, session_dir=self.session_dir)

                # Validate response format
                is_valid, error_msg, parsed_data = validate_agent_response_format(response_text)

                if not is_valid:
                    self.logger.warning(f"Orchestrator format invalid: {error_msg}")

                    # Attempt retry with format correction
                    retry_response = self._retry_with_format_correction(
                        system_prompt, error_msg, current_iteration
                    )

                    if retry_response:
                        response_text = retry_response
                        is_valid = True
                    else:
                        self._update_agent_stats(agent_name, processing_time, failed=True)
                        return None, False, {"error": "Format validation failed after retry",
                                             "processing_time": processing_time}

                if 'plan' in parsed_data:
                    self.plan = parsed_data['plan']

                self.logger.info(f"Processing complete orchestrator reply for iteration {current_iteration}")

                self.context_manager.process_orchestrator_reply(
                    iteration_num=current_iteration,
                    orchestrator_response=response_text,
                    parsed_data=parsed_data
                )

                if current_iteration == 8:
                    return response_text, True, {
                        "agent_name": agent_name,
                        "processing_time": processing_time,
                        "iteration": current_iteration
                    }

                # need to execute the orchestrator reply
                if parsed_data['type'] == 'agent':
                    if parsed_data['agent_name']== 'researcher':
                        # Execute the researcher agent
                        agent_response, agent_success, agent_metadata = self.create_researcher(
                            agent_instructions=parsed_data['agent_prompt'],
                            current_iteration=current_iteration
                        )
                        # Prepare agent_data with both parsed data and agent response
                        if agent_success:
                            agent_data = {
                                'type': 'agent',
                                'agent_name': 'researcher',
                                'agent_prompt': parsed_data['agent_prompt'],
                                'agent_response': agent_response,
                                'agent_success': True,
                                'agent_metadata': agent_metadata
                            }
                        else:
                            raise Exception(f"Researcher agent failed: {agent_metadata.get('error', 'Unknown error')}")

                    if parsed_data['agent_name'] == 'engineer':
                        # Execute the engineer agent
                        agent_response, agent_success, agent_metadata = self.create_engineer(
                            subject=self.context_manager.subject,
                            context_summary=self.context_manager.context,
                            agent_instructions=parsed_data['agent_prompt'],
                            current_iteration=current_iteration
                        )
                        # Prepare agent_data with both parsed data and agent response
                        if agent_success:
                            agent_data = {
                                'type': 'agent',
                                'agent_name': 'engineer',
                                'agent_prompt': parsed_data['agent_prompt'],
                                'agent_response': agent_response,
                                'agent_success': True,
                                'agent_metadata': agent_metadata
                            }
                        else:
                            raise Exception(f"Engineer agent failed: {agent_metadata.get('error', 'Unknown error')}")

                    # Update context with both orchestrator and agent responses (if applicable)
                    processing_time = time.time() - start_time


                # Update performance stats
                self._update_agent_stats(agent_name, processing_time, failed=False)

                metadata = {
                    "agent_name": agent_name,
                    "processing_time": processing_time,
                    "format_valid": is_valid,
                    "parsed_data": parsed_data,
                    "iteration": current_iteration
                }

                self.logger.info(f"Orchestrator completed successfully in {processing_time:.2f}s")
                return response_text, True, metadata

            else:
                self._update_agent_stats(agent_name, processing_time, failed=True)
                error_msg = "API call failed for orchestrator"
                self.logger.error(error_msg)

                save_error_log(self.session_dir, "orchestrator_failure", {
                    "iteration": current_iteration,
                    "error": error_msg,
                    "processing_time": processing_time
                })

                return None, False, {"error": error_msg, "processing_time": processing_time}

        except Exception as e:
            processing_time = time.time() - start_time
            self._update_agent_stats(agent_name, processing_time, failed=True)
            error_msg = f"Orchestrator creation failed: {e}"
            self.logger.error(error_msg)

            save_error_log(self.session_dir, "orchestrator_exception", {
                "iteration": current_iteration,
                "error": str(e),
                "processing_time": processing_time
            })

            return None, False, {"error": error_msg, "processing_time": processing_time}

    def _retry_with_format_correction(self, original_prompt: str, error_msg: str, iteration: int) -> Optional[str]:
        """Retry orchestrator call with format correction guidance"""
        try:
            retry_prompt = f"{original_prompt}\n\n{Config.ERROR_MESSAGES['format_retry']}"

            response, success = retry_api_call(
                self.client.messages.create,
                model=Config.MODEL_NAME,
                max_tokens=Config.MAX_TOKENS,
                temperature=Config.TEMPERATURE,
                system=retry_prompt,
                messages=[{"role": "user",
                           "content": f"Please provide your response for iteration {iteration} in the correct format."}]
            )

            if success and response:
                response_text = extract_text_from_response(response, agent_name='orchestrator', iteration=iteration, session_dir=self.session_dir)
                is_valid, _, _ = validate_agent_response_format(response_text)

                if is_valid:
                    self.logger.info("Format correction successful")
                    return response_text
                else:
                    self.logger.error("Format correction failed")
                    return None
            else:
                self.logger.error("Retry API call failed")
                return None

        except Exception as e:
            self.logger.error(f"Format correction retry failed: {e}")
            return None

    def create_researcher(self, agent_instructions: str,
                          current_iteration: int) -> Tuple[Optional[str], bool, Dict[str, Any]]:
        """
        Create and execute researcher agent
        Returns: (response_text, success, metadata)
        """
        agent_name = "researcher"
        start_time = time.time()
        processing_time = 0.0
        self.logger.info("Conducting a research!")
        chose_web_searcher = "You haven't chosen this yet"
        chose_philosopher = "You haven't chosen this yet"
        chose_perspective = "You haven't chosen this yet"
        try:
            # 3 research iterations:
            for i in range(3):
                iteration_num=i+1
                self.logger.info(f"Researcher starting iteration {iteration_num}")
                # Build system prompt
                system_prompt = Config.AGENT_PROMPTS["researcher"].format(
                    subject=self.context_manager.subject,
                    current_iteration=(iteration_num),
                    context_summary=self.context_manager.context,
                    agent_instructions=agent_instructions,
                    chose_web_searcher=chose_web_searcher,
                    chose_philosopher=chose_philosopher,
                    chose_perspective=chose_perspective
                )

                # Execute researcher
                response, success = retry_api_call(
                    self.client.messages.create,
                    model=Config.MODEL_NAME,
                    max_tokens=Config.MAX_TOKENS,
                    temperature=Config.TEMPERATURE + 0.2, # Higher temperature for creative research
                    system=system_prompt,
                    messages=[{"role": "user", "content": "Please begin/continue your research process."}]
                )

                processing_time = time.time() - start_time

                if success and response:
                    response_text = extract_text_from_response(response, agent_name='research_conductor', iteration=iteration_num, session_dir=self.session_dir)
                    # Validate response format
                    is_valid, error_msg, parsed_data = validate_agent_response_format(response_text, agent_type='researcher')

                    if not is_valid:
                        self.logger.warning(f"Orchestrator format invalid: {error_msg}")

                        # Attempt retry with format correction
                        retry_response = self._retry_with_format_correction(
                            system_prompt, error_msg, current_iteration
                        )

                        if retry_response:
                            response_text = retry_response
                            is_valid, error_msg, parsed_data= validate_agent_response_format(response_text, agent_type='researcher')
                            is_valid = True
                        else:
                            self._update_agent_stats(agent_name, processing_time, failed=True)
                            return None, False, {"error": "Format validation failed after retry",
                                                 "processing_time": processing_time}

                    context_update_success= self.context_manager.add_iteration_result(iteration_num=iteration_num,response=response_text,
                                                              agent_data=parsed_data,
                                                              processing_time=time.time() - start_time, retry_count=0, agent_type='researcher' )

                    if not context_update_success:
                        self.logger.error(f"Failed to update context for iteration {iteration_num}")
                        return None, False, {
                            "error": "Context update failed",
                            "processing_time": time.time() - start_time,
                            "iteration": iteration_num
                        }


                    research_results = self.execute_researcher_tools(parsed_data)
                    if research_results['tool'] == 'WEB_SEARCH':
                        chose_web_searcher = "You already chose web searcher!"
                    if research_results['tool'] == 'PHILOSOPHY':
                        chose_philosopher = "You already chose philosopher!"
                    if research_results['tool'] == 'PERSPECTIVE':
                        chose_perspective = "You already chose perspective!"
                    if not research_results['success']:
                        self.logger.error(f"Researcher tool execution failed: {research_results['result']}")
                        return None, False, {
                            "error": research_results['result'],
                            "processing_time": processing_time,
                            "iteration": iteration_num
                        }

                else:
                    self._update_agent_stats(agent_name, processing_time, failed=True)
                    error_msg = "API call failed for researcher"
                    self.logger.error(error_msg)
                    return None, False, {"error": error_msg, "processing_time": processing_time}

            ## SYNTHESIS iteration - final conclusion of research
            self.logger.info("Starting synthesis iteration")

            # Build system prompt for synthesis
            synthesis_prompt = Config.AGENT_PROMPTS["researcher"].format(
                subject=self.context_manager.subject,
                current_iteration=4,
                context_summary=self.context_manager.context,
                agent_instructions=agent_instructions,
                chose_web_searcher=chose_web_searcher,
                chose_philosopher=chose_philosopher,
                chose_perspective=chose_perspective
            )

            # Execute synthesis
            response, success = retry_api_call(
                self.client.messages.create,
                model=Config.MODEL_NAME,
                max_tokens=Config.MAX_TOKENS,
                temperature=Config.TEMPERATURE,
                system=synthesis_prompt,
                messages=[{"role": "user", "content": "Please provide your final synthesis and conclusions based on all research conducted."}]
            )

            processing_time = time.time() - start_time

            if success and response:
                response_text = extract_text_from_response(response, agent_name='researcher', iteration=4, session_dir=self.session_dir)
                # Validate response format
                is_valid, error_msg, parsed_data = validate_agent_response_format(response_text, agent_type='researcher')


                context_update_success = self.context_manager.add_iteration_result(
                    iteration_num=4,
                    response=response_text,
                    agent_data=parsed_data,
                    processing_time=time.time() - start_time,
                    retry_count=0,
                    agent_type='researcher'
                )

                if not context_update_success:
                    self.logger.error("Failed to update context for synthesis iteration")
                    return None, False, {
                        "error": "Context update failed for synthesis",
                        "processing_time": time.time() - start_time,
                        "iteration": "SYNTHESIS"
                    }

            else:
                self._update_agent_stats(agent_name, processing_time, failed=True)
                error_msg = "API call failed for synthesis"
                self.logger.error(error_msg)
                return None, False, {"error": error_msg, "processing_time": processing_time}
            self._update_agent_stats(agent_name, processing_time, failed=False)

            metadata = {
                "agent_name": agent_name,
                "processing_time": processing_time,
                "iteration": current_iteration
            }

            self.logger.info(f"Researcher completed successfully in {processing_time:.2f}s")
            return response_text, True, metadata

        except Exception as e:
            processing_time = time.time() - start_time
            self._update_agent_stats(agent_name, processing_time, failed=True)
            error_msg = f"Researcher creation failed: {e}"
            self.logger.error(error_msg)
            return None, False, {"error": error_msg, "processing_time": processing_time}



    def create_engineer(self, subject: str, context_summary: str, agent_instructions: str,
                        current_iteration: int) -> Tuple[Optional[str], bool, Dict[str, Any]]:
        """
        Create and execute engineer agent
        Returns: (response_text, success, metadata)
        """
        agent_name = "engineer"
        start_time = time.time()

        try:
            self.logger.info(f"Creating engineer for iteration {current_iteration}")

            # Build system prompt
            system_prompt = Config.AGENT_PROMPTS["engineer"].format(
                subject=subject,
                context_summary=context_summary,
                agent_instructions=agent_instructions
            )

            # Configure reasoning
            thinking_config = {
                "type": "enabled",
                "budget_tokens": 3000
            }

            # Execute engineer
            response, success = retry_api_call(
                self.client.messages.create,
                model=Config.MODEL_NAME,
                max_tokens=Config.MAX_TOKENS,
                temperature=1,
                system=system_prompt,
                thinking=thinking_config,
                messages=[{"role": "user", "content": "Please provide your engineering analysis and recommendations."}]
            )

            processing_time = time.time() - start_time

            if success and response:
                response_text = extract_text_from_response(response, agent_name=agent_name, iteration=current_iteration, session_dir=self.session_dir)

                self._update_agent_stats(agent_name, processing_time, failed=False)

                metadata = {
                    "agent_name": agent_name,
                    "processing_time": processing_time,
                    "iteration": current_iteration
                }

                self.logger.info(f"Engineer completed successfully in {processing_time:.2f}s")
                return response_text, True, metadata

            else:
                self._update_agent_stats(agent_name, processing_time, failed=True)
                error_msg = "API call failed for engineer"
                self.logger.error(error_msg)
                return None, False, {"error": error_msg, "processing_time": processing_time}

        except Exception as e:
            processing_time = time.time() - start_time
            self._update_agent_stats(agent_name, processing_time, failed=True)
            error_msg = f"Engineer creation failed: {e}"
            self.logger.error(error_msg)
            return None, False, {"error": error_msg, "processing_time": processing_time}

    def create_evaluator(self, subject: str, context_summary: str, current_iteration: int,
                         recent_outputs: str) -> Tuple[Optional[str], bool, Dict[str, Any]]:
        """
        Create and execute evaluator agent for strategic checkpoints
        Returns: (response_text, success, metadata)
        """
        agent_name = "evaluator"
        start_time = time.time()

        try:
            self.logger.info(f"Creating evaluator for iteration {current_iteration}")

            # Build system prompt
            system_prompt = Config.AGENT_PROMPTS["evaluator"].format(
                current_iteration=current_iteration,
                subject=subject,
                context_summary=context_summary,
                recent_outputs=recent_outputs
            )

            # Execute evaluator
            response, success = retry_api_call(
                self.client.messages.create,
                model=Config.MODEL_NAME,
                max_tokens=Config.MAX_TOKENS,
                temperature=0.3,  # Lower temperature for more consistent evaluation
                system=system_prompt,
                messages=[{"role": "user", "content": "Please provide your strategic evaluation and recommendations."}]
            )

            processing_time = time.time() - start_time

            if success and response:
                response_text = extract_text_from_response(response, agent_name='evaluator', iteration=current_iteration, session_dir=self.session_dir)

                self._update_agent_stats(agent_name, processing_time, failed=False)

                metadata = {
                    "agent_name": agent_name,
                    "processing_time": processing_time,
                    "evaluation_iteration": current_iteration,
                    "checkpoint_type": "mid_process" if current_iteration == 4 else "pre_final"
                }

                self.logger.info(f"Evaluator completed successfully in {processing_time:.2f}s")
                return response_text, True, metadata

            else:
                self._update_agent_stats(agent_name, processing_time, failed=True)
                error_msg = "API call failed for evaluator"
                self.logger.error(error_msg)
                return None, False, {"error": error_msg, "processing_time": processing_time}

        except Exception as e:
            processing_time = time.time() - start_time
            self._update_agent_stats(agent_name, processing_time, failed=True)
            error_msg = f"Evaluator creation failed: {e}"
            self.logger.error(error_msg)
            return None, False, {"error": error_msg, "processing_time": processing_time}

    from typing import Dict, Any

    def execute_researcher_tools(self, parsed_data: Dict[str, str]) -> Any:
        """
        Execute the appropriate researcher tool based on the parsed data.

        Args:
            parsed_data (Dict[str, str]): The parsed data from the researcher's response.

        Returns:
            Any: The result from the executed tool.
        """
        sub_type = parsed_data['sub_type']

        if sub_type == 'WEB_SEARCHER':
            instructions = parsed_data['instructions']
            return self._execute_web_research_tool(instructions)
        elif sub_type == 'PHILOSOPHER':
            topic = parsed_data['topic']
            return self._execute_philosophy_tool(topic)
        elif sub_type == 'PERSPECTIVE':
            person = parsed_data['person']
            topic = parsed_data['topic']
            return self._execute_perspective_tool(person, topic)
        else:
            raise ValueError(f"Invalid sub_type: {sub_type}")

    def _execute_web_research_tool(self, instructions) -> Dict[str, Any]:
        """Execute web search tool using Claude's web search capabilities"""
        try:
            search_prompt = f"""Your research conductor instructing you to perform a web research.
            first, here is the context of the project: {self.context_manager.context}

            Now here are the instructions: {instructions}
            
            Please use the web_search tool for that.

            """
            # Define the tools specification
            tools = [{
                "type": "web_search_20250305",
                "name": "web_search",
                "max_uses": 1
            }]

            response, success = retry_api_call(
                self.client.messages.create,
                model=Config.MODEL_NAME,
                max_tokens=2000,
                temperature=0.5,
                messages=[{"role": "user", "content": search_prompt}],
                tools=tools
            )

            if success and response:
                response_text = extract_text_from_response(response, agent_name='web_searcher', session_dir=self.session_dir)
                response_dict = {
                    'response' : response_text
                }
                self.context_manager.context_history.append(response_dict)
                self.context_manager._update_context(response_dict)
                return {
                    "tool": "WEB_SEARCH",
                    "result": response_text,
                    "success": True
                }
            else:
                return {
                    "tool": "WEB_SEARCH",
                    "result": "Search failed",
                    "success": False
                }

        except Exception as e:
            self.logger.error(f"Web search tool failed: {e}")
            return {
                "tool": "WEB_SEARCH",
                "result": f"Error: {e}",
                "success": False
            }

    def _execute_philosophy_tool(self, topic: str) -> Dict[str, Any]:
        """Execute philosophy tool for deep conceptual analysis"""
        try:
            philosophy_prompt = f"""Your research conductor needs your philosophy.
            first, here is the context of the project: {self.context_manager.context}

            Now here is the topic: {topic}

            Please philosophize and wonder over this matter."""

            response, success = retry_api_call(
                self.client.messages.create,
                model=Config.MODEL_NAME,
                max_tokens=2000,
                temperature=0.7,
                messages=[{"role": "user", "content": philosophy_prompt}]
            )

            if success and response:
                response_text = extract_text_from_response(response, agent_name='philosopher', session_dir=self.session_dir)
                response_dict = {
                    'response': response_text
                }
                self.context_manager.context_history.append(response_dict)
                self.context_manager._update_context(response_dict)
                return {
                    "tool": "PHILOSOPHY",
                    "topic": topic,
                    "result": response_text,
                    "success": True
                }
            else:
                return {
                    "tool": "PHILOSOPHY",
                    "topic": topic,
                    "result": "Philosophy analysis failed",
                    "success": False
                }

        except Exception as e:
            self.logger.error(f"Philosophy tool failed: {e}")
            return {
                "tool": "PHILOSOPHY",
                "topic": topic,
                "result": f"Error: {e}",
                "success": False
            }

    def _execute_perspective_tool(self, person, topic) -> Dict[str, Any]:
        """Execute perspective tool to channel expert viewpoints"""
        try:
            perspective_prompt = f"""You are helping a research conductor. 
            He needs you to be channeling the perspective of {person} for his project. 
            For now, you are {person}, you have his unique perspective and insights.

            **Context of the project**: {self.context_manager.context}
            **Topic**: {topic}

            Please respond as {person} would. Channel their distinctive thinking style 
            and be persistent with their character as you can possibly be."""

            response, success = retry_api_call(
                self.client.messages.create,
                model=Config.MODEL_NAME,
                max_tokens=2000,
                temperature=0.8,  # Higher temperature for creative perspective
                messages=[{"role": "user", "content": perspective_prompt}]
            )

            if success and response:
                response_text = extract_text_from_response(response,agent_name=f'perspective_{person}', session_dir=self.session_dir)
                response_dict = {
                    'response': response_text
                }
                self.context_manager.context_history.append(response_dict)
                self.context_manager._update_context(response_dict)
                return {
                    "tool": "PERSPECTIVE",
                    "person": person,
                    "topic": topic,
                    "result": response_text,
                    "success": True
                }
            else:
                return {
                    "tool": "PERSPECTIVE",
                    "person": person,
                    "topic": topic,
                    "result": "Perspective analysis failed",
                    "success": False
                }

        except Exception as e:
            self.logger.error(f"Perspective tool failed: {e}")
            return {
                "tool": "PERSPECTIVE",
                "person": person,
                "topic": "",
                "result": f"Error: {e}",
                "success": False
            }

    def _update_agent_stats(self, agent_name: str, processing_time: float, failed: bool = False):
        """Update performance statistics for an agent"""
        try:
            if agent_name in self.agent_stats:
                stats = self.agent_stats[agent_name]
                stats["calls"] += 1

                if failed:
                    stats["failures"] += 1
                else:
                    # Update average response time
                    current_avg = stats["avg_response_time"]
                    current_calls = stats["calls"] - stats["failures"]
                    if current_calls > 0:
                        stats["avg_response_time"] = ((current_avg * (
                                    current_calls - 1)) + processing_time) / current_calls

        except Exception as e:
            self.logger.error(f"Failed to update agent stats: {e}")

    def get_agent_performance(self) -> Dict[str, Any]:
        """Get performance statistics for all agents"""
        try:
            performance = {}
            for agent_name, stats in self.agent_stats.items():
                total_calls = stats["calls"]
                if total_calls > 0:
                    success_rate = (total_calls - stats["failures"]) / total_calls
                    performance[agent_name] = {
                        "total_calls": total_calls,
                        "failures": stats["failures"],
                        "success_rate": success_rate,
                        "avg_response_time": stats["avg_response_time"]
                    }
                else:
                    performance[agent_name] = {
                        "total_calls": 0,
                        "failures": 0,
                        "success_rate": 0.0,
                        "avg_response_time": 0.0
                    }

            return performance

        except Exception as e:
            self.logger.error(f"Failed to get agent performance: {e}")
            return {}

    def validate_agent_response(self, response: str, expected_format: str) -> Tuple[bool, Optional[str]]:
        """
        Validate agent response format
        Returns: (is_valid, error_message)
        """
        try:
            if expected_format == "orchestrator":
                is_valid, error_msg, _ = validate_agent_response_format(response)
                return is_valid, error_msg
            else:
                # For other agents, basic validation
                if response and len(response.strip()) > 10:
                    return True, None
                else:
                    return False, "Response too short or empty"

        except Exception as e:
            return False, f"Validation error: {e}"

    def handle_agent_error(self, agent_name: str, error: Exception, iteration: int) -> Dict[str, Any]:
        """Centralized agent error handling"""
        try:
            error_data = {
                "agent_name": agent_name,
                "error": str(error),
                "iteration": iteration,
                "timestamp": time.time()
            }

            # Save error log
            save_error_log(self.session_dir, f"{agent_name}_error", error_data)

            # Update stats
            self._update_agent_stats(agent_name, 0.0, failed=True)

            # Return formatted error
            return {
                "success": False,
                "error": format_error_message("agent_failure", f"{agent_name}: {error}"),
                "error_data": error_data
            }

        except Exception as e:
            self.logger.error(f"Error handling failed: {e}")
            return {
                "success": False,
                "error": "Critical error in error handling",
                "error_data": {"exception": str(e)}
            }