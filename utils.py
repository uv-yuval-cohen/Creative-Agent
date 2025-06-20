import logging
import json
import os
import time
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from config import Config


def setup_logging():
    """Setup logging configuration with enhanced error tracking"""
    # Create output directory if it doesn't exist
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

    logging.basicConfig(
        level=getattr(logging, Config.LOG_LEVEL),
        format=Config.LOG_FORMAT,
        handlers=[
            logging.FileHandler(f"{Config.OUTPUT_DIR}/orchestrator.log"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def generate_session_id() -> str:
    """Generate a unique session ID for this orchestration run"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def create_session_directory(session_id: str) -> str:
    """Create organized directory structure for a session"""
    session_dir = os.path.join(Config.OUTPUT_DIR, f"session_{session_id}")

    # Create subdirectories
    subdirs = ['iterations', 'context', 'reports', 'agents', 'errors', 'analytics']
    for subdir in subdirs:
        os.makedirs(os.path.join(session_dir, subdir), exist_ok=True)

    return session_dir


def save_json(data: Dict[str, Any], filepath: str) -> bool:
    """Save data to JSON file with error handling and backup"""
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Create backup if file exists
        if os.path.exists(filepath):
            backup_path = f"{filepath}.backup"
            os.rename(filepath, backup_path)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)

        # Remove backup on successful save
        backup_path = f"{filepath}.backup"
        if os.path.exists(backup_path):
            os.remove(backup_path)

        return True
    except Exception as e:
        logging.error(f"Failed to save JSON to {filepath}: {e}")

        # Restore backup if save failed
        backup_path = f"{filepath}.backup"
        if os.path.exists(backup_path):
            os.rename(backup_path, filepath)
            logging.info(f"Restored backup for {filepath}")

        return False


def load_json(filepath: str) -> Optional[Dict[str, Any]]:
    """Load data from JSON file with error handling"""
    try:
        if not os.path.exists(filepath):
            logging.warning(f"File not found: {filepath}")
            return None

        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            logging.debug(f"Successfully loaded JSON from {filepath}")
            return data
    except json.JSONDecodeError as e:
        logging.error(f"JSON decode error in {filepath}: {e}")
        return None
    except Exception as e:
        logging.error(f"Failed to load JSON from {filepath}: {e}")
        return None


def retry_api_call(func, *args, max_retries: int = None, delay: float = None, **kwargs) -> Tuple[Any, bool]:
    """
    Retry API calls with exponential backoff
    Returns: (result, success_boolean)
    """
    max_retries = max_retries or Config.MAX_RETRIES
    delay = delay or Config.RETRY_DELAY

    for attempt in range(max_retries + 1):
        try:
            result = func(*args, **kwargs)
            if validate_api_response(result):
                if attempt > 0:
                    logging.info(f"API call succeeded on retry {attempt}")
                return result, True
            else:
                raise ValueError("Invalid API response structure")

        except Exception as e:
            if attempt == max_retries:
                logging.error(f"API call failed after {max_retries} retries: {e}")
                return None, False
            else:
                wait_time = delay * (2 ** attempt)  # Exponential backoff
                logging.warning(f"API call failed (attempt {attempt + 1}), retrying in {wait_time}s: {e}")
                time.sleep(wait_time)

    return None, False


def validate_api_response(response: Any) -> bool:
    """Validate that API response has expected structure"""
    try:
        if response is None:
            return False
        if hasattr(response, 'content') and response.content:
            return True
        if hasattr(response, 'message') and response.message:
            return True
        return False
    except Exception as e:
        logging.error(f"Error validating API response: {e}")
        return False


def extract_text_from_response(response: Any) -> str:
    """Extract text content from API response including tool usage results"""
    try:
        if not response:
            logging.warning("Empty response provided to extract_text_from_response")
            return ""

        # Handle different response structures
        if hasattr(response, 'content') and response.content:
            if isinstance(response.content, list):
                # Collect all text content blocks
                text_parts = []

                for content_block in response.content:
                    if hasattr(content_block, 'type'):
                        if content_block.type == 'text' and hasattr(content_block, 'text'):
                            text_parts.append(content_block.text)
                        elif content_block.type == 'tool_use':
                            # Log tool usage but don't include in response
                            logging.info(f"Tool used: {getattr(content_block, 'name', 'unknown')}")
                    elif hasattr(content_block, 'text'):
                        text_parts.append(content_block.text)
                    else:
                        text_parts.append(str(content_block))

                return '\n'.join(text_parts) if text_parts else ""

            return str(response.content)

        if hasattr(response, 'message'):
            return str(response.message)

        # Fallback
        return str(response)

    except Exception as e:
        logging.error(f"Failed to extract text from response: {e}")
        return ""


def validate_agent_response_format(response_text: str, agent_type = 'orchestrator') -> Tuple[bool, Optional[str], Optional[Dict[str, str]]]:
    """
    Validate orchestrator response format for agent usage
    Returns: (is_valid, error_message, parsed_data)
    """

    def remove_thoughts_sections(text: str) -> str:
        """Remove all 'Thoughts: ... End_Thoughts' sections from text"""
        # Pattern to match thoughts sections (case insensitive, multiline)
        thoughts_pattern = r"Thoughts:\s*.*?\s*End_Thoughts"
        cleaned_text = re.sub(thoughts_pattern, "", text, flags=re.DOTALL | re.IGNORECASE)
        return cleaned_text.strip()

    try:
        response_text = response_text.strip()

        if agent_type == 'orchestrator':
            # Check for "Using Agent:" format
            agent_pattern = r"Using Agent:\s*(\w+)\.\s*Agent prompt:\s*(.+)"
            agent_match = re.search(agent_pattern, response_text, re.DOTALL | re.IGNORECASE)
            if agent_match:
                agent_name = agent_match.group(1).lower()
                agent_prompt_raw = agent_match.group(2).strip()

                # Remove thoughts sections from agent prompt
                agent_prompt = remove_thoughts_sections(agent_prompt_raw)
                valid_agents = ['researcher', 'engineer']
                if agent_name not in valid_agents:
                    return False, f"Invalid agent name: {agent_name}. Valid agents: {valid_agents}", None
                return True, None, {
                    'type': 'agent',
                    'agent_name': agent_name,
                    'agent_prompt': agent_prompt
                }

            # Check for "Concluding Ideas:" format
            conclude_pattern = r"Concluding Ideas:\s*(.+)"
            conclude_match = re.search(conclude_pattern, response_text, re.DOTALL | re.IGNORECASE)
            if conclude_match:
                reasoning = conclude_match.group(1).strip()
                return True, None, {
                    'type': 'conclude',
                    'reasoning': reasoning
                }

            return False, "Response must start with either 'Using Agent: [name]. Agent prompt: [prompt]' or 'Concluding Ideas: [reasoning]'", None

        elif agent_type == 'researcher':
            # Check for "WEB_SEARCHER:" format
            web_searcher_pattern = r"WEB_SEARCHER:\s*(.+)"
            web_searcher_match = re.match(web_searcher_pattern, response_text, re.DOTALL | re.IGNORECASE)
            if web_searcher_match:
                instructions = web_searcher_match.group(1).strip()
                return True, None, {
                    'sub_type': 'WEB_SEARCHER',
                    'instructions': instructions
                }

            # Check for "PHILOSOPHER:" format
            philosopher_pattern = r"PHILOSOPHER:\s*(.+)"
            philosopher_match = re.match(philosopher_pattern, response_text, re.DOTALL | re.IGNORECASE)
            if philosopher_match:
                topic = philosopher_match.group(1).strip()
                return True, None, {
                    'sub_type': 'PHILOSOPHER',
                    'topic': topic
                }

            # Check for "PERSPECTIVE:" format with colon separator
            perspective_pattern = r"PERSPECTIVE:\s*(.+?):\s*(.+)"
            perspective_match = re.match(perspective_pattern, response_text, re.DOTALL | re.IGNORECASE)
            if perspective_match:
                person = perspective_match.group(1).strip()
                topic = perspective_match.group(2).strip()
                return True, None, {
                    'sub_type': 'PERSPECTIVE',
                    'person': person,
                    'topic': topic
                }

            return False, "Invalid researcher response format. Must start with WEB_SEARCHER:, PHILOSOPHER:, or PERSPECTIVE: followed by the appropriate content.", None

        else:
            return False, f"Unsupported agent_type: {agent_type}", None

    except Exception as e:
        logging.error(f"Error validating agent response format: {e}")
        return False, f"Format validation error: {e}", None


def save_error_log(session_dir: str, error_type: str, error_data: Dict[str, Any]) -> bool:
    """Save error information for debugging and recovery"""
    try:
        error_log = {
            'timestamp': datetime.now().isoformat(),
            'error_type': error_type,
            'error_data': error_data
        }

        error_file = os.path.join(session_dir, 'errors', f"error_{error_type}_{int(time.time())}.json")
        return save_json(error_log, error_file)

    except Exception as e:
        logging.error(f"Failed to save error log: {e}")
        return False


def clean_subject_for_filename(subject: str) -> str:
    """Clean subject string to be safe for use in filenames"""
    # Remove special characters and replace spaces with underscores
    cleaned = re.sub(r'[^\w\s-]', '', subject)
    cleaned = re.sub(r'[-\s]+', '_', cleaned)
    return cleaned[:50]  # Limit length for filesystem compatibility


def get_session_filepath(session_dir: str, file_type: str, iteration: Optional[int] = None) -> str:
    """Generate standardized file paths for session data"""
    if file_type == "iteration" and iteration is not None:
        return os.path.join(session_dir, "iterations", f"iteration_{iteration}.json")
    elif file_type == "context":
        return os.path.join(session_dir, "context", "context_history.json")
    elif file_type == "final_report":
        return os.path.join(session_dir, "reports", "final_report.json")
    elif file_type == "agent_response":
        return os.path.join(session_dir, "agents", f"agent_response_{iteration}.json")
    elif file_type == "analytics":
        return os.path.join(session_dir, "analytics", "session_analytics.json")
    elif file_type == "error_log":
        return os.path.join(session_dir, "errors", "error_summary.json")
    else:
        raise ValueError(f"Unknown file_type: {file_type}")


def check_session_recovery(session_dir: str) -> Tuple[bool, Optional[int], Optional[Dict[str, Any]]]:
    """
    Check if session can be recovered and return last completed iteration
    Returns: (can_recover, last_iteration, session_state)
    """
    try:
        if not os.path.exists(session_dir):
            return False, None, None

        # Check for existing iterations
        iterations_dir = os.path.join(session_dir, "iterations")
        if not os.path.exists(iterations_dir):
            return False, None, None

        # Find the highest iteration number
        iteration_files = [f for f in os.listdir(iterations_dir) if f.startswith("iteration_") and f.endswith(".json")]

        if not iteration_files:
            return False, None, None

        # Extract iteration numbers and find the maximum
        iteration_numbers = []
        for filename in iteration_files:
            try:
                num = int(filename.split("_")[1].split(".")[0])
                iteration_numbers.append(num)
            except:
                continue

        if not iteration_numbers:
            return False, None, None

        last_iteration = max(iteration_numbers)

        # Load session state
        context_file = get_session_filepath(session_dir, "context")
        session_state = load_json(context_file)

        logging.info(f"Session recovery possible. Last completed iteration: {last_iteration}")
        return True, last_iteration, session_state

    except Exception as e:
        logging.error(f"Error checking session recovery: {e}")
        return False, None, None


def format_error_message(error_type: str, details: str) -> str:
    """Format consistent error messages for user display"""
    error_messages = {
        'api_failure': f"API communication failed: {details}",
        'format_error': f"Response format invalid: {details}",
        'file_error': f"File operation failed: {details}",
        'validation_error': f"Data validation failed: {details}",
        'system_error': f"System error occurred: {details}"
    }

    return error_messages.get(error_type, f"Unknown error: {details}")


# Initialize logging when module is imported
logger = setup_logging()