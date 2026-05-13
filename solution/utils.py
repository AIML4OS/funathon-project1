import logging
import datetime


def setup_logging():
    """Configure logging with both console and file handlers"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                f'funathon_aiml4os_project1_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
            )
        ]
    )
    return logging.getLogger(__name__)


def set_seed():
    return 202605
