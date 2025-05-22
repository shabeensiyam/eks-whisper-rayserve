import logging

from whisper_asr import WhisperASR
from whisper_service import WhisperService

logger = logging.getLogger("whisper_service")
logger.setLevel(logging.INFO)


def create_app():
    """Create the composed application."""
    logger.info('Creating application...')
    # Create the Whisper ASR deployment
    whisper_asr = WhisperASR.bind()
    # Create the service layer with FastAPI
    service = WhisperService.bind(whisper_asr)

    logger.info('Application created successfully')
    return service
