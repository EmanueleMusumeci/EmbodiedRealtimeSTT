"""
Language detection and audio enhancement using SpeechBrain
"""
try:
    import torch
    import numpy as np
    from speechbrain.pretrained import LanguageIdentification
    SPEECHBRAIN_AVAILABLE = True
except ImportError:
    SPEECHBRAIN_AVAILABLE = False
    torch = None
    np = None

import logging

logger = logging.getLogger(__name__)

class LanguageDetector:
    def __init__(self, device="cuda" if SPEECHBRAIN_AVAILABLE and torch.cuda.is_available() else "cpu"):
        self.device = device
        self.lang_id = None
        self.last_confidence = 0.0
        
        if not SPEECHBRAIN_AVAILABLE:
            logger.warning("SpeechBrain not available. Language detection disabled.")
            return
            
        try:
            # Initialize SpeechBrain language identification model
            self.lang_id = LanguageIdentification.from_hparams(
                source="speechbrain/lang-id-voxlingua107-ecapa",
                savedir="pretrained_models/lang-id-voxlingua107-ecapa",
                run_opts={"device": device}
            )
            logger.info(f"Language detection model loaded on {device}")
        except Exception as e:
            logger.error(f"Failed to load language detection model: {e}")
            self.lang_id = None
    
    def detect_language(self, audio_data, sample_rate=16000):
        """
        Detect language from audio data
        
        Args:
            audio_data: numpy array of audio samples
            sample_rate: sample rate of audio
            
        Returns:
            tuple: (detected_language_code, confidence_score)
        """
        if self.lang_id is None or not SPEECHBRAIN_AVAILABLE:
            return None, 0.0
            
        try:
            # Ensure audio is the right format
            if isinstance(audio_data, bytes):
                audio_data = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Convert to tensor
            audio_tensor = torch.tensor(audio_data).unsqueeze(0).to(self.device)
            
            # Predict language
            prediction = self.lang_id.classify_batch(audio_tensor)
            
            # Extract language code and confidence
            lang_code = prediction[0][0]
            confidence = float(prediction[1][0].max())
            self.last_confidence = confidence
            
            # Map to common language codes
            lang_mapping = {
                'it': 'it',
                'en': 'en', 
                'es': 'es',
                'fr': 'fr',
                'de': 'de'
            }
            
            mapped_lang = lang_mapping.get(lang_code, lang_code)
            
            return mapped_lang, confidence
            
        except Exception as e:
            logger.error(f"Language detection failed: {e}")
            return None, 0.0
    
    def validate_transcription_language(self, text, expected_language, detected_language, confidence_threshold=0.7):
        """
        Validate if transcription matches expected language
        
        Args:
            text: transcribed text
            expected_language: expected language code
            detected_language: detected language from audio
            confidence_threshold: minimum confidence for language detection
            
        Returns:
            bool: True if language is consistent, False otherwise
        """
        if detected_language is None:
            return True  # Can't validate, assume correct
            
        # If confidence is low, don't make decisions based on detection
        if confidence_threshold and self.last_confidence < confidence_threshold:
            return True
        
        # Simple heuristic: if detected language differs from expected, flag it
        if detected_language != expected_language:
            logger.warning(f"Language mismatch: expected {expected_language}, detected {detected_language}")
            return False
            
        return True
