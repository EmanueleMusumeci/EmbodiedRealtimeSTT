"""
Enhanced recorder thread with timeout and recovery mechanisms
"""
import threading
import time
import signal
import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import json
from datetime import datetime

logger = logging.getLogger(__name__)

class RecorderManager:
    def __init__(self, loop, recorder_config, bcolors, audio_queue, preprocess_text, extended_logging):
        self.loop = loop
        self.recorder_config = recorder_config
        self.bcolors = bcolors
        self.audio_queue = audio_queue
        self.preprocess_text = preprocess_text
        self.extended_logging = extended_logging
        self.recorder = None
        self.stop_recorder = False
        self.transcription_timeout = 30.0  # 30 second timeout for transcription
        self.health_check_interval = 10.0  # Check health every 10 seconds
        self.max_consecutive_failures = 3
        self.consecutive_failures = 0
        self.last_activity_time = time.time()
        self.is_processing = False
        self.executor = ThreadPoolExecutor(max_workers=2)
        
    def initialize_recorder(self):
        """Initialize the recorder with error handling"""
        try:
            from RealtimeSTT import AudioToTextRecorder
            print(f"{self.bcolors.OKGREEN}Initializing RealtimeSTT server with parameters:{self.bcolors.ENDC}")
            for key, value in self.recorder_config.items():
                print(f"    {self.bcolors.OKBLUE}{key}{self.bcolors.ENDC}: {value}")
            
            self.recorder = AudioToTextRecorder(**self.recorder_config)
            print(f"{self.bcolors.OKGREEN}{self.bcolors.BOLD}RealtimeSTT initialized{self.bcolors.ENDC}")
            self.consecutive_failures = 0
            return True
        except Exception as e:
            logger.error(f"Failed to initialize recorder: {e}")
            self.consecutive_failures += 1
            return False
    
    def process_text_with_timeout(self, process_text_func):
        """Process text with timeout protection"""
        try:
            self.is_processing = True
            self.last_activity_time = time.time()
            
            # Submit the text processing to executor with timeout
            future = self.executor.submit(self.recorder.text, process_text_func)
            future.result(timeout=self.transcription_timeout)
            
            self.consecutive_failures = 0
            return True
            
        except TimeoutError:
            logger.error(f"Transcription timed out after {self.transcription_timeout} seconds")
            self.consecutive_failures += 1
            return False
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            self.consecutive_failures += 1
            return False
        finally:
            self.is_processing = False
    
    def recover_recorder(self):
        """Attempt to recover a stuck recorder"""
        logger.warning("Attempting to recover recorder...")
        try:
            if self.recorder:
                # Try to abort current operations
                self.recorder.abort()
                self.recorder.clear_audio_queue()
                time.sleep(1.0)
                
                # Try to restart
                self.recorder.shutdown()
                time.sleep(2.0)
            
            # Reinitialize
            return self.initialize_recorder()
            
        except Exception as e:
            logger.error(f"Recovery failed: {e}")
            return False
    
    def health_check(self):
        """Check if recorder is healthy"""
        current_time = time.time()
        
        # Check if processing has been stuck for too long
        if self.is_processing and (current_time - self.last_activity_time) > self.transcription_timeout:
            logger.warning("Recorder appears to be stuck")
            return False
            
        # Check if too many consecutive failures
        if self.consecutive_failures >= self.max_consecutive_failures:
            logger.warning(f"Too many consecutive failures: {self.consecutive_failures}")
            return False
            
        return True
    
    def process_text(self, full_sentence):
        """Process transcribed text"""
        global prev_text
        prev_text = ""
        full_sentence = self.preprocess_text(full_sentence)
        message = json.dumps({
            'type': 'fullSentence',
            'text': full_sentence
        })
        
        # Send to audio queue
        import asyncio
        asyncio.run_coroutine_threadsafe(self.audio_queue.put(message), self.loop)

        timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]

        if self.extended_logging:
            print(f"  [{timestamp}] Full text: {self.bcolors.BOLD}Sentence:{self.bcolors.ENDC} {self.bcolors.OKGREEN}{full_sentence}{self.bcolors.ENDC}\n", flush=True, end="")
        else:
            print(f"\r[{timestamp}] {self.bcolors.BOLD}Sentence:{self.bcolors.ENDC} {self.bcolors.OKGREEN}{full_sentence}{self.bcolors.ENDC}\n")
    
    def run(self):
        """Main recorder loop with health monitoring"""
        # Initialize recorder
        if not self.initialize_recorder():
            logger.error("Failed to initialize recorder")
            return
        
        last_health_check = time.time()
        
        try:
            while not self.stop_recorder:
                current_time = time.time()
                
                # Periodic health check
                if current_time - last_health_check > self.health_check_interval:
                    if not self.health_check():
                        logger.warning("Health check failed, attempting recovery...")
                        if not self.recover_recorder():
                            logger.error("Recovery failed, stopping recorder")
                            break
                    last_health_check = current_time
                
                # Process text with timeout protection
                if not self.process_text_with_timeout(self.process_text):
                    logger.warning("Text processing failed or timed out")
                    
                    # If too many failures, attempt recovery
                    if self.consecutive_failures >= self.max_consecutive_failures:
                        if not self.recover_recorder():
                            logger.error("Unable to recover, stopping recorder")
                            break
                
                # Small delay to prevent busy loop
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print(f"{self.bcolors.WARNING}Exiting application due to keyboard interrupt{self.bcolors.ENDC}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        try:
            if self.recorder:
                self.recorder.shutdown()
            self.executor.shutdown(wait=True)
        except Exception as e:
            logger.error(f"Cleanup error: {e}")
    
    def stop(self):
        """Stop the recorder"""
        self.stop_recorder = True


def create_enhanced_recorder_thread(loop, recorder_config, bcolors, audio_queue, preprocess_text, extended_logging):
    """Create and start enhanced recorder thread"""
    manager = RecorderManager(loop, recorder_config, bcolors, audio_queue, preprocess_text, extended_logging)
    
    def recorder_thread():
        manager.run()
    
    thread = threading.Thread(target=recorder_thread, name="enhanced_recorder_thread")
    return thread, manager
