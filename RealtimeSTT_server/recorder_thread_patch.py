#!/usr/bin/env python3
"""
Quick fix patch for the server hanging issue.
This replaces the problematic _recorder_thread function with a timeout-protected version.
"""

import time
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

def create_enhanced_recorder_thread(loop, recorder_config, bcolors, audio_queue, preprocess_text, extended_logging):
    """
    Create an enhanced recorder thread that prevents hanging
    """
    def enhanced_recorder_thread():
        global recorder, stop_recorder
        
        # Configuration
        transcription_timeout = 20.0  # 30 seconds timeout
        max_consecutive_failures = 3
        consecutive_failures = 0
        last_activity_time = time.time()
        executor = ThreadPoolExecutor(max_workers=2)
        
        def initialize_recorder_with_retry():
            """Initialize recorder with retry mechanism"""
            nonlocal recorder
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    from RealtimeSTT import AudioToTextRecorder
                    recorder = AudioToTextRecorder(**recorder_config)
                    print(f"{bcolors.OKGREEN}RealtimeSTT initialized successfully{bcolors.ENDC}")
                    return True
                except Exception as e:
                    print(f"{bcolors.FAIL}Failed to initialize recorder (attempt {attempt + 1}/{max_retries}): {e}{bcolors.ENDC}")
                    if attempt < max_retries - 1:
                        time.sleep(2)
            return False
        
        def process_text(full_sentence):
            nonlocal last_activity_time, consecutive_failures
            try:
                import json
                import asyncio
                from datetime import datetime
                
                global prev_text
                prev_text = ""
                full_sentence = preprocess_text(full_sentence)
                message = json.dumps({
                    'type': 'fullSentence',
                    'text': full_sentence
                })
                asyncio.run_coroutine_threadsafe(audio_queue.put(message), loop)

                timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]

                if extended_logging:
                    print(f"  [{timestamp}] Full text: {bcolors.BOLD}Sentence:{bcolors.ENDC} {bcolors.OKGREEN}{full_sentence}{bcolors.ENDC}\n", flush=True, end="")
                else:
                    print(f"\r[{timestamp}] {bcolors.BOLD}Sentence:{bcolors.ENDC} {bcolors.OKGREEN}{full_sentence}{bcolors.ENDC}\n")
                
                # Reset failure counter on success
                consecutive_failures = 0
                last_activity_time = time.time()
                
            except Exception as e:
                print(f"{bcolors.FAIL}Error processing text: {e}{bcolors.ENDC}")
                consecutive_failures += 1
        
        def transcribe_with_timeout():
            """Wrapper for recorder.text() with timeout protection"""
            try:
                if recorder:
                    recorder.text(process_text)
                    return True
                return False
            except Exception as e:
                print(f"{bcolors.FAIL}Transcription error: {e}{bcolors.ENDC}")
                return False
        
        def recover_recorder():
            """Attempt to recover stuck recorder"""
            nonlocal consecutive_failures
            print(f"{bcolors.WARNING}Attempting to recover recorder...{bcolors.ENDC}")
            
            try:
                if recorder:
                    recorder.abort()
                    recorder.clear_audio_queue()
                    time.sleep(1.0)
                    
                return initialize_recorder_with_retry()
                
            except Exception as e:
                print(f"{bcolors.FAIL}Recovery failed: {e}{bcolors.ENDC}")
                consecutive_failures += 1
                return False
        
        # Initialize recorder
        print(f"{bcolors.OKGREEN}Initializing RealtimeSTT server with parameters:{bcolors.ENDC}")
        for key, value in recorder_config.items():
            print(f"    {bcolors.OKBLUE}{key}{bcolors.ENDC}: {value}")
            
        if not initialize_recorder_with_retry():
            print(f"{bcolors.FAIL}Failed to initialize recorder{bcolors.ENDC}")
            return
        
        # Signal that recorder is ready
        recorder_ready.set()
        
        try:
            while not stop_recorder:
                current_time = time.time()
                
                # Health checks
                if current_time - last_activity_time > transcription_timeout:
                    print(f"{bcolors.WARNING}Transcription appears stuck, attempting recovery...{bcolors.ENDC}")
                    if not recover_recorder():
                        print(f"{bcolors.FAIL}Could not recover recorder{bcolors.ENDC}")
                        break
                    last_activity_time = current_time
                    continue
                
                if consecutive_failures >= max_consecutive_failures:
                    print(f"{bcolors.WARNING}Too many consecutive failures ({consecutive_failures}), attempting recovery...{bcolors.ENDC}")
                    if not recover_recorder():
                        print(f"{bcolors.FAIL}Could not recover after {consecutive_failures} failures{bcolors.ENDC}")
                        break
                    continue
                
                # Submit transcription with timeout
                try:
                    future = executor.submit(transcribe_with_timeout)
                    success = future.result(timeout=transcription_timeout)
                    
                    if not success:
                        consecutive_failures += 1
                        print(f"{bcolors.WARNING}Transcription failed (failure count: {consecutive_failures}){bcolors.ENDC}")
                        
                except FutureTimeoutError:
                    print(f"{bcolors.WARNING}Transcription timed out after {transcription_timeout} seconds{bcolors.ENDC}")
                    consecutive_failures += 1
                    
                    # Cancel the future and attempt recovery
                    try:
                        future.cancel()
                    except:
                        pass
                    
                    if not recover_recorder():
                        print(f"{bcolors.FAIL}Could not recover after timeout{bcolors.ENDC}")
                        break
                
                # Small delay to prevent busy loop
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print(f"{bcolors.WARNING}Exiting application due to keyboard interrupt{bcolors.ENDC}")
        finally:
            # Cleanup
            try:
                if recorder:
                    recorder.shutdown()
                executor.shutdown(wait=False)
            except Exception as e:
                print(f"{bcolors.WARNING}Cleanup error: {e}{bcolors.ENDC}")
    
    return threading.Thread(target=enhanced_recorder_thread, name="enhanced_recorder_thread")

# Usage instructions:
# In your stt_server.py, replace the call to:
#   recorder_thread = threading.Thread(target=_recorder_thread, args=(loop,))
# With:
#   recorder_thread = create_enhanced_recorder_thread(loop, recorder_config, bcolors, audio_queue, preprocess_text, extended_logging)
