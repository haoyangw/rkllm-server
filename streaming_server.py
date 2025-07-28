import ctypes
import os
import time
import json
import uuid
import threading
import queue # For thread-safe queue
from flask import Flask, request, Response, jsonify, stream_with_context

# Import settings from config.py
from config import *

# --- Global State & Configuration ---
rkllm_lib = None
llm_handle = ctypes.c_void_p()
rkllm_params_global = None
model_initialized = False
conversation_history_bytes = b""
current_assistant_response_parts = [] 

rkllm_lock = threading.Lock() # Lock for critical RKLLM sections and shared state

# --- RKLLM Constants & Structures ---
LLM_RUN_NORMAL = 0
LLM_RUN_FINISH = 1
LLM_RUN_WAITING = 2
LLM_RUN_ERROR = 3

RKLLMInputType = ctypes.c_int
RKLLMInputType.RKLLM_INPUT_PROMPT      = 0
RKLLMInputType.RKLLM_INPUT_TOKEN       = 1
RKLLMInputType.RKLLM_INPUT_EMBED       = 2
RKLLMInputType.RKLLM_INPUT_MULTIMODAL  = 3

RKLLMInferMode = ctypes.c_int
RKLLMInferMode.RKLLM_INFER_GENERATE = 0
RKLLMInferMode.RKLLM_INFER_GET_LAST_HIDDEN_LAYER = 1
RKLLMInferMode.RKLLM_INFER_GET_LOGITS = 2

class RKLLMExtendParam(ctypes.Structure):
    _fields_ = [
        ("base_domain_id", ctypes.c_int32),
        ("embed_flash", ctypes.c_int8),
        ("enabled_cpus_num", ctypes.c_int8),
        ("enabled_cpus_mask", ctypes.c_uint32),
        ("n_batch", ctypes.c_uint8),
        ("use_cross_attn", ctypes.c_int8),
        ("reserved", ctypes.c_uint8 * 104)
    ]

class RKLLMParam(ctypes.Structure):
    _fields_ = [
        ("model_path", ctypes.c_char_p),
        ("max_context_len", ctypes.c_int32),
        ("max_new_tokens", ctypes.c_int32),
        ("top_k", ctypes.c_int32),
        ("n_keep", ctypes.c_int32),
        ("top_p", ctypes.c_float),
        ("temperature", ctypes.c_float),
        ("repeat_penalty", ctypes.c_float),
        ("frequency_penalty", ctypes.c_float),
        ("presence_penalty", ctypes.c_float),
        ("mirostat", ctypes.c_int32),
        ("mirostat_tau", ctypes.c_float),
        ("mirostat_eta", ctypes.c_float),
        ("skip_special_token", ctypes.c_bool),
        ("is_async", ctypes.c_bool),
        ("img_start", ctypes.c_char_p),
        ("img_end", ctypes.c_char_p),
        ("img_content", ctypes.c_char_p),
        ("extend_param", RKLLMExtendParam),
    ]

class RKLLMLoraAdapter(ctypes.Structure):
    _fields_ = [
        ("lora_adapter_path", ctypes.c_char_p),
        ("lora_adapter_name", ctypes.c_char_p),
        ("scale", ctypes.c_float)
    ]

class RKLLMEmbedInput(ctypes.Structure):
    _fields_ = [
        ("embed", ctypes.POINTER(ctypes.c_float)),
        ("n_tokens", ctypes.c_size_t)
    ]

class RKLLMTokenInput(ctypes.Structure):
    _fields_ = [
        ("input_ids", ctypes.POINTER(ctypes.c_int32)),
        ("n_tokens", ctypes.c_size_t)
    ]

class RKLLMMultiModelInput(ctypes.Structure):
    _fields_ = [
        ("prompt", ctypes.c_char_p),
        ("image_embed", ctypes.POINTER(ctypes.c_float)),
        ("n_image_tokens", ctypes.c_size_t),
        ("n_image", ctypes.c_size_t),
        ("image_width", ctypes.c_size_t),
        ("image_height", ctypes.c_size_t)
    ]

class RKLLMInputUnion(ctypes.Union):
    _fields_ = [
        ("prompt_input", ctypes.c_char_p),
        ("embed_input", RKLLMEmbedInput),
        ("token_input", RKLLMTokenInput),
        ("multimodal_input", RKLLMMultiModelInput)
    ]

class RKLLMInput(ctypes.Structure):
    _fields_ = [
        ("role", ctypes.c_char_p),
        ("enable_thinking", ctypes.c_bool),
        ("input_type", RKLLMInputType),
        ("input_data", RKLLMInputUnion)
    ]

class RKLLMLoraParam(ctypes.Structure):
    _fields_ = [
        ("lora_adapter_name", ctypes.c_char_p)
    ]

class RKLLMPromptCacheParam(ctypes.Structure):
    _fields_ = [
        ("save_prompt_cache", ctypes.c_int),
        ("prompt_cache_path", ctypes.c_char_p)
    ]

class RKLLMInferParam(ctypes.Structure):
    _fields_ = [
        ("mode", RKLLMInferMode),
        ("lora_params", ctypes.POINTER(RKLLMLoraParam)),
        ("prompt_cache_params", ctypes.POINTER(RKLLMPromptCacheParam)),
        ("keep_history", ctypes.c_int)
    ]

class RKLLMResultLastHiddenLayer(ctypes.Structure):
    _fields_ = [
        ("hidden_states", ctypes.POINTER(ctypes.c_float)),
        ("embd_size", ctypes.c_int),
        ("num_tokens", ctypes.c_int)
    ]

class RKLLMResultLogits(ctypes.Structure):
    _fields_ = [
        ("logits", ctypes.POINTER(ctypes.c_float)),
        ("vocab_size", ctypes.c_int),
        ("num_tokens", ctypes.c_int)
    ]

class RKLLMPerfStat(ctypes.Structure):
    _fields_ = [
        ("prefill_time_ms", ctypes.c_float),
        ("prefill_tokens", ctypes.c_int),
        ("generate_time_ms", ctypes.c_float),
        ("generate_tokens", ctypes.c_int),
        ("memory_usage_mb", ctypes.c_float)
    ]

class RKLLMResult(ctypes.Structure):
    _fields_ = [
        ("text", ctypes.c_char_p),
        ("token_id", ctypes.c_int),
        ("last_hidden_layer", RKLLMResultLastHiddenLayer),
        ("logits", RKLLMResultLogits),
        ("perf", RKLLMPerfStat)
    ]
    
LLMResultCallback = ctypes.CFUNCTYPE(None, ctypes.POINTER(RKLLMResult),
    ctypes.c_void_p, ctypes.c_int)

SYS_PROMPT_TEMPLATE = b"<bos>{system_message}"
USER_TURN_START = b"<start_of_turn>user\n"
USER_TURN_END = b"<end_of_turn>\n"
ASSISTANT_TURN_START = b"<start_of_turn>model\n"
ASSISTANT_TURN_END = b"<end_of_turn>\n"
DEFAULT_SYSTEM_MESSAGE = SYSTEM_PROMPT

stream_token_queue = queue.Queue()

@LLMResultCallback
def api_llm_callback(result_ptr, userdata, state):
    global current_assistant_response_parts, stream_token_queue
    # print(f"[Callback {threading.get_ident()}] State: {state}, Qsize: {stream_token_queue.qsize()}")
    if not result_ptr:
        stream_token_queue.put(None) 
        return

    result = result_ptr.contents
    if state == LLM_RUN_NORMAL:
        if result.text:
            try:
                decoded_text = result.text.decode('utf-8', errors='replace')
                current_assistant_response_parts.append(decoded_text) 
                stream_token_queue.put(decoded_text)      
            except Exception as e:
                print(f"[Callback Error] Decoding text: {e}")
                stream_token_queue.put(None) 
    elif state == LLM_RUN_FINISH:
        stream_token_queue.put("__LLM_RUN_FINISH__") 
    elif state == LLM_RUN_ERROR:
        stream_token_queue.put(None) 

def find_library_path(lib_name):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    paths_to_check = ['.', script_dir, os.path.join(script_dir, 'aarch64'), os.path.join(script_dir, 'lib'), '/usr/lib', '/usr/local/lib']
    if 'LD_LIBRARY_PATH' in os.environ: paths_to_check.extend(os.environ['LD_LIBRARY_PATH'].split(os.pathsep))
    for p_item in dict.fromkeys(paths_to_check):
        _lp = os.path.join(p_item, lib_name)
        if os.path.exists(_lp): return _lp
    return None

def init_rkllm_model():
    global rkllm_lib, llm_handle, rkllm_params_global, model_initialized, conversation_history_bytes
    if model_initialized: return True
    
    # Use the library path from config
    lib_path = LIBRARY_PATH
    # Fall back to path search if config path doesn't exist
    if not os.path.exists(lib_path):
        lib_path = find_library_path('librkllmrt.so')
    
    if not lib_path: print("Error: librkllmrt.so not found."); return False
    try: rkllm_lib = ctypes.CDLL(lib_path); print(f"Loaded RKLLM library from '{lib_path}'")
    except OSError as e: print(f"Error loading RKLLM library: {e}"); return False

    rkllm_lib.rkllm_createDefaultParam.restype = RKLLMParam
    rkllm_lib.rkllm_init.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(RKLLMParam), LLMResultCallback]; rkllm_lib.rkllm_init.restype = ctypes.c_int
    rkllm_lib.rkllm_run.argtypes = [ctypes.c_void_p, ctypes.POINTER(RKLLMInput), ctypes.POINTER(RKLLMInferParam), ctypes.c_void_p]; rkllm_lib.rkllm_run.restype = ctypes.c_int
    rkllm_lib.rkllm_destroy.argtypes = [ctypes.c_void_p]; rkllm_lib.rkllm_destroy.restype = ctypes.c_int
    rkllm_lib.rkllm_clear_kv_cache.argtypes = [ctypes.c_void_p, ctypes.c_int]; rkllm_lib.rkllm_clear_kv_cache.restype = ctypes.c_int
    
    rkllm_params_global = rkllm_lib.rkllm_createDefaultParam()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Use model path from config
    model_relative_path = MODEL_PATH
    if not os.path.exists(model_relative_path): 
        model_relative_path = "../model/Qwen3-0.6B-w8a8-opt1-hybrid1-npu3.rkllm"
    
    model_abs_path = os.path.join(script_dir, model_relative_path)
    model_canonical_path = os.path.normpath(model_abs_path)
    if not os.path.exists(model_canonical_path): print(f"Error: Model file not found: '{model_canonical_path}'"); return False
    
    rkllm_params_global.top_k = 64
    rkllm_params_global.top_p = 0.95
    rkllm_params_global.temperature = 0.7
    rkllm_params_global.repeat_penalty = 1.0
    rkllm_params_global.frequency_penalty = 1.0
    rkllm_params_global.presence_penalty = 0.0
    rkllm_params_global.mirostat = 0
    rkllm_params_global.mirostat_tau = 5.0
    rkllm_params_global.mirostat_eta = 0.1
    rkllm_params_global.model_path = model_canonical_path.encode('utf-8')
    rkllm_params_global.max_context_len = MAX_CONTEXT_LENGTH
    rkllm_params_global.n_keep = N_KEEP
    rkllm_params_global.max_new_tokens = MAX_NEW_TOKENS
    
    # --- SETTING is_async TO FALSE (like manufacturer's example) ---
    rkllm_params_global.is_async = False 
    # ---

    os.environ['RKLLM_LOG_LEVEL'] = '1' 
    print(f"RKLLM_LOG_LEVEL: {os.environ['RKLLM_LOG_LEVEL']}")
    print(f"Initializing model with parameters: \n"
          f"  Path: {rkllm_params_global.model_path.decode()}\n"
          f"  Max Context: {rkllm_params_global.max_context_len}\n"
          f"  Max New Tokens: {rkllm_params_global.max_new_tokens}\n"
          f"  N_Keep: {rkllm_params_global.n_keep}\n"
          f"  Is Async (Library Flag): {rkllm_params_global.is_async}") # Log async setting
    
    ret = rkllm_lib.rkllm_init(ctypes.byref(llm_handle), ctypes.byref(rkllm_params_global), api_llm_callback)
    if ret != 0: print(f"Error: rkllm_init failed (code {ret})."); return False
    
    conversation_history_bytes = SYS_PROMPT_TEMPLATE.replace(b"{system_message}", DEFAULT_SYSTEM_MESSAGE.encode('utf-8'))
    model_initialized = True
    print("RKLLM model initialized.")
    return True

app = Flask(__name__)

def format_openai_error_response(message, err_type="invalid_request_error", param=None, code=None):
    error_obj = {"error": {"message": message, "type": err_type}}
    if param: error_obj["error"]["param"] = param
    if code: error_obj["error"]["code"] = code
    return jsonify(error_obj)

# --- Worker function to run RKLLM inference in a separate thread ---
def rkllm_inference_worker(handle, input_obj_ptr, infer_params_obj_ptr):
    """
    This function will be run in a separate thread for streaming requests.
    It calls the blocking rkllm_run. Callbacks will populate the queue.
    """
    thread_id = threading.get_ident()
    print(f"[Worker {thread_id}] Starting rkllm_run. Timestamp: {time.time()}")
    run_ret = rkllm_lib.rkllm_run(handle, input_obj_ptr, infer_params_obj_ptr, None)
    print(f"[Worker {thread_id}] rkllm_run finished with code: {run_ret}. Timestamp: {time.time()}")
    if run_ret != 0:
        # If run fails, callback might not send FINISH. Put an error signal.
        print(f"[Worker {thread_id}] rkllm_run error, ensuring error signal in queue.")
        stream_token_queue.put(None) # Signal error to streamer
    # The __LLM_RUN_FINISH__ or None (for error) should be put by the callback itself.
    # This function just ensures rkllm_run is called.

@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions_handler():
    global conversation_history_bytes, current_assistant_response_parts, rkllm_params_global, llm_handle, rkllm_lib, stream_token_queue
    
    if not model_initialized: return format_openai_error_response("RKLLM model not initialized.", "api_error"), 500
    try: request_data = request.get_json()
    except Exception as e: return format_openai_error_response(f"Invalid JSON: {e}", "invalid_request_error"), 400
    if not request_data: return format_openai_error_response("Request body missing or not JSON.", "invalid_request_error"), 400

    messages = request_data.get("messages", [])
    stream_requested = request_data.get("stream", False)
    
    if not messages or messages[-1].get("role") != "user": return format_openai_error_response("Last message must be 'user'.", "invalid_request_error", param="messages"), 400
    last_user_message_content = messages[-1].get("content", "")
    if not last_user_message_content.strip(): return format_openai_error_response("User content is empty.", "invalid_request_error", param="messages.content"), 400

    # rkllm_lock ensures that context management, rkllm_run call, and history updates are atomic per request.
    with rkllm_lock: 
        current_user_turn_bytes = USER_TURN_START + last_user_message_content.encode('utf-8') + USER_TURN_END
        prompt_for_llm = conversation_history_bytes + current_user_turn_bytes + ASSISTANT_TURN_START
        
        EFFECTIVE_PROMPT_TOKEN_LIMIT = 500 
        estimated_prompt_tokens = len(prompt_for_llm) / 3.5
        
        if estimated_prompt_tokens > EFFECTIVE_PROMPT_TOKEN_LIMIT:
            print(f"\n[API Ctx Mgmt] Est. prompt tokens ({int(estimated_prompt_tokens)}) > limit ({EFFECTIVE_PROMPT_TOKEN_LIMIT}). Clearing KV cache.")
            clear_ret = rkllm_lib.rkllm_clear_kv_cache(llm_handle, 1) 
            if clear_ret == 0:
                print("[API Ctx Mgmt] KV cache cleared.")
                conversation_history_bytes = SYS_PROMPT_TEMPLATE.replace(b"{system_message}", DEFAULT_SYSTEM_MESSAGE.encode('utf-8'))
                prompt_for_llm = conversation_history_bytes + current_user_turn_bytes + ASSISTANT_TURN_START
                print(f"[API Ctx Mgmt] History reset. New prompt est: {int(len(prompt_for_llm)/3.5)} tokens.")
            else:
                print(f"[API Ctx Mgmt] Error clearing KV cache (code: {clear_ret}).")
        
        current_assistant_response_parts.clear()
        while not stream_token_queue.empty():
            try: stream_token_queue.get_nowait()
            except queue.Empty: break
        
        # These objects are passed to the C function, ensure they exist for the duration of the call.
        # For threaded calls, they must persist until the thread is done.
        # ctypes.byref creates a light-weight pointer. The original Python objects must be kept alive.
        rkllm_input_obj = RKLLMInput() 
        rkllm_input_obj.role = "user".encode('utf-8')
        rkllm_input_obj.enable_thinking = ctypes.c_bool(False)
        rkllm_input_obj.input_mode = RKLLMInputType.RKLLM_INPUT_PROMPT
        rkllm_input_obj.input_data.prompt_input = ctypes.c_char_p(prompt_for_llm)
        
        rkllm_infer_params_obj = RKLLMInferParam()
        rkllm_infer_params_obj.mode = RKLLMInferMode.RKLLM_INFER_GENERATE
        rkllm_infer_params_obj.keep_history = 1
        
        request_id = f"chatcmpl-{uuid.uuid4().hex}"
        model_name_to_return = rkllm_params_global.model_path.decode().split('/')[-1] if rkllm_params_global else "rkllm_model"
        created_ts = int(time.time())
        
        run_ret_holder = {"value": 0} # To get return value from worker thread if needed, not strictly used now

        if stream_requested:
            # For streaming, run rkllm_run in a separate thread.
            # The main Flask thread will return the generator that pulls from the queue.
            print(f"[API Handler {threading.get_ident()}] Starting worker thread for rkllm_run. Timestamp: {time.time()}")
            
            # Keep references to pass to thread
            input_obj_ptr = ctypes.byref(rkllm_input_obj)
            infer_params_obj_ptr = ctypes.byref(rkllm_infer_params_obj)

            worker_thread = threading.Thread(target=rkllm_inference_worker, 
                                             args=(llm_handle, input_obj_ptr, infer_params_obj_ptr))
            worker_thread.start()
            # The main thread does not wait for worker_thread to finish here.
            # It proceeds to return the streaming response.

            def generate_stream_response_sse():
                global conversation_history_bytes # To update after stream
                
                print(f"[Stream SSE Gen {threading.get_ident()}] Starting generator. Timestamp: {time.time()}")
                streamed_response_for_history_update = [] 
                try:
                    while True:
                        try:
                            token_or_signal = stream_token_queue.get(timeout=45.0) # Increased timeout
                        except queue.Empty:
                            print(f"[Stream SSE Gen {threading.get_ident()}] Timeout waiting for item from queue. Timestamp: {time.time()}")
                            break 

                        if token_or_signal is None: 
                            print(f"[Stream SSE Gen {threading.get_ident()}] Received None (error/end signal). Timestamp: {time.time()}")
                            break 
                        
                        if token_or_signal == "__LLM_RUN_FINISH__":
                            print(f"[Stream SSE Gen {threading.get_ident()}] Received FINISH signal. Timestamp: {time.time()}")
                            final_chunk = {"id": request_id, "object": "chat.completion.chunk", "created": created_ts, "model": model_name_to_return, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
                            yield f"data: {json.dumps(final_chunk)}\n\n"
                            break
                        
                        streamed_response_for_history_update.append(token_or_signal) 
                        chunk = {"id": request_id, "object": "chat.completion.chunk", "created": created_ts, "model": model_name_to_return, "choices": [{"index": 0, "delta": {"content": token_or_signal}, "finish_reason": None}]}
                        yield f"data: {json.dumps(chunk)}\n\n"
                        # time.sleep(0.005) # Minimal sleep to allow other threads, if any, to process. Can make streaming more "visible".
                
                except Exception as e_stream:
                    print(f"[Stream SSE Gen {threading.get_ident()}] Error: {e_stream}. Timestamp: {time.time()}")
                finally:
                    print(f"[Stream SSE Gen {threading.get_ident()}] Sending [DONE]. Timestamp: {time.time()}")
                    yield f"data: [DONE]\n\n"
                    
                    # Update global conversation history after stream is complete.
                    # This needs to be done carefully with the rkllm_lock.
                    # The worker thread might still be running or just finished.
                    # The lock is acquired by the main request handler.
                    # This update should ideally happen once the worker thread confirms completion.
                    # For now, we assume current_assistant_response_parts got populated by the callbacks
                    # which were triggered by the worker thread.
                    
                    # The lock is held by the parent (chat_completions_handler)
                    # We need to ensure the worker_thread has finished before updating history based on its work.
                    # However, we can't block the stream yield for worker_thread.join().
                    # The current_assistant_response_parts is filled by the callback.
                    # If rkllm_run was successful (worker didn't crash), current_assistant_response_parts should be accurate.
                    
                    # Let's wait for the worker thread here before updating history.
                    # This means the [DONE] signal might be delayed until the worker thread fully exits.
                    print(f"[Stream SSE Gen {threading.get_ident()}] Waiting for worker thread to complete before history update. Timestamp: {time.time()}")
                    worker_thread.join(timeout=5.0) # Wait for worker to finish
                    if worker_thread.is_alive():
                        print(f"[Stream SSE Gen {threading.get_ident()}] WARNING: Worker thread did not complete in time after stream.")

                    # Now that worker is done (or timed out), current_assistant_response_parts should be stable.
                    # The lock is still held by the main request handler.
                    full_response_this_turn = "".join(current_assistant_response_parts) # These parts were for this turn
                    if full_response_this_turn: 
                        new_segment = current_user_turn_bytes + ASSISTANT_TURN_START + full_response_this_turn.encode('utf-8') + ASSISTANT_TURN_END
                        conversation_history_bytes += new_segment
                        print(f"[API History Update after stream generation] History bytes: {len(conversation_history_bytes)}")


            return Response(stream_with_context(generate_stream_response_sse()), mimetype='text/event-stream')
        
        else: # One-shot response (rkllm_run is blocking if is_async=False)
            print(f"[API Handler {threading.get_ident()}] Calling rkllm_run (sync). Timestamp: {time.time()}")
            run_start_time = time.time()
            run_ret = rkllm_lib.rkllm_run(llm_handle, ctypes.byref(rkllm_input_obj), ctypes.byref(rkllm_infer_params_obj), None)
            run_end_time = time.time()
            print(f"[API Handler {threading.get_ident()}] rkllm_run (sync) returned: {run_ret}. Duration: {run_end_time - run_start_time:.4f}s. Qsize: {stream_token_queue.qsize()}. Timestamp: {time.time()}")

            if run_ret != 0:
                print(f"[API Error] rkllm_run (sync) failed (code {run_ret}).")
                return format_openai_error_response(f"RKLLM run failed (code: {run_ret}).", "api_error"), 500

            full_assistant_response_str = "".join(current_assistant_response_parts)
            if run_ret == 0: 
                new_segment = current_user_turn_bytes + ASSISTANT_TURN_START + full_assistant_response_str.encode('utf-8') + ASSISTANT_TURN_END
                conversation_history_bytes += new_segment

            prompt_tokens_est = len(prompt_for_llm) // 4 
            completion_tokens_est = len(full_assistant_response_str) // 4
            response_json = {"id": request_id, "object": "chat.completion", "created": created_ts, "model": model_name_to_return, "choices": [{"index": 0, "message": {"role": "assistant", "content": full_assistant_response_str}, "finish_reason": "stop" }], "usage": {"prompt_tokens": prompt_tokens_est, "completion_tokens": completion_tokens_est, "total_tokens": prompt_tokens_est + completion_tokens_est}}
            return jsonify(response_json)

# --- Main Entry Point ---
if __name__ == '__main__':
    if init_rkllm_model():
        print("Starting Flask server for RKLLM OpenAI-compliant API on http://0.0.0.0:5001/v1/chat/completions")
        app.run(host=SERVER_HOST, port=SERVER_PORT, threaded=True, debug=False) 
    else:
        print("Failed to initialize RKLLM model. Server not starting.")

