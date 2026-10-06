"""Local OpenAI-compatible relay in front of SageMaker AI endpoints (and Bedrock models).

ShoppingBench and lm-eval create an OpenAI client from a fixed base_url / api_key. SageMaker endpoints instead want a
short-lived bearer token (at most 12 h, never longer than the credentials that signed it). Each relay listens on
127.0.0.1:<port>, takes a Chat Completions request, and forwards it with a fresh token, so multi-hour runs never fail
on expiry. A relay can also
  - force fields into every request (force={"chat_template_kwargs": {"enable_thinking": False}} for IFEval), and
  - send the request to a Bedrock model through the Converse API instead (transport="bedrock"), so a frontier model
    without an OpenAI-compatible endpoint runs through the same harness.

    url = start_relay("base", "my-endpoint", 8101, region="us-west-2")      # -> "http://127.0.0.1:8101/v1"
"""
import http.server, json, threading, time
import boto3, requests
from botocore.config import Config

RELAYS = {}


def endpoint_url(region, endpoint):
    return f"https://runtime.sagemaker.{region}.amazonaws.com/endpoints/{endpoint}/openai/v1/chat/completions"


def _call_sagemaker(region, endpoint, body):
    from sagemaker.core.token_generator import generate_token
    body["model"] = ""                                    # the endpoint routes by URL; vLLM accepts an empty name
    for attempt in range(6):                              # retry throttling and transient errors
        r = requests.post(endpoint_url(region, endpoint), json=body, timeout=900,
                          headers={"Authorization": f"Bearer {generate_token(region=region)}"})
        if r.status_code not in (429, 500, 502, 503, 504):
            break
        time.sleep(2 ** attempt)
    return r.status_code, r.content


def _call_bedrock(region, model_id, body):
    """Chat Completions -> Bedrock Converse -> Chat Completions. A fresh client per call picks up refreshed
    credentials during long runs."""
    br = boto3.Session().client("bedrock-runtime", region_name=region,
                                config=Config(read_timeout=900, retries={"max_attempts": 8, "mode": "adaptive"}))
    msgs = body["messages"]
    system = [{"text": m["content"]} for m in msgs if m["role"] == "system"]
    conv = [{"role": m["role"], "content": [{"text": m["content"]}]} for m in msgs if m["role"] in ("user", "assistant")]
    try:
        r = br.converse(modelId=model_id, system=system, messages=conv,
                        inferenceConfig={"maxTokens": int(body.get("max_tokens", 4096)),
                                         "temperature": float(body.get("temperature", 0))})
        text = "".join(b.get("text", "") for b in r["output"]["message"]["content"])
        out = {"id": "bedrock", "object": "chat.completion", "created": int(time.time()), "model": model_id,
               "choices": [{"index": 0, "finish_reason": "length" if r["stopReason"] == "max_tokens" else "stop",
                            "message": {"role": "assistant", "content": text}}]}
        return 200, json.dumps(out).encode()
    except Exception as e:
        return 500, json.dumps({"error": str(e)[:2000]}).encode()


class _Relay(http.server.BaseHTTPRequestHandler):
    region, target, force, transport = None, None, None, "sagemaker"

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        body["stream"] = False
        body.update(self.force or {})
        call = _call_bedrock if self.transport == "bedrock" else _call_sagemaker
        status, content = call(self.region, self.target, body)
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def log_message(self, *a):
        pass


def start_relay(name, target, port, region, force=None, transport="sagemaker"):
    """target: an endpoint name (transport="sagemaker") or a Bedrock model id (transport="bedrock").
    Starting a relay under an existing name replaces it. Returns the base_url for an OpenAI client."""
    stop_relay(name)
    handler = type(f"Relay_{name}", (_Relay,), {"region": region, "target": target, "force": force,
                                                "transport": transport})
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", port), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    RELAYS[name] = srv
    return f"http://127.0.0.1:{port}/v1"


def stop_relay(name):
    srv = RELAYS.pop(name, None)
    if srv:
        srv.shutdown()
        srv.server_close()


def stop_all():
    for name in list(RELAYS):
        stop_relay(name)
