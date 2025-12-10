import subprocess
import time
import argparse
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def display_status(message, status="info"):
    if status == "error":
        logging.error(f"[{status.upper()}] {message}")
    elif status == "warning":
        logging.warning(f"[{status.upper()}] {message}")
    else:
        logging.info(f"[{status.upper()}] {message}")

def install_ollama():
    print("Installing Ollama... This may take a minute...")
    result = subprocess.call("curl -fsSL https://ollama.com/install.sh | sh 2>/dev/null", shell=True)
    if result == 0:
        display_status("✅ Ollama installed successfully!", "success")
    else:
        display_status("⚠️ Ollama installation had warnings but may still work", "warning")

def start_ollama_server():
    print("Starting Ollama server...")
    subprocess.Popen(
        "nohup ollama serve > /tmp/ollama_serve_stdout.log 2>/tmp/ollama_serve_stderr.log &",
        shell=True,
        executable="/bin/bash"
    )
    time.sleep(5)

    running = subprocess.call("pgrep -f 'ollama serve' > /dev/null", shell=True)
    if running == 0:
        display_status("✅ Ollama server is running!", "success")
    else:
        display_status("❌ Ollama server failed to start. Check troubleshooting section.", "error")

def pull_models(models):
    for model in models:
        display_status(f"📦 Pulling model: {model}")
        start_time = time.time()
        result = subprocess.call(["ollama", "pull", model])
        end_time = time.time()

        if result == 0:
            elapsed = end_time - start_time
            display_status(f"✅ Model '{model}' downloaded in {elapsed/60:.1f} minutes!", "success")
        else:
            display_status(f"❌ Failed to download model '{model}'.", "error")

def list_models():
    print("\n📋 Available models:")
    time.sleep(30)
    subprocess.call(["ollama", "list"])

def main():
    parser = argparse.ArgumentParser(description="Install and start Ollama, then pull models.")
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="List of models to pull (e.g. hf.co/Qwen/Qwen3-8B-GGUF:Q8_0)"
    )
    args = parser.parse_args()

    install_ollama()
    start_ollama_server()
    pull_models(args.models)
    list_models()

if __name__ == "__main__":
    main()
