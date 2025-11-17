import subprocess
import time
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Start LiteLLM proxy with specified config file.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the LiteLLM config.yaml file"
    )
    parser.add_argument(
        "--wait",
        type=int,
        default=20,
        help="Seconds to wait for server startup (default: 20)"
    )
    args = parser.parse_args()

    if not os.path.isfile(args.config):
        print(f"❌ Config file not found: {args.config}")
        return

    # Launch LiteLLM in a persistent background shell process
    subprocess.Popen(
        f"nohup litellm --config {args.config} > litellm.log 2>&1 &",
        shell=True,
        executable="/bin/bash"
    )

    print(f"✅ LiteLLM started with config: {args.config}")
    print(f"⏳ Waiting {args.wait} seconds for server to initialize...")
    time.sleep(args.wait)
    os.system("curl http://localhost:4000/models")
    print("🚀 Ready!")

if __name__ == "__main__":
    main()