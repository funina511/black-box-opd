#!/usr/bin/env python3
"""vLLM 0.8.5 server with a read-only identity endpoint on the same process."""
import json
import os
from pathlib import Path


def main():
    import importlib.metadata
    if importlib.metadata.version("vllm") != "0.8.5":
        raise RuntimeError("judge launch/guided JSON was verified with vLLM 0.8.5; validate a new version explicitly")
    import uvloop
    from vllm.entrypoints.openai import api_server
    from vllm.entrypoints.openai.cli_args import make_arg_parser, validate_parsed_serve_args
    from vllm.utils import FlexibleArgumentParser
    from vllm.entrypoints.utils import cli_env_setup
    fingerprint = json.loads(Path(os.environ["GAD_JUDGE_FINGERPRINT_FILE"]).read_text())
    build_app = api_server.build_app
    def build(args):
        app = build_app(args)
        @app.get("/v1/gad_fingerprint")
        async def identity():
            return fingerprint
        return app
    api_server.build_app = build
    cli_env_setup()
    parser = make_arg_parser(FlexibleArgumentParser(description=__doc__))
    args = parser.parse_args()
    validate_parsed_serve_args(args)
    uvloop.run(api_server.run_server(args))


if __name__ == "__main__":
    main()
