import argparse
import socket
import shutil
import subprocess
import time

from pathlib import Path
from urllib.error import URLError, HTTPError
from urllib.request import urlopen


EVAL_DIR = Path(__file__).resolve().parent
DEFAULT_COMPOSE_FILE = EVAL_DIR.parent / "docker-compose.yml"


def _detect_compose_command() -> list[str]:
    docker = shutil.which("docker")
    if docker:
        try:
            subprocess.run(
                [docker, "compose", "version"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
            )
            return [docker, "compose"]
        except subprocess.CalledProcessError:
            pass

    docker_compose = shutil.which("docker-compose")
    if docker_compose:
        return [docker_compose]

    raise RuntimeError("Docker Compose 未安装，无法自动部署 TMem 完整架构依赖")


def _wait_for_http(url: str, timeout_seconds: float) -> None:
    deadline = time.time() + timeout_seconds
    last_error = None

    while time.time() < deadline:
        try:
            with urlopen(url, timeout=3) as response:
                if response.status < 500:
                    return
        except TimeoutError as error:
            last_error = error
        except socket.timeout as error:
            last_error = error
        except HTTPError as error:
            if error.code < 500:
                return
            last_error = error
        except URLError as error:
            last_error = error

        time.sleep(2)

    raise RuntimeError(f"等待服务就绪超时: {url}; last_error={last_error}")


def _wait_for_tcp(host: str, port: int, timeout_seconds: float) -> None:
    deadline = time.time() + timeout_seconds
    last_error = None

    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=3):
                return
        except OSError as error:
            last_error = error

        time.sleep(2)

    raise RuntimeError(f"等待服务就绪超时: {host}:{port}; last_error={last_error}")


def ensure_tmem_full_infra(
    compose_file: str | Path = DEFAULT_COMPOSE_FILE,
    timeout_seconds: float = 120.0,
) -> None:
    compose_path = Path(compose_file).resolve()
    if not compose_path.exists():
        raise FileNotFoundError(f"docker-compose.yml 不存在: {compose_path}")

    compose_cmd = _detect_compose_command()
    subprocess.run(
        [*compose_cmd, "-f", str(compose_path), "up", "-d", "neo4j", "qdrant"],
        cwd=compose_path.parent,
        check=True,
    )

    _wait_for_tcp("127.0.0.1", 17687, timeout_seconds)
    _wait_for_http("http://127.0.0.1:16333/collections", timeout_seconds)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ensure TMem full-architecture Docker infrastructure is running")
    parser.add_argument(
        "--compose-file",
        type=str,
        default=str(DEFAULT_COMPOSE_FILE),
        help="Path to docker-compose.yml",
    )
    parser.add_argument("--timeout", type=float, default=120.0, help="Timeout in seconds")
    args = parser.parse_args()

    ensure_tmem_full_infra(compose_file=args.compose_file, timeout_seconds=args.timeout)


if __name__ == "__main__":
    main()