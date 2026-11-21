"""python -m daisy  ->  serve the demo dataset.

It is served on the IP address of the computer running it (and on localhost), so other computers on the network
can open it;
python -m daisy --host 127.0.0.1 keeps it to this computer only.
"""
import argparse
import socket
import threading
import webbrowser

from shiny import run_app
from .app import demo_app


def network_address() -> str:
    """This computer's address on the local network (what other computers type in their browser)."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.connect(("10.255.255.255", 1))  # no data is sent: it only picks the outgoing network interface
            return probe.getsockname()[0]
    except OSError:
        return "127.0.0.1"


def serve(
    app,
    host: str | None = None,
    port: int = 8000,
    launch_browser: bool = True,
    ssl_certfile: str | None = None,
    ssl_keyfile: str | None = None,
) -> None:
    """Run a Daisy app on this computer's IP address and on localhost (every address, when ``host`` is None).

    Pass ``host="127.0.0.1"`` to keep the app to this computer only. Pass ``ssl_certfile`` and ``ssl_keyfile`` to
    serve HTTPS on the same single port instead of HTTP.
    """
    if bool(ssl_certfile) != bool(ssl_keyfile):
        raise ValueError("HTTPS needs both a certificate file and a key file")
    scheme = "https" if ssl_certfile else "http"
    host = host or "0.0.0.0"
    everywhere = host in ("0.0.0.0", "::")
    address = network_address() if everywhere else host
    print(f"Daisy is running at {scheme}://{address}:{port}", flush=True)
    if everywhere and address != "127.0.0.1":
        print(f"On this computer you can also use {scheme}://localhost:{port}", flush=True)
    if launch_browser:  # opened here because a browser cannot open the listening address 0.0.0.0 on every system
        threading.Timer(1.5, webbrowser.open, args=(f"{scheme}://{address}:{port}",)).start()
    extra = {"ssl_certfile": ssl_certfile, "ssl_keyfile": ssl_keyfile} if ssl_certfile else {}
    run_app(app, host=host, port=port, launch_browser=False, **extra)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="python -m daisy", description="Serve the Daisy demo dataset.")
    parser.add_argument(
        "--host",
        default=None,
        help="address to listen on (default: every address of this computer, so its IP address "
        "and localhost both work; 127.0.0.1 for this computer only)",
    )
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--cert",
        dest="ssl_certfile",
        default=None,
        help="TLS certificate file: with --key, serves HTTPS on the same port",
    )
    parser.add_argument("--key", dest="ssl_keyfile", default=None, help="TLS private key file for --cert")
    parser.add_argument("--no-browser", action="store_true", help="do not open a browser on this computer")
    args = parser.parse_args()
    serve(demo_app(), args.host, args.port, not args.no_browser, args.ssl_certfile, args.ssl_keyfile)
