#! /usr/bin/env python
"""Minimal UCI adapter to talk to Stockfish (or any UCI engine).

Example usage::

    from uci_adapter import UCIEngine

    with UCIEngine(path="stockfish") as engine:
        engine.new_game()
        engine.set_position(fen="8/8/8/8/8/8/8/K6k w - - 0 1")
        bestmove, ponder, _ = engine.go(depth=12)
        print("Engine move:", bestmove)
"""

from __future__ import annotations

import queue
import shlex
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Iterable, List, Optional


@dataclass
class GoResult:
    bestmove: Optional[str]
    ponder: Optional[str]
    info_lines: List[str]


class UCIEngine:
    """Very small helper for driving a UCI engine via stdin/stdout."""

    def __init__(
        self,
        path: str = "stockfish",
        *,
        options: Optional[dict[str, str | int | float]] = None,
        startup_timeout: float = 5.0,
    ) -> None:
        self.path = path
        self.options = options or {}
        self.startup_timeout = startup_timeout
        # We rely on line-buffered text IO for low-latency command exchange.
        self._proc = subprocess.Popen(
            shlex.split(path),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        if self._proc.stdin is None or self._proc.stdout is None:
            raise RuntimeError("Failed to open pipes to engine")

        self._queue: queue.Queue[str] = queue.Queue()
        self._reader = threading.Thread(
            target=self._drain_stdout,
            name="uci-engine-reader",
            daemon=True,
        )
        self._reader.start()

        self._initialize()

    # ------------------------------------------------------------------
    # Context-manager helpers
    def __enter__(self) -> "UCIEngine":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[override]
        self.close()

    # ------------------------------------------------------------------
    def _drain_stdout(self) -> None:
        stdout = self._proc.stdout
        assert stdout is not None
        for line in stdout:
            self._queue.put(line.rstrip("\n"))
        # Once the child process exits we stop enqueuing lines.

    def _send(self, command: str) -> None:
        stdin = self._proc.stdin
        assert stdin is not None
        stdin.write(command + "\n")
        stdin.flush()

    def _collect_until(self, token: str, timeout: float) -> List[str]:
        deadline = time.time() + timeout
        collected: List[str] = []
        while time.time() < deadline:
            try:
                line = self._queue.get(timeout=max(0.0, deadline - time.time()))
            except queue.Empty:
                break
            collected.append(line)
            if token in line:
                return collected
        raise TimeoutError(f"Timed out waiting for '{token}' from engine")

    def _initialize(self) -> None:
        self._send("uci")
        self._collect_until("uciok", self.startup_timeout)
        for name, value in self.options.items():
            self.set_option(name, value)
        self.is_ready()

    # ------------------------------------------------------------------
    def is_alive(self) -> bool:
        return self._proc.poll() is None

    def is_ready(self, timeout: float = 5.0) -> None:
        self._send("isready")
        self._collect_until("readyok", timeout)

    def set_option(self, name: str, value: str | int | float) -> None:
        self._send(f"setoption name {name} value {value}")

    def new_game(self) -> None:
        self._send("ucinewgame")
        # Engines typically expect an isready after ucinewgame.
        self.is_ready()

    def set_position(
        self,
        *,
        startpos: bool = False,
        fen: Optional[str] = None,
        moves: Optional[Iterable[str]] = None,
    ) -> None:
        if startpos and fen:
            raise ValueError("Provide either startpos or fen, not both")
        if not startpos and not fen:
            raise ValueError("Provide a FEN string when startpos is False")

        if startpos:
            command = "position startpos"
        else:
            assert fen is not None
            command = f"position fen {fen}"
        if moves:
            moves_str = " ".join(moves)
            command += f" moves {moves_str}"
        self._send(command)

    def go(
        self,
        *,
        depth: Optional[int] = None,
        movetime: Optional[int] = None,
        nodes: Optional[int] = None,
        ponder: bool = False,
        timeout: float = 30.0,
    ) -> GoResult:
        pieces: List[str] = ["go"]
        if depth is not None:
            pieces.extend(["depth", str(depth)])
        if movetime is not None:
            pieces.extend(["movetime", str(movetime)])
        if nodes is not None:
            pieces.extend(["nodes", str(nodes)])
        if ponder:
            pieces.append("ponder")
        command = " ".join(pieces)
        self._send(command)

        lines = self._collect_until("bestmove", timeout)
        bestmove = None
        ponder_move = None
        for line in lines:
            if line.startswith("bestmove"):
                tokens = line.split()
                if len(tokens) >= 2:
                    bestmove = tokens[1]
                if len(tokens) >= 4 and tokens[2] == "ponder":
                    ponder_move = tokens[3]
                break
        return GoResult(bestmove=bestmove, ponder=ponder_move, info_lines=lines)

    def stop(self) -> None:
        if self.is_alive():
            self._send("stop")

    def close(self) -> None:
        if self.is_alive():
            try:
                self._send("quit")
            except BrokenPipeError:
                pass
        # Give engine a moment to exit cleanly.
        try:
            self._proc.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            self._proc.kill()


def interactive_cli() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Play a quick game against a UCI engine."
    )
    parser.add_argument(
        "engine", nargs="?", default="stockfish", help="Path to the engine executable"
    )
    parser.add_argument("--fen", default="startpos", help="Starting FEN or 'startpos'")
    parser.add_argument(
        "--movetime",
        type=int,
        default=None,
        help="Engine thinking time in milliseconds",
    )
    parser.add_argument(
        "--depth", type=int, default=12, help="Fixed search depth for the engine"
    )
    parser.add_argument(
        "--human",
        choices=("white", "black"),
        default="white",
        help="Human side to play",
    )
    args = parser.parse_args()

    human_side = args.human
    engine_side = "black" if human_side == "white" else "white"

    moves: List[str] = []
    fen_arg = None if args.fen == "startpos" else args.fen

    print("Type moves in long algebraic (e.g. e2e4, g1f3). Enter 'quit' to stop.")

    with UCIEngine(path=args.engine) as engine:
        engine.new_game()
        engine.set_position(startpos=fen_arg is None, fen=fen_arg)
        side_to_move = (
            "white" if (fen_arg is None or fen_arg.split()[1] == "w") else "black"
        )

        while True:
            if side_to_move == human_side:
                user_move = input(f"Your move ({side_to_move}): ").strip()
                if user_move.lower() in {"quit", "exit"}:
                    print("Stopping game.")
                    break
                if len(user_move) < 4:
                    print("Please enter moves in coordinate format like e2e4.")
                    continue
                moves.append(user_move)
                engine.set_position(startpos=fen_arg is None, fen=fen_arg, moves=moves)
                side_to_move = engine_side
            else:
                result = engine.go(depth=args.depth, movetime=args.movetime)
                if result.bestmove is None:
                    print("Engine resigns or has no move.")
                    break
                moves.append(result.bestmove)
                print(f"Engine plays ({side_to_move}): {result.bestmove}")
                engine.set_position(startpos=fen_arg is None, fen=fen_arg, moves=moves)
                side_to_move = human_side


if __name__ == "__main__":
    interactive_cli()
