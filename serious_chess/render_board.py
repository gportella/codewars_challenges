#!/usr/bin/env python
import sys
from base64 import standard_b64encode
from typing import Optional
import cairosvg
import os
import chess.svg
from imgcat import imgcat

from types_and_masks import U64, iter_bits


def serialize_gr_command(**cmd):
    payload = cmd.pop("payload", None)
    cmd = ",".join(f"{k}={v}" for k, v in cmd.items())
    ans = []
    w = ans.append
    w(b"\033_G"), w(cmd.encode("ascii"))
    if payload:
        w(b";")
        w(payload)
    w(b"\033\\")
    return b"".join(ans)


def write_chunked(**cmd):
    data = standard_b64encode(cmd.pop("data"))
    while data:
        chunk, data = data[:4096], data[4096:]
        m = 1 if data else 0
        sys.stdout.buffer.write(serialize_gr_command(payload=chunk, m=m, **cmd))
        sys.stdout.flush()
        cmd.clear()
    sys.stdout.write("\n")


def detect_terminal():
    """
    Detects the terminal emulator the Python script is running in.
    """
    # Check for VS Code integrated terminal
    if "TERM_PROGRAM" in os.environ and os.environ["TERM_PROGRAM"] == "vscode":
        return "VS Code Integrated Terminal"
    if "TERM" in os.environ and "kitty" in os.environ["TERM"].lower():
        return "kitty"


def show_fancy_board(fen, attacks: Optional[U64] = None, size=300):
    terminal = detect_terminal()
    board = chess.Board(fen)

    squares = {x: "#da131364" for x in iter_bits(int(attacks))} if attacks else None
    if squares:
        cb = chess.svg.board(board, fill=squares)
    else:
        cb = chess.svg.board(board)
    ib = cairosvg.svg2png(cb, output_width=size, output_height=size)
    if terminal == "kitty":
        write_chunked(a="T", f=100, data=ib)
    else:
        imgcat(ib)


if __name__ == "__main__":
    show_fancy_board("r1bqkbnr/pppppppp/n7/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
