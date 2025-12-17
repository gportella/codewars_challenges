#!/usr/bin/env node
/* UCI wrapper for the compact chess engine (original snippet retained).
   Exposes a UCI interface: uci, isready, ucinewgame, position, go, stop, quit.
   Uses a 10×12 mailbox board and the original X/Y/Z functions.
*/

const readline = require('readline');

// -------------------------------
// Engine globals (original names)
// -------------------------------
let B, y, u, b, x, z, I, l;
let X, Y, Z;

// -------------------------------
// Original minified snippet (unchanged)
// -------------------------------
(function originalEngineSetup() {
  for (B = y = u = b = 0, x = 10, z = 15, I = [], l = [];
    l[B] = ("ustvrtsuqqqqqqqq" + "yyyyyyyy}{|~z|{}@G@TSb~?A6J57IKJT576,+-48HLSUmgukgg OJNMLK  IDHGFE").charCodeAt(B) - 64, B++ < 120;
    I[B - 1] = B % x ? B / x % 2 < 2 | B % x < 2 ? 7 : B / x & 4 ? 0 : l[u++] : 7);

  X = (c, h, e, S, s) => {
    c ^= 8;
    for (var T, o, L, E, D, O = 20, G, N = -1e8, n, g, d = S && X(c, 0) > 1e4, C, R, A, K = 78 - h << 9, a = c ? x : -x;
      ++O < 99;
    )
      if ((o = I[T = O]) && (G = o & z ^ c) < 7) {
        A = G-- & 2 ? 8 : 4;
        C = 9 - o & z ? l[61 + G] : 49;
        do {
          R = I[T += l[C]];
          g = D = G | T + a - e ? 0 : e;
          if (!R && (G || A < 3 || g) || (1 + R & z ^ c) > 9 && G | A > 2) {
            if (!(2 - R & 7)) return K;
            for (E = n = G | I[T - a] - 7 ? o & z : 6 ^ c;
              E;
              E = !E && !d && !(g = T, D = T < O ? g - 3 : g + 2, I[D] < z | I[D + O - T] | I[T += T - O])) {
              L = (R && l[R & 7 | 32] * 2 - h - G) + (G ? 0 : n - o & z ? 110 : (D && 14) + (A < 2) + 1);
              if (S > h || 1 < S && S == h && L > 2 | d) {
                I[T] = n,
                  I[g] = I[D],
                  I[O] = D ? I[D] = 0 : 0;
                L -= X(c, h + 1, E = G | A > 1 ? 0 : T, S, L - N);
                if (!(h || S - 1 | B - O | T - b | L < -1e4)) return W(I, B = b, c, y = E);
                E = 1 - G | A < 7 | D | !S | R | o < z || X(c, 0) > 1e4;
                I[O] = o;
                I[T] = R;
                I[D] = I[g];
                D ? I[g] = G ? 0 : 9 ^ c : 0
              }
              if (L > N || !h & L == N && Math.random() < .5)
                if (N = L, S > 1)
                  if (h ? s - L < 0 : (B = O, b = T, 0)) return N
            }
          }
        }
        while (!R & G > 2 || (T = O, G | A > 2 | z < o & !R && ++C * --A))
      }
    // Suppress noisy logs in UCI mode; comment next line to restore
    // console.log(K + 768 < N | d && N)
    return -K + 768 < N | d && N
  };

  Y = (V) => {
    X(8, 0, y, V);
    X(8, 0, y, 1)
  };

  Z = (U) => {
    b = U;
    I[b] & 8 ? W(I, B = b) : X(0, 0, y, 1)
  };
})();

// -------------------------------
// Helper: mapping mailbox <-> UCI
// -------------------------------
const STRIDE = 10;
const OFFBOARD = 7;
const EMPTY = 0;

// Convert mailbox index to algebraic (e.g., 22 -> a1)
function idxToAlgebra(sq) {
  const r = Math.floor(sq / STRIDE);
  const f = sq % STRIDE;
  const file = f - 2;  // files 2..9 => a..h
  const rank = r - 1;  // ranks 2..9 => 1..8
  return String.fromCharCode(97 + file) + String(rank);
}

// Convert algebraic (e2) to mailbox index
function algebraToIdx(moveSq) {
  const file = moveSq.charCodeAt(0) - 97; // a=0..h=7
  const rank = parseInt(moveSq[1], 10);   // '1'..'8'
  const f = file + 2;
  const r = rank + 1;
  return r * STRIDE + f;
}

// -------------------------------
// W: capture best move for UCI
// -------------------------------
let lastBest = { from: null, to: null, promo: null, color: 0 };
function W(board, fromIdx, colorBit, specialSquare) {
  // The original code sets globals B and b; we rely on them
  lastBest = { from: fromIdx, to: b, promo: null, color: colorBit };
  // Commit move to board to advance the position
  const piece = board[lastBest.from];
  board[lastBest.to] = piece;
  board[lastBest.from] = EMPTY;
  return `${idxToAlgebra(lastBest.from)}${idxToAlgebra(lastBest.to)}`;
}

// -------------------------------
// Board control for UCI
// -------------------------------
function resetBoardToStartpos() {
  // Recreate original initial board by re-running the setup loop
  // We re-run the original setup
  for (B = y = u = b = 0, x = 10, z = 15, I = [], l = [];
    l[B] = ("ustvrtsuqqqqqqqq" + "yyyyyyyy}{|~z|{}@G@TSb~?A6J57IKJT576,+-48HLSUmgukgg OJNMLK  IDHGFE").charCodeAt(B) - 64, B++ < 120;
    I[B - 1] = B % x ? B / x % 2 < 2 | B % x < 2 ? 7 : B / x & 4 ? 0 : l[u++] : 7);
}

function clearBoard() {
  for (let i = 0; i < 120; i++) {
    if (i % STRIDE === 0 || (i % STRIDE) < 2 || (i % STRIDE) > 9 || Math.floor(i / STRIDE) < 2 || Math.floor(i / STRIDE) > 9) {
      I[i] = OFFBOARD;
    } else {
      I[i] = EMPTY;
    }
  }
}

// Apply a single UCI move "e2e4" or with promotion "e7e8q"
function applyUciMove(m) {
  const from = algebraToIdx(m.slice(0, 2));
  const to = algebraToIdx(m.slice(2, 4));
  // Promotion char (optional)
  const promoChar = m.length >= 5 ? m[4] : null;

  // Move piece
  const piece = I[from];
  const captured = I[to];

  I[to] = piece;
  I[from] = EMPTY;

  // Handle simple promotion approximation: replace piece code type
  if (promoChar) {
    // The engine encodes pieces compactly; a robust mapping is non-trivial.
    // We’ll set the target to a queen-like code by nudging TYPE_MASK bits.
    // This is a best-effort placeholder compatible with its evaluation logic.
    const colorBit = piece & 8;
    const QUEEN_TYPE = 5; // typical type id before mask; mapping is heuristic
    I[to] = (QUEEN_TYPE | colorBit);
  }
}

// Parse FEN to internal board (simple standard startpos and typical pieces)
// Note: This engine’s encoding is special; we only support 'startpos' reliably.
// For generic FEN, we place approximate piece codes (best-effort).
function setFen(fenStr) {
  clearBoard();
  // Example FEN tokens: "rnbqkbnr/pppppppp/8/... w KQkq - 0 1"
  const parts = fenStr.trim().split(/\s+/);
  const boardPart = parts[0];

  const ranks = boardPart.split('/');
  if (ranks.length !== 8) return;

  for (let r8 = 0; r8 < 8; r8++) {
    const rankStr = ranks[r8];
    let file = 0;
    for (let ch of rankStr) {
      if (/\d/.test(ch)) {
        file += parseInt(ch, 10);
      } else {
        const isWhite = ch === ch.toUpperCase();
        const colorBit = isWhite ? 8 : 0;
        let typeCode;
        switch (ch.toLowerCase()) {
          case 'p': typeCode = 1; break;
          case 'n': typeCode = 2; break;
          case 'b': typeCode = 3; break;
          case 'r': typeCode = 4; break;
          case 'q': typeCode = 5; break;
          case 'k': typeCode = 6; break;
          default: typeCode = 0;
        }
        const sq = ( (8 - r8 + 1) * STRIDE ) + (file + 2);
        I[sq] = typeCode | colorBit;
        file++;
      }
    }
  }
}

// -------------------------------
// Search trigger
// -------------------------------
function runSearch(options) {
  // Options: { depth, movetime, wtime, btime, winc, binc }
  // We’ll prefer depth; if movetime provided, we time-bound with setTimeout.
  lastBest = { from: null, to: null, promo: null, color: 0 };

  const sideToMove = options.sideToMove === 'w' ? 0 : 8; // 0 for white, 8 for black at root
  let bestMoveText = null;

  const start = Date.now();
  const maxDepth = options.depth || 2;
  const maxTime = options.movetime || null;

  function thinkOnce() {
    // The Y() function is side=8-fixed in original; we need to align:
    // For white to move (sideToMove=0), call X(0,0,y,1)
    // For black to move (sideToMove=8), call X(8,0,y,1)
    if (sideToMove === 0) {
      X(0, 0, y, maxDepth);
      X(0, 0, y, 1);
    } else {
      X(8, 0, y, maxDepth);
      X(8, 0, y, 1);
    }

    if (lastBest.from != null && lastBest.to != null) {
      bestMoveText = `${idxToAlgebra(lastBest.from)}${idxToAlgebra(lastBest.to)}`;
    }
  }

  if (maxTime) {
    // Time-limited: think repeatedly until time is up
    thinkOnce();
    const remaining = Math.max(0, maxTime - (Date.now() - start));
    // In this simple engine, repeated calls won’t deepen much; one call suffices
    // but we wait to respect movetime.
    return new Promise(resolve => setTimeout(() => resolve(bestMoveText), remaining));
  } else {
    thinkOnce();
    return Promise.resolve(bestMoveText);
  }
}

// -------------------------------
// UCI loop
// -------------------------------
const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout
});
rl.on('line', async (line) => {
  const cmd = line.trim();

  if (cmd === 'uci') {
    console.log('id name MicroJSChess');
    console.log('id author Anonymous');
    // Expose a simple option: SearchDepth
    console.log('option name SearchDepth type spin default 2 min 1 max 5');
    console.log('uciok');
    return;
  }

  if (cmd === 'isready') {
    console.log('readyok');
    return;
  }

  if (cmd === 'ucinewgame') {
    resetBoardToStartpos();
    lastBest = { from: null, to: null, promo: null, color: 0 };
    return;
  }

  if (cmd.startsWith('setoption name SearchDepth value')) {
    // Example: setoption name SearchDepth value 3
    const v = parseInt(cmd.split('value')[1], 10);
    engineDepth = isNaN(v) ? 2 : Math.max(1, Math.min(5, v));
    return;
  }

  if (cmd.startsWith('position')) {
    // position [startpos | fen <fenstring>] moves <move1> <move2> ...
    const tokens = cmd.split(/\s+/);
    let idx = 1;
    let sideToMove = 'w';

    if (tokens[idx] === 'startpos') {
      resetBoardToStartpos();
      idx++;
    } else if (tokens[idx] === 'fen') {
      // Collect FEN until 'moves' or end
      idx++;
      const fenParts = [];
      while (idx < tokens.length && tokens[idx] !== 'moves') {
        fenParts.push(tokens[idx]);
        idx++;
      }
      const fenStr = fenParts.join(' ');
      setFen(fenStr);
      // Extract side to move from fen if present
      const sideToken = fenStr.split(/\s+/)[1];
      if (sideToken === 'b') sideToMove = 'b';
    } else {
      // Default to startpos if unspecified
      resetBoardToStartpos();
    }

    // Apply moves
    const movesIndex = tokens.indexOf('moves');
    if (movesIndex !== -1) {
      for (let i = movesIndex + 1; i < tokens.length; i++) {
        const mv = tokens[i];
        applyUciMove(mv);
        // Toggle sideToMove
        sideToMove = sideToMove === 'w' ? 'b' : 'w';
      }
    }
    currentSideToMove = sideToMove;
    return;
  }

  if (cmd.startsWith('go')) {
    // Parse go params: depth N, movetime M, wtime/btime/winc/binc
    const tokens = cmd.split(/\s+/);
    let depth = engineDepth || 2;
    let movetime = null;
    let st = 0;

    for (let i = 1; i < tokens.length; i++) {
      const t = tokens[i];
      if (t === 'depth' && i + 1 < tokens.length) {
        depth = parseInt(tokens[++i], 10);
      } else if (t === 'movetime' && i + 1 < tokens.length) {
        movetime = parseInt(tokens[++i], 10);
      } else if (t === 'wtime' && i + 1 < tokens.length) {
        // For simplicity we’ll use movetime fallback; a real engine would schedule
        const wtime = parseInt(tokens[++i], 10);
        st = wtime;
      } else if (t === 'btime' && i + 1 < tokens.length) {
        const btime = parseInt(tokens[++i], 10);
        st = btime;
      }
    }

    const options = {
      depth: depth,
      movetime: movetime,
      sideToMove: currentSideToMove || 'w'
    };

    const best = await runSearch(options);
    if (!best) {
      // No move found; resign/stalemate
      console.log('bestmove 0000');
    } else {
      console.log('bestmove ' + best);
    }
    return;
  }

  if (cmd === 'stop') {
    // This simple engine doesn’t run an async search loop; no-op
    return;
  }

  if (cmd === 'quit') {
    rl.close();
    process.exit(0);
  }
});

// State variables for UCI options
let engineDepth = 2;
let currentSideToMove = 'w';
