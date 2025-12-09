#! /usr/bin/env python

import operator
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Protocol, Dict, List, Optional, Union, Tuple, Deque
import logging

LOG_LEVEL = logging.ERROR
logging.basicConfig(level=LOG_LEVEL, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger()

MsgSegment = Tuple[bool, str]


def parse_literal(token: str) -> Optional[int]:
    try:
        return int(token)
    except ValueError:
        return None


class Registers:
    def __init__(self, names=(), width=32):
        self.width = width
        self.mask = (1 << width) - 1
        self._r = {name: 0 for name in names}
        self.cmp_lhs: Optional[int] = None
        self.cmp_rhs: Optional[int] = None

    def read(self, name):
        return self._r.setdefault(name, 0)

    def read_signed(self, name):
        raw = self._r.setdefault(name, 0)
        sign_bit = 1 << (self.width - 1)
        return raw - (1 << self.width) if raw & sign_bit else raw

    def write(self, name, value):
        self._r[name] = value & self.mask

    def show(self):
        return {r: self.read_signed(r) for r in self._r}

    def __repr__(self) -> str:
        return f"{self._r}"


@dataclass
class ExecutionContext:
    registers: Registers
    memory: Optional[Dict[int, int]] = None
    labels: Optional[Dict[str, int]] = field(default_factory=dict)
    ip: int = 0
    call_stack: Deque[int] = field(default_factory=deque)
    output_parts: List[str] = field(default_factory=list)
    terminated: bool = False
    max_len: int = 0  # to keep of total number of instructions


class Instruction(Protocol):
    def exe(self, ctx: ExecutionContext) -> None: ...

    def __repr__(self) -> str:
        return super().__repr__()


@dataclass(frozen=True)
class ArithmeticInstruction(Instruction):
    """Operate content of reg to val (reg/value) and store in reg"""

    reg: str
    val: Union[int, str]
    op: Callable[[int, int], int]

    def exe(self, ctx: ExecutionContext):
        lhs = ctx.registers.read(self.reg)
        rhs = ctx.registers.read(self.val) if isinstance(self.val, str) else self.val
        ctx.registers.write(self.reg, self.op(lhs, rhs))
        ctx.ip += 1


@dataclass(frozen=True)
class mov(Instruction):
    """Store val (reg/value) in reg"""

    reg: str
    val: Union[int, str]

    def exe(self, ctx: ExecutionContext):
        if isinstance(self.val, str):
            ctx.registers.write(self.reg, ctx.registers.read(self.val))
        else:
            ctx.registers.write(self.reg, self.val)
        ctx.ip += 1


@dataclass(frozen=True)
class inc(Instruction):
    """Incrememnt register by 1"""

    reg: str

    def exe(self, ctx: ExecutionContext):
        ctx.registers.write(self.reg, ctx.registers.read(self.reg) + 1)
        ctx.ip += 1


@dataclass(frozen=True)
class dec(Instruction):
    """Decrement register by 1"""

    reg: str

    def exe(self, ctx: ExecutionContext):
        ctx.registers.write(self.reg, ctx.registers.read(self.reg) - 1)
        ctx.ip += 1


@dataclass(frozen=True)
class put(Instruction):
    """Print out the contents to stdout"""

    reg: str

    def exe(self, ctx: ExecutionContext):
        print(ctx.registers.read(self.reg))
        ctx.ip += 1


@dataclass(frozen=True)
class cmp(Instruction):
    """Compare val/reg and store it in cmp for later use"""

    val_a: Union[str, int]
    val_b: Union[str, int]

    def exe(self, ctx: ExecutionContext):
        lhs = (
            ctx.registers.read_signed(self.val_a)
            if isinstance(self.val_a, str)
            else self.val_a
        )
        rhs = (
            ctx.registers.read_signed(self.val_b)
            if isinstance(self.val_b, str)
            else self.val_b
        )
        ctx.registers.cmp_lhs = lhs
        ctx.registers.cmp_rhs = rhs
        ctx.ip += 1


@dataclass(frozen=True)
class Jump(Instruction):
    """Jump to specific place"""

    lbl: str
    op: Optional[Callable[[int, int], bool]] = None
    is_call: bool = False

    def __post_init__(self):
        if self.op is not None and self.is_call:
            raise ValueError("Jump cannot set both op and is_call")

    def exe(self, ctx: ExecutionContext):
        if not ctx.labels or self.lbl not in ctx.labels:
            raise Exception(f"No labels defined or unexisting label {self.lbl}")

        if self.op is None:
            if self.is_call:
                ctx.call_stack.append(ctx.ip + 1)
            ctx.ip = ctx.labels[self.lbl]
            return

        if ctx.registers.cmp_lhs is None or ctx.registers.cmp_rhs is None:
            raise Exception("Attempting condition jump without prior cmp")

        if self.op(ctx.registers.cmp_lhs, ctx.registers.cmp_rhs):
            ctx.ip = ctx.labels[self.lbl]
        else:
            ctx.ip += 1


@dataclass(frozen=True)
class Return(Instruction):
    """Return to caller using the stored call stack"""

    def exe(self, ctx: ExecutionContext):
        if not ctx.call_stack:
            raise Exception("Found ret outside subroutine")
        next_ip = ctx.call_stack.pop()
        if next_ip < 0 or next_ip >= ctx.max_len:
            raise ValueError(f"Terrible, {next_ip} is out of bounds")

        ctx.ip = next_ip
        return


@dataclass(frozen=True)
class Msg(Instruction):
    """Append literal or register contents to the program output buffer"""

    segments: Tuple[MsgSegment, ...]

    def exe(self, ctx: ExecutionContext):
        for is_register, token in self.segments:
            if is_register:
                ctx.output_parts.append(str(ctx.registers.read_signed(token)))
            else:
                ctx.output_parts.append(token)
        ctx.ip += 1


@dataclass(frozen=True)
class End(Instruction):
    """Mark program termination"""

    def exe(self, ctx: ExecutionContext):
        ctx.terminated = True
        ctx.ip = ctx.max_len


@dataclass(frozen=True)
class jnz(Instruction):
    """Jump to step if value or register is different than 0"""

    reg: Union[int, str]
    step: int

    def exe(self, ctx: ExecutionContext):
        value = (
            ctx.registers.read_signed(self.reg)
            if isinstance(self.reg, str)
            else self.reg
        )
        ctx.ip += self.step if value != 0 else 1


@dataclass(frozen=True)
class ShiftInstruction(Instruction):
    """Shift register value left or right by an optional amount"""

    reg: str
    amount: Union[int, str] = 1
    direction: str = "left"

    def exe(self, ctx: ExecutionContext):
        raw_shift = (
            ctx.registers.read_signed(self.amount)
            if isinstance(self.amount, str)
            else self.amount
        )
        if raw_shift < 0:
            raise ValueError("Shift amount must be non-negative")

        value = ctx.registers.read_signed(self.reg)
        if self.direction == "left":
            shifted = value << raw_shift
        else:
            shifted = value >> raw_shift

        ctx.registers.write(self.reg, shifted)
        ctx.ip += 1


def strip_comments(line: str) -> str:
    cleaned = line
    for marker in (";", "#"):
        cleaned = cleaned.split(marker, 1)[0]
    return cleaned.strip()


def tokenize_instruction(line: str) -> List[str]:
    cleaned = strip_comments(line)
    if not cleaned:
        return []
    tokens: List[str] = []
    for token in cleaned.split():
        trimmed = token.rstrip(",")
        if trimmed:
            tokens.append(trimmed)
    return tokens


def parse_labels(line: str) -> Optional[str]:
    """Find location of labels"""
    parts = tokenize_instruction(line)
    if not parts:
        raise ValueError("Empty instruction line")
    op = parts[0].lower()
    if ":" in op[-1]:
        return op.replace(":", "")


def parse_msg_segments(argument_str: str) -> Tuple[MsgSegment, ...]:
    segments: List[MsgSegment] = []
    idx = 0
    length = len(argument_str)
    while idx < length:
        char = argument_str[idx]
        if char.isspace() or char == ",":
            idx += 1
            continue
        if char == "'":
            end_idx = argument_str.find("'", idx + 1)
            if end_idx == -1:
                raise ValueError("Unterminated string literal in msg")
            segments.append((False, argument_str[idx + 1 : end_idx]))
            idx = end_idx + 1
            continue
        next_sep = idx
        while (
            next_sep < length
            and not argument_str[next_sep].isspace()
            and argument_str[next_sep] != ","
        ):
            next_sep += 1
        token = argument_str[idx:next_sep]
        literal = parse_literal(token)
        if literal is not None:
            segments.append((False, str(literal)))
        else:
            segments.append((True, token))
        idx = next_sep
    return tuple(segments)


def parse_instruction(line: str) -> Optional[Instruction]:
    """Well skip labels here"""
    stripped = strip_comments(line)
    parts = [p.rstrip(",") for p in stripped.split() if p.rstrip(",")]
    if not parts:
        raise ValueError("Empty instruction line")

    op = parts[0].lower()
    if op == "mov" and len(parts) == 3:
        literal = parse_literal(parts[2])
        val = literal if literal is not None else parts[2]
        return mov(parts[1], val)
    if op == "add" and len(parts) == 3:
        literal = parse_literal(parts[2])
        val = literal if literal is not None else parts[2]
        return ArithmeticInstruction(parts[1], val, operator.add)
    if op == "sub" and len(parts) == 3:
        literal = parse_literal(parts[2])
        val = literal if literal is not None else parts[2]
        return ArithmeticInstruction(parts[1], val, operator.sub)
    if op == "mul" and len(parts) == 3:
        literal = parse_literal(parts[2])
        val = literal if literal is not None else parts[2]
        return ArithmeticInstruction(parts[1], val, operator.mul)
    if op == "div" and len(parts) == 3:
        literal = parse_literal(parts[2])
        val = literal if literal is not None else parts[2]
        return ArithmeticInstruction(parts[1], val, operator.floordiv)
    if op == "cmp" and len(parts) == 3:
        literal = parse_literal(parts[2])
        val = literal if literal is not None else parts[2]
        return cmp(parts[1], val)
    if op == "inc" and len(parts) == 2:
        return inc(parts[1])
    if op == "dec" and len(parts) == 2:
        return dec(parts[1])
    if op == "put" and len(parts) == 2:
        return put(parts[1])
    if op in {"shl", "shr"} and len(parts) in {2, 3}:
        amount: Union[int, str]
        if len(parts) == 2:
            amount = 1
        else:
            literal = parse_literal(parts[2])
            amount = literal if literal is not None else parts[2]
        direction = "left" if op == "shl" else "right"
        return ShiftInstruction(parts[1], amount, direction)
    if op == "jmp" and len(parts) == 2:
        return Jump(lbl=parts[1])
    if op == "jne" and len(parts) == 2:
        return Jump(lbl=parts[1], op=operator.ne)
    if op == "je" and len(parts) == 2:
        return Jump(lbl=parts[1], op=operator.eq)
    if op == "jge" and len(parts) == 2:
        return Jump(lbl=parts[1], op=operator.ge)
    if op == "jg" and len(parts) == 2:
        return Jump(lbl=parts[1], op=operator.gt)
    if op == "jle" and len(parts) == 2:
        return Jump(lbl=parts[1], op=operator.le)
    if op == "jl" and len(parts) == 2:
        return Jump(lbl=parts[1], op=operator.lt)
    if op == "call" and len(parts) == 2:
        return Jump(lbl=parts[1], is_call=True)
    if op == "ret" and len(parts) == 1:
        return Return()
    if op == "end" and len(parts) == 1:
        return End()
    if op == "jnz" and len(parts) == 3:
        literal = parse_literal(parts[2])
        if literal is None:
            raise ValueError(f"jnz requires numeric step, got {parts[2]!r}")
        condition = parse_literal(parts[1])
        condition_operand = condition if condition is not None else parts[1]
        return jnz(condition_operand, literal)
    if op == "msg":
        argument_str = stripped[len(parts[0]) :].strip()
        segments = parse_msg_segments(argument_str)
        return Msg(segments)

    if ":" in op[-1]:
        return None

    raise ValueError(f"Unsupported instruction format: {line!r}")


def build_instructions(
    lines: Union[str, List[str]],
) -> Tuple[List[Instruction], Dict[str, int]]:
    program_src = lines if isinstance(lines, str) else "\n".join(lines)
    instructions: List[Instruction] = []
    labels: Dict[str, int] = {}
    for line in program_src.splitlines():
        if not tokenize_instruction(line):
            continue
        inst = parse_instruction(line)
        if inst is not None:
            instructions.append(inst)
        label = parse_labels(line)
        if label:
            labels[label] = len(instructions)
    return instructions, labels


@dataclass(frozen=True)
class Program:
    ctx: ExecutionContext
    exec_plan: List[Instruction] = field(default_factory=list)

    def __iter__(self):
        return self

    def __next__(self):
        xx = self.exec_plan[self.ctx.ip]
        xx.exe(self.ctx)
        if self.ctx.ip >= len(self.exec_plan):
            raise StopIteration
        return xx


def assembler_interpreter(program):
    inst, lbls = build_instructions(program)
    registers = Registers()
    ctx = ExecutionContext(registers=registers, labels=lbls, max_len=len(inst))
    prg = Program(ctx=ctx, exec_plan=inst)
    for _inst in prg:
        logger.debug(f"stck: {ctx.ip} inst: {_inst}")
        logger.debug(registers.show())
    if ctx.terminated:
        return "".join(ctx.output_parts)
    return -1


if __name__ == "__main__":
    import sys
    import time
    import textwrap

    if len(sys.argv) > 1:
        program_path = sys.argv[1]
        with open(program_path, "r", encoding="utf-8") as asm_file:
            source = asm_file.read()
        start = time.perf_counter_ns()
        result = assembler_interpreter(source)
        end = time.perf_counter_ns()
        print(result)
        print(f"Elapsed: {end - start} ns")
    else:
        program = textwrap.dedent(
            """
                mov   a, 3
                mov   b, 9
                mov   c, a
                mov   d, b
                call  proc_func
                call  print
                end

                proc_func:
                    cmp   d, 1
                    je    continue
                    mul   c, a
                    dec   d
                    call  proc_func

                continue:
                    ret

                print:
                    msg a, '^', b, ' = ', c
                    ret
            """
        )
        start = time.perf_counter_ns()
        result = assembler_interpreter(program)
        end = time.perf_counter_ns()
        print(result)
        print(f"Elapsed: {end - start} ns")
