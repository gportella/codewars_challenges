#! /usr/bin/env python

import operator
from dataclasses import dataclass, field
from typing import Callable, Protocol, Dict, List, Optional, Union, Tuple
import logging

LOG_LEVEL = logging.ERROR
logging.basicConfig(level=LOG_LEVEL, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger()


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
    next_ip: Optional[int] = (
        None  # this is to return to the right place after a function
    )
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
                ctx.next_ip = ctx.ip + 1
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
    """Jumps back to next_ip, should be used after hitting ret"""

    def exe(self, ctx: ExecutionContext):
        if ctx.next_ip is None:
            raise Exception("Found ret outside subroutine")
        if ctx.next_ip < 0 or ctx.next_ip >= ctx.max_len:
            raise ValueError(f"Terrible, {ctx.next_ip} is out of bounds")

        ctx.ip = ctx.next_ip
        ctx.next_ip = None
        return


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


def tokenize_instruction(line: str) -> List[str]:
    cleaned = line
    for marker in (";", "#"):
        cleaned = cleaned.split(marker, 1)[0]
    cleaned = cleaned.strip()
    return cleaned.split() if cleaned else []


def parse_labels(line: str) -> Optional[str]:
    """Find location of labels"""
    parts = tokenize_instruction(line)
    if not parts:
        raise ValueError("Empty instruction line")
    op = parts[0].lower()
    if ":" in op[-1]:
        return op.replace(":", "")


def parse_instruction(line: str) -> Optional[Instruction]:
    """Well skip labels here"""
    parts = tokenize_instruction(line)
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
    if op in {"ret", "end"} and len(parts) == 1:
        return Return()
    if op == "jnz" and len(parts) == 3:
        literal = parse_literal(parts[2])
        if literal is None:
            raise ValueError(f"jnz requires numeric step, got {parts[2]!r}")
        condition = parse_literal(parts[1])
        condition_operand = condition if condition is not None else parts[1]
        return jnz(condition_operand, literal)

    if ":" in op[-1]:
        return None

    raise ValueError(f"Unsupported instruction format: {line!r}")


def build_instructions(lines: List[str]) -> Tuple[List[Instruction], Dict[str, int]]:
    program_src = "\n".join(lines)
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


def simple_assembler(program):
    inst, lbls = build_instructions(program)
    registers = Registers()
    ctx = ExecutionContext(registers=registers, labels=lbls, max_len=len(inst))
    prg = Program(ctx=ctx, exec_plan=inst)
    for _inst in prg:
        logger.debug(f"stck: {ctx.ip} inst: {_inst}")
        logger.debug(registers.show())
    reg = ctx.registers.show()
    return reg


if __name__ == "__main__":
    code = """\
mov c 12
mov b 0
mov a 200
dec a
inc b
jnz a -2
dec c
mov a b
jnz c -5
jnz 0 1
mov c a"""
    program_lines = code.splitlines()
    program_lines_2 = [
        "mov a 5",
        "inc a",
        "dec a",
        "put a",
        "jnz a -2",
        "inc a",
        "put a",
    ]
    program_lines_3 = ["mov a -10", "mov b a", "inc a", "dec b", "jnz a -2"]
    registers = simple_assembler(program_lines_3)
    print(registers)
