#! /usr/bin/env python

from dataclasses import dataclass, field
from typing import Protocol, Dict, List, Optional, Union
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
    ip: int = 0  # instruction pointer, if you need sequencing later


class Instruction(Protocol):
    def exe(self, ctx: ExecutionContext) -> None: ...

    def __repr__(self) -> str:
        return super().__repr__()


@dataclass(frozen=True)
class mov(Instruction):
    reg: str
    val: Union[int, str]

    def exe(self, ctx: ExecutionContext):
        if isinstance(self.val, str):
            ctx.registers.write(self.reg, ctx.registers.read(self.val))
        else:
            logger.debug(f"Copy {self.val} -> {self.reg}")
            ctx.registers.write(self.reg, self.val)
        ctx.ip += 1


@dataclass(frozen=True)
class inc(Instruction):
    reg: str

    def exe(self, ctx: ExecutionContext):
        ctx.registers.write(self.reg, ctx.registers.read(self.reg) + 1)
        ctx.ip += 1


@dataclass(frozen=True)
class dec(Instruction):
    reg: str

    def exe(self, ctx: ExecutionContext):
        ctx.registers.write(self.reg, ctx.registers.read(self.reg) - 1)
        ctx.ip += 1


@dataclass(frozen=True)
class put(Instruction):
    reg: str

    def exe(self, ctx: ExecutionContext):
        print(ctx.registers.read(self.reg))
        ctx.ip += 1


@dataclass(frozen=True)
class jnz(Instruction):
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
    cleaned = line.split("#", 1)[0].strip()
    return cleaned.split() if cleaned else []


def parse_instruction(line: str) -> Instruction:
    parts = tokenize_instruction(line)
    if not parts:
        raise ValueError("Empty instruction line")

    op = parts[0].lower()
    if op == "mov" and len(parts) == 3:
        literal = parse_literal(parts[2])
        logger.debug(f"Literal for mov is {literal}")
        val = literal if literal is not None else parts[2]
        logger.debug(f"Val for mov is {val}")
        return mov(parts[1], val)
    if op == "inc" and len(parts) == 2:
        return inc(parts[1])
    if op == "dec" and len(parts) == 2:
        return dec(parts[1])
    if op == "put" and len(parts) == 2:
        return put(parts[1])
    if op == "jnz" and len(parts) == 3:
        literal = parse_literal(parts[2])
        if literal is None:
            raise ValueError(f"jnz requires numeric step, got {parts[2]!r}")
        condition = parse_literal(parts[1])
        condition_operand = condition if condition is not None else parts[1]
        return jnz(condition_operand, literal)

    raise ValueError(f"Unsupported instruction format: {line!r}")


def build_instructions(lines: List[str]) -> List[Instruction]:
    program_src = "\n".join(lines)
    return [
        parse_instruction(line)
        for line in program_src.splitlines()
        if tokenize_instruction(line)
    ]


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
    inst = build_instructions(program)
    registers = Registers()
    ctx = ExecutionContext(registers=registers)
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
