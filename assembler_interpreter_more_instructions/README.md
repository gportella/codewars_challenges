# Description

Fancier interpreter compared to [`simple_assembler_interpreter`](../simple_assembler_interpreter/).

## Status

> WIP!! just started from simpler version. Let's see how it goes.

## Instruction Set

We want to create an interpreter of assembler which will support the following instructions:

- `mov x, y`: copy y (either an integer or the value of a register) into register x.
- `inc x`: increase the content of register x by one.
- `dec x`: decrease the content of register x by one.
- `add x, y`: add the content of the register x with y (either an integer or the value of a register) and store the result in x (i.e. register[x] += y).
- `sub x, y`: subtract y (either an integer or the value of a register) from register x and store the result in x (i.e. register[x] -= y).
- `mul x, y`: multiply register x by y (either an integer or the value of a register) and store the result in x (i.e. register[x] *= y).
- `div x, y`: divide register x by y using integer division (i.e. register[x] /= y).
- `label:`: define a label position (`label` = identifier + ":", an identifier being a string that does not match any other command). Jump commands and `call` target these label positions in the program.
- `jmp lbl`: jump to the label `lbl`.
- `cmp x, y`: compare x (either an integer or the value of a register) and y (either an integer or the value of a register). The result is used in the conditional jumps (`jne`, `je`, `jge`, `jg`, `jle`, `jl`).
- `jne lbl`: jump to the label `lbl` if the values of the previous `cmp` command were not equal.
- `je lbl`: jump to the label `lbl` if the values of the previous `cmp` command were equal.
- `jge lbl`: jump to the label `lbl` if x was greater or equal than y in the previous `cmp` command.
- `jg lbl`: jump to the label `lbl` if x was greater than y in the previous `cmp` command.
- `jle lbl`: jump to the label `lbl` if x was less or equal than y in the previous `cmp` command.
- `jl lbl`: jump to the label `lbl` if x was less than y in the previous `cmp` command.
- `call lbl`: call the subroutine identified by `lbl`. When a `ret` is found in a subroutine, the instruction pointer should return to the instruction next to this `call` command.
- `ret`: return from the current subroutine to the instruction that requested it.
- `msg 'Register: ', x`: store output for the program. It may contain text strings (delimited by single quotes) and registers. The number of arguments is not limited and will vary depending on the program.
- `end`: indicate that the program ends correctly, so the stored output is returned (if the program terminates without this instruction it should return the default output; see below).
- `; comment`: ignore any text following `;` on the current line during execution.

## Output Format

- The normal output format is a string returned with the `end` command.
- If the program finishes without executing `end`, return `-1` (integer).

## Input Format

The function/method will take as input a multiline string of instructions, delimited with EOL characters. Please, note that the instructions may also have indentation for readability purposes.

### Example Program

```python
program = """
; My first program
mov  a, 5
inc  a
call function
msg  '(5+1)/2 = ', a    ; output message
end

function:
    div  a, 2
    ret
"""
```

```
assembler_interpreter(program)
```

### Execution Walkthrough

The above code sets register `a` to 5, increases its value by 1, calls the subroutine `function`, divides the value by 2, returns to the first `call` instruction, prepares the output of the program, and then returns it with the `end` instruction. In this case, the output is `(5+1)/2 = 3`.
