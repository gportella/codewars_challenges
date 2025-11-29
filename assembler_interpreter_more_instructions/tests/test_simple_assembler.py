"""Basic tests for the simple assembler interpreter."""

from assembler_interpreter import (
    ExecutionContext,
    Registers,
    build_instructions,
    cmp,
    jnz,
    inc,
    mov,
    simple_assembler,
)


def assert_raises(expected_exception, func, *args, **kwargs):
    try:
        func(*args, **kwargs)
    except expected_exception:
        return
    except (
        Exception
    ) as err:  # pragma: no cover - only triggered on unexpected exception types
        raise AssertionError(
            f"Expected {expected_exception.__name__}, but got {type(err).__name__}: {err}"
        ) from err
    else:
        raise AssertionError(f"Expected {expected_exception.__name__} to be raised")


class TestBasicOperations:
    def test_mov_and_inc_creates_expected_value(self):
        program = [
            "mov a 1",
            "inc a",
            "inc a",
        ]

        registers = simple_assembler(program)

        assert registers == {"a": 3}

    def test_mov_copies_between_registers(self):
        program = [
            "mov a -10",
            "mov b a",
        ]

        registers = simple_assembler(program)

        assert registers == {"a": -10, "b": -10}


class TestArithmeticOperations:
    def test_add_with_literal_and_register_operand(self):
        program = [
            "mov a 5",
            "mov b 3",
            "add a 4",
            "add b a",
        ]

        registers = simple_assembler(program)

        assert registers == {"a": 9, "b": 12}

    def test_sub_with_literal_and_register_operand(self):
        program = [
            "mov a 10",
            "mov b 3",
            "sub a 4",
            "sub b a",
        ]

        registers = simple_assembler(program)

        assert registers == {"a": 6, "b": -3}

    def test_mul_with_literal_and_register_operand(self):
        program = [
            "mov a 2",
            "mov b 3",
            "mul a 4",
            "mul b a",
        ]

        registers = simple_assembler(program)

        assert registers == {"a": 8, "b": 24}

    def test_div_with_literal_and_register_operand(self):
        program = [
            "mov a 12",
            "mov b 3",
            "div a 4",
            "div b a",
        ]

        registers = simple_assembler(program)

        assert registers == {"a": 3, "b": 1}


class TestConditionalLogic:
    def test_jnz_with_loop_decrements_to_zero(self):
        program = [
            "mov a 3",
            "dec a",
            "jnz a -1",
        ]

        registers = simple_assembler(program)

        assert registers == {"a": 0}

    def test_jnz_with_literal_jump_skips_instruction(self):
        program = [
            "mov a 5",
            "jnz 1 2",
            "mov a 0",
            "inc a",
        ]

        registers = simple_assembler(program)

        assert registers == {"a": 6}

    def test_jmp_unconditional_jump_to_label(self):
        program = [
            "start:",
            "mov a 1",
            "jmp end",
            "inc a",
            "end:",
            "mov b a",
        ]

        registers = simple_assembler(program)

        assert registers == {"a": 1, "b": 1}

    def test_je_jumps_when_values_are_equal(self):
        program = [
            "mov a 5",
            "mov b 5",
            "cmp a b",
            "je equal",
            "mov a 0",
            "jmp end",
            "equal:",
            "inc a",
            "end:",
        ]

        registers = simple_assembler(program)

        assert registers == {"a": 6, "b": 5}

    def test_jne_jumps_when_values_are_different(self):
        program = [
            "mov a 5",
            "mov b 3",
            "cmp a b",
            "jne not_equal",
            "jmp end",
            "not_equal:",
            "sub a 1",
            "end:",
        ]

        registers = simple_assembler(program)

        assert registers == {"a": 4, "b": 3}

    def test_jg_and_jl_follow_cmp_sign(self):
        program = [
            "mov a -1",
            "mov b 2",
            "cmp a b",
            "jg greater",
            "jl less",
            "jmp end",
            "greater:",
            "mov c 1",
            "jmp end",
            "less:",
            "mov c -1",
            "end:",
        ]

        registers = simple_assembler(program)

        assert registers == {"a": -1, "b": 2, "c": -1}

    def test_jge_and_jle_trigger_on_equality(self):
        program = [
            "mov a 7",
            "mov b 7",
            "cmp a b",
            "jge ge_label",
            "jmp end",
            "ge_label:",
            "jle le_label",
            "jmp end",
            "le_label:",
            "mov c 1",
            "end:",
        ]

        registers = simple_assembler(program)

        assert registers == {"a": 7, "b": 7, "c": 1}

    def test_jump_raises_when_label_missing(self):
        program = [
            "mov a 1",
            "jmp nowhere",
        ]

        instructions, labels = build_instructions(program)
        registers = Registers()
        ctx = ExecutionContext(registers=registers, labels=labels)

        assert_raises(Exception, instructions[1].exe, ctx)

    def test_conditional_jump_with_unknown_label_raises(self):
        program = [
            "mov a 1",
            "cmp a 1",
            "je nowhere",
        ]

        instructions, labels = build_instructions(program)
        registers = Registers()
        ctx = ExecutionContext(registers=registers, labels=labels)

        instructions[0].exe(ctx)
        instructions[1].exe(ctx)

        assert_raises(Exception, instructions[2].exe, ctx)

    def test_conditional_jump_without_cmp_raises(self):
        program = [
            "mov a 1",
            "jne somewhere",
            "somewhere:",
        ]

        instructions, labels = build_instructions(program)
        registers = Registers()
        ctx = ExecutionContext(registers=registers, labels=labels)

        assert_raises(Exception, instructions[1].exe, ctx)


class TestComparisonOperations:
    def test_cmp_stores_operands_for_future_jumps(self):
        registers = Registers()
        registers.write("a", 5)
        ctx = ExecutionContext(registers=registers)

        cmp("a", 5).exe(ctx)
        assert registers.cmp_lhs == 5
        assert registers.cmp_rhs == 5
        assert ctx.ip == 1

        registers.write("b", -3)
        cmp("a", "b").exe(ctx)
        assert registers.cmp_lhs == 5
        assert registers.cmp_rhs == -3
        assert ctx.ip == 2


class TestCommentHandling:
    def test_inline_comment_is_ignored(self):
        program = [
            "mov a 1 ; initialize",
            "inc a ; increase",
            "inc a ; bump",
        ]

        registers = simple_assembler(program)

        assert registers == {"a": 3}

    def test_full_line_comment_is_ignored(self):
        program = [
            "; setup",
            "mov a 2",
            "; another comment",
            "dec a ; decrement once",
        ]

        registers = simple_assembler(program)

        assert registers == {"a": 1}


class TestLabelHandling:
    def test_labels_are_skipped_and_positions_recorded(self):
        program = [
            "start:",
            "mov a 1",
            "loop:",
            "inc a",
            "jnz a -1",
            "end:",
        ]

        instructions, labels = build_instructions(program)

        assert len(instructions) == 3
        assert len(labels) == 3
        assert isinstance(instructions[0], type(mov("a", 0)))
        assert isinstance(instructions[1], type(inc("a")))
        assert isinstance(instructions[2], type(jnz(0, 0)))


class TestCallAndReturn:
    def test_call_enters_subroutine_and_returns_after_ret(self):
        program = [
            "mov a 5",
            "call double",
            "jmp done",
            "double:",
            "mul a 2",
            "ret",
            "done:",
            "inc a",
        ]

        registers = simple_assembler(program)

        assert registers == {"a": 11}

    def test_ret_without_prior_call_raises(self):
        program = [
            "ret",
        ]

        instructions, labels = build_instructions(program)
        registers = Registers()
        ctx = ExecutionContext(
            registers=registers, labels=labels, max_len=len(instructions)
        )

        assert_raises(Exception, instructions[0].exe, ctx)
