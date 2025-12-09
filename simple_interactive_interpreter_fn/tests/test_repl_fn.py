import pytest

from repl_fn import Interpreter


@pytest.fixture
def interp():
    return Interpreter()


def test_variable_assignment_and_usage(interp):
    assert interp.input("x = 7") == 7
    assert interp.input("x + 3") == 10


def test_unknown_variable_raises(interp):
    with pytest.raises(SystemExit) as excinfo:
        interp.input("y + 1")
    assert "No variable with name 'y'" in str(excinfo.value)


def test_function_declaration_and_call(interp):
    assert interp.input("fn add x y => x + y") == ""
    assert interp.input("add 2 3") == 5


def test_function_chain_right_associative(interp):
    interp.input("fn echo x => x")
    interp.input("fn add x y => x + y")
    assert interp.input("add echo 4 echo 3") == 7


def test_function_body_invalid_identifier(interp):
    with pytest.raises(SystemExit) as excinfo:
        interp.input("fn add x y => x + z")
    assert "Invalid identifier 'z'" in str(excinfo.value)


def test_name_conflict_variable_then_function(interp):
    interp.input("x = 5")
    with pytest.raises(ValueError) as excinfo:
        interp.input("fn x => x")
    assert "Name conflict" in str(excinfo.value)


def test_name_conflict_function_then_variable(interp):
    interp.input("fn inc x => x + 1")
    with pytest.raises(ValueError) as excinfo:
        interp.input("inc = 2")
    assert "Name conflict" in str(excinfo.value)


def test_function_redefinition_overwrites(interp):
    interp.input("fn inc x => x + 1")
    assert interp.input("inc 2") == 3
    interp.input("fn inc x => x + 2")
    assert interp.input("inc 2") == 4


def test_function_wrong_arity(interp):
    interp.input("fn inc x => x + 1")
    with pytest.raises(ValueError) as excinfo:
        interp.input("inc")
    assert "expects 1 arguments" in str(excinfo.value)


def test_zero_arg_function(interp):
    interp.input("fn four => 4")
    assert interp.input("four") == 4


def test_function_reference_other_function(interp):
    interp.input("fn inc x => x + 1")
    interp.input("fn twice x => inc inc x")
    assert interp.input("twice 3") == 5


def test_function_arguments_are_local(interp):
    interp.input("x = 10")
    interp.input("fn echo x => x")
    assert interp.input("echo 2") == 2


def test_assignment_with_function_call(interp):
    interp.input("fn add x y => x + y")
    assert interp.input("z = add 2 3") == 5
    assert interp.input("z") == 5


def test_function_cannot_capture_global(interp):
    interp.input("g = 10")
    with pytest.raises(SystemExit) as excinfo:
        interp.input("fn use => g")
    assert "Invalid identifier 'g'" in str(excinfo.value)


def test_complex_right_associative_chain(interp):
    interp.input("fn f1 a1 a2 => a1 * a2")
    interp.input("fn f2 a1 a2 a3 => a1 * a2 * a3")
    assert interp.input("f2 f2 1 2 3 f1 4 5 f1 6 7") == 5040


def test_duplicate_function_params_disallowed(interp):
    with pytest.raises(ValueError) as excinfo:
        interp.input("fn bad x x => x")
    assert "Duplicate parameter name" in str(excinfo.value)
