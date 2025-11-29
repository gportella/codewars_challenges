# Description

>> **WIP**: Not finished yet, but not sure I will get it to a point to pass the tests in the kata. Too many ways of expressing the same expression, might end up fighting against the test suite and not sure worth the pain. 

Differentiate a mathematical expression given as a string in prefix notation. The result should be the derivative of the expression returned in prefix notation.

To simplify things we will use a simple list format made up of parentesis and spaces.

The expression format is (func arg1) or (op arg1 arg2) where op means operator, func means function and arg1, arg2 are aguments to the operator or function. For example `(+ x 1)` or `(cos x)`

The expressions will always have balanced parentesis and with spaces between list items.

Expression operators, functions and arguments will all be lowercase.

Expressions are single variable expressions using `x` as the variable.

Expressions can have nested arguments at any depth for example `(+ (* 1 x) (* 2 (+ x 1)))`

The operators and functions you are required to implement are `+ - * / ^ cos sin tan exp ln`

In addition to returning the derivative your solution must also do some simplifications of the result but only what is specified below.

The returned expression should not have unecessary 0 or 1 factors. For example it should not return `(* 1 (+ x 1))` but simply the term `(+ x 1)` similarly it should not return `(* 0 (+ x 1))` instead it should return just 0

Results with two constant values such as for example `(+ 2 2)` should be evaluated and returned as a single value 4

Any argument raised to the zero power should return 1 and if raised to 1 should return the same value or variable. For example `(^ x 0)` should return 1 and `(^ x 1)` should return `x`

No simplifications are expected for functions like `cos`, `sin`, `exp`, `ln`... (but their arguments might require a simplification).


