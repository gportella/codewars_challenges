# Description

`N.B.`: I had a tokenizer/parser from the numerical derivative kata, and from the symbolic derivative one. It was easy to port it for this one, and it was worth quite some points.


Given a mathematical expression as a string you must return the result as a number.

## Numbers

Number may be both whole numbers and/or decimal numbers. The same goes for the returned result.

## Operators

You need to support the following mathematical operators:

- Multiplication `*`
- Division `/` (as floating point division)
- Addition `+`
- Subtraction `-`

Operators are always evaluated from left-to-right, and `*` and `/` must be evaluated before `+` and `-`.

## Parentheses

You need to support multiple levels of nested parentheses, ex. `(2 / (2 + 3.33) * 4) - -6`
