; Demonstrate shift instructions in assembler_interpreter
; Computes (value << 3) + (value >> 1) for value = 12

mov a, 12         ; base value
mov b, a          ; keep a copy for right shift

shl a             ; a = 12 << 1 = 24
shl a, 2          ; a = 24 << 2 = 96

shr b             ; b = 12 >> 1 = 6

add a, b          ; combine results -> 96 + 6 = 102

msg 'Result: ', a
end
