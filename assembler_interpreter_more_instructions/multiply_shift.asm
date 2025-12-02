mov a, 17
mov b, 15

mov h, b
mov c, 0
mov d, a

main_loop:
    cmp b, 0
    jle finish

    mov e, b
    shr e
    shl e
    cmp e, b
    je skip_add
    add c, d

skip_add:
    shl d
    shr b
    jmp main_loop

finish:
    msg a, ' * ', h, ' = ', c
end
