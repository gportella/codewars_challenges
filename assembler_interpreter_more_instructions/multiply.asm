mov a, 17     
mov b, 15  
mov c, 0      
mov d, b   

loop:
	cmp d, 0     
	jle done   
	add c, a  
	dec d     
	jmp loop  

done:
	msg a, ' * ', b, ' = ', c
end