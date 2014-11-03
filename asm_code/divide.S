	.text
	.align	2
	@@ The divide calls no other functions and only
        @@ uses r0-r3. We don't need to use the stack
	.global	divide
divide: mov     r2,r1           @ move divisor to r2
        mov     r1,r0           @ move dividend to r1
        mov     r0,#0           @ clear R0 to accumulate result
        mov     r3,#1           @ set bit 0 in R3
divstrt:cmp     r2,r1           @ shift R2 left until it is >= R1
        movls   r2,r2,lsl#1     @ shift R3 left in parallel to 
        movls   r3,r3,lsl#1     @ track how far we have to go
        bls     divstrt
divloop:cmp     r1,r2           @ carry set if R1>R2
        subcs   r1,r1,r2        @ subtract R2 from R1 if this would
                                @ give a positive answer
        addcs   r0,r0,r3        @ and add the current bit in R3 to
                                @ the accumulating answer in R0
        movs    r3,r3,lsr#1     @ Shift R3 right into carry flag
        movcc   r2,r2,lsr#1     @ if bit 0 of R3 was zero, also
                                @ shift R2 right
        bcc     divloop         @ If carry not clear, R3 has shifted
                                @ one bit past where it started, and we
                                @ are done.
        mov     pc,lr
        .end

