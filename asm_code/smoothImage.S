
	.data

test:	.asciz "%u    %u \n"
	.align 2
allocerr: .asciz "Unable to allocate space for temporary image"
	.align 2

	.text
	.global smoothImage
smoothImage:
	stmfd	sp!,{r4-r11,lr}

	mov r4, r0
	ldr r5, [r4, $12]	/*load rows into re5*/
	ldr r6, [r4, $16]	/*load columns into r6*/
	ldr r7, [r4, $20]	/*load image into r7*/

/*allocate memory for the temporary image*/
	mov r0, r5	/*move nmber of rows into r0*/
	bl malloc 	/*allocate memory for temp image*/
	mov r11, r0	
	cmp r0, $0	/*if memory wasnt allocared print error message*/
	ldreq r0, =allocerr
	bleq printf
	beq return

	mov r8, $0
for_alloc: /*allocate memory for each row in the image*/
	
	mov r0, r6
	bl malloc
	mov r10, r0
	cmp r0, $0
	ldreq r0, =allocerr
	bleq printf
	beq return
	
	str r10, [r11, r8, lsl $2]
	
	add r8, $1

	cmp r8, r5
	blt for_alloc
/*end memory allocation for temporary image..... as of here registers
	r0,r1,r2,r3 and r8,r9, and r10 can be used.. but r11 is the pointer
	to the base of the temporary image*/
	
	mov r8, $0
rowloop:
	mov r9, $0

	ldr r1, [r7,r8,lsl $2]		/*pointer to row in image*/
	ldr r10, [r11,r8, lsl $2]	/*pointer to row in temp image*/

	add r8, $1
colloop:
	mov r11, $0
	mov r3, $3
	mul r3,r9,r3		/*load in at an offset of 3*/
	//ldr r2,[r1, r3] 	/*read in 1 word from iage*/
	
	stmfd sp!,{r0-r3}
	/*call function to find sum and shift into master sum*/

	
	
	ldmfd sp!,{r0-r3}

	add r9, $1

	cmp r9, r6	/*loop through all columns*/
	blt colloop
	
	cmp r8, r5	/*loop through all rows*/
	blt rowloop

	b return
	
return: /*return from the function */
	ldmfd	sp!,{r4-r11,lr}
	mov	pc,lr
/*function to get the sum of the neighboring, red green or blur pixels*/
/*returns the sum which is shifted into 8 bits of a master sum to be stored
in the temporary image ----
takes parematers, r0 = i r1 = j r2 = tempimage r3 = original image 	*/
sum:
	ldmfd sp!,{r4-r11, lr}
	
	ldr r11, [r3, r0, lsl $2]

	mov r10, $3
	mul r10, r1, r10
	ldr r7, [r11 , r10]

	mov r4, r7, lsr $24 /*get red pixel*/
	
	mov r5, r7, lsl $16
	lsr r5, $24

	mov r6, r7, lsl $8
	lsr r6, $24
	
	

	stmfd sp!,{r4-r11, lr}
	mov pc, lr

