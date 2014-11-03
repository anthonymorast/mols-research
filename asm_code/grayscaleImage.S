/***************************************************************************//**
 * \file grayscale.S 
 *
 * \brief this function will covert a colored image to a grayscale image
 *
 * \author Anthony Morast
 *
 * \date Friday May 2, 2014
 *
 * This function will convert a colored image to a grayscale image. It will 
 * also return the min and max values to be used in contrast later (I think).
 * The function starts by interpruting the magic number it is 1 or 4 the image
 * cannot be converted to grayscale so the function returns. If it can be 
 * grayscaled the program will allocate the memory for each row and column in
 * the new grayimage. Then the original image is read in 3 bytes at a time. 
 * The individual pixels are shifted out, altered, and written into the gray
 * image. Then the min max is determined and returned from the function. Also
 * AT THE end of the function the magic number is incremented so that the image
 * is output as grayscale later on.  
 *
 *****************************************************************************/
 	.data

test:	.asciz "%d %d \n"
	.align 2
max:	.word 0
	.align 2
min: 	.word 255
	.align 2	
cantgs: .asciz "Cannot convert bitmap to grayscale, returning."
	.align 2
memalloc:.asciz "Unable to allocate memory" 
	.align 2
	.text
	.global grayscaleImage
grayscaleImage:
	stmfd	sp!,{r4-r11,lr}
	
        mov r4, r0

	ldr r5, [r4, $12]  /*get rowz */
	ldr r6, [r4, $16]  /*get columns */
	ldr r7, [r4, $20]  /*get image pointers */
	ldr r8, [r4]  /*get magic number*/

	/*if the magic number indicates bitmap to grayscale */
 	cmp r8, $1	
	beq return_cantgrayscale
	cmp r8, $4
	beq return_cantgrayscale

	ldr r9,[r4,$24] /*get grayscale image loaded*/
	
	/*find min and max values if the image exists return them*/
	mov r0, r5
	mov r1, r6
	mov r2, r9
	cmp r8, $2
	bleq findMinMax
	cmp r8, $5
	bleq findMinMax

	/*can freely use r8 now, r9 is the grayscale image pointer*/
	
	/*call allocGrayScale to allocate the memory for the grayscale image*/
	cmp r9,$0	/*if grayimage is null*/
	moveq r0, r5
	lsleq r0, $2
	bleq malloc 	/*allocate 4 times the rows*/
	mov r11, r0
	cmp r0,$0
	ldreq r0, =memalloc	/*if no memory addresses are reuturned*/
	bleq printf 	/*print error message*/
	beq return	/*return*/
	str r11,[r4, $24] 	/*store in memory address*/
	mov r8, $0	/*counter*/

	ldr r11, [r4,$24]

for_alloc:	/*for each row allocate amount of columns*/
	mov r0, r6
	bl malloc
	mov r10,r0
	cmp r0, $0
	ldreq r0,=memalloc
	bleq printf
	beq return
	str r10, [r11,r8,lsl $2]

	add r8, $1

	cmp r8, r5
	blt for_alloc
	
	/*grayscale the iamge*/
	/*can use r10 and r11 and r8, r0, r1, r2, and r3*/
	/*instead of .3, .6, .1, .5  --- times all by 10,multiply values by 3 6 1, add 5, divide by 10*/
	mov r0, $0
	ldr r9, [r4, $24]
rowsloop:
	mov r1, $0

	ldr r2, [r7,r0,lsl $2]	/*pointer to a column in the color iage*/
	ldr r10, [r9,r0, lsl $2] /*pointer to grayscale image*/

	add r0,$1
colsloop:
	mov r8, $3
	mul r8, r1, r8
	ldr r3, [r2,r8]	
		
	stmfd sp!, {r10,r11}
	/*shift out first 3 bits, disregard last*/
	mov r8,r3, lsr $24	/*red pixel, mul by 3*/
	mov r10, $3
	mul r8,r10

	mov r10, r3, lsl $16	/*green pixel mul by 6*/
	lsr r10, $24
	mov r11, $6
	mul r10, r11
	
	mov r11, r3, lsl $8	/*blue pixel mul by 1*/
	lsr r11, $24
	
	add r8, r10	/*add r8 + r10 + r11*/
	add r8, r11
	add r8, $5

	/*divide by 10*/
	stmfd sp!, {r0-r3}
	mov r0, r8
	mov r1, $10
	bl divide
	mov r8, r0
	ldmfd sp!, {r0-r3}	

	cmp r8, $255	/*crop*/
	movgt r8, $255

	ldmfd sp!, {r10, r11}
	
	str r8, [r10,r1]	/*store in grayscale image*/

	/*can use r10, r11, probably r3 now*/
	/*store the new value in the gray image*/
	add r1, $1

	cmp r1, r6
	blt colsloop

	cmp r0, r5
	blt rowsloop

	mov r0, r5	/*recompute min and max*/
	mov r1, r6
	mov r2, r9
	bl findMinMax
	
	b return

return_cantgrayscale:	/*depending on magic num may be unable to grayscale*/
	ldr r0, =cantgs
	bl printf
	mov 	r0,$0
	mov  	r1,$1		/*so return*/
	ldmfd	sp!,{r4-r11,lr}
	mov	pc,lr

/*function that will walk through the rows ans columns of the image. The 
* function will take r0 = rows, r1 = cols, and r2 = grayimage pointer as
 parameters */
findMinMax:  
	stmfd	sp!,{r4-r11,lr}

	mov r10, $3	/*do 3/4 the rows here since we are stepping*/
	mul r1, r10	/*through the whole image*/
	lsr r1, $2

	ldr r10,=min
	ldr r11,=max
	ldr r3, [r10]
	ldr r4,[r11]

	mov r4, $0
rowloop:
	mov r5, $0
	
	ldr r6,[r2, r4, lsl $2]

	add r4,$1
colloop:
	ldr r7,[r6, r5, lsl $2]
	add r5,$1	

	/*compare last pixel with min and max value */
	mov r8, r7, lsr $24
	cmp r8,r3
	movlt r3, r8
	cmp r8, r4
	movgt r4, r8

	/*compare second to last pixel with current min and max value*/
	mov r8,r7, lsl $8
	lsr r8,$24
        cmp r8,r3
        movlt r3, r8
        cmp r8, r4
        movgt r4, r8
	
	/* compare the 3rd to last pixel with current min and max value*/
	mov r8, r7, lsl $16
	lsr r8, $24
        cmp r8,r3
        movlt r3, r8
        cmp r8, r4
        movgt r4, r8

	/*compare first pixel with min and max*/
	mov r8, r7, lsl $24
	lsr r8, $24
        cmp r8,r3
        movlt r3, r8
        cmp r8, r4
        movgt r4, r8


	cmp r5, r1
	blt colloop

	cmp r4,r0
	blt rowloop

	str r3,[r10]
	str r4,[r11]		

	ldmfd 	sp!,{r4-r11,lr}
	mov pc, lr

return:
	mov r6, r4
	ldr r5, [r4]
	sub r5, $1
	str r5, [r4]
	
	ldr r5, =min
	ldr r0, [r5]
	ldr r5, =max
	ldr r1, [r5]

	ldmfd sp!,{r4-r11,lr}
	mov pc, lr


