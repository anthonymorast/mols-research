/***************************************************************************//**
 * \file brightenImage.S
 *	
 * \brief birghtens an image  
 *
 * \author Anthony Morast
 *
 * \date Friday May 2, 2014
 *
 * This function will brighten an image passed in from the command line.
 * Somewhere else the image will be printed to another file whose name is also
 * a command line parameter. This function will walk through the existing image
 * and read in a word at a time, extract the individual pixels, 1 byte each, 
 * then add the brighten factor (also a command line parameter) to each pixel
 * the pixels are cropped and then put back into a single word. This word is 
 * stored back in the image in the structure.
 *
 *****************************************************************************/
	.data

test:	.asciz "%u \n"
	.align 2

	.text
	.global brightenImage
brightenImage:
	stmfd	sp!,{r4-r11,lr}
	
	mov r7,r0

	mov r0,r1
	bl  atoi
	mov r11, r0	/*convert second param to integer, store in r10*/
	
	ldr r4,[r7,$12]	/*load in rows*/
	ldr r5,[r7,$16] /*load in columns*/
	ldr r6,[r7,$20]	/*load pointer to image*/

	mov r0,$3	/*mult cols by 3/4 to account for reading 4 bytes*/
	mul r5,r0	
	lsr r5,$2
	
	mov r8, $0 	/*row counter */

rowloop:
	mov r9,$0	/*column counter*/
	ldr r7,[r6,r8,lsl $2] /*load pointer to a row*/

	add r8,$1	/*increment counter*/

colloop:
	ldr r3,[r7,r9,lsl $2] /*read in a word at a time*/	

	mov r0,r3 	/*move word to be passed to function*/
	mov r1,r11
	bl brighten		
	mov r3, r0
	
	str r3,[r7,r9,lsl $2] /*store brightened values back to image*/
	
	add r9,$1	/*increment counter*/
		
	cmp r9,r5
	blt colloop	/*loop through columns*/

	cmp r8,r4
	blt rowloop	/*loop through rows*/

	ldmfd	sp!,{r4-r11,lr}
	mov	pc,lr
/*takes the current word loaded from the image as parameter 1 and takes
* the number to brighten by as the second argument, thus r0= word loaded
* and r1 = brighten value*/
brighten:
	stmfd sp!,{r4-r5,lr}

	mov r2,r0,lsr $24	/*get last 8 bits*/
	mov r3,r0,lsl $8	
	lsr r3,$24		/*get second to last 8 bits*/
	mov r4,r0,lsl $16
	lsr r4,$24		/*get third to last 8 bits*/
	mov r5,r0,lsl $24
	lsr r5,$24		/*get first 8 bits*/

	/*add brighten amount r1 and crop*/
	add r2, r1
	cmp r2,$255
	movgt r2, $255
	
	add r3,r1
	cmp r3, $255
	movgt r3, $255

        add r4,r1
        cmp r4, $255
        movgt r4, $255

        add r5,r1
        cmp r5, $255
        movgt r5, $255

	/*shift values back into r0 and return*/
	mov r0, r2, lsl $24
	add r0, r3, lsl $16
	add r0, r4, lsl $8
	add r0, r5	

	ldmfd sp!,{r4-r5,lr}
	mov pc,lr
