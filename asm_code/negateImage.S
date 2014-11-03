/***************************************************************************//**
 * \file negate.S 
 *
 * \brief will negate an image
 *
 * \author Anthony Morast
 *
 * \date Friday May 2, 2014
 *
 * This function will get the image, the rows, and the columns from the struct
 * passed into the function. The image is walked through 4 bytes at a time. The
 * word will be XOR'ed with a word of all 1's. This will essentially subtract
 * each pixel value from 255 causing the image to become negated. This is done
 * and then each word is written back to the image. 
 *
 *****************************************************************************/

	.data

test:	.asciz "%d\n"
	.align 2
	
	.text
	.global negateImage
negateImage:	
	stmfd	sp!,{r4-r11,lr}
	
	mov r4, r0	/*store copies of struct*/
	
	ldr r1,[r4,$12]   /*rows of picture*/
	ldr r2,[r4,$16]	  /*columns of picture*/
	ldr r5,[r4,$20]	  /*image stored in structure*/	
	
	mov r0,$3	/*multiply number of columns by 3/4*/
	mul r2,r0	/*this will fix, loadin 4 out of 3*/
	lsr r2, $2 

	mov r8,$0
rowsloop: 

	mov r9,$0

	ldr r6,[r5,r8,lsl $2]	/*memory address of beginning of current row.*/

	add r8,$1	/*increment row counter*/

colsloop:
	/*make r2 the columns / 1.333333333333 in order to take proper amount of steps
	do remainder at end*/
	ldr r7,[r6,r9, lsl $2]
	
	
	ldr r0,=0xFFFFFFFF	/*load maximum of 32 bit int*/
	eor r7,r0		/*exclusive to subtract*/

	/*write register back to exactly where it is loaded from*/
	str r7,[r6,r9, lsl $2]

	add r9,$1	/*increment columns counter*/

	cmp r9,r2
	blt colsloop	/*for 3/4 of the columns*/

	cmp r8,r1
	blt rowsloop	/*loop through rows*/

	ldmfd	sp!,{r4-r11,lr}
	mov	pc,lr
