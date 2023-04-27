I have implemented the Laplacian of Gaussian - Spatial Filter using python with the Command Line interface (CLI).
In this work,
	-> The function convolve method takes an image and mask and convolutes the image using the mask over all the pixels present in the image.
	-> The main function takes the image path input from the CLI, deine the LoG filter parameters, intialization of the kernel. Next we create a Laplacian Kernel and call the convolve the image with the Gaussian Kernel we get the smoothed image.
	-> Next we convolve the smoothed image with the Laplacian kernel and we get laplacian image. Then we Convolve the Laplacian image with the Gaussian kernel to compute the LoG response and Normalize the output image.
	-> Finally, we get the LOG image and save the image.