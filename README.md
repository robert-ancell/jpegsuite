This is a collection of JPEG images created to test JPEG applications like viewers, converters and editors. It is inspired by [PngSuite](http://www.schaik.com/pngsuite/).

JPEG is specified in [ISO/IEC 10918](https://www.iso.org/standard/18902.html) which is not freely available, but is freely available as [ITU T.81](https://www.w3.org/Graphics/JPEG/itu-t81.pdf) and [JFIF](https://www.w3.org/Graphics/JPEG/jfif3.pdf).

The reference JPEG implementation is [libjpeg](https://github.com/thorfdbg/libjpeg).

## Baseline Images

`32x32x8_grayscale.jpg`
The reference grayscale image.

`32x32x8_ycbcr.jpg`
`32x32x8_rgb.jpg`
`32x32x8_cmyk.jpg`
The reference color image in YCbCr, RGB and CMYK format with a single scan per channel.

`32x32x8_ycbcr_interleaved.jpg`
`32x32x8_rgb_interleaved.jpg`
`32x32x8_cmyk_interleaved.jpg`
The reference color image as above, but with a single scan interleaving each channel.

`32x32x8_comment.jpg`
`32x32x8_comments.jpg`
The reference grayscale image with one and two comments.

`32x32x8_dnl.jpg`
The reference grayscale image with the the height set to zero in the *start of frame* and instead sent in the *define number of lines* after the scan.

`32x32x8_restarts.jpg`
The reference grayscale image sent in four sections with restart markers.

`32x32x8_ycbcr_2x2_1x1_1x1.jpg`
`32x32x8_ycbcr_2x2_2x1_1x2.jpg`
`32x32x8_ycbcr_2x2_1x1_1x1_interleaved.jpg`
`32x32x8_ycbcr_2x2_2x1_1x2_interleaved.jpg`
The reference color image with the color channels using different sampling factors.

## Extended Images

Contains the same images as baseline in both Huffman and Arithmetic encoding, and additionally:

`32x32x12_grayscale.jpg`
`32x32x12_ycbcr.jpg`
`32x32x12_ycbcr_interleaved.jpg`
The reference grayscale and color images with 12 bit samples.
