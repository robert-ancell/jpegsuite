This is a collection of JPEG images created to test JPEG applications like viewers, converters and editors. It is inspired by [PngSuite](http://www.schaik.com/pngsuite/).

JPEG is specified in [ISO/IEC 10918](https://www.iso.org/standard/18902.html) which is not freely available, but is freely available as [ITU T.81](https://www.w3.org/Graphics/JPEG/itu-t81.pdf) and [JFIF](https://www.w3.org/Graphics/JPEG/jfif3.pdf).

The reference JPEG implementation is [libjpeg](https://github.com/thorfdbg/libjpeg).

The JPEG files used in this test suite are licensed under the [CC0 license](https://creativecommons.org/public-domain/cc0/) so they should be suitable for including in your own projects.

## Baseline DCT Images

These are images that all JPEG decoders should support.

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

`32x32x8_grayscale_quantization.jpg`
`32x32x8_ycbcr_quantization.jpg`
The reference images quantized using the quantization tables in the JPEG specification.

`8x8x8_grayscale_zero_coefficients.jpg`
An 8x8 gray image encoded using a single data unit with zero coefficients.

`NxMx8_grayscale.jpg`
Small images of size 1x1 to 16x16.

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

## Extended DCT Images

Contains the same images as baseline DCT in both Huffman and Arithmetic encoding, and additionally:

`32x32x12_grayscale.jpg`
`32x32x12_ycbcr.jpg`
`32x32x12_ycbcr_interleaved.jpg`
The reference grayscale and color images with 12 bit samples.

`32x32x8_conditioning_bounds_4_6.jpg`
`32x32x8_conditioning_kx_6.jpg`
The reference grayscale image with non-default arithmetic conditioning.

## Progressive DCT Images

Contains the same images as extended DCT in both Huffman and Arithmetic encoding, and additionally:

`32x32x8_grayscale_spectral.jpg`
The reference grayscale image with DC coefficients sent in one scan, and AC coefficients in another.

`32x32x8_grayscale_spectral_all.jpg`
The reference grayscale image with a DC coefficient scan followed by 63 scans each containing AC coefficients in the order 1-63.

`32x32x8_grayscale_spectral_all_reverse.jpg`
The same as the previous image, except the AC coefficients are sent in reverse order (63-1).

`32x32x8_grayscale_successive_dc.jpg`
The reference grayscale image with the lower 4 bits of each DC coefficient sent in separate scans.

`32x32x8_grayscale_successive_ac.jpg`
The reference grayscale image with the lower 4 bits of each AC coefficient sent in separate scans.

`32x32x8_grayscale_successive.jpg`
The reference grayscale image with the lower 4 bits of each DC and AC coefficients sent in separate scans.

## Lossless Images

`32x32xN_grayscale.jpg`
The reference greyscale image in bit depths from 2-16 bits.

`NxMx8_grayscale.jpg`
Small images of size 1x1 to 16x16.

`32x32x8_grayscale_predictorN.jpg`
The reference greyscale image using each of the predictor methods.

`32x32x8_restarts.jpg`
The reference grayscale image sent in four sections with restart markers.

`32x32x8_rgb.jpg`
The reference color image in RGB format with a single scan per channel.

`32x32x8_dnl.jpg`
The reference grayscale image with the the height set to zero in the *start of frame* and instead sent in the *define number of lines* after the scan.
