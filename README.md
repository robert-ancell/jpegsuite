This is a collection of JPEG images created to test JPEG applications like viewers, converters and editors. It is inspired by [PngSuite](http://www.schaik.com/pngsuite/).

JPEG is specified in [ISO/IEC 10918](https://www.iso.org/standard/18902.html) which is not freely available, but is freely available as [ITU T.81](https://www.w3.org/Graphics/JPEG/itu-t81.pdf) and [JFIF](https://www.w3.org/Graphics/JPEG/jfif3.pdf).

The reference C++ JPEG implementation is [libjpeg](https://github.com/thorfdbg/libjpeg).
[pyjpeg](https://github.com/robert-ancell/pyjpeg) is a pure-Python implementation that was developed to generate these images and be useful to understand how JPEG works.

The JPEG files used in this test suite are licensed under the [CC0 license](https://creativecommons.org/public-domain/cc0/) so they should be suitable for including in your own projects.

## Baseline DCT Images

These are images that all JPEG decoders should support.

Each `.jpg` file has a matching `.json` file with an exact description of what it contains; the summary below is just an overview of the cases covered.

- The reference grayscale and color (YCbCr, RGB, CMYK) images, each in both a single interleaved scan and one scan per channel, with channels and scans in normal and reversed order.
- Chroma subsampling: several sampling factor combinations.
- Restart markers.
- Standard JPEG quantization tables.
- Small (1x1 up to 16x16) and tiny single-block (8x8) images, including all-black, all-white, all-gray, checkerboard, and all-zero-coefficient content.
- Random noise images (full noise and 1-bit "bitmap" noise).
- Comments (single and multiple), the DNL segment, and no/JFIF/Adobe/Exif header combinations, including Exif combined with an Adobe color-transform marker on RGB/CMYK images.

## Extended DCT Images

Contains the same images as baseline DCT in both Huffman and Arithmetic encoding, plus:

- The reference grayscale and color images at 12 bit precision.
- Non-default arithmetic conditioning parameters (bounds and Kx).

## Progressive DCT Images

Contains the same images as extended DCT in both Huffman and Arithmetic encoding, plus:

- Spectral selection: a DC scan followed by 63 single-coefficient AC scans, in both forward and reverse order.
- Successive approximation: the DC coefficients, AC coefficients, or both, sent as separate high/low-bit scans.

## Lossless Images

Available in both Huffman and Arithmetic encoding.

- The reference grayscale image at every precision from 2-16 bits, and using each of the 7 predictor methods.
- The reference color image in YCbCr and RGB format, each in both a single interleaved scan and one scan per channel.
- Small (1x1 up to 16x16) images, restart markers, and the DNL segment.
- Single-block all-black/white/gray/checkerboard images, and random noise images.

## JPEG-LS Images

- The reference grayscale image at every precision from 2-16 bits.
- Small (1x1 up to 16x16) images.
- The reference color image in YCbCr and RGB format, with no interleaving, line interleaving, and sample interleaving.
- Near-lossless encoding at several difference bounds.
- LSE preset-parameter edge cases: fully default, fully empty, and each individual parameter (MAXVAL, T1-T3, RESET) left empty on its own so a decoder must substitute its default.
- Restart markers, the oversize-image segment, mapping tables, and the DNL segment.
- Single-block all-black/white/gray/checkerboard images, and random noise images.
