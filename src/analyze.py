#!/usr/bin/env python3

import sys

import pyjpeg


def print_data_unit(data_unit: list[int]) -> None:
    values = pyjpeg.dct.unzig_zag(data_unit)

    cols = []
    for x in range(8):
        col = []
        for y in range(8):
            value = values[y * 8 + x]
            col.append(f"{value}")
        cols.append(col)

    col_widths = []
    for x in range(8):
        width = 0
        for y in range(8):
            width = max(width, len(cols[x][y]))
        col_widths.append(width)

    for y in range(8):
        row = []
        for x in range(8):
            row.append(cols[x][y].rjust(col_widths[x]))
        print("  " + " ".join(row))


if len(sys.argv) != 2:
    print("Usage: analyze.py <filename.jpg>")
    sys.exit(1)

with open(sys.argv[1], "rb") as f:
    data = f.read()
reader = pyjpeg.BufferedReader(data)
stream = pyjpeg.Stream.read(reader)

is_lossless = False
is_ls = False
for segment in stream.segments:
    if isinstance(segment, pyjpeg.StartOfImage):
        print("SOI Start of Image")
    elif isinstance(segment, pyjpeg.JfifHeader):
        print(f"APP{segment.n} JFIF")
        print(f" Version: {segment.version[0]}.{segment.version[1]}")
        if segment.density.unit == pyjpeg.JfifDensityUnit.ASPECT_RATIO:
            print(f" Aspect Ratio: {segment.density.x}x{segment.density.y}")
        elif segment.density.unit == pyjpeg.JfifDensityUnit.DPI:
            print(f" Density: {segment.density.x}x{segment.density.y}dpi")
        elif segment.density.unit == pyjpeg.JfifDensityUnit.DPCM:
            print(f" Density: {segment.density.x}x{segment.density.y}dpcm")
        if len(segment.thumbnail_data) > 0:
            # FIXME: Support RGB thumbnails
            s = f" Thumbnail {segment.thumbnail_size[0]}x{segment.thumbnail_size[0]}:"
            for i in range(0, len(segment.thumbnail_data), 3):
                if i % (segment.thumbnail_size[0] * 3) == 0:
                    s += "\n "
                r = segment.thumbnail_data[i]
                g = segment.thumbnail_data[i + 1]
                b = segment.thumbnail_data[i + 2]
                s += f" {r},{g},{b}"
            print(s)
    elif isinstance(segment, pyjpeg.JfifJpegThumbnail):
        print(f"APP{segment.n} JPEG Thumbnail")
        print(f" Data: {segment.data!r}")
    elif isinstance(segment, pyjpeg.JfifPalletizedThumbnail):
        print(f"APP{segment.n} Palletized Thumbnail")
        print(f" Width: {segment.width}")
        print(f" Height: {segment.height}")
        print(f" Data: {segment.data}")
    elif isinstance(segment, pyjpeg.JfifRgbThumbnail):
        print(f"APP{segment.n} RGB Thumbnail")
        print(f" Width: {segment.width}")
        print(f" Height: {segment.height}")
        print(f" Data: {segment.data}")
    elif isinstance(segment, pyjpeg.SpiffHeader):
        print(f"APP{segment.n} SPIFF")
        print(f" Version: {segment.version[0]}.{segment.version[1]}")
        print(f" Profile: {segment.profile}")
        print(f" Number of Components: {segment.number_of_components}")
        print(f" Height: {segment.height}")
        print(f" Width: {segment.width}")
        print(f" Color Space: {segment.color_space}")
        print(f" Bits per Sample: {segment.bits_per_sample}")
        print(f" Compression Type: {segment.compression_type}")
        print(f" Resolution Units: {segment.resolution_units}")
        print(f" Vertical Resolution: {segment.vertical_resoution}")
        print(f" Horizontal Resolution: {segment.horizontal_resolution}")
    elif isinstance(segment, pyjpeg.ExifHeader):
        print(f"APP{segment.n} EXIF")
        print(f" Data: {segment.data!r}")
    elif isinstance(segment, pyjpeg.AdobeHeader):
        print(f"APP{segment.n} Adobe")
        print(f" Version: {segment.version}")
        print(f" Flags 0: {segment.flags0:04x}")
        print(f" Flags 1: {segment.flags1:04x}")
        colorspace_str = {
            pyjpeg.AdobeColorSpace.RGB_OR_CMYK: "RGB or CMYK",
            pyjpeg.AdobeColorSpace.Y_CB_CR: "YCbCr",
            pyjpeg.AdobeColorSpace.Y_CB_CR_K: "YCbCrK",
        }.get(segment.color_space, f"{segment.color_space}")
        print(f" Colorspace: {colorspace_str}")
    elif isinstance(segment, pyjpeg.UnknownApplicationSpecificData):
        print(f"APP{segment.n} Application Specific Data")
        s = " Data: "
        for d in segment.data:
            s += f"{d:02X}"
        print(s)
    elif isinstance(segment, pyjpeg.Comment):
        print("COM Comment")
        print(f" Data: {segment.data!r}")
    elif isinstance(segment, pyjpeg.DefineQuantizationTables):
        print("DQT Define Quantization Tables")
        for quantization_table in segment.tables:
            print(f" Table {quantization_table.destination}:")
            print(f"  Precision: {quantization_table.precision} bits")
            print_data_unit(quantization_table.values)
    elif isinstance(segment, pyjpeg.DefineHuffmanTables):
        print("DHT Define Huffman Tables")
        for huffman_table in segment.tables:
            class_name = {0: "DC", 1: "AC"}[huffman_table.table_class]
            print(f" {class_name} Table {huffman_table.destination}:")
            for i, symbols in enumerate(huffman_table.table):
                if len(symbols) > 0:
                    s = f"  Symbols of length {i + 1}:"
                    for symbol in symbols:
                        s += f" {symbol:02x}"
                    print(s)
    elif isinstance(segment, pyjpeg.DefineArithmeticConditioning):
        print("DAC Define Arithmetic Conditioning")
        for conditioning in segment.tables:
            class_name = {0: "DC", 1: "AC"}[conditioning.table_class]
            print(
                f" {class_name} Table {conditioning.destination}: {conditioning.value}"
            )
    elif isinstance(segment, pyjpeg.DefineRestartInterval):
        print("DRI Define Restart Interval")
        print(f" Restart interval: {segment.restart_interval}")
    elif isinstance(segment, pyjpeg.ExpandReferenceComponents):
        print("EXP Expand Reference Components")
        print(
            " Expand Horizontal: {}".format(
                {False: "No", True: "Yes"}[segment.expand_horizontal != 0]
            )
        )
        print(
            " Expand Vertical: {}".format(
                {False: "No", True: "Yes"}[segment.expand_vertical != 0]
            )
        )
    elif isinstance(segment, pyjpeg.StartOfFrame):
        is_lossless = segment.n in (3, 7, 11, 15)
        is_ls = segment.n == 55
        frame_name = {
            pyjpeg.FrameType.BASELINE: "Baseline DCT",
            pyjpeg.FrameType.EXTENDED_HUFFMAN: "Extended sequential DCT, Huffman coding",
            pyjpeg.FrameType.PROGRESSIVE_HUFFMAN: "Progressive DCT, Huffman coding",
            pyjpeg.FrameType.LOSSLESS_HUFFMAN: "Lossless (sequential), Huffman coding",
            pyjpeg.FrameType.DIFFERENTIAL_SEQUENTIAL_HUFFMAN: "Differential sequential DCT, Huffman coding",
            pyjpeg.FrameType.DIFFERENTIAL_PROGRESSIVE_HUFFMAN: "Differential progressive DCT, Huffman coding",
            pyjpeg.FrameType.DIFFERENTIAL_LOSSLESS_HUFFMAN: "Differential lossless (sequential), Huffman coding",
            pyjpeg.FrameType.EXTENDED_ARITHMETIC: "Extended sequential DCT, Arithmetic coding",
            pyjpeg.FrameType.PROGRESSIVE_ARITHMETIC: "Progressive DCT, Arithmetic coding",
            pyjpeg.FrameType.LOSSLESS_ARITHMETIC: "Lossless (sequential), Arithmetic coding",
            pyjpeg.FrameType.DIFFERENTIAL_SEQUENTIAL_ARITHMETIC: "Differential sequential DCT, Arithmetic coding",
            pyjpeg.FrameType.DIFFERENTIAL_PROGRESSIVE_ARITHMETIC: "Differential progressive DCT, Arithmetic coding",
            pyjpeg.FrameType.DIFFERENTIAL_LOSSLESS_ARITHMETIC: "Differential lossless (sequential), Arithmetic coding",
            pyjpeg.FrameType.LS: "JPEG-LS",
        }[segment.n]
        print(f"SOF{segment.n} Start of Frame, {frame_name}")
        print(f" Precision: {segment.precision} bits")
        print(
            f" Number of lines: {segment.number_of_lines}"
        )  # FIXME: Note if zero defined later
        print(f" Number of samples per line: {segment.samples_per_line}")
        for frame_component in segment.components:
            print(" Component:")
            print(f"  Id: {frame_component.id}")
            print(
                f"  Sampling Factor: {frame_component.sampling_factor[0]}x{frame_component.sampling_factor[1]}"
            )
            if not is_lossless and not is_ls:
                print(
                    f"  Quantization Table: {frame_component.quantization_table_index}"
                )
    elif isinstance(segment, pyjpeg.StartOfScan):
        print("SOS Start of Scan")
        for scan_component in segment.components:
            print(" Component:")
            print(f"  Id: {scan_component.component_selector}")
            if is_ls:
                print(f"  Mapping table: {scan_component.get_mapping_table()}")
            else:
                print(f"  DC Table: {scan_component.dc_table}")
                if not is_lossless:
                    print(f"  AC Table: {scan_component.ac_table}")
        if is_lossless:
            print(f" Predictor: {segment.spectral_selection[0]}")
        elif is_ls:
            print(f" Near: {segment.spectral_selection[0]}")
            interleave_mode = segment.spectral_selection[1]
            interleave_mode_str = {
                pyjpeg.LSInterleaveMode.NONE: "None",
                pyjpeg.LSInterleaveMode.LINE: "Line",
                pyjpeg.LSInterleaveMode.SAMPLE: "Sample",
            }.get(
                interleave_mode,
                f"{interleave_mode}",
            )
            print(f" Interleave Mode: {interleave_mode_str}")
        else:
            print(
                f" Spectral Selection: {segment.spectral_selection[0]}-{segment.spectral_selection[1]}"
            )

        if (segment.point_transform & 0xF0) != 0:
            print(f" Previous Point Transform: {segment.point_transform >> 4}")
        print(f" Point Transform: {segment.point_transform & 0xF}")
    elif isinstance(segment, (pyjpeg.HuffmanDCTScan, pyjpeg.ArithmeticDCTScan)):
        for data_unit in segment.data_units:
            print_data_unit(data_unit)
    elif isinstance(
        segment,
        (pyjpeg.HuffmanLosslessScan, pyjpeg.ArithmeticLosslessScan, pyjpeg.LSScan),
    ):
        s = " Samples:"
        for sample in segment.samples:
            s += f" {sample}"
        print(s)
    elif isinstance(segment, pyjpeg.Restart):
        print("RST{segment.index} Restart")
    elif isinstance(segment, pyjpeg.DefineNumberOfLines):
        print("DNL Define Number of Lines")
        print(f" Number of lines: {segment.number_of_lines}")
    elif isinstance(segment, pyjpeg.LSCodingParameters):
        print("LSE Coding Parameters")
        print(f" Maximum value: {segment.maxval}")
        t1, t2, t3 = segment.gradient_thresholds
        print(" Gradient thresholds: {t1}, {t2}, {t3}")
        print(f" Reset: {segment.reset}")
    elif isinstance(segment, pyjpeg.LSMappingTable):
        print("LSE Mapping Table")
        print(f" Table ID: {segment.table_id}")
        print(f" Weight: {segment.weight}")
        print(f" Table Data: {segment.table.hex()}")
    elif isinstance(segment, pyjpeg.LSMappingTableContinuation):
        print("LSE Mapping Table Continuation")
        print(f" Table ID: {segment.table_id}")
        print(f" Weight: {segment.weight}")
        print(f" Table Data: {segment.table.hex()}")
    elif isinstance(segment, pyjpeg.LSOversizeImageDimensions):
        print("LSE Oversize Image Dimensions")
        print(f" Number of lines: {segment.number_of_lines}")
        print(f" Number of samples per line: {segment.samples_per_line}")
    elif isinstance(segment, pyjpeg.LSUnknownPresetParameters):
        print("LSE Preset Parameters")
        print(f" ID: {segment.id}")
        print(f" Data: {segment.data.hex()}")
    elif isinstance(segment, pyjpeg.EndOfImage):
        print("EOI End of Image")
    else:
        print(segment)
