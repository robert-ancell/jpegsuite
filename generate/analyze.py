#!/usr/bin/env python3

import struct
import sys

import jpeg
from jpeg_segments import *
from jpeg_decoder import *


def print_du(du):
    cols = []
    for x in range(8):
        col = []
        for y in range(8):
            col.append("%d" % du[y * 8 + x])
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
        print("  %s" % " ".join(row))


data = open(sys.argv[1], "rb").read()
decoder = Decoder(data)
decoder.decode()

for segment in decoder.segments:
    if isinstance(segment, StartOfImage):
        print("SOI Start of Image")
    elif isinstance(segment, ApplicationSpecificData):
        print("APP%d Application Specific Data" % segment.n)
    elif isinstance(segment, Comment):
        print("COM Comment")
        print(" Data: %s" % repr(segment.data))
    elif isinstance(segment, DefineQuantizationTables):
        print("DQT Define Quantization Tables")
        for precision, destination, values in segment.tables:
            print(" Table %d:" % destination)
            print("  Precision: %d bits" % {0: 8, 1: 16}[precision])
            print_du(values)
    elif isinstance(segment, DefineHuffmanTables):
        print("DHT Define Huffman Tables")
        for table in segment.tables:
            print(
                " %s Table %d:"
                % ({0: "DC", 1: "AC"}[table.table_class], table.identifier)
            )

            def tobitstring(bits):
                s = ""
                for b in bits:
                    s += str(b)
                return s

            for code in table.table.keys():
                print("  %02x: %s" % (table.table[code], tobitstring(code)))
    elif isinstance(segment, DefineArithmeticConditioning):
        print("DAC Define Arithmetic Conditioning")
        for conditioning in segment.tables:
            print(
                " %s Table %d: %s"
                % (
                    {0: "DC", 1: "AC"}[conditioning.table_class],
                    conditioning.identifier,
                    repr(conditioning.value),
                )
            )
    elif isinstance(segment, DefineRestartInterval):
        print("DRI Define Restart Interval")
        print(" Restart interval: %d" % segment.restart_interval)
    elif isinstance(segment, StartOfFrame):
        print(
            "SOF%d Start of Frame, %s"
            % (
                segment.n,
                {
                    0: "Baseline DCT",
                    1: "Extended sequential DCT, Huffman coding",
                    2: "Progressive DCT, Huffman coding",
                    3: "Lossless (sequential), Huffman coding",
                    5: "Differential sequential DCT, Huffman coding",
                    6: "Differential progressive DCT, Huffman coding",
                    7: "Differential lossless (sequential), Huffman coding",
                    9: "Extended sequential DCT, arithmetic coding",
                    10: "Progressive DCT, arithmetic coding",
                    11: "Lossless (sequential), arithmetic coding",
                    13: "Differential sequential DCT, arithmetic coding",
                    14: "Differential progressive DCT, arithmetic coding",
                    15: "Differential lossless (sequential), arithmetic coding",
                }[segment.n],
            )
        )
        print(" Precision: %d bits" % segment.precision)
        print(
            " Number of lines: %d" % segment.number_of_lines
        )  # FIXME: Note if zero defined later
        print(" Number of samples per line: %d" % segment.samples_per_line)
        for component in segment.components:
            print(" Component %d:" % component.id)
            print(
                "  Sampling Factor: %dx%d"
                % (component.sampling_factor[0], component.sampling_factor[1])
            )
            print("  Quantization Table: %d" % component.quantization_table_index)
    elif isinstance(segment, StartOfStream):
        print("SOS Start of Stream")
        for component in segment.components:
            print(" Component %d:" % component.component_selector)
            print("  DC Table: %d" % component.dc_table)
            print("  AC Table: %d" % component.ac_table)
        print(" Spectral Selection: %d-%d" % (segment.ss, segment.se))
        print(" Previous Point Transform: %d" % segment.ah)
        print(" Point Transform: %d" % segment.al)
    elif isinstance(segment, DCTDataUnit):
        print(" Data Unit:")
        print_du(segment.coefficients)
    elif isinstance(segment, Restart):
        print("RST%d Restart" % segment.n)
    elif isinstance(segment, DefineNumberOfLines):
        print("DNL Define Number of Lines")
        print(" Number of lines: %d" % segment.number_of_lines)
    elif isinstance(segment, EndOfImage):
        print("EOI End of Image")
    else:
        print(segment)