#!/usr/bin/env python3

import json
import math

import pyjpeg

from pnm import *

WIDTH = 32
HEIGHT = 32

# Simplest Exif structure, containing no tags.
MINIMAL_EXIF_DATA = b"II*\x00\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"


def clamp(value: int, min_value: int, max_value: int) -> int:
    return max(min(value, max_value), min_value)


def rgb_to_ycbcr(r: int, g: int, b: int, precision: int) -> tuple[int, int, int]:
    offset = 1 << (precision - 1)
    y = round(0.299 * r + 0.587 * g + 0.114 * b)
    cb = round(-0.1687 * r - 0.3313 * g + 0.5 * b + offset)
    cr = round(0.5 * r - 0.4187 * g - 0.0813 * b + offset)
    max_value = (1 << precision) - 1
    return (clamp(y, 0, max_value), clamp(cb, 0, max_value), clamp(cr, 0, max_value))


def rgb_to_cmyk(r: int, g: int, b: int, precision: int) -> tuple[int, int, int, int]:
    max_value = 1 << precision
    rf = r / max_value
    gf = g / max_value
    bf = b / max_value
    k = 1 - max(rf, gf, bf)
    if k == 1:
        c, m, y = 0.0, 0.0, 0.0
    else:
        c = (1 - rf - k) / (1 - k)
        m = (1 - gf - k) / (1 - k)
        y = (1 - bf - k) / (1 - k)
    return (
        round(c * max_value),
        round(m * max_value),
        round(y * max_value),
        round(k * max_value),
    )


def make_grayscale(precision: int) -> list[int]:
    width, height, max_value, channels, raw_samples = read_pnm(
        "data/32x32x16_grayscale.pgm"
    )
    assert width == WIDTH
    assert height == HEIGHT
    assert channels == 1
    samples = []
    for s in raw_samples:
        samples.append(round(s * ((1 << precision) - 1) / max_value))
    return samples


def make_solid(width: int, height: int, value: int) -> list[int]:
    return [value] * width * height


def make_check(width: int, height: int, white: int) -> list[int]:
    samples = []
    for y in range(width):
        for x in range(height):
            if (x + y) % 2 == 0:
                samples.append(0)
            else:
                samples.append(white)
    return samples


def make_mapped_samples(samples: list[list[int]]) -> tuple[list[int], bytes]:
    sample_to_index: dict[bytes, int] = {}
    index_to_sample: dict[int, bytes] = {}
    weight = len(samples)
    for i in range(len(samples[0])):
        sample = b""
        for j in range(weight):
            sample += bytes([samples[j][i]])
        if sample not in sample_to_index:
            index = len(sample_to_index)
            sample_to_index[sample] = index
            index_to_sample[index] = sample
    mapping_table = b""
    for i in range(len(index_to_sample)):
        mapping_table += index_to_sample[i]
    mapped_samples = []
    for i in range(len(samples[0])):
        sample = b""
        for j in range(weight):
            sample += bytes([samples[j][i]])
        mapped_samples.append(sample_to_index[sample])
    return mapped_samples, mapping_table


grayscale_samples8 = make_grayscale(8)
grayscale_samples12 = make_grayscale(12)
grayscale_components8 = [(grayscale_samples8, (1, 1))]
grayscale_components12 = [(grayscale_samples12, (1, 1))]


def make_rgb(precision: int) -> list[list[int]]:
    width, height, max_value, channels, raw_samples = read_pnm("data/32x32x16_rgb.ppm")
    assert width == WIDTH
    assert height == HEIGHT
    assert channels == 3
    r_samples = []
    g_samples = []
    b_samples = []
    for i in range(0, len(raw_samples), 3):
        r_samples.append(round(raw_samples[i] * ((1 << precision) - 1) / max_value))
        g_samples.append(round(raw_samples[i + 1] * ((1 << precision) - 1) / max_value))
        b_samples.append(round(raw_samples[i + 2] * ((1 << precision) - 1) / max_value))
    return [r_samples, g_samples, b_samples]


rgb_samples8 = make_rgb(8)
rgb_components8 = [
    (rgb_samples8[0], (1, 1)),
    (rgb_samples8[1], (1, 1)),
    (rgb_samples8[2], (1, 1)),
]


def make_ycbcr(precision: int) -> list[list[int]]:
    r_samples, g_samples, b_samples = make_rgb(precision)
    y_samples = []
    cb_samples = []
    cr_samples = []
    for i in range(len(r_samples)):
        (y, cb, cr) = rgb_to_ycbcr(r_samples[i], g_samples[i], b_samples[i], precision)
        y_samples.append(y)
        cb_samples.append(cb)
        cr_samples.append(cr)
    return [y_samples, cb_samples, cr_samples]


ycbcr_samples8 = make_ycbcr(8)
ycbcr_samples12 = make_ycbcr(12)
ycbcr_components8 = [
    (ycbcr_samples8[0], (1, 1)),
    (ycbcr_samples8[1], (1, 1)),
    (ycbcr_samples8[2], (1, 1)),
]
ycbcr_components12 = [
    (ycbcr_samples12[0], (1, 1)),
    (ycbcr_samples12[1], (1, 1)),
    (ycbcr_samples12[2], (1, 1)),
]


def make_cmyk(precision: int) -> list[list[int]]:
    r_samples, g_samples, b_samples = make_rgb(precision)
    c_samples = []
    m_samples = []
    y_samples = []
    k_samples = []
    for i in range(len(r_samples)):
        (c, m, y, k) = rgb_to_cmyk(r_samples[i], g_samples[i], b_samples[i], precision)
        c_samples.append(c)
        m_samples.append(m)
        y_samples.append(y)
        k_samples.append(k)
    return [c_samples, m_samples, y_samples, k_samples]


cmyk_samples8 = make_cmyk(8)
cmyk_components8 = [
    (cmyk_samples8[0], (1, 1)),
    (cmyk_samples8[1], (1, 1)),
    (cmyk_samples8[2], (1, 1)),
    (cmyk_samples8[3], (1, 1)),
]


def scale_samples(
    width: int, height: int, samples: list[int], h_max: int, h: int, v_max: int, v: int
) -> list[int]:
    if h == h_max and v == v_max:
        return samples
    assert h_max % h == 0
    assert v_max % v == 0
    out_samples = []
    for y in range(0, height, v_max // v):
        for x in range(0, width, h_max // h):
            out_samples.append(samples[y * width + x])
    return out_samples


def segments_to_json(segments: list[pyjpeg.Segment]) -> list[dict[str, object]]:
    s = []
    for segment in segments:
        value: dict[str, object] = {}
        if isinstance(segment, pyjpeg.StartOfImage):
            value["type"] = "SOI"
        elif isinstance(segment, pyjpeg.ApplicationSpecificData):
            value["type"] = f"APP{segment.n}"
            if isinstance(segment, pyjpeg.JfifHeader):
                value.update(
                    {
                        "format": "JFIF",
                        "version": f"{segment.version[0]}.{segment.version[1]}",
                    }
                )
                if segment.density.unit == pyjpeg.JfifDensityUnit.ASPECT_RATIO:
                    value["aspect-ratio"] = f"{segment.density.x}x{segment.density.y}"
                elif segment.density.unit == pyjpeg.JfifDensityUnit.DPI:
                    value["dpi"] = f"{segment.density.x}x{segment.density.y}"
                elif segment.density.unit == pyjpeg.JfifDensityUnit.DPCM:
                    value["dpcm"] = f"{segment.density.x}x{segment.density.y}"
                else:
                    value["density"] = {
                        "unit": segment.density.unit,
                        "x": segment.density.x,
                        "y": segment.density.y,
                    }
                if segment.thumbnail_size != (0, 0) or len(segment.thumbnail_data) != 0:
                    value["thumbnail"] = {
                        "size": segment.thumbnail_size,
                        "data": list(segment.thumbnail_data),
                    }
            elif isinstance(segment, pyjpeg.AdobeHeader):
                value["format"] = "Adobe"
                color_space_value = {
                    pyjpeg.AdobeColorSpace.RGB_OR_CMYK: "RGB or CMYK",
                    pyjpeg.AdobeColorSpace.Y_CB_CR: "YCbCr",
                    pyjpeg.AdobeColorSpace.Y_CB_CR_K: "YCbCrK",
                }.get(segment.color_space, segment.color_space)
                value.update(
                    {
                        "version": segment.version,
                        "flags0": segment.flags0,
                        "flags1": segment.flags1,
                        "color-space": color_space_value,
                    }
                )
            elif isinstance(segment, pyjpeg.UnknownApplicationSpecificData):
                value["data"] = list(segment.data)
            elif isinstance(segment, pyjpeg.ExifHeader):
                value.update({"format": "Exif", "data": list(segment.data)})
        elif isinstance(segment, pyjpeg.Comment):
            value.update({"type": "COM", "data": str(segment.data, "ascii")})
        elif isinstance(segment, pyjpeg.DefineQuantizationTables):
            tables = []
            for quantization_table in segment.tables:
                quantization_table_values = pyjpeg.dct.unzig_zag(
                    quantization_table.values
                )
                values = []
                for y in range(8):
                    row: list[int] = []
                    values.append(row)
                    for x in range(8):
                        row.append(quantization_table_values[y * 8 + x])
                tables.append(
                    {
                        "destination": quantization_table.destination,
                        "precision": quantization_table.precision,
                        "values": values,
                    }
                )
            value.update({"type": "DQT", "tables": tables})
        elif isinstance(segment, pyjpeg.DefineHuffmanTables):
            tables = []
            for huffman_table in segment.tables:
                tables.append(
                    {
                        "class": {0: "dc", 1: "ac"}[huffman_table.table_class],
                        "destination": huffman_table.destination,
                        "symbols": huffman_table.table,
                    }
                )
            value.update({"type": "DHT", "tables": tables})
        elif isinstance(segment, pyjpeg.DefineArithmeticConditioning):
            tables = []
            for arithmetic_table in segment.tables:
                table_value = {
                    "class": {0: "dc", 1: "ac"}[arithmetic_table.table_class],
                    "destination": arithmetic_table.destination,
                }
                if arithmetic_table.table_class == 0:
                    table_value["lower"] = arithmetic_table.value & 0xF
                    table_value["upper"] = arithmetic_table.value >> 4
                else:
                    table_value["kx"] = arithmetic_table.value
                tables.append(table_value)
            value.update({"type": "DAC", "tables": tables})
        elif isinstance(segment, pyjpeg.DefineRestartInterval):
            value.update({"type": "DRI", "restart_interval": segment.restart_interval})
        elif isinstance(segment, pyjpeg.StartOfFrame):
            components = []
            for frame_component in segment.components:
                components.append(
                    {
                        "id": frame_component.id,
                        "sampling_factor": frame_component.sampling_factor,
                        "quantization_table": frame_component.quantization_table_index,
                    }
                )
            value.update(
                {
                    "type": f"SOF{segment.n}",
                    "precision": segment.precision,
                    "number_of_lines": segment.number_of_lines,
                    "samples_per_line": segment.samples_per_line,
                    "components": components,
                }
            )
        elif isinstance(segment, pyjpeg.StartOfScan):
            components = []
            for scan_component in segment.components:
                components.append(
                    {
                        "component_id": scan_component.component_selector,
                        "dc_table": scan_component.dc_table,
                        "ac_table": scan_component.ac_table,
                    }
                )
            value.update(
                {
                    "type": "SOS",
                    "components": components,
                    "spectral_selection": segment.spectral_selection,
                }
            )
            if segment.point_transform & 0xF0 != 0:
                value["previous_point_transform"] = segment.point_transform >> 4
            value["point_transform"] = segment.point_transform & 0xF
        elif isinstance(segment, (pyjpeg.HuffmanDCTScan, pyjpeg.ArithmeticDCTScan)):
            value["type"] = "DCT"
        elif isinstance(
            segment, (pyjpeg.HuffmanLosslessScan, pyjpeg.ArithmeticLosslessScan)
        ):
            value["type"] = "Lossless"
        elif isinstance(segment, pyjpeg.LSScan):
            value["type"] = "LS"
        elif isinstance(
            segment,
            (
                pyjpeg.HuffmanDCTDCSuccessiveScan,
                pyjpeg.HuffmanDCTACSuccessiveScan,
                pyjpeg.ArithmeticDCTDCSuccessiveScan,
                pyjpeg.ArithmeticDCTACSuccessiveScan,
            ),
        ):
            value["type"] = "DCTSuccessive"
        elif isinstance(segment, pyjpeg.Restart):
            value["type"] = f"RST{segment.index}"
        elif isinstance(segment, pyjpeg.DefineNumberOfLines):
            value.update({"type": "DNL", "number_of_lines": segment.number_of_lines})
        elif isinstance(segment, pyjpeg.EndOfImage):
            value["type"] = "EOI"
        elif isinstance(segment, pyjpeg.LSCodingParameters):
            value.update(
                {
                    "type": "LSE",
                    "subtype": "Coding parameters",
                    "maxval": segment.maxval,
                    "gradient-thresholds": segment.gradient_thresholds,
                    "reset": segment.reset,
                }
            )
        elif isinstance(segment, pyjpeg.LSMappingTable):
            # FIXME: Contents
            value.update(
                {
                    "type": "LSE",
                    "subtype": "Mapping table",
                    "weight": segment.weight,
                    "table": segment.table.hex(),
                }
            )
        elif isinstance(segment, pyjpeg.LSOversizeImageDimensions):
            value.update(
                {
                    "type": "LSE",
                    "subtype": "Oversize image dimensions",
                    "number-of-lines": segment.number_of_lines,
                    "samples-per-line": segment.samples_per_line,
                }
            )
        else:
            assert False
        s.append(value)
    return s


def make_dct_data_units(
    width: int,
    height: int,
    precision: int,
    samples: list[int],
    quantization_table: list[int],
) -> list[list[int]]:
    data_units = []
    for du_y in range(0, height, 8):
        for du_x in range(0, width, 8):
            values = []
            for y in range(8):
                for x in range(8):
                    px = du_x + x
                    py = du_y + y
                    if px >= width:
                        px = width - 1
                    if py >= height:
                        py = height - 1
                    p = samples[py * width + px]
                    values.append(p)

            data_unit = pyjpeg.dct.fdct(values, precision, quantization_table)
            data_units.append(data_unit)

    return data_units


def generate_dct(
    section: str,
    name: str,
    description: str,
    width: int,
    height: int,
    components: list[tuple[list[int], tuple[int, int]]],
    scans: list[tuple[list[int], int, int, int]],
    precision: int = 8,
    luminance_quantization_table: list[int] = [1] * 64,
    chrominance_quantization_table: list[int] = [1] * 64,
    use_dnl: bool = False,
    restart_interval: int = 0,
    color_space: int = pyjpeg.AdobeColorSpace.Y_CB_CR,
    comments: list[bytes] | None = None,
    jfif: bool = True,
    exif: bool = False,
    adobe: bool = False,
    extended: bool = False,
    progressive: bool = False,
    arithmetic: bool = False,
    arithmetic_conditioning_bounds: tuple[
        tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]
    ] = (
        (0, 1),
        (0, 1),
        (0, 1),
        (0, 1),
    ),
    arithmetic_conditioning_kx: tuple[int, int, int, int] = (5, 5, 5, 5),
) -> None:
    if arithmetic:
        assert extended or progressive

    n_components = len(components)

    max_h_sampling_factor = 0
    max_v_sampling_factor = 0
    for _, sampling_factor in components:
        h, v = sampling_factor
        max_h_sampling_factor = max(h, max_h_sampling_factor)
        max_v_sampling_factor = max(v, max_v_sampling_factor)

    component_sizes = []
    scaled_component_samples = []
    for samples, sampling_factor in components:
        w = math.ceil(width * sampling_factor[0] / max_h_sampling_factor)
        h = math.ceil(height * sampling_factor[1] / max_v_sampling_factor)
        component_sizes.append((w, h))
        scaled_component_samples.append(
            scale_samples(
                width,
                height,
                samples,
                max_h_sampling_factor,
                sampling_factor[0],
                max_v_sampling_factor,
                sampling_factor[1],
            )
        )

    if color_space == pyjpeg.AdobeColorSpace.RGB_OR_CMYK:
        assert n_components in (3, 4)
    elif color_space == pyjpeg.AdobeColorSpace.Y_CB_CR:
        assert n_components in (1, 3)
    elif color_space == pyjpeg.AdobeColorSpace.Y_CB_CR_K:
        assert n_components == 4

    if color_space == pyjpeg.AdobeColorSpace.Y_CB_CR and n_components == 3:
        use_chrominance = True
    else:
        use_chrominance = False

    quantization_tables = [
        pyjpeg.QuantizationTable(0, luminance_quantization_table),
    ]
    if use_chrominance:
        quantization_tables.append(
            pyjpeg.QuantizationTable(1, chrominance_quantization_table)
        )
    component_quantization_tables = []
    for i in range(n_components):
        if i == 0 or not use_chrominance:
            table_index = 0
        else:
            table_index = 1
        component_quantization_tables.append(table_index)

    data_units = []
    for i in range(n_components):
        if i == 0 or not use_chrominance:
            quantization_table = luminance_quantization_table
        else:
            quantization_table = chrominance_quantization_table
        data_units.append(
            make_dct_data_units(
                component_sizes[i][0],
                component_sizes[i][1],
                precision,
                scaled_component_samples[i],
                quantization_table,
            )
        )

    sof_components = []
    for i, (_, sampling_factor) in enumerate(components):
        sof_components.append(
            pyjpeg.FrameComponent.dct(
                i + 1,
                sampling_factor=sampling_factor,
                quantization_table_index=component_quantization_tables[i],
            )
        )

    # FIXME: Split into restart intervals

    # Generate scans
    jpeg_scans = []
    scan_components = []
    for i in range(n_components):
        if arithmetic or i == 0 or not use_chrominance:
            dc_table_index = 0
            ac_table_index = 0
        else:
            dc_table_index = 1
            ac_table_index = 1
        scan_components.append(
            pyjpeg.ScanComponent.dct(
                i + 1, dc_table=dc_table_index, ac_table=ac_table_index
            )
        )
    if restart_interval == 0:
        n_intervals = 1
    else:
        n_intervals = len(data_units[0]) // restart_interval
    for scan_index, (component_indexes, start, end, point_transform) in enumerate(
        scans
    ):
        sos_components = []
        for i in component_indexes:
            sos_components.append(scan_components[i])
        selection = (start, end)
        successive = False
        previous_point_transform = 0
        for i in range(scan_index):
            (c, s, e, p) = scans[i]
            if (c, s, e) == (component_indexes, start, end) and p != 0:
                successive = True
                previous_point_transform = p
        if successive and start == 0:
            assert end == 0
        sos = pyjpeg.StartOfScan.dct(
            sos_components,
            spectral_selection=selection,
            point_transform=point_transform,
            previous_point_transform=previous_point_transform,
        )
        scan_data: list[pyjpeg.Segment] = []
        if arithmetic:
            if successive:
                assert len(component_indexes) == 1
                if start == 0:
                    scan_data.append(
                        pyjpeg.ArithmeticDCTDCSuccessiveScan(
                            data_units[component_indexes[0]],
                            point_transform=point_transform,
                        )
                    )
                else:
                    scan_data.append(
                        pyjpeg.ArithmeticDCTACSuccessiveScan(
                            data_units[component_indexes[0]],
                            spectral_selection=selection,
                            point_transform=point_transform,
                        )
                    )
            else:
                for interval in range(n_intervals):
                    arithmetic_components = []
                    mcu_data_units = []
                    for i in component_indexes:
                        # MCU is 1 in non-interleaved
                        if len(component_indexes) == 1:
                            sampling_factor = (1, 1)
                        else:
                            _, sampling_factor = components[i]
                        interval_length = len(data_units[i]) // n_intervals
                        interval_start = interval * interval_length
                        mcu_data_units.append(
                            pyjpeg.dct.order_mcu_dct_data_units(
                                component_sizes[i][0],
                                component_sizes[i][1] // n_intervals,
                                data_units[i][
                                    interval_start : interval_start + interval_length
                                ],
                                sampling_factor,
                            )
                        )
                        arithmetic_components.append(
                            pyjpeg.ArithmeticDCTScanComponent(
                                sampling_factor=sampling_factor,
                                conditioning_bounds=arithmetic_conditioning_bounds[
                                    scan_components[i].dc_table
                                ],
                                kx=arithmetic_conditioning_kx[
                                    scan_components[i].ac_table
                                ],
                            )
                        )
                    if interval != 0:
                        scan_data.append(pyjpeg.Restart((interval - 1) % 8))
                    # FIXME: Interleave earlier
                    data_units_ = []
                    while len(mcu_data_units[0]) > 0:
                        for i, arithmetic_scan_component in enumerate(
                            arithmetic_components
                        ):
                            for _ in range(
                                arithmetic_scan_component.sampling_factor[0]
                                * arithmetic_scan_component.sampling_factor[1]
                            ):
                                data_units_.append(mcu_data_units[i].pop(0))
                    scan_data.append(
                        pyjpeg.ArithmeticDCTScan(
                            data_units_,
                            components=arithmetic_components,
                            spectral_selection=selection,
                            point_transform=point_transform,
                        )
                    )
        else:
            if successive:
                assert len(component_indexes) == 1
                if start == 0:
                    scan_data.append(
                        pyjpeg.HuffmanDCTDCSuccessiveScan(
                            data_units[component_indexes[0]],
                            point_transform=point_transform,
                        )
                    )
                else:
                    table = pyjpeg.huffman.make_huffman_table([1] * 256)
                    scan_data.append(
                        pyjpeg.HuffmanDCTACSuccessiveScan(
                            data_units[component_indexes[0]],
                            table,
                            spectral_selection=selection,
                            point_transform=point_transform,
                        )
                    )
            else:
                for interval in range(n_intervals):
                    huffman_components = []
                    mcu_data_units = []
                    for i in component_indexes:
                        # MCU is 1 in non-interleaved
                        if len(component_indexes) == 1:
                            sampling_factor = (1, 1)
                        else:
                            _, sampling_factor = components[i]
                        interval_length = len(data_units[i]) // n_intervals
                        interval_start = interval * interval_length
                        mcu_data_units.append(
                            pyjpeg.dct.order_mcu_dct_data_units(
                                component_sizes[i][0],
                                component_sizes[i][1] // n_intervals,
                                data_units[i][
                                    interval_start : interval_start + interval_length
                                ],
                                sampling_factor,
                            )
                        )
                        if precision > 8:
                            dc_table = pyjpeg.huffman.make_huffman_table([1] * 256)
                            ac_table = pyjpeg.huffman.make_huffman_table([1] * 256)
                        elif i == 0 or not use_chrominance:
                            dc_table = pyjpeg.standard_luminance_dc_huffman_table
                            ac_table = pyjpeg.standard_luminance_ac_huffman_table
                        else:
                            dc_table = pyjpeg.standard_chrominance_dc_huffman_table
                            ac_table = pyjpeg.standard_chrominance_ac_huffman_table
                        huffman_components.append(
                            pyjpeg.HuffmanDCTScanComponent(
                                sampling_factor=sampling_factor,
                                dc_table=dc_table,
                                ac_table=ac_table,
                            )
                        )
                    if interval != 0:
                        scan_data.append(pyjpeg.Restart((interval - 1) % 8))
                    # FIXME: Interleave earlier
                    data_units_ = []
                    while len(mcu_data_units[0]) > 0:
                        for i, huffman_scan_component in enumerate(huffman_components):
                            for _ in range(
                                huffman_scan_component.sampling_factor[0]
                                * huffman_scan_component.sampling_factor[1]
                            ):
                                data_units_.append(mcu_data_units[i].pop(0))
                    scan_data.append(
                        pyjpeg.HuffmanDCTScan(
                            data_units_,
                            components=huffman_components,
                            spectral_selection=selection,
                            point_transform=point_transform,
                        )
                    )
        jpeg_scans.append((sos, scan_data))

    segments: list[pyjpeg.Segment] = [pyjpeg.StartOfImage()]
    if comments is not None:
        for comment in comments:
            segments.append(pyjpeg.Comment(comment))
    if jfif:
        segments.append(pyjpeg.JfifHeader())
    if exif:
        segments.append(pyjpeg.ExifHeader(MINIMAL_EXIF_DATA))
    if adobe:
        segments.append(pyjpeg.AdobeHeader(color_space=color_space))
    segments.append(pyjpeg.DefineQuantizationTables(quantization_tables))
    if use_dnl:
        number_of_lines = 0
    else:
        number_of_lines = height
    if extended:
        segments.append(
            pyjpeg.StartOfFrame.extended(
                number_of_lines,
                width,
                sof_components,
                precision=precision,
                arithmetic=arithmetic,
            )
        )
    elif progressive:
        segments.append(
            pyjpeg.StartOfFrame.progressive(
                number_of_lines,
                width,
                sof_components,
                precision=precision,
                arithmetic=arithmetic,
            )
        )
    else:
        segments.append(
            pyjpeg.StartOfFrame.baseline(number_of_lines, width, sof_components)
        )
    if arithmetic:
        conditioning = []
        for i, bounds in enumerate(arithmetic_conditioning_bounds):
            if bounds != (0, 1):
                conditioning.append(pyjpeg.ArithmeticConditioning.dc(i, bounds))
        for i, kx in enumerate(arithmetic_conditioning_kx):
            if kx != 5:
                conditioning.append(pyjpeg.ArithmeticConditioning.ac(i, kx))
        if len(conditioning) > 0:
            segments.append(pyjpeg.DefineArithmeticConditioning(conditioning))
    else:
        tables = [
            pyjpeg.HuffmanTable.dc(0, pyjpeg.standard_luminance_dc_huffman_table),
            pyjpeg.HuffmanTable.ac(0, pyjpeg.standard_luminance_ac_huffman_table),
        ]
        if use_chrominance:
            tables.append(
                pyjpeg.HuffmanTable.dc(1, pyjpeg.standard_chrominance_dc_huffman_table)
            )
            tables.append(
                pyjpeg.HuffmanTable.ac(1, pyjpeg.standard_chrominance_ac_huffman_table)
            )
        segments.append(pyjpeg.DefineHuffmanTables(tables))
    if restart_interval != 0:
        segments.append(pyjpeg.DefineRestartInterval(restart_interval))
    for i, (sos, scan_data) in enumerate(jpeg_scans):
        segments.append(sos)
        segments.extend(scan_data)
        if i == 0 and use_dnl:
            segments.append(pyjpeg.DefineNumberOfLines(height))
    segments.append(pyjpeg.EndOfImage())
    segments = pyjpeg.huffman_optimize(segments)
    write_jpeg(segments, section, width, height, precision, name, description)


def generate_lossless(
    section: str,
    name: str,
    description: str,
    width: int,
    height: int,
    component_samples: list[list[int]],
    scans: list[list[int]],
    precision: int = 8,
    use_dnl: bool = False,
    color_space: int = pyjpeg.AdobeColorSpace.Y_CB_CR,
    jfif: bool = True,
    adobe: bool = False,
    predictor: int = 1,
    restart_interval: int = 0,
    arithmetic: bool = False,
) -> None:
    conditioning_bounds = (0, 1)
    segments: list[pyjpeg.Segment] = [pyjpeg.StartOfImage()]
    if jfif:
        segments.append(pyjpeg.JfifHeader())
    if adobe:
        segments.append(pyjpeg.AdobeHeader(color_space=color_space))
    if use_dnl:
        number_of_lines = 0
    else:
        number_of_lines = height
    sof_components = []
    for i in range(len(component_samples)):
        sof_components.append(pyjpeg.FrameComponent.lossless(i + 1))
    segments.append(
        pyjpeg.StartOfFrame.lossless(
            number_of_lines,
            width,
            sof_components,
            precision=precision,
            arithmetic=arithmetic,
        )
    )
    huffman_table: list[list[int]] = [[] * 255]
    if not arithmetic:
        # Need large table to handle all bit depths
        huffman_table = pyjpeg.huffman.make_huffman_table([1] * 256)
        tables = []
        for i in range(len(component_samples)):
            tables.append(
                pyjpeg.HuffmanTable.dc(
                    i,
                    huffman_table,
                )
            )
        segments.append(pyjpeg.DefineHuffmanTables(tables))
    if restart_interval != 0:
        segments.append(pyjpeg.DefineRestartInterval(restart_interval))
    all_scan_components = []
    for i, samples in enumerate(component_samples):
        if arithmetic:
            table = 0
        else:
            table = i
        all_scan_components.append(pyjpeg.ScanComponent.lossless(i + 1, table=table))
    for scan_index, component_indexes in enumerate(scans):
        sos_components = []
        arithmetic_scan_components = []
        huffman_scan_components = []
        for c in component_indexes:
            sos_components.append(all_scan_components[c])
            if arithmetic:
                arithmetic_scan_components.append(
                    pyjpeg.ArithmeticLosslessScanComponent(
                        conditioning_bounds=conditioning_bounds
                    )
                )
            else:
                huffman_scan_components.append(
                    pyjpeg.HuffmanLosslessScanComponent(huffman_table)
                )
        segments.append(
            pyjpeg.StartOfScan.lossless(
                components=sos_components,
                predictor=predictor,
            )
        )
        n_samples = width * height
        if restart_interval == 0:
            segment_length = n_samples
        else:
            segment_length = restart_interval
        for offset in range(0, n_samples, segment_length):
            # Interleave
            samples = []
            for i in range(segment_length):
                for c in component_indexes:
                    samples.append(component_samples[c][offset + i])
            if offset != 0:
                index = (offset // segment_length) - 1
                segments.append(pyjpeg.Restart(index % 8))
            if arithmetic:
                segments.append(
                    pyjpeg.ArithmeticLosslessScan(
                        width,
                        samples,
                        arithmetic_scan_components,
                        precision=precision,
                        predictor=predictor,
                    )
                )
            else:
                segments.append(
                    pyjpeg.HuffmanLosslessScan(
                        width,
                        samples,
                        huffman_scan_components,
                        precision=precision,
                        predictor=predictor,
                    )
                )
            if offset == 0 and scan_index == 0 and use_dnl:
                segments.append(pyjpeg.DefineNumberOfLines(height))
    segments.append(pyjpeg.EndOfImage())
    segments = pyjpeg.huffman_optimize(segments)
    write_jpeg(segments, section, width, height, precision, name, description)


def generate_ls(
    section: str,
    name: str,
    description: str,
    width: int,
    height: int,
    component_samples: list[list[int]],
    scans: list[tuple[int, int, list[int]]],
    precision: int = 8,
    use_dnl: bool = False,
    number_of_lines_number_of_bytes: int = 2,
    color_space: int = pyjpeg.AdobeColorSpace.Y_CB_CR,
    jfif: bool = True,
    adobe: bool = False,
    restart_interval: int = 0,
    restart_interval_number_of_bytes: int = 2,
    mapping_tables: list[tuple[int, int, bytes]] | None = None,
    maxval: int = 0,
    gradient_thresholds: tuple[int, int, int] = (0, 0, 0),
    reset: int = 0,
    always_parameters: bool = False,
    use_oversize_image_dimensions: bool = False,
    oversize_image_dimensions_number_of_bytes: int = 2,
) -> None:
    if mapping_tables is None:
        mapping_tables = []
    segments: list[pyjpeg.Segment] = [pyjpeg.StartOfImage()]
    if jfif:
        segments.append(pyjpeg.JfifHeader())
    if adobe:
        segments.append(pyjpeg.AdobeHeader(color_space=color_space))
    if use_dnl or use_oversize_image_dimensions:
        number_of_lines = 0
    else:
        number_of_lines = height
    if use_oversize_image_dimensions:
        samples_per_line = 0
    else:
        samples_per_line = width
    sof_components = []
    for i in range(len(component_samples)):
        sof_components.append(pyjpeg.FrameComponent.lossless(i + 1))
    segments.append(
        pyjpeg.StartOfFrame.ls(
            number_of_lines, samples_per_line, sof_components, precision=precision
        )
    )
    if use_oversize_image_dimensions:
        segments.append(
            pyjpeg.LSOversizeImageDimensions(
                width,
                height,
                number_of_bytes=oversize_image_dimensions_number_of_bytes,
            )
        )
    if (
        maxval != 0
        or gradient_thresholds != (0, 0, 0)
        or reset != 0
        or always_parameters
    ):
        segments.append(
            pyjpeg.LSCodingParameters(
                maxval=maxval, gradient_thresholds=gradient_thresholds, reset=reset
            )
        )
    for table_id, weight, table in mapping_tables:
        # FIXME: Support table continuation
        segments.append(pyjpeg.LSMappingTable(table_id, table, weight=weight))
    if restart_interval != 0:
        segments.append(
            pyjpeg.DefineRestartInterval(
                restart_interval, number_of_bytes=restart_interval_number_of_bytes
            )
        )
    all_scan_components = []
    for i, samples in enumerate(component_samples):
        if len(mapping_tables) > 0:
            mapping_table = i + 1
        else:
            mapping_table = 0
        all_scan_components.append(
            pyjpeg.ScanComponent.ls(i + 1, mapping_table=mapping_table)
        )
    for scan_index, (difference_bound, interleave_mode, component_indexes) in enumerate(
        scans
    ):
        sos_components = []
        scan_components = []
        for c in component_indexes:
            sos_components.append(all_scan_components[c])
            scan_components.append(pyjpeg.LSScanComponent())
        segments.append(
            pyjpeg.StartOfScan.ls(
                components=sos_components,
                difference_bound=difference_bound,
                interleave_mode=interleave_mode,
            )
        )
        n_samples = width * height
        if restart_interval == 0:
            segment_length = n_samples
        else:
            segment_length = restart_interval
        for offset in range(0, n_samples, segment_length):
            # Interleave
            samples = []
            for i in range(segment_length):
                for c in component_indexes:
                    samples.append(component_samples[c][offset + i])
            if offset != 0:
                index = (offset // segment_length) - 1
                segments.append(pyjpeg.Restart(index % 8))
            if maxval != 0:
                scan_maxval = maxval
            else:
                scan_maxval = (1 << precision) - 1
            segments.append(
                pyjpeg.LSScan(
                    width,
                    samples,
                    scan_components,
                    interleave_mode=interleave_mode,
                    difference_bound=difference_bound,
                    maxval=scan_maxval,
                    gradient_thresholds=gradient_thresholds,
                    reset=reset,
                )
            )
            if offset == 0 and scan_index == 0 and use_dnl:
                segments.append(
                    pyjpeg.DefineNumberOfLines(
                        height, number_of_bytes=number_of_lines_number_of_bytes
                    )
                )
    segments.append(pyjpeg.EndOfImage())
    write_jpeg(segments, section, width, height, precision, name, description)


def write_jpeg(
    segments: list[pyjpeg.Segment],
    section: str,
    width: int,
    height: int,
    precision: int,
    name: str,
    description: str,
) -> None:
    writer = pyjpeg.BufferedWriter()
    for segment in segments:
        segment.write(writer)
    basename = f"../jpeg/{section}/{width}x{height}x{precision}_{name}"
    with open(basename + ".jpg", "wb") as f:
        f.write(writer.data)
    j = {
        "description": description,
        "width": width,
        "height": height,
        "segments": segments_to_json(segments),
    }
    with open(basename + ".json", "w") as f:
        f.write(json.dumps(j, indent=2))


for mode, encoding in [
    ("baseline", "huffman"),
    ("extended", "huffman"),
    ("extended", "arithmetic"),
    ("progressive", "huffman"),
    ("progressive", "arithmetic"),
]:
    extended = mode == "extended"
    progressive = mode == "progressive"
    arithmetic = encoding == "arithmetic"
    if mode != "baseline":
        section = f"{mode}_{encoding}"
    else:
        section = "baseline"
    if not progressive:
        dct_one_channel_scans = [([0], 0, 63, 0)]
        three_channel_scans = [([0], 0, 63, 0), ([1], 0, 63, 0), ([2], 0, 63, 0)]
        three_channel_interleaved_scans = [([0, 1, 2], 0, 63, 0)]
        four_channel_scans = [
            ([0], 0, 63, 0),
            ([1], 0, 63, 0),
            ([2], 0, 63, 0),
            ([3], 0, 63, 0),
        ]
        four_channel_interleaved_scans = [([0, 1, 2, 3], 0, 63, 0)]
    else:
        dct_one_channel_scans = [([0], 0, 0, 0), ([0], 1, 63, 0)]
        three_channel_scans = [
            ([0], 0, 0, 0),
            ([1], 0, 0, 0),
            ([2], 0, 0, 0),
            ([0], 1, 63, 0),
            ([1], 1, 63, 0),
            ([2], 1, 63, 0),
        ]
        three_channel_interleaved_scans = [
            ([0, 1, 2], 0, 0, 0),
            ([0], 1, 63, 0),
            ([1], 1, 63, 0),
            ([2], 1, 63, 0),
        ]
        four_channel_scans = [
            ([0], 0, 0, 0),
            ([1], 0, 0, 0),
            ([2], 0, 0, 0),
            ([3], 0, 0, 0),
            ([0], 1, 63, 0),
            ([1], 1, 63, 0),
            ([2], 1, 63, 0),
            ([3], 1, 63, 0),
        ]
        four_channel_interleaved_scans = [
            ([0, 1, 2, 3], 0, 0, 0),
            ([0], 1, 63, 0),
            ([1], 1, 63, 0),
            ([2], 1, 63, 0),
            ([3], 1, 63, 0),
        ]
    generate_dct(
        section,
        "grayscale",
        "DCT encoded grayscale image.",
        WIDTH,
        HEIGHT,
        grayscale_components8,
        scans=dct_one_channel_scans,
        extended=extended,
        progressive=progressive,
        arithmetic=arithmetic,
    )
    generate_dct(
        section,
        "grayscale_quantization",
        "DCT encoded grayscale image encoded using standard JPEG quantization values.",
        WIDTH,
        HEIGHT,
        grayscale_components8,
        luminance_quantization_table=pyjpeg.standard_luminance_quantization_table,
        scans=dct_one_channel_scans,
        extended=extended,
        progressive=progressive,
        arithmetic=arithmetic,
    )
    generate_dct(
        section,
        "y_cb_cr",
        "DCT encoded color image encoded in YCbCr format in three scans.",
        WIDTH,
        HEIGHT,
        ycbcr_components8,
        scans=three_channel_scans,
        extended=extended,
        progressive=progressive,
        arithmetic=arithmetic,
    )
    generate_dct(
        section,
        "cr_cb_y",
        "DCT encoded color image encoded in YCbCr format in three scans, but scan order is in reverse.",
        WIDTH,
        HEIGHT,
        ycbcr_components8,
        scans=[([2], 0, 63, 0), ([1], 0, 63, 0), ([0], 0, 63, 0)],
        extended=extended,
        progressive=progressive,
        arithmetic=arithmetic,
    )
    generate_dct(
        section,
        "y_cb_cr_quantization",
        "DCT encoded color image encoded in YCbCr format in three scans using standard JPEG quantization values.",
        WIDTH,
        HEIGHT,
        ycbcr_components8,
        luminance_quantization_table=pyjpeg.standard_luminance_quantization_table,
        chrominance_quantization_table=pyjpeg.standard_chrominance_quantization_table,
        scans=three_channel_scans,
        extended=extended,
        progressive=progressive,
        arithmetic=arithmetic,
    )
    generate_dct(
        section,
        "ycbcr",
        "DCT encoded color image encoded in YCbCr format in one interleaved scan.",
        WIDTH,
        HEIGHT,
        ycbcr_components8,
        scans=three_channel_interleaved_scans,
        extended=extended,
        progressive=progressive,
        arithmetic=arithmetic,
    )
    generate_dct(
        section,
        "crcby",
        "DCT encoded color image encoded in YCbCr format in one interleaved scan with components in reverse order.",
        WIDTH,
        HEIGHT,
        ycbcr_components8,
        scans=[([2, 1, 0], 0, 63, 0)],
        extended=extended,
        progressive=progressive,
        arithmetic=arithmetic,
    )
    # FIXME: Greyscale sampling
    generate_dct(
        section,
        "y_cb_cr_2x2_1x1_1x1",
        "DCT encoded color image encoded in YCbCr format in three scans. The color channels are at half resolution.",
        WIDTH,
        HEIGHT,
        [
            (ycbcr_samples8[0], (2, 2)),
            (ycbcr_samples8[1], (1, 1)),
            (ycbcr_samples8[2], (1, 1)),
        ],
        scans=three_channel_scans,
        extended=extended,
        progressive=progressive,
        arithmetic=arithmetic,
    )
    generate_dct(
        section,
        "ycbcr_2x2_1x1_1x1",
        "DCT encoded color image encoded in YCbCr format in one interleaved scan. The color channels are at half resolution.",
        WIDTH,
        HEIGHT,
        [
            (ycbcr_samples8[0], (2, 2)),
            (ycbcr_samples8[1], (1, 1)),
            (ycbcr_samples8[2], (1, 1)),
        ],
        scans=three_channel_interleaved_scans,
        extended=extended,
        progressive=progressive,
        arithmetic=arithmetic,
    )
    generate_dct(
        section,
        "y_cb_cr_2x2_2x1_1x2",
        "DCT encoded color image encoded in YCbCr format in three scans. The Cb channel half vertical resolution. The Cr channel half horizontal resolution.",
        WIDTH,
        HEIGHT,
        [
            (ycbcr_samples8[0], (2, 2)),
            (ycbcr_samples8[1], (2, 1)),
            (ycbcr_samples8[2], (1, 2)),
        ],
        scans=three_channel_scans,
        extended=extended,
        progressive=progressive,
        arithmetic=arithmetic,
    )
    generate_dct(
        section,
        "ycbcr_2x2_2x1_1x2",
        "DCT encoded color image encoded in YCbCr format in one interleaved scan. The Cb channel half vertical resolution. The Cr channel half horizontal resolution.",
        WIDTH,
        HEIGHT,
        [
            (ycbcr_samples8[0], (2, 2)),
            (ycbcr_samples8[1], (2, 1)),
            (ycbcr_samples8[2], (1, 2)),
        ],
        scans=three_channel_interleaved_scans,
        extended=extended,
        progressive=progressive,
        arithmetic=arithmetic,
    )
    generate_dct(
        section,
        "grayscale_zero_coefficients",
        "DCT encoded image where all coefficients are zeros",
        8,
        8,
        [([128] * 64, (1, 1))],
        scans=dct_one_channel_scans,
        extended=extended,
        progressive=progressive,
        arithmetic=arithmetic,
    )
    generate_dct(
        section,
        "grayscale_black",
        "DCT encoded single channel image with all black pixels.",
        8,
        8,
        [(make_solid(8, 8, 0), (1, 1))],
        scans=dct_one_channel_scans,
        extended=extended,
        progressive=progressive,
        arithmetic=arithmetic,
    )
    generate_dct(
        section,
        "grayscale_white",
        "DCT encoded single channel image with all white pixels.",
        8,
        8,
        [(make_solid(8, 8, 255), (1, 1))],
        scans=dct_one_channel_scans,
        extended=extended,
        progressive=progressive,
        arithmetic=arithmetic,
    )
    generate_dct(
        section,
        "grayscale_gray",
        "DCT encoded single channel image with all mid-level gray pixels.",
        8,
        8,
        [(make_solid(8, 8, 127), (1, 1))],
        scans=dct_one_channel_scans,
        extended=extended,
        progressive=progressive,
        arithmetic=arithmetic,
    )
    generate_dct(
        section,
        "grayscale_check",
        "DCT encoded single channel image with in a checkered format.",
        8,
        8,
        [(make_check(8, 8, 255), (1, 1))],
        scans=dct_one_channel_scans,
        extended=extended,
        progressive=progressive,
        arithmetic=arithmetic,
    )
    for size in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16):
        (width, height, _, channels, samples) = read_pnm(
            f"data/{size}x{size}x8_grayscale.pgm"
        )
        assert width == height == size
        assert channels == 1
        generate_dct(
            section,
            "grayscale",
            "DCT encoded grayscale image.",
            width,
            height,
            [(samples, (1, 1))],
            scans=dct_one_channel_scans,
            extended=extended,
            progressive=progressive,
            arithmetic=arithmetic,
        )
    generate_dct(
        section,
        "comment",
        "DCT encoded grayscale image with a comment tag.",
        WIDTH,
        HEIGHT,
        grayscale_components8,
        scans=dct_one_channel_scans,
        comments=[b"Hello World"],
        extended=extended,
        progressive=progressive,
        arithmetic=arithmetic,
    )
    generate_dct(
        section,
        "comments",
        "DCT encoded grayscale image with mutiple comment tags.",
        WIDTH,
        HEIGHT,
        grayscale_components8,
        scans=dct_one_channel_scans,
        comments=[b"Hello", b"World"],
        extended=extended,
        progressive=progressive,
        arithmetic=arithmetic,
    )
    generate_dct(
        section,
        "r_g_b",
        "DCT encoded color image encoded in RGB format in three scans.",
        WIDTH,
        HEIGHT,
        rgb_components8,
        scans=three_channel_scans,
        color_space=pyjpeg.AdobeColorSpace.RGB_OR_CMYK,
        jfif=False,
        adobe=True,
        extended=extended,
        progressive=progressive,
        arithmetic=arithmetic,
    )
    generate_dct(
        section,
        "b_g_r",
        "DCT encoded color image encoded in RGB format in three scans. Channels are in reverse order.",
        WIDTH,
        HEIGHT,
        rgb_components8,
        scans=[([2], 0, 63, 0), ([1], 0, 63, 0), ([0], 0, 63, 0)],
        color_space=pyjpeg.AdobeColorSpace.RGB_OR_CMYK,
        jfif=False,
        adobe=True,
        extended=extended,
        progressive=progressive,
        arithmetic=arithmetic,
    )
    generate_dct(
        section,
        "rgb",
        "DCT encoded color image encoded in RGB format in one interleaved scan.",
        WIDTH,
        HEIGHT,
        rgb_components8,
        scans=three_channel_interleaved_scans,
        color_space=pyjpeg.AdobeColorSpace.RGB_OR_CMYK,
        jfif=False,
        adobe=True,
        extended=extended,
        progressive=progressive,
        arithmetic=arithmetic,
    )
    generate_dct(
        section,
        "bgr",
        "DCT encoded color image encoded in RGB format in one interleaved scan. Channels are in reverse order.",
        WIDTH,
        HEIGHT,
        rgb_components8,
        scans=[([2, 1, 0], 0, 63, 0)],
        color_space=pyjpeg.AdobeColorSpace.RGB_OR_CMYK,
        jfif=False,
        adobe=True,
        extended=extended,
        progressive=progressive,
        arithmetic=arithmetic,
    )
    generate_dct(
        section,
        "c_m_y_k",
        "DCT encoded color image encoded in CMYK format in four scans.",
        WIDTH,
        HEIGHT,
        cmyk_components8,
        scans=four_channel_scans,
        color_space=pyjpeg.AdobeColorSpace.RGB_OR_CMYK,
        jfif=False,
        adobe=True,
        extended=extended,
        progressive=progressive,
        arithmetic=arithmetic,
    )
    generate_dct(
        section,
        "cmyk",
        "DCT encoded color image encoded in CMYK format in one interleaved scan.",
        WIDTH,
        HEIGHT,
        cmyk_components8,
        scans=four_channel_interleaved_scans,
        color_space=pyjpeg.AdobeColorSpace.RGB_OR_CMYK,
        jfif=False,
        adobe=True,
        extended=extended,
        progressive=progressive,
        arithmetic=arithmetic,
    )
    generate_dct(
        section,
        "dnl",
        "DCT encoded grayscale image without the height defined after the scan data in a DNL segment.",
        WIDTH,
        HEIGHT,
        grayscale_components8,
        scans=dct_one_channel_scans,
        use_dnl=True,
        extended=extended,
        progressive=progressive,
        arithmetic=arithmetic,
    )
    generate_dct(
        section,
        "restarts",
        "DCT encoded grayscale image with restart markers.",
        WIDTH,
        HEIGHT,
        grayscale_components8,
        scans=dct_one_channel_scans,
        restart_interval=4,
        extended=extended,
        progressive=progressive,
        arithmetic=arithmetic,
    )
    generate_dct(
        section,
        "exif",
        "DCT encoded grayscale image with Exif header instead of JFIF.",
        WIDTH,
        HEIGHT,
        grayscale_components8,
        scans=dct_one_channel_scans,
        jfif=False,
        exif=True,
        extended=extended,
        progressive=progressive,
        arithmetic=arithmetic,
    )

    if arithmetic:
        generate_dct(
            section,
            "conditioning_bounds_4_6",
            "DCT encoded grayscale image using non-standard Arithmetic conditioning bounds.",
            WIDTH,
            HEIGHT,
            grayscale_components8,
            scans=dct_one_channel_scans,
            extended=extended,
            progressive=progressive,
            arithmetic=True,
            arithmetic_conditioning_bounds=((4, 6), (4, 6), (4, 6), (4, 6)),
        )

        generate_dct(
            section,
            "conditioning_kx_6",
            "DCT encoded grayscale image using non-standard Arithmetic Kx values.",
            WIDTH,
            HEIGHT,
            grayscale_components8,
            scans=dct_one_channel_scans,
            extended=extended,
            progressive=progressive,
            arithmetic=True,
            arithmetic_conditioning_kx=(6, 6, 6, 6),
        )

    if mode != "baseline":
        generate_dct(
            section,
            "grayscale",
            "DCT encoded grayscale image with 12 bit samples.",
            WIDTH,
            HEIGHT,
            grayscale_components12,
            scans=dct_one_channel_scans,
            precision=12,
            extended=extended,
            progressive=progressive,
            arithmetic=arithmetic,
        )
        generate_dct(
            section,
            "y_cb_cr",
            "DCT encoded color image with 12 bit samples encoded with three scans.",
            WIDTH,
            HEIGHT,
            ycbcr_components12,
            scans=three_channel_scans,
            precision=12,
            extended=extended,
            progressive=progressive,
            arithmetic=arithmetic,
        )
        generate_dct(
            section,
            "ycbcr",
            "DCT encoded color image with 12 bit samples encded with one interleaved scan.",
            WIDTH,
            HEIGHT,
            ycbcr_components12,
            scans=three_channel_interleaved_scans,
            precision=12,
            extended=extended,
            progressive=progressive,
            arithmetic=arithmetic,
        )
        generate_dct(
            section,
            "grayscale_black",
            "DCT encoded grayscale image with 12 bit samples, all pixels are black.",
            8,
            8,
            [(make_solid(8, 8, 0), (1, 1))],
            scans=dct_one_channel_scans,
            precision=12,
            extended=extended,
            progressive=progressive,
            arithmetic=arithmetic,
        )
        generate_dct(
            section,
            "grayscale_white",
            "DCT encoded grayscale image with 12 bit samples, all pixels are white.",
            8,
            8,
            [(make_solid(8, 8, 4095), (1, 1))],
            scans=dct_one_channel_scans,
            precision=12,
            extended=extended,
            progressive=progressive,
            arithmetic=arithmetic,
        )
        generate_dct(
            section,
            "grayscale_gray",
            "DCT encoded grayscale image with 12 bit samples, all pixels are mid-level gray.",
            8,
            8,
            [(make_solid(8, 8, 2047), (1, 1))],
            scans=dct_one_channel_scans,
            precision=12,
            extended=extended,
            progressive=progressive,
            arithmetic=arithmetic,
        )
        generate_dct(
            section,
            "grayscale_check",
            "DCT encoded grayscale image with 12 bit samples, checkered pattern.",
            8,
            8,
            [(make_check(8, 8, 4095), (1, 1))],
            scans=dct_one_channel_scans,
            precision=12,
            extended=extended,
            progressive=progressive,
            arithmetic=arithmetic,
        )

    if mode == "progressive":
        all_selection = [([0], 0, 0, 0)]
        all_reverse_selection = [([0], 0, 0, 0)]
        for i in range(1, 64):
            all_selection.append(([0], i, i, 0))
            all_reverse_selection.append(([0], 64 - i, 64 - i, 0))
        generate_dct(
            section,
            "grayscale_spectral_all",
            "DCT encoded grayscale image with scans for each coefficient.",
            WIDTH,
            HEIGHT,
            grayscale_components8,
            scans=all_selection,
            extended=extended,
            progressive=progressive,
            arithmetic=arithmetic,
        )
        generate_dct(
            section,
            "grayscale_spectral_all_reverse",
            "DCT encoded grayscale image with scans for each coefficient. The coefficients are encoded in reverse order.",
            WIDTH,
            HEIGHT,
            grayscale_components8,
            scans=all_reverse_selection,
            extended=extended,
            progressive=progressive,
            arithmetic=arithmetic,
        )
        generate_dct(
            section,
            "grayscale_successive_dc",
            "DCT encoded grayscale image where DC bits are sent in multiple scans.",
            WIDTH,
            HEIGHT,
            grayscale_components8,
            scans=[
                ([0], 0, 0, 4),
                ([0], 0, 0, 3),
                ([0], 0, 0, 2),
                ([0], 0, 0, 1),
                ([0], 0, 0, 0),
                ([0], 1, 63, 0),
            ],
            extended=extended,
            progressive=progressive,
            arithmetic=arithmetic,
        )
        generate_dct(
            section,
            "grayscale_successive_ac",
            "DCT encoded grayscale image where AC bits are sent in multiple scans.",
            WIDTH,
            HEIGHT,
            grayscale_components8,
            scans=[
                ([0], 0, 0, 0),
                ([0], 1, 63, 4),
                ([0], 1, 63, 3),
                ([0], 1, 63, 2),
                ([0], 1, 63, 1),
                ([0], 1, 63, 0),
            ],
            extended=extended,
            progressive=progressive,
            arithmetic=arithmetic,
        )
        generate_dct(
            section,
            "grayscale_successive",
            "DCT encoded grayscale image where both DC and AC bits are sent in multiple scans.",
            WIDTH,
            HEIGHT,
            grayscale_components8,
            scans=[
                ([0], 0, 0, 4),
                ([0], 0, 0, 3),
                ([0], 0, 0, 2),
                ([0], 0, 0, 1),
                ([0], 0, 0, 0),
                ([0], 1, 63, 4),
                ([0], 1, 63, 3),
                ([0], 1, 63, 2),
                ([0], 1, 63, 1),
                ([0], 1, 63, 0),
            ],
            extended=extended,
            progressive=progressive,
            arithmetic=arithmetic,
        )
        # FIXME: successive 3, 2, 1
        # FIXME: successive with restarts

for encoding in ["huffman", "arithmetic"]:
    arithmetic = encoding == "arithmetic"
    section = f"lossless_{encoding}"
    for predictor in range(1, 8):
        generate_lossless(
            section,
            f"grayscale_predictor{predictor}",
            f"Lossless encoded grayscale image using predictor #{predictor}.",
            WIDTH,
            HEIGHT,
            [grayscale_samples8],
            scans=[[0]],
            predictor=predictor,
            arithmetic=arithmetic,
        )
    for precision in range(2, 17):
        generate_lossless(
            section,
            "grayscale",
            f"Lossless encoded grayscale image using {precision} bits of precision.",
            WIDTH,
            HEIGHT,
            [make_grayscale(precision)],
            scans=[[0]],
            precision=precision,
            predictor=1,
            arithmetic=arithmetic,
        )
    for size in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16):
        (width, height, _, channels, samples) = read_pnm(
            f"data/{size}x{size}x8_grayscale.pgm"
        )
        assert width == height == size
        assert channels == 1
        generate_lossless(
            section,
            "grayscale",
            f"Lossless encoded grayscale image of size {width}x{height} pixels.",
            width,
            height,
            [samples],
            scans=[[0]],
            precision=8,
            predictor=1,
            arithmetic=arithmetic,
        )
    generate_lossless(
        section,
        "y_cb_cr",
        "Lossless encoded color image encoded in YCbCr format in three scans.",
        WIDTH,
        HEIGHT,
        ycbcr_samples8,
        scans=[[0], [1], [2]],
        predictor=1,
        arithmetic=arithmetic,
    )
    generate_lossless(
        section,
        "ycbcr",
        "Lossless encoded color image encoded in YCbCr format in one interleaved scan.",
        WIDTH,
        HEIGHT,
        ycbcr_samples8,
        scans=[[0, 1, 2]],
        predictor=1,
        arithmetic=arithmetic,
    )
    generate_lossless(
        section,
        "r_g_b",
        "Lossless encoded color image encoded in RGB format in three scans.",
        WIDTH,
        HEIGHT,
        rgb_samples8,
        scans=[[0], [1], [2]],
        color_space=pyjpeg.AdobeColorSpace.RGB_OR_CMYK,
        jfif=False,
        adobe=True,
        predictor=1,
        arithmetic=arithmetic,
    )
    generate_lossless(
        section,
        "rgb",
        "Lossless encoded color image encoded in RGB format in one interleaved scan.",
        WIDTH,
        HEIGHT,
        rgb_samples8,
        scans=[[0, 1, 2]],
        color_space=pyjpeg.AdobeColorSpace.RGB_OR_CMYK,
        jfif=False,
        adobe=True,
        predictor=1,
        arithmetic=arithmetic,
    )
    generate_lossless(
        section,
        "restarts",
        "Lossless encoded grayscale image with restart markers.",
        WIDTH,
        HEIGHT,
        [grayscale_samples8],
        scans=[[0]],
        predictor=1,
        restart_interval=32 * 8,
        arithmetic=arithmetic,
    )
    generate_lossless(
        section,
        "dnl",
        "Lossless encoded grayscale image without the height defined after the scan data in a DNL segment.",
        WIDTH,
        HEIGHT,
        [grayscale_samples8],
        scans=[[0]],
        use_dnl=True,
        predictor=1,
        arithmetic=arithmetic,
    )

section = "ls"
ls_one_channel_scans = [(0, pyjpeg.LSInterleaveMode.NONE, [0])]
for precision in range(2, 17):
    generate_ls(
        section,
        "grayscale",
        f"JPEG-LS encoded grayscale image using {precision} bit precision.",
        WIDTH,
        HEIGHT,
        [make_grayscale(precision)],
        scans=ls_one_channel_scans,
        precision=precision,
    )
for size in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16):
    (width, height, _, channels, samples) = read_pnm(
        f"data/{size}x{size}x8_grayscale.pgm"
    )
    assert width == height == size
    assert channels == 1
    generate_ls(
        section,
        "grayscale",
        f"JPEG-LS encoded grayscale image of size {width}x{height} pixels.",
        width,
        height,
        [samples],
        scans=ls_one_channel_scans,
    )
grayscale_mapped_samples, mapping_table = make_mapped_samples([grayscale_samples8])
generate_ls(
    section,
    "grayscale_mapping_table",
    "JPEG-LS encoded grayscale image encoded using a mapping table.",
    WIDTH,
    HEIGHT,
    [grayscale_mapped_samples],
    mapping_tables=[(1, 1, mapping_table)],
    maxval=len(mapping_table) - 1,
    scans=ls_one_channel_scans,
)
generate_ls(
    section,
    "y_cb_cr",
    "JPEG-LS encoded color image encoded in YCbCr format in three scans.",
    WIDTH,
    HEIGHT,
    ycbcr_samples8,
    scans=[
        (0, pyjpeg.LSInterleaveMode.NONE, [0]),
        (0, pyjpeg.LSInterleaveMode.NONE, [1]),
        (0, pyjpeg.LSInterleaveMode.NONE, [2]),
    ],
)
generate_ls(
    section,
    "ycbcr_line_interleaved",
    "JPEG-LS encoded color image encoded in YCbCr format in one line interleaved scan.",
    WIDTH,
    HEIGHT,
    ycbcr_samples8,
    scans=[(0, pyjpeg.LSInterleaveMode.LINE, [0, 1, 2])],
)
generate_ls(
    section,
    "ycbcr_sample_interleaved",
    "JPEG-LS encoded color image encoded in YCbCr format in one sample interleaved scan.",
    WIDTH,
    HEIGHT,
    ycbcr_samples8,
    scans=[(0, pyjpeg.LSInterleaveMode.SAMPLE, [0, 1, 2])],
)
generate_ls(
    section,
    "r_g_b",
    "JPEG-LS encoded color image encoded in RGB format in three scans.",
    WIDTH,
    HEIGHT,
    rgb_samples8,
    scans=[
        (0, pyjpeg.LSInterleaveMode.NONE, [0]),
        (0, pyjpeg.LSInterleaveMode.NONE, [1]),
        (0, pyjpeg.LSInterleaveMode.NONE, [2]),
    ],
    color_space=pyjpeg.AdobeColorSpace.RGB_OR_CMYK,
    jfif=False,
    adobe=True,
)
rgb_mapped_samples, mapping_table = make_mapped_samples(rgb_samples8)
generate_ls(
    section,
    "rgb_mapping_table",
    "JPEG-LS encoded color image encoded in RGB format using a mapping table.",
    WIDTH,
    HEIGHT,
    [rgb_mapped_samples],
    mapping_tables=[(1, 3, mapping_table)],
    maxval=len(mapping_table) - 1,
    scans=ls_one_channel_scans,
)
generate_ls(
    section,
    "rgb_line_interleaved",
    "JPEG-LS encoded color image encoded in RGB format in one line interleaved scan.",
    WIDTH,
    HEIGHT,
    rgb_samples8,
    scans=[(0, pyjpeg.LSInterleaveMode.LINE, [0, 1, 2])],
    color_space=pyjpeg.AdobeColorSpace.RGB_OR_CMYK,
    jfif=False,
    adobe=True,
)
generate_ls(
    section,
    "rgb_sample_interleaved",
    "JPEG-LS encoded color image encoded in RGB format in one sample interleaved scan.",
    WIDTH,
    HEIGHT,
    rgb_samples8,
    scans=[(0, pyjpeg.LSInterleaveMode.SAMPLE, [0, 1, 2])],
    color_space=pyjpeg.AdobeColorSpace.RGB_OR_CMYK,
    jfif=False,
    adobe=True,
)
generate_ls(
    section,
    "oversize",
    "JPEG-LS encoded grayscale image encoded where the size is specificed in an oversize image segment.",
    WIDTH,
    HEIGHT,
    [grayscale_samples8],
    scans=ls_one_channel_scans,
    use_oversize_image_dimensions=True,
)
generate_ls(
    section,
    "oversize3",
    "JPEG-LS encoded grayscale image encoded where the size is specificed in an oversize image segment. Each dimension is encoded in 3 bytes.",
    WIDTH,
    HEIGHT,
    [grayscale_samples8],
    scans=ls_one_channel_scans,
    use_oversize_image_dimensions=True,
    oversize_image_dimensions_number_of_bytes=3,
)
generate_ls(
    section,
    "oversize4",
    "JPEG-LS encoded grayscale image encoded where the size is specificed in an oversize image segment. Each dimension is encoded in 4 bytes.",
    WIDTH,
    HEIGHT,
    [grayscale_samples8],
    scans=ls_one_channel_scans,
    use_oversize_image_dimensions=True,
    oversize_image_dimensions_number_of_bytes=4,
)
generate_ls(
    section,
    "restarts",
    "JPEG-LS encoded grayscale image with restart markers.",
    WIDTH,
    HEIGHT,
    [grayscale_samples8],
    scans=ls_one_channel_scans,
    restart_interval=32 * 8,
)
generate_ls(
    section,
    "restarts3",
    "JPEG-LS encoded grayscale image with restart markers. The restart interval is encoded with 3 bytes.",
    WIDTH,
    HEIGHT,
    [grayscale_samples8],
    scans=ls_one_channel_scans,
    restart_interval=32 * 8,
    restart_interval_number_of_bytes=3,
)
generate_ls(
    section,
    "restarts4",
    "JPEG-LS encoded grayscale image with restart markers. The restart interval is encoded with 4 bytes.",
    WIDTH,
    HEIGHT,
    [grayscale_samples8],
    scans=ls_one_channel_scans,
    restart_interval=32 * 8,
    restart_interval_number_of_bytes=4,
)
generate_ls(
    section,
    "dnl",
    "JPEG-LS encoded grayscale image without the height defined after the scan data in a DNL segment.",
    WIDTH,
    HEIGHT,
    [grayscale_samples8],
    scans=ls_one_channel_scans,
    use_dnl=True,
)
generate_ls(
    section,
    "dnl3",
    "JPEG-LS encoded grayscale image without the height defined after the scan data in a DNL segment. The height is encoded with 3 bytes.",
    WIDTH,
    HEIGHT,
    [grayscale_samples8],
    scans=ls_one_channel_scans,
    use_dnl=True,
    number_of_lines_number_of_bytes=3,
)
generate_ls(
    section,
    "dnl4",
    "JPEG-LS encoded grayscale image without the height defined after the scan data in a DNL segment. The height is encoded with 4 bytes.",
    WIDTH,
    HEIGHT,
    [grayscale_samples8],
    scans=ls_one_channel_scans,
    use_dnl=True,
    number_of_lines_number_of_bytes=4,
)
generate_ls(
    section,
    "empty_parameters",
    "JPEG-LS encoded grayscale image with encoding parameters specified with empty values. A decoder should replace these empty values with the JPEG-LS defaults.",
    WIDTH,
    HEIGHT,
    [grayscale_samples8],
    scans=ls_one_channel_scans,
    always_parameters=True,
)
generate_ls(
    section,
    "empty_maxval",
    "JPEG-LS encoded grayscale image with default encoding parameters specified except for maxval which is empty. A decoder should replace this empty values with the JPEG-LS default.",
    WIDTH,
    HEIGHT,
    [grayscale_samples8],
    scans=ls_one_channel_scans,
    maxval=0,
    gradient_thresholds=(3, 7, 21),
    reset=64,
)
generate_ls(
    section,
    "empty_t1",
    "JPEG-LS encoded grayscale image with default encoding parameters specified except for T1 which is empty. A decoder should replace this empty values with the JPEG-LS default.",
    WIDTH,
    HEIGHT,
    [grayscale_samples8],
    scans=ls_one_channel_scans,
    maxval=255,
    gradient_thresholds=(0, 7, 21),
    reset=64,
)
generate_ls(
    section,
    "empty_t2",
    "JPEG-LS encoded grayscale image with default encoding parameters specified except for T2 which is empty. A decoder should replace this empty values with the JPEG-LS default.",
    WIDTH,
    HEIGHT,
    [grayscale_samples8],
    scans=ls_one_channel_scans,
    maxval=255,
    gradient_thresholds=(3, 0, 21),
    reset=64,
)
generate_ls(
    section,
    "empty_t3",
    "JPEG-LS encoded grayscale image with default encoding parameters specified except for T3 which is empty. A decoder should replace this empty values with the JPEG-LS default.",
    WIDTH,
    HEIGHT,
    [grayscale_samples8],
    scans=ls_one_channel_scans,
    maxval=255,
    gradient_thresholds=(3, 7, 0),
    reset=64,
)
generate_ls(
    section,
    "empty_reset",
    "JPEG-LS encoded grayscale image with default encoding parameters specified except for reset which is empty. A decoder should replace this empty values with the JPEG-LS default.",
    WIDTH,
    HEIGHT,
    [grayscale_samples8],
    scans=ls_one_channel_scans,
    maxval=255,
    gradient_thresholds=(3, 7, 21),
    reset=0,
)
generate_ls(
    section,
    "default_parameters",
    "JPEG-LS encoded grayscale image with all encoding parameters specified but set to the JPEG-LS default values.",
    WIDTH,
    HEIGHT,
    [grayscale_samples8],
    scans=ls_one_channel_scans,
    maxval=255,
    gradient_thresholds=(3, 7, 21),
    reset=64,
)
generate_ls(
    section,
    "non_default_parameters",
    "JPEG-LS encoded grayscale image with all encoding parameters specified and set to values that are not the JPEG-LS defaults.",
    WIDTH,
    HEIGHT,
    [grayscale_samples8],
    scans=ls_one_channel_scans,
    maxval=255,
    gradient_thresholds=(4, 8, 22),
    reset=63,
)
for difference_bound in (1, 2, 3, 7, 20):
    generate_ls(
        section,
        f"near_lossless_{difference_bound}",
        f"JPEG-LS near-lossless encoded grayscale image with samples within {difference_bound} of original values.",
        WIDTH,
        HEIGHT,
        [grayscale_samples8],
        scans=[(difference_bound, pyjpeg.LSInterleaveMode.NONE, [0])],
    )

# 3 channel, red, green, blue, white, mixed color
# version 1.1
# density
# thumbnail
# multiple huffman tables
# arithmetic properties
# Large images (black to compress well)
# Generate with both JFIF and Adobe headers
