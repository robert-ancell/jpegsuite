#!/usr/bin/env python3

import math
import jpeg
import jpeg_encoder
import huffman
from pgm import *
from quantization_tables import *
from jpeg_segments import *


WIDTH = 32
HEIGHT = 32


def rgb_to_ycbcr(r, g, b, precision):
    offset = 1 << (precision - 1)
    y = round(0.299 * r + 0.587 * g + 0.114 * b)
    cb = round(-0.1687 * r - 0.3313 * g + 0.5 * b + offset)
    cr = round(0.5 * r - 0.4187 * g - 0.0813 * b + offset)
    return (y, cb, cr)


def rgb_to_cmyk(r, g, b, precision):
    max_value = 1 << (precision - 1)
    rf = r / max_value
    gf = g / max_value
    bf = b / max_value
    k = 1 - max(rf, gf, bf)
    if k == 1:
        c, m, y = 0, 0, 0
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


def make_grayscale(precision):
    width, height, max_value, raw_samples = read_pgm("32x32x16_grayscale.pgm")
    assert width == WIDTH
    assert height == HEIGHT
    samples = []
    for s in raw_samples:
        samples.append(round(s * ((1 << precision) - 1) / max_value))
    return samples


grayscale_samples8 = make_grayscale(8)
grayscale_samples12 = make_grayscale(12)
grayscale_components8 = [(grayscale_samples8, (1, 1))]
grayscale_components12 = [(grayscale_samples12, (1, 1))]


def make_rgb(precision):
    width, height, max_value, raw_samples = read_pgm("32x32x16_rgb.ppm")
    assert width == WIDTH
    assert height == HEIGHT
    r_samples = []
    g_samples = []
    b_samples = []
    for r, g, b in raw_samples:
        r_samples.append(round(r * ((1 << precision) - 1) / max_value))
        g_samples.append(round(g * ((1 << precision) - 1) / max_value))
        b_samples.append(round(b * ((1 << precision) - 1) / max_value))
    return (r_samples, g_samples, b_samples)


rgb_samples8 = make_rgb(8)
rgb_components8 = [
    (rgb_samples8[0], (1, 1)),
    (rgb_samples8[1], (1, 1)),
    (rgb_samples8[2], (1, 1)),
]


def make_ycbcr(precision):
    r_samples, g_samples, b_samples = make_rgb(precision)
    y_samples = []
    cb_samples = []
    cr_samples = []
    for i in range(len(r_samples)):
        (y, cb, cr) = rgb_to_ycbcr(r_samples[i], g_samples[i], b_samples[i], precision)
        y_samples.append(y)
        cb_samples.append(cb)
        cr_samples.append(cr)
    return (y_samples, cb_samples, cr_samples)


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


def make_cmyk(precision):
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
    return (c_samples, m_samples, y_samples, k_samples)


cmyk_samples8 = make_cmyk(8)
cmyk_components8 = [
    (cmyk_samples8[0], (1, 1)),
    (cmyk_samples8[1], (1, 1)),
    (cmyk_samples8[2], (1, 1)),
    (cmyk_samples8[3], (1, 1)),
]


def scale_samples(width, height, samples, h_max, h, v_max, v):
    if h == h_max and v == v_max:
        return samples
    assert h_max % h == 0
    assert v_max % v == 0
    out_samples = []
    for y in range(0, height, v_max // v):
        for x in range(0, width, h_max // h):
            out_samples.append(samples[y * height + x])
    return out_samples


def make_dct_sequential(
    width,
    height,
    components=[],
    precision=8,
    luminance_quantization_table=[1] * 64,
    chrominance_quantization_table=[1] * 64,
    use_dnl=False,
    restart_interval=0,
    color_space=None,
    scans=[],
    comments=[],
    extended=False,
    progressive=False,
    arithmetic=False,
    arithmetic_conditioning_bounds=[(0, 1), (0, 1), (0, 1), (0, 1)],
    arithmetic_conditioning_kx=[5, 5, 5, 5],
):
    if arithmetic:
        assert extended or progressive

    max_h_sampling_factor = 0
    max_v_sampling_factor = 0
    for _, sampling_factor in components:
        h, v = sampling_factor
        max_h_sampling_factor = max(h, max_h_sampling_factor)
        max_v_sampling_factor = max(v, max_v_sampling_factor)

    if color_space is None:
        assert len(components) in (1, 3)
    elif color_space == ADOBE_COLOR_SPACE_RGB_OR_CMYK:
        assert len(components) in (3, 4)
    elif color_space == ADOBE_COLOR_SPACE_Y_CB_CR:
        assert len(components) == 3
    elif color_space == ADOBE_COLOR_SPACE_Y_CB_CR_K:
        assert len(components) == 4

    if (
        color_space is None and len(components) == 3
    ) or color_space == ADOBE_COLOR_SPACE_Y_CB_CR:
        use_chrominance = True
    else:
        use_chrominance = False

    quantization_tables = [
        QuantizationTable(0, luminance_quantization_table),
    ]
    if use_chrominance:
        quantization_tables.append(QuantizationTable(1, chrominance_quantization_table))
    component_quantization_tables = []
    for i in range(len(components)):
        if i == 0 or not use_chrominance:
            table_index = 0
        else:
            table_index = 1
        component_quantization_tables.append(table_index)

    component_data_units = []
    component_mcu_order_data_units = []
    for i, (samples, sampling_factor) in enumerate(components):
        w = math.ceil(width * sampling_factor[0] / max_h_sampling_factor)
        h = math.ceil(height * sampling_factor[1] / max_v_sampling_factor)
        scaled_samples = scale_samples(
            width,
            height,
            samples,
            max_h_sampling_factor,
            sampling_factor[0],
            max_v_sampling_factor,
            sampling_factor[1],
        )

        if i == 0 or not use_chrominance:
            quantization_table = luminance_quantization_table
        else:
            quantization_table = chrominance_quantization_table

        (width_in_data_units, height_in_data_units, data_units) = (
            jpeg.make_dct_data_units(
                w,
                h,
                precision,
                scaled_samples,
                sampling_factor,
                quantization_table,
            )
        )
        component_data_units.append(data_units)
        component_mcu_order_data_units.append(
            jpeg.mcu_order_dct_data_units(
                width_in_data_units, height_in_data_units, data_units, sampling_factor
            )
        )

    sof_components = []
    for i, (_, sampling_factor) in enumerate(components):
        sof_components.append(
            FrameComponent(
                i + 1,
                sampling_factor,
                component_quantization_tables[i],
            )
        )

    # Generate scans
    jpeg_scans = []
    scan_components = []
    for i in range(len(components)):
        if arithmetic or i == 0 or not use_chrominance:
            dc_table_index = 0
            ac_table_index = 0
        else:
            dc_table_index = 1
            ac_table_index = 1
        scan_components.append(ScanComponent(i + 1, dc_table_index, ac_table_index))
    luminance_dc_symbol_frequencies = [0] * 256
    luminance_ac_symbol_frequencies = [0] * 256
    chrominance_dc_symbol_frequencies = [0] * 256
    chrominance_ac_symbol_frequencies = [0] * 256
    for scan_index, (component_indexes, start, end, point_transform) in enumerate(
        scans
    ):
        sos_components = []
        for i in component_indexes:
            sos_components.append(scan_components[i])
        spectral_selection = (start, end)
        successive = False
        previous_point_transform = 0
        for i in range(scan_index):
            (c, s, e, p) = scans[i]
            if (c, s, e) == (component_indexes, start, end) and p != 0:
                successive = True
                previous_point_transform = p
        if successive:
            if start == 0:
                assert end == 0
        sos = StartOfScan.dct(
            sos_components,
            spectral_selection=spectral_selection,
            point_transform=point_transform,
            previous_point_transform=previous_point_transform,
        )
        if successive:
            scan_data_units = []
            assert len(component_indexes) == 1
            if start == 0:
                scan_data = b""
                # scan_data = jpeg.arithmetic_dct_dc_scan_successive(
                #    component_data_units[component_indexes[0]],
                #    point_transform,
                # )
            else:
                scan_data = b""
                # scan_data = jpeg.arithmetic_dct_ac_scan_successive(
                #    component_data_units[component_indexes[0]],
                #    spectral_selection,
                #    point_transform,
                # )
        else:
            if len(component_indexes) == 1:
                scan_data_units = component_data_units[component_indexes[0]]
            else:
                scan_data_units = []
                next_data_unit = [0] * len(component_indexes)
                while next_data_unit[0] < len(component_mcu_order_data_units[0]):
                    for index in component_indexes:
                        (_, sampling_factor) = components[index]
                        for _ in range(sampling_factor[0] * sampling_factor[1]):
                            scan_data_units.append(
                                component_mcu_order_data_units[index][
                                    next_data_unit[index]
                                ]
                            )
                            next_data_unit[index] += 1

        # Calculate symbol frequencies for Huffman tables.
        if not arithmetic:
            for component_index in component_indexes:
                encoder = jpeg_encoder.HuffmanDCTScanEncoder(
                    spectral_selection=spectral_selection
                )
                if len(component_indexes) == 1:
                    data_units = component_data_units[component_index]
                else:
                    data_units = component_mcu_order_data_units[component_index]
                for data_unit in data_units:
                    encoder.write_data_unit(0, data_unit, 0, 0)
                if component_index == 0 or not use_chrominance:
                    for i in range(256):
                        luminance_dc_symbol_frequencies[
                            i
                        ] += encoder.dc_symbol_frequencies[i]
                        luminance_ac_symbol_frequencies[
                            i
                        ] += encoder.ac_symbol_frequencies[i]
                else:
                    for i in range(256):
                        chrominance_dc_symbol_frequencies[
                            i
                        ] += encoder.dc_symbol_frequencies[i]
                        chrominance_ac_symbol_frequencies[
                            i
                        ] += encoder.ac_symbol_frequencies[i]

        scan_data = DCTScan(scan_data_units)
        jpeg_scans.append((sos, scan_data))

    # Generate Huffman tables.
    if not arithmetic:
        dc_huffman_table = huffman.make_huffman_table(luminance_dc_symbol_frequencies)
        ac_huffman_table = huffman.make_huffman_table(luminance_ac_symbol_frequencies)
        huffman_tables = [
            HuffmanTable.dc(0, dc_huffman_table),
            HuffmanTable.ac(0, ac_huffman_table),
        ]
        if use_chrominance:
            dc_huffman_table = huffman.make_huffman_table(
                chrominance_dc_symbol_frequencies
            )
            ac_huffman_table = huffman.make_huffman_table(
                chrominance_ac_symbol_frequencies
            )
            huffman_tables.extend(
                [
                    HuffmanTable.dc(1, dc_huffman_table),
                    HuffmanTable.ac(1, ac_huffman_table),
                ]
            )

    segments = []
    segments.append(StartOfImage())
    for comment in comments:
        segments.append(Comment(comment))
    if color_space is None:
        segments.append(ApplicationSpecificData.jfif())
    else:
        segments.append(ApplicationSpecificData.adobe(color_space=color_space))
    segments.append(DefineQuantizationTables(quantization_tables))
    if use_dnl:
        number_of_lines = 0
    else:
        number_of_lines = height
    if extended:
        segments.append(
            StartOfFrame.extended(
                number_of_lines, width, precision, sof_components, arithmetic=arithmetic
            )
        )
    elif progressive:
        segments.append(
            StartOfFrame.progressive(
                number_of_lines, width, precision, sof_components, arithmetic=arithmetic
            )
        )
    else:
        segments.append(StartOfFrame.baseline(number_of_lines, width, sof_components))
    if arithmetic:
        conditioning = []
        for i, bounds in enumerate(arithmetic_conditioning_bounds):
            if bounds != (0, 1):
                conditioning.append(ArithmeticConditioning.dc(i, bounds))
        for i, kx in enumerate(arithmetic_conditioning_kx):
            if kx != 5:
                conditioning.append(ArithmeticConditioning.ac(i, kx))
        if len(conditioning) > 0:
            segments.append(DefineArithmeticConditioning(conditioning))
    else:
        segments.append(DefineHuffmanTables(huffman_tables))
    if restart_interval != 0:
        segments.append(DefineRestartInterval(restart_interval))
    for i, (sos, scan_data) in enumerate(jpeg_scans):
        segments.append(sos)
        segments.append(scan_data)
        if i == 0 and use_dnl:
            segments.append(DefineNumberOfLines(height))
    segments.append(EndOfImage())

    encoder = jpeg_encoder.Encoder(segments)
    encoder.encode()
    return encoder.data


def make_lossless(
    width,
    height,
    component_samples,
    precision=8,
    use_dnl=False,
    predictor=1,
    restart_interval=0,
    arithmetic=False,
):
    conditioning_bounds = (0, 1)
    components = []
    jpeg_scans = []
    for i, samples in enumerate(component_samples):
        values = jpeg.make_lossless_values(
            predictor, width, precision, samples, restart_interval=restart_interval
        )
        if arithmetic:
            table = 0
            scan_data = jpeg.arithmetic_lossless_scan(
                conditioning_bounds,
                width,
                values,
                restart_interval=restart_interval,
            )
        else:
            table = i
            scan_data = jpeg.huffman_lossless_scan_data(
                table,
                values,
                restart_interval=restart_interval,
            )
        components.append(jpeg.Component(id=i + 1))
        sos = jpeg.start_of_scan_lossless(
            components=[jpeg.ScanComponent.lossless(i + 1, table=table)],
            predictor=predictor,
        )
        jpeg_scans.append((sos, scan_data))

    # Generate Huffman tables and encode scans.
    if not arithmetic:
        all_huffman_bits = []
        for _, scan_data in jpeg_scans:
            all_huffman_bits.extend(scan_data)
        huffman_tables = []
        for i in range(len(components)):
            huffman_tables.append(
                jpeg.HuffmanTable.dc(
                    i, jpeg.make_dct_huffman_dc_table(all_huffman_bits, i)
                )
            )
        for i, (sos, scan_data) in enumerate(jpeg_scans):
            jpeg_scans[i] = (
                sos,
                jpeg.huffman_lossless_scan(huffman_tables, scan_data),
            )

    data = jpeg.start_of_image()
    if use_dnl:
        number_of_lines = 0
    else:
        number_of_lines = height
    data += jpeg.start_of_frame_lossless(
        width, number_of_lines, precision, components, arithmetic=arithmetic
    )
    if not arithmetic:
        data += jpeg.define_huffman_tables(tables=huffman_tables)
    if restart_interval != 0:
        data += jpeg.define_restart_interval(restart_interval)
    for i, (sos, scan_data) in enumerate(jpeg_scans):
        data += sos + scan_data
        if i == 0 and use_dnl:
            data += jpeg.define_number_of_lines(height)
    data += jpeg.end_of_image()
    return data


def generate_dct(
    section,
    description,
    width,
    height,
    components,
    precision=8,
    luminance_quantization_table=[1] * 64,
    chrominance_quantization_table=[1] * 64,
    restart_interval=0,
    use_dnl=False,
    color_space=None,
    scans=[],
    comments=[],
    extended=False,
    progressive=False,
    arithmetic=False,
    arithmetic_conditioning_bounds=[(0, 1), (0, 1), (0, 1), (0, 1)],
    arithmetic_conditioning_kx=[5, 5, 5, 5],
):
    open(
        "../jpeg/%s/%dx%dx%d_%s.jpg" % (section, width, height, precision, description),
        "wb",
    ).write(
        make_dct_sequential(
            width,
            height,
            components,
            precision=precision,
            luminance_quantization_table=luminance_quantization_table,
            chrominance_quantization_table=chrominance_quantization_table,
            restart_interval=restart_interval,
            use_dnl=use_dnl,
            color_space=color_space,
            scans=scans,
            comments=comments,
            extended=extended,
            progressive=progressive,
            arithmetic=arithmetic,
            arithmetic_conditioning_bounds=arithmetic_conditioning_bounds,
            arithmetic_conditioning_kx=arithmetic_conditioning_kx,
        )
    )


def generate_lossless(
    section,
    description,
    width,
    height,
    component_samples,
    use_dnl=False,
    precision=8,
    restart_interval=0,
    predictor=1,
    arithmetic=False,
):
    open(
        "../jpeg/%s/%dx%dx%d_%s.jpg" % (section, width, height, precision, description),
        "wb",
    ).write(
        make_lossless(
            width,
            height,
            component_samples,
            use_dnl=use_dnl,
            precision=precision,
            restart_interval=restart_interval,
            predictor=predictor,
            arithmetic=arithmetic,
        )
    )


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
        section = "%s_%s" % (mode, encoding)
    else:
        section = "baseline"
    generate_dct(
        section,
        "grayscale",
        WIDTH,
        HEIGHT,
        grayscale_components8,
        scans=[([0], 0, 63, 0)],
        extended=extended,
        progressive=progressive,
        arithmetic=arithmetic,
    )
    generate_dct(
        section,
        "grayscale_quantization",
        WIDTH,
        HEIGHT,
        grayscale_components8,
        luminance_quantization_table=standard_luminance_quantization_table,
        scans=[([0], 0, 63, 0)],
        extended=extended,
        progressive=progressive,
        arithmetic=arithmetic,
    )
    generate_dct(
        section,
        "ycbcr",
        WIDTH,
        HEIGHT,
        ycbcr_components8,
        scans=[([0], 0, 63, 0), ([1], 0, 63, 0), ([2], 0, 63, 0)],
        extended=extended,
        progressive=progressive,
        arithmetic=arithmetic,
    )
    generate_dct(
        section,
        "ycbcr_quantization",
        WIDTH,
        HEIGHT,
        ycbcr_components8,
        luminance_quantization_table=standard_luminance_quantization_table,
        chrominance_quantization_table=standard_chrominance_quantization_table,
        scans=[([0], 0, 63, 0), ([1], 0, 63, 0), ([2], 0, 63, 0)],
        extended=extended,
        progressive=progressive,
        arithmetic=arithmetic,
    )
    generate_dct(
        section,
        "ycbcr_interleaved",
        WIDTH,
        HEIGHT,
        ycbcr_components8,
        scans=[([0, 1, 2], 0, 63, 0)],
        extended=extended,
        progressive=progressive,
        arithmetic=arithmetic,
    )
    # FIXME: Greyscale sampling
    generate_dct(
        section,
        "ycbcr_2x2_1x1_1x1",
        WIDTH,
        HEIGHT,
        [
            (ycbcr_samples8[0], (2, 2)),
            (ycbcr_samples8[1], (1, 1)),
            (ycbcr_samples8[2], (1, 1)),
        ],
        scans=[([0], 0, 63, 0), ([1], 0, 63, 0), ([2], 0, 63, 0)],
        extended=extended,
        progressive=progressive,
        arithmetic=arithmetic,
    )
    generate_dct(
        section,
        "ycbcr_2x2_1x1_1x1_interleaved",
        WIDTH,
        HEIGHT,
        [
            (ycbcr_samples8[0], (2, 2)),
            (ycbcr_samples8[1], (1, 1)),
            (ycbcr_samples8[2], (1, 1)),
        ],
        scans=[([0, 1, 2], 0, 63, 0)],
        extended=extended,
        progressive=progressive,
        arithmetic=arithmetic,
    )
    generate_dct(
        section,
        "ycbcr_2x2_2x1_1x2",
        WIDTH,
        HEIGHT,
        [
            (ycbcr_samples8[0], (2, 2)),
            (ycbcr_samples8[1], (2, 1)),
            (ycbcr_samples8[2], (1, 2)),
        ],
        scans=[([0], 0, 63, 0), ([1], 0, 63, 0), ([2], 0, 63, 0)],
        extended=extended,
        progressive=progressive,
        arithmetic=arithmetic,
    )
    generate_dct(
        section,
        "ycbcr_2x2_2x1_1x2_interleaved",
        WIDTH,
        HEIGHT,
        [
            (ycbcr_samples8[0], (2, 2)),
            (ycbcr_samples8[1], (2, 1)),
            (ycbcr_samples8[2], (1, 2)),
        ],
        scans=[([0, 1, 2], 0, 63, 0)],
        extended=extended,
        progressive=progressive,
        arithmetic=arithmetic,
    )
    generate_dct(
        section,
        "grayscale_zero_coefficients",
        8,
        8,
        [([128] * 64, (1, 1))],
        scans=[([0], 0, 63, 0)],
        extended=extended,
        progressive=progressive,
        arithmetic=arithmetic,
    )
    generate_dct(
        section,
        "grayscale_black",
        8,
        8,
        [([0] * 64, (1, 1))],
        scans=[([0], 0, 63, 0)],
        extended=extended,
        progressive=progressive,
        arithmetic=arithmetic,
    )
    generate_dct(
        section,
        "grayscale_white",
        8,
        8,
        [([255] * 64, (1, 1))],
        scans=[([0], 0, 63, 0)],
        extended=extended,
        progressive=progressive,
        arithmetic=arithmetic,
    )
    for size in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16):
        (width, height, _, samples) = read_pgm("%dx%dx8_grayscale.pgm" % (size, size))
        assert width == height == size
        generate_dct(
            section,
            "grayscale",
            width,
            height,
            [(samples, (1, 1))],
            scans=[([0], 0, 63, 0)],
            extended=extended,
            progressive=progressive,
            arithmetic=arithmetic,
        )
    generate_dct(
        section,
        "comment",
        WIDTH,
        HEIGHT,
        grayscale_components8,
        scans=[([0], 0, 63, 0)],
        comments=[b"Hello World"],
        extended=extended,
        progressive=progressive,
        arithmetic=arithmetic,
    )
    generate_dct(
        section,
        "comments",
        WIDTH,
        HEIGHT,
        grayscale_components8,
        scans=[([0], 0, 63, 0)],
        comments=[b"Hello", b"World"],
        extended=extended,
        progressive=progressive,
        arithmetic=arithmetic,
    )
    generate_dct(
        section,
        "rgb",
        WIDTH,
        HEIGHT,
        rgb_components8,
        scans=[([0], 0, 63, 0), ([1], 0, 63, 0), ([2], 0, 63, 0)],
        color_space=ADOBE_COLOR_SPACE_RGB_OR_CMYK,
        extended=extended,
        progressive=progressive,
        arithmetic=arithmetic,
    )
    generate_dct(
        section,
        "rgb_interleaved",
        WIDTH,
        HEIGHT,
        rgb_components8,
        scans=[([0, 1, 2], 0, 63, 0)],
        color_space=ADOBE_COLOR_SPACE_RGB_OR_CMYK,
        extended=extended,
        progressive=progressive,
        arithmetic=arithmetic,
    )
    generate_dct(
        section,
        "cmyk",
        WIDTH,
        HEIGHT,
        cmyk_components8,
        scans=[([0], 0, 63, 0), ([1], 0, 63, 0), ([2], 0, 63, 0), ([3], 0, 63, 0)],
        color_space=ADOBE_COLOR_SPACE_RGB_OR_CMYK,
        extended=extended,
        progressive=progressive,
        arithmetic=arithmetic,
    )
    generate_dct(
        section,
        "cmyk_interleaved",
        WIDTH,
        HEIGHT,
        cmyk_components8,
        scans=[([0, 1, 2, 3], 0, 63, 0)],
        color_space=ADOBE_COLOR_SPACE_RGB_OR_CMYK,
        extended=extended,
        progressive=progressive,
        arithmetic=arithmetic,
    )
    generate_dct(
        section,
        "dnl",
        WIDTH,
        HEIGHT,
        grayscale_components8,
        scans=[([0], 0, 63, 0)],
        use_dnl=True,
        extended=extended,
        progressive=progressive,
        arithmetic=arithmetic,
    )
    generate_dct(
        section,
        "restarts",
        WIDTH,
        HEIGHT,
        grayscale_components8,
        scans=[([0], 0, 63, 0)],
        restart_interval=4,
        extended=extended,
        progressive=progressive,
        arithmetic=arithmetic,
    )

    if arithmetic:
        generate_dct(
            section,
            "conditioning_bounds_4_6",
            WIDTH,
            HEIGHT,
            grayscale_components8,
            scans=[([0], 0, 63, 0)],
            extended=extended,
            progressive=progressive,
            arithmetic=True,
            arithmetic_conditioning_bounds=[(4, 6), (4, 6), (4, 6), (4, 6)],
        )

        generate_dct(
            section,
            "conditioning_kx_6",
            WIDTH,
            HEIGHT,
            grayscale_components8,
            scans=[([0], 0, 63, 0)],
            extended=extended,
            progressive=progressive,
            arithmetic=True,
            arithmetic_conditioning_kx=[6, 6, 6, 6],
        )

    if mode != "baseline":
        generate_dct(
            section,
            "grayscale",
            WIDTH,
            HEIGHT,
            grayscale_components12,
            scans=[([0], 0, 63, 0)],
            precision=12,
            extended=extended,
            progressive=progressive,
            arithmetic=arithmetic,
        )
        generate_dct(
            section,
            "ycbcr",
            WIDTH,
            HEIGHT,
            ycbcr_components12,
            scans=[([0], 0, 63, 0), ([1], 0, 63, 0), ([2], 0, 63, 0)],
            precision=12,
            extended=extended,
            progressive=progressive,
            arithmetic=arithmetic,
        )
        generate_dct(
            section,
            "ycbcr_interleaved",
            WIDTH,
            HEIGHT,
            ycbcr_components12,
            scans=[([0, 1, 2], 0, 63, 0)],
            precision=12,
            extended=extended,
            progressive=progressive,
            arithmetic=arithmetic,
        )

    if mode == "progressive":
        generate_dct(
            section,
            "grayscale_spectral",
            WIDTH,
            HEIGHT,
            grayscale_components8,
            scans=[([0], 0, 0, 0), ([0], 1, 63, 0)],
            extended=extended,
            progressive=progressive,
            arithmetic=arithmetic,
        )
        all_selection = [([0], 0, 0, 0)]
        all_reverse_selection = [([0], 0, 0, 0)]
        for i in range(1, 64):
            all_selection.append(([0], i, i, 0))
            all_reverse_selection.append(([0], 64 - i, 64 - i, 0))
        generate_dct(
            section,
            "grayscale_spectral_all",
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
    section = "lossless_%s" % encoding
    for predictor in range(1, 8):
        generate_lossless(
            section,
            "grayscale_predictor%d" % predictor,
            WIDTH,
            HEIGHT,
            [grayscale_samples8],
            predictor=predictor,
            arithmetic=arithmetic,
        )
    for precision in range(2, 17):
        generate_lossless(
            section,
            "grayscale",
            WIDTH,
            HEIGHT,
            [make_grayscale(precision)],
            precision=precision,
            predictor=1,
            arithmetic=arithmetic,
        )
    for size in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16):
        (width, height, _, samples) = read_pgm("%dx%dx8_grayscale.pgm" % (size, size))
        assert width == height == size
        generate_lossless(
            section,
            "grayscale",
            width,
            height,
            [samples],
            precision=8,
            predictor=1,
            arithmetic=arithmetic,
        )
    generate_lossless(
        section,
        "rgb",
        WIDTH,
        HEIGHT,
        rgb_samples8,
        predictor=1,
        arithmetic=arithmetic,
    )
    generate_lossless(
        section,
        "restarts",
        WIDTH,
        HEIGHT,
        [grayscale_samples8],
        predictor=1,
        restart_interval=32 * 8,
        arithmetic=arithmetic,
    )
    generate_lossless(
        section,
        "dnl",
        WIDTH,
        HEIGHT,
        [grayscale_samples8],
        use_dnl=True,
        predictor=1,
        arithmetic=arithmetic,
    )

# 3 channel, red, green, blue, white, mixed color
# version 1.1
# density
# thumbnail
# multiple huffman tables
# arithmetic properties
# lossless interleaved
