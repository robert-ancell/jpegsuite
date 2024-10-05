#!/usr/bin/env python3

import math
import jpeg
from pgm import *


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
    width, height, max_value, raw_samples = read_pgm("test-face.pgm")
    assert width == WIDTH
    assert height == HEIGHT
    samples = []
    for s in raw_samples:
        samples.append(round(s * ((1 << precision) - 1) / max_value))
    return samples


grayscale_samples8 = make_grayscale(8)
grayscale_samples12 = make_grayscale(12)


def make_rgb(precision):
    width, height, max_value, raw_samples = read_pgm("test-face.ppm")
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


# FIXME: Merge sampling factors into samples
def make_dct_sequential(
    width,
    height,
    component_samples,
    sampling_factors=None,
    precision=8,
    use_dnl=False,
    restart_interval=0,
    color_space=None,
    scans=[],
    comments=[],
    extended=False,
    progressive=False,
    arithmetic=False,
):
    if arithmetic:
        assert extended or progressive

    n_components = len(component_samples)

    if sampling_factors is None:
        sampling_factors = [(1, 1)] * n_components

    max_h_sampling_factor = 0
    max_v_sampling_factor = 0
    for h, v in sampling_factors:
        max_h_sampling_factor = max(h, max_h_sampling_factor)
        max_v_sampling_factor = max(v, max_v_sampling_factor)

    component_sizes = []
    scaled_component_samples = []
    for i, samples in enumerate(component_samples):
        sampling_factor = sampling_factors[i]
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

    if color_space is None:
        assert n_components in (1, 3)
    elif color_space == jpeg.ADOBE_COLOR_SPACE_RGB_OR_CMYK:
        assert n_components in (3, 4)
    elif color_space == jpeg.ADOBE_COLOR_SPACE_Y_CB_CR:
        assert n_components == 3
    elif color_space == jpeg.ADOBE_COLOR_SPACE_Y_CB_CR_K:
        assert n_components == 4
    assert len(sampling_factors) == n_components

    if (
        color_space is None and n_components == 3
    ) or color_space == jpeg.ADOBE_COLOR_SPACE_Y_CB_CR:
        use_chrominance = True
    else:
        use_chrominance = False

    luminance_quantization_table = [1] * 64  # FIXME: Using nothing at this point
    chrominance_quantization_table = [1] * 64  # FIXME: Using nothing at this point
    quantization_tables = [
        jpeg.QuantizationTable(destination=0, data=luminance_quantization_table),
    ]
    if use_chrominance:
        quantization_tables.append(
            jpeg.QuantizationTable(destination=1, data=chrominance_quantization_table)
        )
    component_quantization_tables = []
    for i in range(n_components):
        if i == 0 or not use_chrominance:
            table_index = 0
        else:
            table_index = 1
        component_quantization_tables.append(table_index)

    coefficients = []
    for i in range(n_components):
        if i == 0 or not use_chrominance:
            quantization_table = luminance_quantization_table
        else:
            quantization_table = chrominance_quantization_table
        coefficients.append(
            jpeg.make_dct_coefficients(
                component_sizes[i][0],
                component_sizes[i][1],
                precision,
                scaled_component_samples[i],
                quantization_table,
            )
        )

    sof_components = []
    for i in range(n_components):
        sof_components.append(
            jpeg.Component(
                id=i + 1,
                sampling_factor=sampling_factors[i],
                quantization_table=component_quantization_tables[i],
            )
        )

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
            jpeg.ScanComponent(i + 1, dc_table=dc_table_index, ac_table=ac_table_index)
        )
    for scan_index, (components, start, end, point_transform) in enumerate(scans):
        sos_components = []
        for i in components:
            sos_components.append(scan_components[i])
        selection = (start, end)
        successive = False
        previous_point_transform = 0
        for i in range(scan_index):
            (c, s, e, p) = scans[i]
            if (c, s, e) == (components, start, end) and p != 0:
                successive = True
                previous_point_transform = p
        if successive:
            if start == 0:
                assert end == 0
        sos = jpeg.start_of_scan_dct(
            components=sos_components,
            selection=selection,
            point_transform=point_transform,
            previous_point_transform=previous_point_transform,
        )
        if arithmetic:
            if successive:
                assert len(components) == 1
                if start == 0:
                    scan_data = jpeg.arithmetic_dct_dc_scan_successive(
                        coefficients[components[0]],
                        point_transform,
                    )
                else:
                    scan_data = jpeg.arithmetic_dct_ac_scan_successive(
                        coefficients[components[0]],
                        selection,
                        point_transform,
                    )
            else:
                assert len(components) == 1
                scan_data = jpeg.arithmetic_dct_scan(
                    components=[jpeg.ArithmeticDCTComponent(coefficients=coefficients[components[0]], sampling_factor=sampling_factors[0])],
                    restart_interval=restart_interval,
                    selection=selection,
                    point_transform=point_transform,
                )
        else:
            huffman_components = []
            for i in components:
                # MCU is 1 in non-interleaved
                if len(components) == 1:
                    sampling_factor = (1, 1)
                else:
                    sampling_factor = sampling_factors[i]
                mcu_coefficients = jpeg.order_mcu_dct_coefficients(
                    component_sizes[i][0],
                    component_sizes[i][1],
                    coefficients[i],
                    sampling_factor,
                )
                huffman_components.append(
                    jpeg.HuffmanDCTComponent(
                        dc_table=scan_components[i].dc_table,
                        ac_table=scan_components[i].ac_table,
                        coefficients=mcu_coefficients,
                        sampling_factor=sampling_factor,
                    )
                )
            if successive:
                assert len(components) == 1
                if start == 0:
                    scan_data = jpeg.huffman_dct_dc_scan_successive_data(
                        coefficients=coefficients[components[0]],
                        point_transform=point_transform,
                    )
                else:
                    scan_data = jpeg.huffman_dct_ac_scan_successive_data(
                        coefficients=coefficients[components[0]],
                        selection=selection,
                        point_transform=point_transform,
                    )
            else:
                scan_data = jpeg.huffman_dct_scan_data(
                    components=huffman_components,
                    restart_interval=restart_interval,
                    selection=selection,
                    point_transform=point_transform,
                )
        jpeg_scans.append((sos, scan_data))

    # Generate Huffman tables and encode scans.
    if not arithmetic:
        all_huffman_bits = []
        for _, scan_data in jpeg_scans:
            all_huffman_bits.extend(scan_data)
        huffman_tables = [
            jpeg.HuffmanTable.dc(
                0, jpeg.make_dct_huffman_dc_table(all_huffman_bits, 0)
            ),
            jpeg.HuffmanTable.ac(
                0, jpeg.make_dct_huffman_ac_table(all_huffman_bits, 0)
            ),
        ]
        if use_chrominance:
            huffman_tables.append(
                jpeg.HuffmanTable.dc(
                    1, jpeg.make_dct_huffman_dc_table(all_huffman_bits, 1)
                )
            )
            huffman_tables.append(
                jpeg.HuffmanTable.ac(
                    1, jpeg.make_dct_huffman_ac_table(all_huffman_bits, 1)
                )
            )

        for i, (sos, scan_data) in enumerate(jpeg_scans):
            jpeg_scans[i] = (
                sos,
                jpeg.huffman_dct_scan(huffman_tables, scan_data),
            )

    data = jpeg.start_of_image()
    for comment in comments:
        data += jpeg.comment(comment)
    if color_space is None:
        data += jpeg.jfif()
    else:
        data += jpeg.adobe(color_space=color_space)
    data += jpeg.define_quantization_tables(tables=quantization_tables)
    if use_dnl:
        number_of_lines = 0
    else:
        number_of_lines = height
    if extended:
        data += jpeg.start_of_frame_extended(
            width, number_of_lines, precision, sof_components, arithmetic=arithmetic
        )
    elif progressive:
        data += jpeg.start_of_frame_progressive(
            width, number_of_lines, precision, sof_components, arithmetic=arithmetic
        )
    else:
        data += jpeg.start_of_frame_baseline(width, number_of_lines, sof_components)
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
    conditioning_range = (0, 1)
    components = []
    jpeg_scans = []
    for i, samples in enumerate(component_samples):
        values = jpeg.make_lossless_values(
            predictor, width, precision, samples, restart_interval=restart_interval
        )
        if arithmetic:
            table = 0
            scan_data = jpeg.arithmetic_lossless_scan(
                conditioning_range,
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
    component_samples,
    precision=8,
    restart_interval=0,
    use_dnl=False,
    sampling_factors=None,
    color_space=None,
    scans=[],
    comments=[],
    extended=False,
    progressive=False,
    arithmetic=False,
):
    open(
        "../jpeg/%s/%dx%dx%d_%s.jpg" % (section, width, height, precision, description),
        "wb",
    ).write(
        make_dct_sequential(
            width,
            height,
            component_samples,
            precision=precision,
            restart_interval=restart_interval,
            use_dnl=use_dnl,
            sampling_factors=sampling_factors,
            color_space=color_space,
            scans=scans,
            comments=comments,
            extended=extended,
            progressive=progressive,
            arithmetic=arithmetic,
        )
    )


def generate_lossless(
    section,
    description,
    width,
    height,
    component_samples,
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
            precision=precision,
            restart_interval=restart_interval,
            predictor=predictor,
            arithmetic=arithmetic,
        )
    )


# Encodings that work in all modes
for mode, encoding in [
    ("baseline", "huffman"),
    ("extended", "huffman"),
    ("extended", "arithmetic"),
]:
    extended = mode == "extended"
    arithmetic = encoding == "arithmetic"
    if extended:
        section = "extended_%s" % encoding
    else:
        section = "baseline"
    generate_dct(
        section,
        "grayscale",
        WIDTH,
        HEIGHT,
        [grayscale_samples8],
        scans=[([0], 0, 63, 0)],
        extended=extended,
        arithmetic=arithmetic,
    )
    generate_dct(
        section,
        "ycbcr",
        WIDTH,
        HEIGHT,
        ycbcr_samples8,
        scans=[([0], 0, 63, 0), ([1], 0, 63, 0), ([2], 0, 63, 0)],
        extended=extended,
        arithmetic=arithmetic,
    )
    # FIXME: Support interleaved arithmetic
    if not arithmetic:
        generate_dct(
            section,
            "ycbcr_interleaved",
            WIDTH,
            HEIGHT,
            ycbcr_samples8,
            scans=[([0, 1, 2], 0, 63, 0)],
            extended=extended,
            arithmetic=arithmetic,
        )
    # FIXME: Greyscale sampling
    generate_dct(
        section,
        "ycbcr_2x2_1x1_1x1",
        WIDTH,
        HEIGHT,
        ycbcr_samples8,
        scans=[([0], 0, 63, 0), ([1], 0, 63, 0), ([2], 0, 63, 0)],
        sampling_factors=[(2, 2), (1, 1), (1, 1)],
        extended=extended,
        arithmetic=arithmetic,
    )
    if not arithmetic:
        generate_dct(
            section,
            "ycbcr_2x2_1x1_1x1_interleaved",
            WIDTH,
            HEIGHT,
            ycbcr_samples8,
            scans=[([0, 1, 2], 0, 63, 0)],
            sampling_factors=[(2, 2), (1, 1), (1, 1)],
            extended=extended,
            arithmetic=arithmetic,
        )
    generate_dct(
        section,
        "ycbcr_2x2_2x1_1x2",
        WIDTH,
        HEIGHT,
        ycbcr_samples8,
        scans=[([0], 0, 63, 0), ([1], 0, 63, 0), ([2], 0, 63, 0)],
        sampling_factors=[(2, 2), (2, 1), (1, 2)],
        extended=extended,
        arithmetic=arithmetic,
    )
    if not arithmetic:
        generate_dct(
            section,
            "ycbcr_2x2_2x1_1x2_interleaved",
            WIDTH,
            HEIGHT,
            ycbcr_samples8,
            scans=[([0, 1, 2], 0, 63, 0)],
            sampling_factors=[(2, 2), (2, 1), (1, 2)],
            extended=extended,
            arithmetic=arithmetic,
        )
    generate_dct(
        section,
        "comment",
        WIDTH,
        HEIGHT,
        [grayscale_samples8],
        scans=[([0], 0, 63, 0)],
        comments=[b"Hello World"],
        extended=extended,
        arithmetic=arithmetic,
    )
    generate_dct(
        section,
        "comments",
        WIDTH,
        HEIGHT,
        [grayscale_samples8],
        scans=[([0], 0, 63, 0)],
        comments=[b"Hello", b"World"],
        extended=extended,
        arithmetic=arithmetic,
    )
    generate_dct(
        section,
        "rgb",
        WIDTH,
        HEIGHT,
        rgb_samples8,
        scans=[([0], 0, 63, 0), ([1], 0, 63, 0), ([2], 0, 63, 0)],
        color_space=jpeg.ADOBE_COLOR_SPACE_RGB_OR_CMYK,
        extended=extended,
        arithmetic=arithmetic,
    )
    if not arithmetic:
        generate_dct(
            section,
            "rgb_interleaved",
            WIDTH,
            HEIGHT,
            rgb_samples8,
            scans=[([0, 1, 2], 0, 63, 0)],
            color_space=jpeg.ADOBE_COLOR_SPACE_RGB_OR_CMYK,
            extended=extended,
            arithmetic=arithmetic,
        )
    generate_dct(
        section,
        "cmyk",
        WIDTH,
        HEIGHT,
        cmyk_samples8,
        scans=[([0], 0, 63, 0), ([1], 0, 63, 0), ([2], 0, 63, 0), ([3], 0, 63, 0)],
        color_space=jpeg.ADOBE_COLOR_SPACE_RGB_OR_CMYK,
        extended=extended,
        arithmetic=arithmetic,
    )
    if not arithmetic:
        generate_dct(
            section,
            "cmyk_interleaved",
            WIDTH,
            HEIGHT,
            cmyk_samples8,
            scans=[([0, 1, 2, 3], 0, 63, 0)],
            color_space=jpeg.ADOBE_COLOR_SPACE_RGB_OR_CMYK,
            extended=extended,
            arithmetic=arithmetic,
        )
    generate_dct(
        section,
        "dnl",
        WIDTH,
        HEIGHT,
        [grayscale_samples8],
        scans=[([0], 0, 63, 0)],
        use_dnl=True,
        extended=extended,
        arithmetic=arithmetic,
    )
    generate_dct(
        section,
        "restarts",
        WIDTH,
        HEIGHT,
        [grayscale_samples8],
        scans=[([0], 0, 63, 0)],
        restart_interval=4,
        extended=extended,
        arithmetic=arithmetic,
    )

for encoding in ["huffman", "arithmetic"]:
    arithmetic = encoding == "arithmetic"

    # Encodings that require extended (12 bit)
    section = "extended_%s" % encoding
    generate_dct(
        section,
        "grayscale",
        WIDTH,
        HEIGHT,
        [grayscale_samples12],
        scans=[([0], 0, 63, 0)],
        precision=12,
        extended=True,
        arithmetic=arithmetic,
    )
    generate_dct(
        section,
        "ycbcr",
        WIDTH,
        HEIGHT,
        ycbcr_samples12,
        scans=[([0], 0, 63, 0), ([1], 0, 63, 0), ([2], 0, 63, 0)],
        precision=12,
        extended=True,
        arithmetic=arithmetic,
    )

    section = "progressive_%s" % encoding
    generate_dct(
        section,
        "grayscale_spectral",
        WIDTH,
        HEIGHT,
        [grayscale_samples8],
        scans=[([0], 0, 0, 0), ([0], 1, 63, 0)],
        progressive=True,
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
        [grayscale_samples8],
        scans=all_selection,
        progressive=True,
        arithmetic=arithmetic,
    )
    generate_dct(
        section,
        "grayscale_spectral_all_reverse",
        WIDTH,
        HEIGHT,
        [grayscale_samples8],
        scans=all_reverse_selection,
        progressive=True,
        arithmetic=arithmetic,
    )
    generate_dct(
        section,
        "grayscale_successive_dc",
        WIDTH,
        HEIGHT,
        [grayscale_samples8],
        scans=[
            ([0], 0, 0, 4),
            ([0], 0, 0, 3),
            ([0], 0, 0, 2),
            ([0], 0, 0, 1),
            ([0], 0, 0, 0),
            ([0], 1, 63, 0),
        ],
        progressive=True,
        arithmetic=arithmetic,
    )
    generate_dct(
        section,
        "grayscale_successive_ac",
        WIDTH,
        HEIGHT,
        [grayscale_samples8],
        scans=[
            ([0], 0, 0, 0),
            ([0], 1, 63, 4),
            ([0], 1, 63, 3),
            ([0], 1, 63, 2),
            ([0], 1, 63, 1),
            ([0], 1, 63, 0),
        ],
        progressive=True,
        arithmetic=arithmetic,
    )
    generate_dct(
        section,
        "grayscale_successive",
        WIDTH,
        HEIGHT,
        [grayscale_samples8],
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
        progressive=True,
        arithmetic=arithmetic,
    )
    # FIXME: successive 3, 2, 1
    # FIXME: successive with restarts

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

# 3 channel, red, green, blue, white, mixed color
# version 1.1
# density
# thumbnail
# multiple tables
# arithmetic interleaved
