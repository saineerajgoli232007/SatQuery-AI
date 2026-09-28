import streamlit as st
from PIL import Image
import numpy as np
import rasterio
import re
import json


st.set_page_config(
    page_title="SatQuery AI",
    page_icon="🛰️",
    layout="wide"
)


# ============================================================
# IMAGE PROCESSING
# ============================================================

def scale_to_uint8(array):
    array = np.asarray(array)

    if array.ndim == 2:
        array = array.astype(np.float32)

        low = np.percentile(array, 2)
        high = np.percentile(array, 98)

        if high <= low:
            low = float(array.min())
            high = float(array.max())

        if high > low:
            array = (array - low) / (high - low) * 255.0
        else:
            array = np.zeros_like(array)

        return np.clip(array, 0, 255).astype(np.uint8)

    array = array.astype(np.float32)
    result = np.zeros_like(array, dtype=np.uint8)

    for band in range(array.shape[2]):
        channel = array[:, :, band]

        low = np.percentile(channel, 2)
        high = np.percentile(channel, 98)

        if high <= low:
            low = float(channel.min())
            high = float(channel.max())

        if high > low:
            scaled = (
                (channel - low)
                / (high - low)
                * 255.0
            )
        else:
            scaled = np.zeros_like(channel)

        result[:, :, band] = np.clip(
            scaled,
            0,
            255
        ).astype(np.uint8)

    return result


def grayscale_from_image(image_array):
    array = np.asarray(image_array)

    if array.ndim == 2:
        gray = array.astype(np.float32)
    else:
        gray = (
            array[:, :, :3]
            .astype(np.float32)
            .mean(axis=2)
        )

    low = np.percentile(gray, 2)
    high = np.percentile(gray, 98)

    if high <= low:
        high = low + 1.0

    gray = (
        (gray - low)
        / (high - low)
        * 255.0
    )

    return np.clip(
        gray,
        0,
        255
    ).astype(np.uint8)


# ============================================================
# CONNECTED COMPONENT DETECTION
# ============================================================

def connected_components(
    binary,
    min_area=20,
    max_area=100000
):
    height, width = binary.shape

    visited = np.zeros_like(
        binary,
        dtype=bool
    )

    components = []

    for y in range(height):

        xs = np.flatnonzero(
            binary[y] & ~visited[y]
        )

        for x in xs:

            if visited[y, x]:
                continue

            stack = [
                (int(y), int(x))
            ]

            visited[y, x] = True

            min_x = max_x = int(x)
            min_y = max_y = int(y)

            count = 0

            while stack:

                cy, cx = stack.pop()

                count += 1

                min_x = min(min_x, cx)
                max_x = max(max_x, cx)

                min_y = min(min_y, cy)
                max_y = max(max_y, cy)

                for dy in (-1, 0, 1):

                    ny = cy + dy

                    if ny < 0 or ny >= height:
                        continue

                    for dx in (-1, 0, 1):

                        if dx == 0 and dy == 0:
                            continue

                        nx = cx + dx

                        if nx < 0 or nx >= width:
                            continue

                        if (
                            binary[ny, nx]
                            and not visited[ny, nx]
                        ):
                            visited[ny, nx] = True
                            stack.append(
                                (ny, nx)
                            )

            if min_area <= count <= max_area:

                components.append(
                    {
                        "min_x": min_x,
                        "max_x": max_x,
                        "min_y": min_y,
                        "max_y": max_y,
                        "area": count
                    }
                )

    return components


# ============================================================
# MASK DILATION
# ============================================================

def dilate_mask(mask, radius):

    radius = max(
        0,
        int(radius)
    )

    if radius == 0:
        return mask.copy()

    result = mask.copy()

    for shift in range(
        1,
        radius + 1
    ):

        result[:, shift:] |= (
            mask[:, :-shift]
        )

        result[:, :-shift] |= (
            mask[:, shift:]
        )

        result[shift:, :] |= (
            mask[:-shift, :]
        )

        result[:-shift, :] |= (
            mask[shift:, :]
        )

    return result


def make_road_buffer(
    mask,
    radius_pixels
):
    return dilate_mask(
        mask,
        radius_pixels
    )


# ============================================================
# QUERY DISTANCE
# ============================================================

def extract_distance_m(query_text):

    pattern = (
        r"(\d+(?:\.\d+)?)"
        r"\s*(?:metres?|meters?|m)\b"
    )

    match = re.search(
        pattern,
        query_text.lower()
    )

    if match:
        return float(
            match.group(1)
        )

    return None


# ============================================================
# PIXEL → MAP COORDINATES
# ============================================================

def pixel_to_map(transform, row, col):
    """Convert raster row/column to map coordinates using the affine transform."""
    x = (
        float(transform.a) * float(col)
        + float(transform.b) * float(row)
        + float(transform.c)
    )
    y = (
        float(transform.d) * float(col)
        + float(transform.e) * float(row)
        + float(transform.f)
    )
    return x, y


def pixel_polygon(
    transform,
    min_x,
    min_y,
    max_x,
    max_y
):
    """Create a GeoJSON polygon without rasterio.transform.xy."""
    corners = [
        pixel_to_map(transform, min_y, min_x),
        pixel_to_map(transform, min_y, max_x + 1),
        pixel_to_map(transform, max_y + 1, max_x + 1),
        pixel_to_map(transform, max_y + 1, min_x),
    ]

    coordinates = [
        [float(x), float(y)]
        for x, y in corners
    ]
    coordinates.append(coordinates[0])
    return [coordinates]


# ============================================================
# DETECTION OVERLAY
# ============================================================

def add_overlay_boxes(
    image,
    buildings,
    matching_ids,
    roads
):

    if image.ndim == 2:

        output = np.stack(
            [
                image,
                image,
                image
            ],
            axis=2
        ).copy()

    else:

        output = (
            image[:, :, :3]
            .copy()
        )

    output = output.astype(
        np.uint8
    )

    # Buildings
    for index, building in enumerate(
        buildings
    ):

        min_x = building["min_x"]
        max_x = building["max_x"]

        min_y = building["min_y"]
        max_y = building["max_y"]

        if index in matching_ids:

            thickness = 4

            values = np.array(
                [255, 60, 60],
                dtype=np.uint8
            )

        else:

            thickness = 2

            values = np.array(
                [255, 180, 0],
                dtype=np.uint8
            )

        for t in range(thickness):

            y1 = max(
                0,
                min_y - t
            )

            y2 = min(
                output.shape[0] - 1,
                max_y + t
            )

            x1 = max(
                0,
                min_x - t
            )

            x2 = min(
                output.shape[1] - 1,
                max_x + t
            )

            output[
                y1,
                x1:x2 + 1
            ] = values

            output[
                y2,
                x1:x2 + 1
            ] = values

            output[
                y1:y2 + 1,
                x1
            ] = values

            output[
                y1:y2 + 1,
                x2
            ] = values

    # Roads
    for road in roads:

        min_x = road["min_x"]
        max_x = road["max_x"]

        min_y = road["min_y"]
        max_y = road["max_y"]

        values = np.array(
            [0, 220, 255],
            dtype=np.uint8
        )

        output[
            min_y:min(
                min_y + 3,
                output.shape[0]
            ),
            min_x:max_x + 1
        ] = values

        output[
            max(
                0,
                max_y - 2
            ):max_y + 1,
            min_x:max_x + 1
        ] = values

        output[
            min_y:max_y + 1,
            min_x:min(
                min_x + 3,
                output.shape[1]
            )
        ] = values

        output[
            min_y:max_y + 1,
            max(
                0,
                max_x - 2
            ):max_x + 1
        ] = values

    return output


# ============================================================
# SATELLITE IMAGE ANALYSIS
# ============================================================

def analyze_satellite_image(
    image_array,
    pixel_width,
    pixel_height,
    query_text,
    road_sensitivity=65
):

    gray = grayscale_from_image(
        image_array
    )

    image_size = gray.size


    # ========================================================
    # BUILDING DETECTION
    # ========================================================

    building_threshold = np.percentile(
        gray,
        80
    )

    building_mask = (
        gray >= building_threshold
    )

    building_components = connected_components(
        building_mask,
        min_area=max(
            12,
            int(image_size * 0.000015)
        ),
        max_area=max(
            100,
            int(image_size * 0.01)
        )
    )

    buildings = []

    for component in building_components:

        width = (
            component["max_x"]
            - component["min_x"]
            + 1
        )

        height = (
            component["max_y"]
            - component["min_y"]
            + 1
        )

        aspect = max(
            width / height,
            height / width
        )

        if width < 4:
            continue

        if height < 4:
            continue

        if aspect > 5.0:
            continue

        buildings.append(
            {
                **component,
                "center_x": (
                    component["min_x"]
                    + component["max_x"]
                ) / 2.0,

                "center_y": (
                    component["min_y"]
                    + component["max_y"]
                ) / 2.0,

                "width_px": width,
                "height_px": height
            }
        )

    buildings = sorted(
        buildings,
        key=lambda item: item["area"],
        reverse=True
    )[:250]


    # ========================================================
    # IMPROVED ROAD DETECTION
    # ========================================================

    sensitivity = int(
        np.clip(
            road_sensitivity,
            0,
            100
        )
    )

    # Dark roads
    dark_percentile = (
        52
        - (
            sensitivity
            * 12
            / 100
        )
    )

    # Bright roads
    bright_percentile = (
        55
        + (
            sensitivity
            * 12
            / 100
        )
    )

    dark_threshold = np.percentile(
        gray,
        dark_percentile
    )

    bright_threshold = np.percentile(
        gray,
        bright_percentile
    )

    dark_mask = (
        gray <= dark_threshold
    )

    bright_mask = (
        gray >= bright_threshold
    )


    min_road_area = max(
        25,
        int(image_size * 0.00002)
    )

    max_road_area = max(
        300,
        int(image_size * 0.08)
    )


    dark_components = connected_components(
        dark_mask,
        min_area=min_road_area,
        max_area=max_road_area
    )

    bright_components = connected_components(
        bright_mask,
        min_area=min_road_area,
        max_area=max_road_area
    )


    roads = []


    def add_road_candidates(
        components,
        source
    ):

        for component in components:

            width = (
                component["max_x"]
                - component["min_x"]
                + 1
            )

            height = (
                component["max_y"]
                - component["min_y"]
                + 1
            )

            aspect = max(
                width / height,
                height / width
            )

            bbox_area = (
                width * height
            )

            fill_ratio = (
                component["area"]
                / max(
                    1,
                    bbox_area
                )
            )

            # Roads should be elongated.
            if aspect < 3.0:
                continue

            if component["area"] < min_road_area:
                continue

            score = (
                np.log1p(
                    component["area"]
                )
                * min(
                    aspect,
                    25.0
                )
                * (
                    1.15
                    - min(
                        fill_ratio,
                        1.0
                    )
                )
            )

            roads.append(
                {
                    **component,
                    "aspect": aspect,
                    "fill_ratio": fill_ratio,
                    "score": float(score),
                    "source": source
                }
            )


    add_road_candidates(
        dark_components,
        "dark"
    )

    add_road_candidates(
        bright_components,
        "bright"
    )


    # ========================================================
    # REMOVE DUPLICATE ROAD BOXES
    # ========================================================

    roads.sort(
        key=lambda item: item["score"],
        reverse=True
    )

    selected_roads = []

    for candidate in roads:

        duplicate = False

        for selected in selected_roads:

            x_overlap = max(
                0,
                min(
                    candidate["max_x"],
                    selected["max_x"]
                )
                -
                max(
                    candidate["min_x"],
                    selected["min_x"]
                )
                + 1
            )

            y_overlap = max(
                0,
                min(
                    candidate["max_y"],
                    selected["max_y"]
                )
                -
                max(
                    candidate["min_y"],
                    selected["min_y"]
                )
                + 1
            )

            intersection = (
                x_overlap
                * y_overlap
            )

            candidate_area = (
                (
                    candidate["max_x"]
                    - candidate["min_x"]
                    + 1
                )
                *
                (
                    candidate["max_y"]
                    - candidate["min_y"]
                    + 1
                )
            )

            overlap_ratio = (
                intersection
                /
                max(
                    1,
                    candidate_area
                )
            )

            if overlap_ratio > 0.70:

                duplicate = True
                break

        if not duplicate:

            selected_roads.append(
                candidate
            )

        if len(selected_roads) >= 20:
            break

    roads = selected_roads


    # ========================================================
    # MAIN ROAD SELECTION
    # ========================================================

    query_lower_for_road = query_text.lower()

    main_road_requested = "main road" in query_lower_for_road

    if roads:
        if main_road_requested:
            main_road = max(
                roads,
                key=lambda item: item["score"]
            )
            buffer_roads = [main_road]
        else:
            main_road = None
            buffer_roads = roads
    else:
        main_road = None
        buffer_roads = []


    # ========================================================
    # ROAD MASK
    # ========================================================

    road_mask = np.zeros_like(
        gray,
        dtype=bool
    )

    for road in buffer_roads:

        road_mask[
            road["min_y"]:
            road["max_y"] + 1,

            road["min_x"]:
            road["max_x"] + 1
        ] = True


    # Connect fragmented road candidates.
    if buffer_roads:

        connection_radius = max(
            1,
            min(
                4,
                int(
                    round(
                        min(gray.shape)
                        * 0.004
                    )
                )
            )
        )

        road_mask = dilate_mask(
            road_mask,
            connection_radius
        )


    # ========================================================
    # DISTANCE BUFFER
    # ========================================================

    distance_m = extract_distance_m(
        query_text
    )

    if distance_m is None:
        distance_m = 100.0


    average_pixel_size = (
        abs(float(pixel_width))
        +
        abs(float(pixel_height))
    ) / 2.0

    if average_pixel_size <= 0:
        average_pixel_size = 1.0


    radius_pixels = int(
        np.ceil(
            distance_m
            /
            average_pixel_size
        )
    )

    radius_pixels = max(
        1,
        min(
            radius_pixels,
            max(gray.shape)
        )
    )


    buffer_mask = make_road_buffer(
        road_mask,
        radius_pixels
    )


    # ========================================================
    # FIND BUILDINGS INSIDE BUFFER
    # ========================================================

    matching_ids = []

    for index, building in enumerate(
        buildings
    ):

        x = int(
            round(
                building["center_x"]
            )
        )

        y = int(
            round(
                building["center_y"]
            )
        )

        x = min(
            max(
                x,
                0
            ),
            gray.shape[1] - 1
        )

        y = min(
            max(
                y,
                0
            ),
            gray.shape[0] - 1
        )

        if buffer_mask[y, x]:

            matching_ids.append(
                index
            )


    return {
        "gray": gray,
        "buildings": buildings,
        "roads": roads,
        "road_mask": road_mask,
        "buffer_mask": buffer_mask,
        "matching_ids": matching_ids,
        "distance_m": distance_m,
        "radius_pixels": radius_pixels,
        "dark_threshold": float(
            dark_threshold
        ),
        "bright_threshold": float(
            bright_threshold
        ),
        "main_road_requested": main_road_requested
    }


# ============================================================
# GEOJSON
# ============================================================

def build_geojson(
    buildings,
    matching_ids,
    transform,
    crs
):

    features = []

    for output_id, building_index in enumerate(
        matching_ids,
        start=1
    ):

        building = buildings[
            building_index
        ]

        geometry = {
            "type": "Polygon",
            "coordinates": pixel_polygon(
                transform,
                building["min_x"],
                building["min_y"],
                building["max_x"],
                building["max_y"]
            )
        }

        features.append(
            {
                "type": "Feature",

                "properties": {
                    "id": output_id,
                    "source_detection_id":
                        building_index + 1,
                    "area_pixels":
                        building["area"],
                    "width_pixels":
                        building["width_px"],
                    "height_pixels":
                        building["height_px"],
                    "within_query_distance":
                        True
                },

                "geometry": geometry
            }
        )

    return {
        "type": "FeatureCollection",

        "name":
            "SatQuery_AI_buildings",

        "crs": (
            {
                "type": "name",
                "properties": {
                    "name": str(crs)
                }
            }
            if crs
            else None
        ),

        "features": features
    }


# ============================================================
# CUSTOM CSS
# ============================================================

def apply_custom_css():
    st.markdown("""
    <style>
    /* Dark Theme Basics */
    .stApp {
        background-color: #0B0F19;
        color: #E2E8F0;
    }
    /* Hide Default Elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {background-color: transparent !important;}
    
    /* Cards */
    .metric-card {
        background: linear-gradient(145deg, #111827, #1F2937);
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #374151;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        text-align: center;
        transition: transform 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: #3B82F6;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #60A5FA;
        margin: 10px 0;
    }
    .metric-title {
        font-size: 0.9rem;
        color: #9CA3AF;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .metric-subtitle {
        font-size: 0.8rem;
        color: #6B7280;
    }
    
    /* Legend */
    .legend-item {
        display: inline-block;
        margin-right: 15px;
        font-size: 0.9rem;
    }
    .legend-box {
        display: inline-block;
        width: 12px;
        height: 12px;
        margin-right: 5px;
        border-radius: 2px;
    }
    
    /* Pipeline */
    .pipeline-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: #111827;
        padding: 15px 25px;
        border-radius: 8px;
        border: 1px solid #374151;
        margin-bottom: 20px;
    }
    .pipeline-step {
        text-align: center;
        font-size: 0.85rem;
        color: #6B7280;
        position: relative;
        flex: 1;
    }
    .pipeline-step.active {
        color: #10B981;
        font-weight: bold;
    }
    .pipeline-arrow {
        color: #374151;
    }
    
    /* Status */
    .status-online {
        color: #10B981;
        font-size: 0.8rem;
        font-weight: bold;
        letter-spacing: 1px;
    }
    
    /* Buttons */
    .stButton>button {
        border-radius: 6px;
        font-weight: bold;
        border: none;
        transition: all 0.2s;
    }
    .stButton>button:hover {
        box-shadow: 0 0 10px rgba(59, 130, 246, 0.5);
    }
    
    /* Map Container */
    .map-container {
        border: 1px solid #374151;
        border-radius: 12px;
        padding: 10px;
        background: #111827;
    }
    </style>
    """, unsafe_allow_html=True)

apply_custom_css()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("""
    <h3 style='margin-bottom:0;'>🛰️ SATQUERY AI</h3>
    <div class='status-online'>● SYSTEM ONLINE</div>
    """, unsafe_allow_html=True)
    st.divider()
    
    st.markdown("**INPUT**<br><span style='color:#9CA3AF'>Satellite GeoTIFF</span>", unsafe_allow_html=True)
    st.markdown("<br>**ANALYSIS**<br><span style='color:#9CA3AF'>AI Vision + GIS</span>", unsafe_allow_html=True)
    st.markdown("<br>**OUTPUT**<br><span style='color:#9CA3AF'>Detection Map<br>GeoJSON</span>", unsafe_allow_html=True)
    
    st.divider()
    st.header("⚙️ Settings")
    road_sensitivity = st.slider(
        "Road detection sensitivity",
        min_value=0, max_value=100, value=65,
        help="Increase this if roads are not being detected. It considers a wider range of dark and bright linear structures."
    )


# ============================================================
# HEADER
# ============================================================

st.markdown("""
    <div style='margin-bottom: 2rem;'>
        <h1 style='margin-bottom: 0;'>🛰️ SATQUERY AI</h1>
        <h3 style='color: #60A5FA; margin-top: 0;'>AI-Powered Satellite Intelligence & Geospatial Reasoning</h3>
        <p style='color: #9CA3AF;'>Ask natural-language questions about satellite imagery and receive spatially-aware results.</p>
    </div>
""", unsafe_allow_html=True)


# ============================================================
# MAIN LAYOUT
# ============================================================

col_upload, col_query = st.columns([1, 1])

with col_upload:
    st.markdown("### 🛰️ Satellite Image")
    st.markdown("<p style='color: #9CA3AF; font-size: 0.9rem;'>Upload a georeferenced GeoTIFF for analysis.</p>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Accepted format: GeoTIFF", type=["jpg", "jpeg", "png", "tif", "tiff"], label_visibility="collapsed")
    
    if uploaded_file:
        st.markdown(f"""
        <div style='background: #111827; border: 1px solid #10B981; border-radius: 8px; padding: 15px;'>
            <div style='color: #10B981; font-weight: bold; margin-bottom: 5px;'>✓ Image loaded</div>
            <div style='color: #E2E8F0; font-size: 0.9rem;'>{uploaded_file.name}</div>
            <div style='color: #9CA3AF; font-size: 0.8rem;'>{uploaded_file.type} • ready for analysis</div>
        </div>
        """, unsafe_allow_html=True)

with col_query:
    st.markdown("### 🔎 Query Workspace")
    st.markdown("<p style='color: #9CA3AF; font-size: 0.9rem;'>Describe what you want to find in the satellite image.</p>", unsafe_allow_html=True)
    query = st.text_area("Prompt", placeholder="Example: Show all buildings within 100 metres of the main road.", label_visibility="collapsed")
    
    analyze_pressed = st.button("🚀 Analyze Image", type="primary", use_container_width=True)
    st.markdown("<p style='text-align: center; color: #6B7280; font-size: 0.8rem;'>Supports natural-language spatial queries.</p>", unsafe_allow_html=True)


# ============================================================
# PROCESSING PIPELINE
# ============================================================

file_extension = None

if analyze_pressed or query.strip():
    if not query.strip():
        st.info("Enter a natural-language question to begin analysis.")
    elif uploaded_file is None:
        st.error("❌ Please upload a satellite image before analysis.")
    else:
        file_extension = uploaded_file.name.lower().split(".")[-1]
        
        if file_extension not in ["tif", "tiff"]:
            st.warning("⚠️ This spatial query requires a georeferenced GeoTIFF.")
        else:
            # Show pipeline visualization
            st.markdown("""
            <div class='pipeline-container'>
                <div class='pipeline-step active'>01 Natural Language</div><div class='pipeline-arrow'>→</div>
                <div class='pipeline-step active'>02 Vision Analysis</div><div class='pipeline-arrow'>→</div>
                <div class='pipeline-step active'>03 Georeferencing</div><div class='pipeline-arrow'>→</div>
                <div class='pipeline-step active'>04 GIS Reasoning</div><div class='pipeline-arrow'>→</div>
                <div class='pipeline-step active'>05 Detection Map</div><div class='pipeline-arrow'>→</div>
                <div class='pipeline-step active'>06 GeoJSON</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # QUERY UNDERSTANDING
            query_lower = query.lower()
            has_building = "building" in query_lower or "buildings" in query_lower
            has_road = "road" in query_lower or "roads" in query_lower
            distance_m = extract_distance_m(query)
            
            if distance_m is None and has_road and has_building:
                distance_m = 100.0
                
            if not (has_building and has_road):
                st.info("💡 The current spatial prototype is configured for building + road queries, for example: 'Show all buildings within 100 metres of the main road.'")
            else:
                st.markdown("### 🧠 Query Understanding")
                
                # Query understanding cards
                qc1, qc2, qc3 = st.columns(3)
                with qc1:
                    st.markdown(f"""
                    <div class='metric-card'>
                        <div class='metric-title'>TARGET</div>
                        <div class='metric-value'>{"Buildings" if has_building else "None"}</div>
                    </div>
                    """, unsafe_allow_html=True)
                with qc2:
                    st.markdown(f"""
                    <div class='metric-card'>
                        <div class='metric-title'>REFERENCE</div>
                        <div class='metric-value'>{"Main Road" if "main road" in query_lower else ("Road" if has_road else "None")}</div>
                    </div>
                    """, unsafe_allow_html=True)
                with qc3:
                    st.markdown(f"""
                    <div class='metric-card'>
                        <div class='metric-title'>DISTANCE</div>
                        <div class='metric-value'>{f"{distance_m:g} metres" if distance_m else "N/A"}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                try:
                    uploaded_file.seek(0)
                    with rasterio.open(uploaded_file) as src:
                        if src.crs is None:
                            st.error("❌ The uploaded GeoTIFF has no CRS. Real-world distance calculations cannot be performed reliably.")
                        else:
                            pixel_width = abs(src.transform.a)
                            pixel_height = abs(src.transform.e)
                            
                            # Vision Analysis
                            if src.count >= 3:
                                raw = src.read([1, 2, 3])
                                analysis_array = np.transpose(raw, (1, 2, 0))
                            else:
                                analysis_array = src.read(1)
                                
                            analysis = analyze_satellite_image(
                                analysis_array, pixel_width, pixel_height, query, road_sensitivity
                            )
                            
                            buildings = analysis["buildings"]
                            roads = analysis["roads"]
                            matching_ids = analysis["matching_ids"]
                            
                            st.markdown("---")
                            
                            # RESULTS
                            st.markdown("### 🎯 Vision Analysis Results")
                            rc1, rc2, rc3 = st.columns(3)
                            
                            with rc1:
                                st.markdown(f"""
                                <div class='metric-card'>
                                    <div class='metric-title'>🏢 BUILDING CANDIDATES</div>
                                    <div class='metric-value'>{len(buildings)}</div>
                                    <div class='metric-subtitle'>Detected</div>
                                </div>
                                """, unsafe_allow_html=True)
                            with rc2:
                                st.markdown(f"""
                                <div class='metric-card'>
                                    <div class='metric-title'>🛣 ROAD CANDIDATES</div>
                                    <div class='metric-value'>{len(roads)}</div>
                                    <div class='metric-subtitle'>Detected</div>
                                </div>
                                """, unsafe_allow_html=True)
                            with rc3:
                                st.markdown(f"""
                                <div class='metric-card'>
                                    <div class='metric-title'>📍 MATCHING BUILDINGS</div>
                                    <div class='metric-value'>{len(matching_ids)}</div>
                                    <div class='metric-subtitle'>Within query distance</div>
                                </div>
                                """, unsafe_allow_html=True)
                                
                            if analysis.get("main_road_requested", False):
                                st.info("🛣️ Main-road query: the strongest detected road candidate was used as the reference road.")
                                
                            if not roads:
                                st.warning("⚠️ No road candidates were detected in this image. Try increasing the Road Detection Sensitivity in the sidebar.")
                                
                            st.markdown("---")
                            
                            # GEOSPATIAL REASONING
                            st.markdown("### 🌍 Geospatial Reasoning")
                            average_pixel_size = (pixel_width + pixel_height) / 2.0
                            
                            gc1, gc2 = st.columns(2)
                            with gc1:
                                st.markdown(f"""
                                <div style='background: #111827; border: 1px solid #374151; border-radius: 8px; padding: 15px;'>
                                    <div style='color: #9CA3AF; font-size: 0.85rem; margin-bottom: 5px;'>CRS</div>
                                    <div style='color: #E2E8F0; font-family: monospace;'>{src.crs}</div>
                                    <div style='color: #9CA3AF; font-size: 0.85rem; margin-top: 15px; margin-bottom: 5px;'>Pixel Size</div>
                                    <div style='color: #E2E8F0; font-family: monospace;'>{average_pixel_size:.4f} map units/pixel</div>
                                </div>
                                """, unsafe_allow_html=True)
                                
                            with gc2:
                                st.markdown(f"""
                                <div style='background: #111827; border: 1px solid #374151; border-radius: 8px; padding: 15px;'>
                                    <div style='color: #9CA3AF; font-size: 0.85rem; margin-bottom: 5px;'>Requested Distance</div>
                                    <div style='color: #E2E8F0; font-family: monospace;'>{analysis['distance_m']:g} metres</div>
                                    <div style='color: #9CA3AF; font-size: 0.85rem; margin-top: 15px; margin-bottom: 5px;'>Buffer Radius</div>
                                    <div style='color: #E2E8F0; font-family: monospace;'>{analysis['radius_pixels']} pixels</div>
                                </div>
                                """, unsafe_allow_html=True)
                                
                            if roads:
                                msg = "Main road detected and the requested distance buffer was created." if analysis.get("main_road_requested", False) else "Road candidates detected and the requested distance buffer was created."
                                st.markdown(f"<div style='color: #10B981; margin-top: 10px; font-size: 0.9rem;'>✓ {msg}</div>", unsafe_allow_html=True)
                            else:
                                st.warning("⚠️ Buffer contains no road candidate because no road was detected.")
                                
                            st.markdown("---")
                            
                            # DETECTION MAP
                            st.markdown("### 🗺️ Detection Intelligence Map")
                            st.markdown("""
                            <div style='margin-bottom: 10px;'>
                                <div class='legend-item'><div class='legend-box' style='background: #FF3C3C;'></div>🟥 Matching buildings</div>
                                <div class='legend-item'><div class='legend-box' style='background: #FFB400;'></div>🟨 Other building candidates</div>
                                <div class='legend-item'><div class='legend-box' style='background: #00DCFF;'></div>🟦 Road candidates</div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            overlay = add_overlay_boxes(
                                scale_to_uint8(analysis_array), buildings, set(matching_ids), roads
                            )
                            
                            st.markdown("<div class='map-container'>", unsafe_allow_html=True)
                            st.image(overlay, use_container_width=True)
                            st.markdown("</div>", unsafe_allow_html=True)
                            
                            st.markdown("---")
                            
                            # BUILDINGS TABLE
                            st.markdown(f"### 🏢 Buildings Within {analysis['distance_m']:g} Metres")
                            st.markdown(f"<p style='color: #9CA3AF;'>{len(matching_ids)} buildings were identified within {analysis['distance_m']:g} metres of the detected road.</p>", unsafe_allow_html=True)
                            
                            if matching_ids:
                                rows = []
                                for result_id, building_index in enumerate(matching_ids, start=1):
                                    building = buildings[building_index]
                                    center_x = int(round(building["center_x"]))
                                    center_y = int(round(building["center_y"]))
                                    world_x, world_y = pixel_to_map(
                                        src.transform, center_y + 0.5, center_x + 0.5
                                    )
                                    rows.append({
                                        "Building": result_id,
                                        "Pixel X": center_x,
                                        "Pixel Y": center_y,
                                        "Map X": round(float(world_x), 3),
                                        "Map Y": round(float(world_y), 3),
                                        "Area (pixels)": building["area"]
                                    })
                                st.dataframe(rows, use_container_width=True)
                            else:
                                st.warning("No building candidates were found inside the requested buffer.")
                                
                            st.markdown("---")
                            
                            # GEOJSON
                            st.markdown("### 📦 Geospatial Export")
                            st.markdown("<p style='color: #9CA3AF;'>Export the detected buildings as GeoJSON for use in GIS software.</p>", unsafe_allow_html=True)
                            
                            geojson = build_geojson(buildings, matching_ids, src.transform, src.crs)
                            geojson_text = json.dumps(geojson, indent=2)
                            
                            export_col1, export_col2 = st.columns([1, 2])
                            with export_col1:
                                st.download_button(
                                    label="⬇️ Download GeoJSON",
                                    data=geojson_text,
                                    file_name="satquery_buildings.geojson",
                                    mime="application/geo+json",
                                    type="primary",
                                    use_container_width=True
                                )
                                st.markdown("<div style='color: #10B981; text-align: center; font-size: 0.85rem; margin-top: 10px;'>✓ GeoJSON generated successfully</div>", unsafe_allow_html=True)
                                
                            with st.expander("View generated GeoJSON"):
                                st.code(geojson_text, language="json")
                                
                except Exception as error:
                    st.markdown("""
                    <div style='background: rgba(239, 68, 68, 0.1); border: 1px solid #EF4444; border-radius: 8px; padding: 15px; margin-top: 20px;'>
                        <h4 style='color: #EF4444; margin-top: 0;'>⚠ Analysis Error</h4>
                    """, unsafe_allow_html=True)
                    st.error(f"{error}")
                    st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# FOOTER
# ============================================================
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
    <div style='text-align: center; color: #6B7280; font-size: 0.85rem; padding: 20px 0; border-top: 1px solid #374151;'>
        <strong>SatQuery AI</strong><br>
        AI-powered satellite intelligence & geospatial reasoning<br>
        Prototype • Remote Sensing • GIS
    </div>
""", unsafe_allow_html=True)
