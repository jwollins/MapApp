import os
import glob
import random
import hashlib
import math

import pandas as pd
import streamlit as st
import geopandas as gpd
import pydeck as pdk


# --------------------------------------------------
# Streamlit setup (MUST be first Streamlit call)
# --------------------------------------------------
st.set_page_config(page_title="Multi-layer Globe Viewer", layout="wide")
st.title("🌍 Multi-layer Globe Viewer (auto-load from ./files)")


# --------------------------------------------------
# Mapbox token (prefer Streamlit secrets, fallback to env)
# --------------------------------------------------
MAPBOX_KEY = (
    st.secrets.get("MAPBOX_API_KEY")
    or st.secrets.get("MAPBOX_ACCESS_TOKEN")
    or os.getenv("MAPBOX_API_KEY")
    or os.getenv("MAPBOX_ACCESS_TOKEN")
)

st.write("Token loaded:", bool(MAPBOX_KEY))

if MAPBOX_KEY:
    pdk.settings.mapbox_api_key = MAPBOX_KEY
else:
    st.warning("No Mapbox token detected (MAPBOX_API_KEY / MAPBOX_ACCESS_TOKEN). Basemap may be blank.")


STYLE_OPTIONS = {
    "Satellite": "mapbox://styles/mapbox/satellite-v9",
    "Satellite + Labels": "mapbox://styles/mapbox/satellite-streets-v12",
    "Light": "mapbox://styles/mapbox/light-v11",
    "Dark": "mapbox://styles/mapbox/dark-v11",
    "Outdoors": "mapbox://styles/mapbox/outdoors-v12",
}


# --------------------------------------------------
# Helpers
# --------------------------------------------------
def layer_color(name: str, alpha: int = 120) -> list[int]:
    """Stable RGBA color derived from the layer filename."""
    h = hashlib.md5(name.encode("utf-8")).hexdigest()
    r = 40 + (int(h[0:2], 16) % 171)  # 40..210
    g = 40 + (int(h[2:4], 16) % 171)
    b = 40 + (int(h[4:6], 16) % 171)
    return [r, g, b, alpha]


def to_hex(rgb: list[int]) -> str:
    return "#%02x%02x%02x" % (rgb[0], rgb[1], rgb[2])


def hex_to_rgb(hex_color: str) -> list[int]:
    h = hex_color.lstrip("#")
    return [int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)]


def rgba_for_attribute_numeric(values: pd.Series, alpha: int) -> list[list[int]]:
    s = pd.to_numeric(values, errors="coerce")
    if s.count() == 0:
        return [[150, 150, 150, alpha] for _ in range(len(values))]

    vmin, vmax = float(s.min()), float(s.max())
    if vmax == vmin:
        return [[120, 120, 120, alpha] for _ in range(len(values))]

    out: list[list[int]] = []
    for v in s:
        if pd.isna(v):
            out.append([150, 150, 150, alpha])
        else:
            t = (float(v) - vmin) / (vmax - vmin)
            g = int(40 + t * 180)  # grayscale ramp
            out.append([g, g, g, alpha])
    return out


def rgba_for_attribute_categorical(values: pd.Series, alpha: int) -> list[list[int]]:
    out: list[list[int]] = []
    for v in values.astype(str):
        rnd = random.Random(v)
        out.append([rnd.randint(40, 210), rnd.randint(40, 210), rnd.randint(40, 210), alpha])
    return out


@st.cache_data(show_spinner=False)
def load_layers(file_paths: list[str], default_epsg_if_missing: int = 27700) -> dict[str, gpd.GeoDataFrame]:
    layers: dict[str, gpd.GeoDataFrame] = {}
    for path in sorted(file_paths):
        name = os.path.basename(path)
        gdf = gpd.read_file(path)
        gdf = gdf[gdf.geometry.notnull()].copy()

        # If CRS missing, assume UK BNG (change if needed)
        if gdf.crs is None:
            gdf = gdf.set_crs(epsg=default_epsg_if_missing)

        # Mapbox expects lon/lat (EPSG:4326)
        gdf = gdf.to_crs(epsg=4326)
        gdf["source_file"] = name
        layers[name] = gdf
    return layers


def zoom_from_bounds(bounds, max_zoom=16, min_zoom=1):
    """
    Rough zoom estimator from lon/lat bounds.
    bounds = [minx, miny, maxx, maxy] in EPSG:4326.
    """
    minx, miny, maxx, maxy = bounds
    # avoid zero span
    lon_span = max(1e-9, abs(maxx - minx))
    lat_span = max(1e-9, abs(maxy - miny))
    span = max(lon_span, lat_span)

    # Heuristic: smaller span -> larger zoom
    # 360 degrees ~ zoom 0; ~0.01 deg ~ zoom ~ 14-ish
    z = math.log2(360.0 / span)
    return float(max(min_zoom, min(max_zoom, z)))


# --------------------------------------------------
# Load files from folder
# --------------------------------------------------
DATA_FOLDER = "files"
paths = (
    glob.glob(os.path.join(DATA_FOLDER, "*.shp"))
    + glob.glob(os.path.join(DATA_FOLDER, "*.geojson"))
    + glob.glob(os.path.join(DATA_FOLDER, "*.json"))
)

if not paths:
    st.error(f"No .shp/.geojson/.json files found in ./{DATA_FOLDER}/")
    st.stop()

layers_by_name = load_layers(paths)


# --------------------------------------------------
# Sidebar controls
# --------------------------------------------------
st.sidebar.header("Controls")

style_name = st.sidebar.selectbox("Background style", list(STYLE_OPTIONS.keys()), index=1)
map_style = STYLE_OPTIONS[style_name]

marker_px = st.sidebar.slider("Overview marker size (px)", 4, 40, 12, 1)
poly_opacity = st.sidebar.slider("Polygon opacity", 0, 255, 90, 5)
marker_opacity = st.sidebar.slider("Marker opacity", 0, 255, 230, 5)

st.sidebar.subheader("Layers to display")
enabled_names = []
for lname in layers_by_name.keys():
    if st.sidebar.checkbox(lname, value=True):
        enabled_names.append(lname)

if not enabled_names:
    st.warning("No layers selected. Enable at least one layer in the sidebar.")
    st.stop()

st.sidebar.subheader("Layer colors (auto defaults + manual override)")
layer_rgb: dict[str, list[int]] = {}
for lname in enabled_names:
    default_rgb = layer_color(lname, alpha=255)[:3]
    chosen_hex = st.sidebar.color_picker(lname, value=to_hex(default_rgb), key=f"color_{lname}")
    layer_rgb[lname] = hex_to_rgb(chosen_hex)

use_globe = st.sidebar.checkbox("Globe view", value=True)


# --------------------------------------------------
# Tabs
# --------------------------------------------------
tab_map, tab_info = st.tabs(["Map", "Information"])

with tab_info:
    st.subheader("Information options")

    selected_layer_name = st.selectbox("Select a layer", enabled_names)
    sel_gdf = layers_by_name[selected_layer_name]

    cols = [c for c in sel_gdf.columns if c != "geometry"]
    if not cols:
        st.info("This layer has no attribute columns (besides geometry).")
        tooltip_fields = ["source_file"]
        color_field = None
    else:
        st.markdown("### Tooltip fields")
        tooltip_fields = st.multiselect(
            "Choose which fields to show on hover",
            options=cols,
            default=[c for c in ["source_file"] if c in cols] or [cols[0]],
        )

        st.markdown("### Styling (optional)")
        color_field = st.selectbox(
            "Colour polygons by attribute (applies to selected layer only)",
            options=["(none)"] + cols,
            index=0,
        )
        if color_field == "(none)":
            color_field = None

    st.markdown("### Preview data")
    st.dataframe(sel_gdf[cols].head(50) if cols else sel_gdf.head(50), use_container_width=True)


# --------------------------------------------------
# Tooltip
# --------------------------------------------------
if not tooltip_fields:
    tooltip_fields = ["source_file"]

tooltip_html = "".join([f"<b>{f}:</b> " + "{properties." + f + "}<br/>" for f in tooltip_fields])
tooltip = {
    "html": tooltip_html,
    "style": {"backgroundColor": "rgba(0,0,0,0.85)", "color": "white"},
}


# --------------------------------------------------
# Center + zoom on enabled layers
# --------------------------------------------------
enabled_gdfs = [layers_by_name[n] for n in enabled_names]
combined = gpd.GeoDataFrame(pd.concat(enabled_gdfs, ignore_index=True), crs="EPSG:4326")

bounds = combined.total_bounds  # [minx, miny, maxx, maxy]
center_lon = float((bounds[0] + bounds[2]) / 2.0)
center_lat = float((bounds[1] + bounds[3]) / 2.0)
zoom = zoom_from_bounds(bounds)


# --------------------------------------------------
# Build Deck layers
# --------------------------------------------------
deck_layers = []

for lname in enabled_names:
    gdf = layers_by_name[lname]
    rgb = layer_rgb.get(lname, layer_color(lname, alpha=255)[:3])

    # Optionally colour polygons by attribute for selected layer
    if lname == selected_layer_name and color_field and color_field in gdf.columns:
        gdf_local = gdf.copy()
        s = gdf_local[color_field]
        if pd.api.types.is_numeric_dtype(s):
            gdf_local["_rgba"] = rgba_for_attribute_numeric(s, alpha=poly_opacity)
        else:
            gdf_local["_rgba"] = rgba_for_attribute_categorical(s, alpha=poly_opacity)

        geojson = gdf_local.__geo_interface__
        shape_layer = pdk.Layer(
            "GeoJsonLayer",
            data=geojson,
            stroked=True,
            filled=True,
            get_fill_color="properties._rgba",
            get_line_color=[0, 0, 0, 200],
            get_line_width=2,
            line_width_min_pixels=1,
            pickable=True,
            auto_highlight=True,
        )
    else:
        geojson = gdf.__geo_interface__
        shape_layer = pdk.Layer(
            "GeoJsonLayer",
            data=geojson,
            stroked=True,
            filled=True,
            get_fill_color=[rgb[0], rgb[1], rgb[2], poly_opacity],
            get_line_color=[0, 0, 0, 200],
            get_line_width=2,
            line_width_min_pixels=1,
            pickable=True,
            auto_highlight=True,
        )

    deck_layers.append(shape_layer)

    # Markers
    pts = gdf.copy()
    pts["geometry"] = pts.geometry.representative_point()
    pts["lon"] = pts.geometry.x.astype(float)
    pts["lat"] = pts.geometry.y.astype(float)

    keep_cols = ["lon", "lat"] + [c for c in tooltip_fields if c in pts.columns]
    marker_records = pts[keep_cols].to_dict("records")

    deck_layers.append(
        pdk.Layer(
            "ScatterplotLayer",
            data=marker_records,
            get_position=["lon", "lat"],
            radius_units="pixels",
            get_radius=marker_px,
            radius_min_pixels=marker_px,
            radius_max_pixels=marker_px,
            get_fill_color=[rgb[0], rgb[1], rgb[2], marker_opacity],
            get_line_color=[255, 255, 255, min(255, marker_opacity + 10)],
            line_width_min_pixels=1,
            pickable=True,
            auto_highlight=True,
        )
    )


# --------------------------------------------------
# Render map
# --------------------------------------------------
with tab_map:
    views = [pdk.View(type="GlobeView")] if use_globe else None

    deck = pdk.Deck(
        layers=deck_layers,
        initial_view_state=pdk.ViewState(
            longitude=center_lon,
            latitude=center_lat,
            zoom=zoom,      # <-- auto zoom to your data
            pitch=35,
            bearing=0,
        ),
        tooltip=tooltip,
        map_provider="mapbox",
        map_style=map_style,
        views=views,
    )

    st.pydeck_chart(deck, use_container_width=True)

st.write("bounds:", bounds)
st.write("center:", center_lat, center_lon, "zoom:", zoom)

import pydeck
st.write("pydeck version:", pydeck.__version__)