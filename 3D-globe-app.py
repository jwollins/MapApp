import os
import glob
import random
import hashlib
import math

import pandas as pd
import streamlit as st
import geopandas as gpd
import pydeck as pdk


# ----------------------------
# Streamlit setup (must be first)
# ----------------------------
st.set_page_config(page_title="Multi-layer Map Viewer", layout="wide")
st.title("🗺️ Multi-layer Map Viewer (auto-load from ./files)")


# ----------------------------
# Mapbox token (secrets first)
# ----------------------------
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
    # ---- CARTO (NO TOKEN REQUIRED) ----
    "Light (Carto Positron)": "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
    "Dark (Carto Dark Matter)": "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
    "Voyager (Carto Streets)": "https://basemaps.cartocdn.com/gl/voyager-gl-style/style.json",

    # ---- OPTIONAL: MAPBOX (ONLY WORKS IF TOKEN VALID) ----
    "Mapbox Streets": "mapbox://styles/mapbox/streets-v12",
    "Mapbox Satellite + Labels": "mapbox://styles/mapbox/satellite-streets-v12",
}


# ----------------------------
# Helpers
# ----------------------------
def layer_color(name: str, alpha: int = 120) -> list[int]:
    h = hashlib.md5(name.encode("utf-8")).hexdigest()
    r = 40 + (int(h[0:2], 16) % 171)
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
            g = int(40 + t * 180)
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

        if gdf.crs is None:
            gdf = gdf.set_crs(epsg=default_epsg_if_missing)

        gdf = gdf.to_crs(epsg=4326)
        gdf["source_file"] = name
        layers[name] = gdf
    return layers


def zoom_from_bounds(bounds, max_zoom=16, min_zoom=1):
    minx, miny, maxx, maxy = bounds
    lon_span = max(1e-9, abs(maxx - minx))
    lat_span = max(1e-9, abs(maxy - miny))
    span = max(lon_span, lat_span)
    z = math.log2(360.0 / span)
    return float(max(min_zoom, min(max_zoom, z)))


# ----------------------------
# Load geodata from ./files
# ----------------------------
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


# ----------------------------
# Sidebar controls
# ----------------------------
st.sidebar.header("Controls")

style_name = st.sidebar.selectbox("Background style", list(STYLE_OPTIONS.keys()), index=0)
map_style = STYLE_OPTIONS[style_name]

marker_px = st.sidebar.slider("Marker size (px)", 2, 40, 10, 1)
poly_opacity = st.sidebar.slider("Polygon opacity", 0, 255, 90, 5)
marker_opacity = st.sidebar.slider("Marker opacity", 0, 255, 230, 5)

st.sidebar.subheader("Layers to display")
enabled_names = [lname for lname in layers_by_name.keys() if st.sidebar.checkbox(lname, value=True)]

if not enabled_names:
    st.warning("No layers selected.")
    st.stop()

st.sidebar.subheader("Layer colors")
layer_rgb: dict[str, list[int]] = {}
for lname in enabled_names:
    default_rgb = layer_color(lname, alpha=255)[:3]
    chosen_hex = st.sidebar.color_picker(lname, value=to_hex(default_rgb), key=f"color_{lname}")
    layer_rgb[lname] = hex_to_rgb(chosen_hex)

# IMPORTANT: default OFF on Cloud (GlobeView can blank the map)
use_globe = st.sidebar.checkbox("Try Globe view (may not work on Cloud)", value=False)


# ----------------------------
# Tabs
# ----------------------------
tab_map, tab_info = st.tabs(["Map", "Information"])

with tab_info:
    st.subheader("Info / styling")
    selected_layer_name = st.selectbox("Select a layer", enabled_names)
    sel_gdf = layers_by_name[selected_layer_name]

    cols = [c for c in sel_gdf.columns if c != "geometry"]
    if not cols:
        tooltip_fields = ["source_file"]
        color_field = None
        st.info("This layer has no attribute columns (besides geometry).")
    else:
        tooltip_fields = st.multiselect(
            "Tooltip fields",
            options=cols,
            default=[c for c in ["source_file"] if c in cols] or [cols[0]],
        )
        color_field = st.selectbox("Colour polygons by attribute (selected layer only)",
                                   options=["(none)"] + cols, index=0)
        if color_field == "(none)":
            color_field = None

    st.dataframe(sel_gdf[cols].head(50) if cols else sel_gdf.head(50), use_container_width=True)


# ----------------------------
# Tooltip config
# ----------------------------
if not tooltip_fields:
    tooltip_fields = ["source_file"]

tooltip_html = "".join([f"<b>{f}:</b> " + "{properties." + f + "}<br/>" for f in tooltip_fields])
tooltip = {"html": tooltip_html, "style": {"backgroundColor": "rgba(0,0,0,0.85)", "color": "white"}}


# ----------------------------
# Center + zoom
# ----------------------------
enabled_gdfs = [layers_by_name[n] for n in enabled_names]
combined = gpd.GeoDataFrame(pd.concat(enabled_gdfs, ignore_index=True), crs="EPSG:4326")

bounds = combined.total_bounds
center_lon = float((bounds[0] + bounds[2]) / 2.0)
center_lat = float((bounds[1] + bounds[3]) / 2.0)
zoom = 1.5

# Helpful debug
st.caption(f"Center: {center_lat:.6f}, {center_lon:.6f} | Zoom: {zoom:.2f}")


# ----------------------------
# Build layers
# ----------------------------
deck_layers = []

for lname in enabled_names:
    gdf = layers_by_name[lname]
    rgb = layer_rgb.get(lname, layer_color(lname, alpha=255)[:3])

    # polygon styling
    if lname == selected_layer_name and color_field and color_field in gdf.columns:
        gdf_local = gdf.copy()
        s = gdf_local[color_field]
        gdf_local["_rgba"] = (
            rgba_for_attribute_numeric(s, alpha=poly_opacity)
            if pd.api.types.is_numeric_dtype(s)
            else rgba_for_attribute_categorical(s, alpha=poly_opacity)
        )
        geojson = gdf_local.__geo_interface__
        fill = "properties._rgba"
    else:
        geojson = gdf.__geo_interface__
        fill = [rgb[0], rgb[1], rgb[2], poly_opacity]

    deck_layers.append(
        pdk.Layer(
            "GeoJsonLayer",
            data=geojson,
            stroked=True,
            filled=True,
            get_fill_color=fill,
            get_line_color=[0, 0, 0, 200],
            get_line_width=2,
            line_width_min_pixels=1,
            pickable=True,
            auto_highlight=True,
        )
    )

    # marker layer
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


# ----------------------------
# Render
# ----------------------------
with tab_map:
    views = [pdk.View(type="GlobeView")] if use_globe else [pdk.View(type="MapView")]

    deck = pdk.Deck(
        layers=deck_layers,
        initial_view_state=pdk.ViewState(
            longitude=center_lon,
            latitude=center_lat,
            zoom=zoom,
            pitch=55,   # gives “3D” feel on MapView
            bearing=0,
        ),
        tooltip=tooltip,
        map_provider="mapbox",
        map_style=map_style,
        views=views,
    )

    st.pydeck_chart(deck, use_container_width=True)