# 1.import required libraies
import streamlit as st
import numpy as np
import math
import uuid
from astropy import units as u
from astropy.time import Time, TimeDelta
from poliastro.bodies import Earth
from poliastro.twobody.orbit import Orbit
from poliastro.util import norm
import plotly.graph_objects as go
from typing import List, Dict

# page layout 
st.set_page_config(page_title="AI Space Traffic — MVP", layout="wide")
st.title("🚀 AI-Enabled Space Traffic Management — MVP")

# ------------------------------------
# Utilities: create and sample orbits
# ------------------------------------
def create_orbit(a_km: float, ecc: float, inc_deg: float, raan_deg: float, argp_deg: float, nu_deg: float, epoch: Time = None) -> Orbit:
    if epoch is None:
        epoch = Time.now()
    return Orbit.from_classical(Earth, a_km * u.km, ecc * u.one,
                                inc_deg * u.deg, raan_deg * u.deg,
                                argp_deg * u.deg, nu_deg * u.deg, epoch=epoch)

def sample_positions(orbit: Orbit, times: List[Time]) -> np.ndarray:
    pts = []
    for t in times:
        dt = t - orbit.epoch
        # propagate returns an Orbit; .rv() returns r,v astropy Quantity
        try:
            orb_t = orbit.propagate(dt)
        except Exception:
            # fallback: try small-step propagation
            orb_t = orbit.propagate(dt)
        r, v = orb_t.rv()
        pts.append(r.to(u.km).value)
    return np.array(pts)  # shape (N,3) in km

# ------------------------
# Conjunction detection
# ------------------------
def pairwise_min_distance(positions: Dict[str, np.ndarray]) -> List[Dict]:
    # positions: {sat_id: np.ndarray(times,3) km}
    sat_ids = list(positions.keys())
    n = len(sat_ids)
    results = []
    if n < 2:
        return results
    times_len = positions[sat_ids[0]].shape[0]
    for i in range(n):
        for j in range(i+1, n):
            a = sat_ids[i]; b = sat_ids[j]
            dists_m = np.linalg.norm(positions[a] - positions[b], axis=1) * 1000.0  # m
            idx = int(np.argmin(dists_m))
            results.append({
                "sat_a": a,
                "sat_b": b,
                "min_dist_m": float(dists_m[idx]),
                "t_idx": idx
            })
    # sort by closest
    results.sort(key=lambda x: x["min_dist_m"])
    return results

def risk_from_distance(m: float) -> float:
    # heuristic risk proxy; output 0..1
    if m < 50:
        return 0.99
    if m < 200:
        return 0.8
    if m < 500:
        return 0.6
    if m < 2000:
        return 0.2
    return 0.01

# ---------------------------------------------------------
# Toy planner: along-track dv -> phase shift approximation
# ---------------------------------------------------------
def apply_along_track_phase_shift(orbit: Orbit, dv_m_s: float, duration_s: float) -> Orbit:
    """
    Toy model: convert a small along-track dv to a phase shift in true anomaly.
    This is an approximation for demonstration only.
    """
    a = orbit.a.to(u.km).value
    mu = Earth.k.to(u.km**3 / u.s**2).value
    v_circ = math.sqrt(mu / a)  # km/s approx circular velocity
    dv_km_s = dv_m_s / 1000.0
    # orbital period
    T = 2 * math.pi * math.sqrt(a**3 / mu)
    frac = duration_s / T
    phase_shift_rad = (dv_km_s / v_circ) * frac * 2 * math.pi
    phase_shift_deg = math.degrees(phase_shift_rad)
    new_nu = (orbit.nu.to(u.deg).value + phase_shift_deg) % 360.0
    new_orbit = create_orbit(a, float(orbit.ecc.value), float(orbit.inc.to(u.deg).value),
                             float(orbit.raan.to(u.deg).value), float(orbit.argp.to(u.deg).value),
                             new_nu, epoch=orbit.epoch)
    return new_orbit

# ------------------------
#    UI: parameters sidebar
# ------------------------
st.sidebar.header("Simulation Parameters")
num_sats = st.sidebar.slider("Number of satellites", 2, 40, 8)
duration_min = st.sidebar.slider("Propagation window (minutes)", 10, 720, 120)
steps = st.sidebar.slider("Time steps", 20, 800, 240)
miss_threshold_m = st.sidebar.slider("Conjunction alert threshold (m)", 10, 200, 500)

seed = st.sidebar.number_input("Random seed", value=42)
np.random.seed(int(seed))

# ---------------------------------
# Generate synthetic constellation
# ----------------------------------
st.sidebar.markdown("### Orbital generation options")
preset = st.sidebar.selectbox("Preset scenario", ["Clustered polar sun-synchronous", "Random LEO band", "Two close satellites (demo)"])

start_epoch = Time.now()
times = start_epoch + TimeDelta(np.linspace(0, duration_min * 60, steps) * u.s)

sats = {}  # id -> dict orbit and metadata

if st.sidebar.button("Generate / Reset Orbits") or "sats_generated" not in st.session_state:
    st.session_state["sats_generated"] = True
    sats = {}
    if preset == "Clustered polar sun-synchronous":
        # many similar orbits, slight phasing differences
        base_a = 7000.0  # km
        for i in range(num_sats):
            nu = float((i * (360.0 / max(1,num_sats))) + np.random.normal(0, 2.0))
            a = base_a + np.random.normal(0, 1.0)
            inc = 98.6 + np.random.normal(0, 0.1)
            raan = np.random.uniform(0, 360)
            argp = np.random.uniform(0, 360)
            sat_id = f"SAT-{i+1:03d}"
            sats[sat_id] = {"orbit": create_orbit(a, 0.001 + np.random.normal(0, 1e-4), inc, raan, argp, nu, epoch=start_epoch)}
    elif preset == "Random LEO band":
        for i in range(num_sats):
            a = np.random.uniform(6600, 7200)  # km
            ecc = np.random.uniform(0.0, 0.01)
            inc = np.random.uniform(40, 100)
            raan = np.random.uniform(0, 360)
            argp = np.random.uniform(0, 360)
            nu = np.random.uniform(0, 360)
            sat_id = f"SAT-{i+1:03d}"
            sats[sat_id] = {"orbit": create_orbit(a, ecc, inc, raan, argp, nu, epoch=start_epoch)}
    else:  # Two close satellites demo
        sats = {}
        sat1 = create_orbit(7000, 0.001, 98.6, 0.0, 0.0, 0.0, epoch=start_epoch)
        sat2 = create_orbit(7000, 0.001, 98.6, 0.0, 0.0, 0.8, epoch=start_epoch)  # close 0.8 deg separation
        sats["SAT-001"] = {"orbit": sat1}
        sats["SAT-002"] = {"orbit": sat2}
        # add extras if requested
        for i in range(2, num_sats):
            nu = np.random.uniform(0, 360)
            sats[f"SAT-{i+1:03d}"] = {"orbit": create_orbit(7100, 0.002, 53.0, np.random.uniform(0,360), np.random.uniform(0,360), nu, epoch=start_epoch)}
    st.session_state["sats"] = sats

sats = st.session_state.get("sats", {})
if not sats:
    st.info("Generate or reset orbits using the sidebar.")
    st.stop()

# -----------------------------------
# Sample positions for all satellites
# -----------------------------------
positions_km = {}
for sid, info in sats.items():
    positions_km[sid] = sample_positions(info["orbit"], times)

# ------------------------
# Conjunctions
# ------------------------
conjs = pairwise_min_distance(positions_km)

# Metrics
num_conjs_under_thresh = sum(1 for c in conjs if c["min_dist_m"] < miss_threshold_m)
min_overall = conjs[0]["min_dist_m"] if conjs else float("inf")

col1, col2, col3 = st.columns(3)
col1.metric("Satellites", len(sats))
col2.metric("Conjunctions < threshold", num_conjs_under_thresh)
col3.metric("Closest approach (m)", f"{min_overall:,.1f}")

st.markdown("---")
st.subheader("Conjunctions (closest first)")
if conjs:
    # show top 10
    for c in conjs[:min(20, len(conjs))]:
        st.write(f"**{c['sat_a']} ↔ {c['sat_b']}**  |  Min distance: **{c['min_dist_m']:.1f} m**  |  Time index: {c['t_idx']}  | Risk: {risk_from_distance(c['min_dist_m']):.2f}")

else:
    st.write("No conjunctions detected in this window.")

# ----------------------------
# 3D visualization (Plotly)
# ----------------------------
st.markdown("---")
st.subheader("3D Orbit Visualization")
fig = go.Figure()

# draw Earth (sphere)
sphere_u = np.linspace(0, 2 * np.pi, 60)
sphere_v = np.linspace(0, np.pi, 30)
R = Earth.R.to(u.km).value
x = (R * np.outer(np.cos(sphere_u), np.sin(sphere_v)))
y = (R * np.outer(np.sin(sphere_u), np.sin(sphere_v)))
z = (R * np.outer(np.ones_like(sphere_u), np.cos(sphere_v)))
fig.add_trace(go.Surface(x=x, y=y, z=z, colorscale="Blues", showscale=False, opacity=0.7))

colors = px_colors = [
    "red","orange","gold","green","cyan","royalblue","magenta","purple","brown","pink"
]
i = 0
for sid, pos in positions_km.items():
    xs, ys, zs = pos[:,0], pos[:,1], pos[:,2]
    color = colors[i % len(colors)]
    fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", name=sid, line=dict(width=2, color=color)))
    # mark current position (last sample)
    fig.add_trace(go.Scatter3d(x=[xs[-1]], y=[ys[-1]], z=[zs[-1]], mode="markers", marker=dict(size=3, color=color), showlegend=False))
    i += 1

fig.update_layout(scene=dict(aspectmode='data'), height=700, margin=dict(l=0,r=0,t=30,b=0))
st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------
# Planner UI: pick a satellite, propose dv, simulate re-eval
# -----------------------------------------------------------
st.markdown("---")
st.subheader("Planner — propose along-track delta-v to a single satellite")

colA, colB = st.columns([2,1])
with colA:
    sat_list = list(sats.keys())
    chosen_sat = st.selectbox("Choose SAT to maneuver", sat_list)
    dv_m_s = st.slider("Delta-v (m/s)", 0.0, 10.0, 0.5, 0.1)
    apply_duration_min = st.slider("Duration to propagate after maneuver (minutes)", 1, duration_min, min(30, duration_min))
    if st.button("Simulate maneuver"):
        # apply toy phase shift
        orb = sats[chosen_sat]["orbit"]
        new_orb = apply_along_track_phase_shift(orb, float(dv_m_s), float(apply_duration_min * 60))
        # create new sat copy and sample positions
        positions_km_new = dict(positions_km)  # shallow copy of arrays
        # recompute positions for chosen sat
        positions_km_new[chosen_sat] = sample_positions(new_orb, times)
        # recompute conjunctions
        conjs_new = pairwise_min_distance(positions_km_new)
        min_before = min(c["min_dist_m"] for c in conjs) if conjs else float("inf")
        min_after = min(c["min_dist_m"] for c in conjs_new) if conjs_new else float("inf")
        num_before = sum(1 for c in conjs if c["min_dist_m"] < miss_threshold_m)
        num_after = sum(1 for c in conjs_new if c["min_dist_m"] < miss_threshold_m)
        delta_min = min_after - min_before
        st.success(f"Done. Closest approach BEFORE: {min_before:.1f} m; AFTER: {min_after:.1f} m  (Δ {delta_min:.1f} m)")
        st.write(f"Conjunctions under threshold BEFORE: {num_before}; AFTER: {num_after}")
        # show top few changed pairs
        st.subheader("Top changed conjunctions (before -> after)")
        # build a map for quick lookup
        lookup_before = {(c["sat_a"], c["sat_b"]): c for c in conjs}
        lookup_after = {(c["sat_a"], c["sat_b"]): c for c in conjs_new}
        # union keys
        all_keys = set(list(lookup_before.keys()) + list(lookup_after.keys()))
        changes = []
        for k in all_keys:
            a,b = k
            before = lookup_before.get(k)
            after = lookup_after.get(k)
            before_d = before["min_dist_m"] if before else float("inf")
            after_d = after["min_dist_m"] if after else float("inf")
            changes.append((k, before_d, after_d, after_d - before_d))
        changes.sort(key=lambda x: x[2])  # sort by after distance
        for (a,b),bd,ad,delta in changes[:10]:
            st.write(f"{a} ↔ {b} : before {bd:.1f} m -> after {ad:.1f} m  (Δ {delta:.1f} m)")

with colB:
    st.markdown("💡 Planner notes")
    st.markdown("""
- This planner uses a toy **along-track phase-shift** model for demo purposes.
- For real ops you must compute an impulsive dv vector, apply it at a precise epoch, and propagate using validated propagators.
- Next steps: support multi-satellite optimization (minimize total Δv), covariance-based collision probability, and automated maneuver scheduling.
""")

st.markdown("---")
st.caption("Prototype — for demo only. Not for operational use. Use validated tools & human oversight for real satellite operations.")
