import streamlit as st
import numpy as np
import logic  # Ensure logic.so is in the same directory
import plotly.graph_objects as go
import io
import stl               
from stl import mesh 
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap

st.set_page_config(page_title="Shell Topology Opt", layout="centered")

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: 800; color: #0f172a; margin-bottom: 0rem; }
    .tag-container { display: flex; gap: 10px; margin-bottom: 1.5rem; }
    .tag { background-color: #f1f5f9; color: #475569; padding: 4px 12px; border-radius: 9999px; font-size: 0.85rem; font-weight: 500; border: 1px solid #e2e8f0; }
    .section-header { font-size: 1.25rem; font-weight: 700; color: #1e293b; margin-top: 1rem; margin-bottom: 0.5rem; }
    /* Forces the updating image to match the Plotly charts */
    [data-testid="stImage"] img {
        max-height: 600px;
        width: auto;
        object-fit: contain;
        margin-left: auto;
        margin-right: auto;
        display: block;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# PART 1: HEADER & OBJECTIVE
# ==========================================
st.markdown('<div class="main-header">Shell Topology Optimization</div>', unsafe_allow_html=True)
st.markdown('<div class="tag-container"><span class="tag">Shell</span><span class="tag">Optimization</span><span class="tag">FEA Engine</span></div>', unsafe_allow_html=True)

with st.expander("🎯 App Objective", expanded=False):
    st.markdown("""
    **Objective:** Distribute a constant amount of material to maximize the stiffness of a shell-type structure 
    under external distributed load and self-weight.
    * The amount of material distributed is given by a fraction of the solid with sides: Domain X, Domain y, Min Thickness.

    
    📄 [**Read the detailed code explanation and documentation here (PDF)**](https://github.com/sebaspozo94/ShellOpt/blob/main/ShellOpt.pdf)
    """)

# --- SETUP SESSION STATE ---
if 'run_finished' not in st.session_state:
    st.session_state.run_finished = False
    st.session_state.history = None
    st.session_state.X = None
    st.session_state.Y = None

if "bc_df" not in st.session_state:
    st.session_state.bc_df = pd.DataFrame(
        [[48.0, 156.0, 4.0, 4.0, "Pinned"], [48.0, 36.0, 4.0, 4.0, "Pinned"], [192.0, 156.0, 4.0, 4.0, "Pinned"], [192.0, 36.0, 4.0, 4.0, "Pinned"]],
        columns=["X (in)", "Y (in)", "Width", "Height", "Type"]
    )

if "run_bc_df" not in st.session_state:
    st.session_state.run_bc_df = st.session_state.bc_df.copy()
    
if "show_labels" not in st.session_state:
    st.session_state.show_labels = False

# Plotly toolbar config dictionary
PLOTLY_CONFIG = {
    'displayModeBar': True,
    'scrollZoom': True
}

# ==========================================
# PART 2: MODEL CONFIGURATION
# ==========================================
st.markdown('<div class="section-header">⚙️ Model Configuration</div>', unsafe_allow_html=True)

conf_col1, conf_col2, conf_col3, conf_col4 = st.columns(4)

with conf_col1:
    with st.expander("📏 Domain & Mesh", expanded=False):
        dimx = st.number_input("Domain X (in)", value=240, step=4, min_value=1)
        dimy = st.number_input("Domain Y (in)", value=192, step=4, min_value=1)
        mesh_size = st.number_input("Mesh Size (in)", value=4.0, step=0.5, min_value=0.1)
        
        # Calculate elements based on mesh size
        nelx = int(dimx / mesh_size)
        nely = int(dimy / mesh_size)
        total_elements = nelx * nely
        
        if total_elements > 20000:
            st.error(f"🚨 Mesh is too fine! Total elements: {total_elements:,}. The max allowed is 20,000.")
            st.stop()
        else:
            st.success(f"Grid: {nelx} x {nely}\n\nTotal: {total_elements:,}")

with conf_col2:
    with st.expander("🧪 Materials & Loads", expanded=False):
        E = st.number_input("Elastic Modulus (psi)", value=1500000, step=100000)
        nu = st.slider("Poisson's Ratio (v)", 0.0, 0.5, 0.30)
        rho = st.number_input("Material Density (p)", value=0.010, format="%.3f")
        w_u = st.number_input("Distributed Load (w_u)", value=0.2778, format="%.4f")
        self_weight = st.checkbox("Include Self-Weight", value=True)

with conf_col3:
    with st.expander("🎯 Optimization Settings", expanded=False):
        vol_frac = st.slider("Volume Fraction", 0.05, 1.0, 0.3)
        rmin = st.number_input("Filter Radius (rmin)", value=5.0, step=1.0)
        itmax = st.number_input("Max Iterations", value=50, step=10)

with conf_col4:
    with st.expander("📐 Thickness Limits", expanded=False):
        tmin = st.number_input("Min Thickness (in)", value=2.0, step=0.5)
        tmax = st.number_input("Max Thickness (in)", value=12.0, step=0.5)

st.markdown("---")

# ==========================================
# PART 3: BOUNDARY CONDITIONS & RUN OPTIMIZATION
# ==========================================
col_bc, col_run = st.columns(2)

# --- 3A. INTERACTIVE PLOT COLUMN ---
with col_bc:
    st.markdown('<div class="section-header">🎛️ Setup (Click to edit)</div>', unsafe_allow_html=True)
    
    if 'add_t' not in st.session_state: st.session_state.add_t = False
    if 'del_t' not in st.session_state: st.session_state.del_t = False

    def on_add_toggle():
        if st.session_state.add_t: st.session_state.del_t = False

    def on_del_toggle():
        if st.session_state.del_t: st.session_state.add_t = False

    col_t1, col_t2 = st.columns(2)
    add_mode = col_t1.toggle("➕ ADD Support", key="add_t", on_change=on_add_toggle)
    del_mode = col_t2.toggle("➖ DEL Support", key="del_t", on_change=on_del_toggle)

    fig2d = go.Figure()

    fig2d.add_shape(type="rect", x0=0, y0=0, x1=dimx, y1=dimy, 
                    line=dict(color="#0f172a", width=2, dash="dash"), fillcolor="rgba(0,0,0,0)")

    for i, row in st.session_state.bc_df.iterrows():
        hx, hy = row['Width'] / 2.0, row['Height'] / 2.0
        color = 'red' 
        fig2d.add_shape(type="rect", x0=row['X (in)']-hx, y0=row['Y (in)']-hy, x1=row['X (in)']+hx, y1=row['Y (in)']+hy, 
                        line=dict(color=color, width=2), fillcolor=color, opacity=0.6)
        
        if st.session_state.show_labels:
            fig2d.add_annotation(x=row['X (in)'], y=row['Y (in)'], text=f"S{i+1}", showarrow=False, 
                                 font=dict(color="black", size=11, family="Arial Black"))

    grid_spacing = 12
    grid_x, grid_y = np.meshgrid(np.arange(0, dimx + 1, grid_spacing), np.arange(0, dimy + 1, grid_spacing))
    gx, gy = grid_x.flatten(), grid_y.flatten()

    grid_opacity = 0.3 if (add_mode or del_mode) else 0.0
    grid_color = 'green' if add_mode else 'blue'

    fig2d.add_trace(go.Scatter(
        x=gx, y=gy, mode='markers',
        marker=dict(size=12, color=grid_color, opacity=grid_opacity, symbol='square'),
        hoverinfo='text', text="Click here", name="Grid"
    ))

    fig2d.update_layout(
        height=600,
        autosize=True,
        xaxis=dict(range=[-10, dimx+10], constrain='domain'), 
        yaxis=dict(range=[-10, dimy+10], scaleanchor="x", scaleratio=1, constrain='domain'), 
        clickmode='event+select', 
        margin=dict(l=0, r=0, t=0, b=0), showlegend=False
    )

    event = st.plotly_chart(fig2d, on_select="rerun", key="setup_map", use_container_width=True, config=PLOTLY_CONFIG)

    if event and "selection" in event and len(event["selection"]["points"]) > 0:
        pt = event["selection"]["points"][0]
        cx, cy = pt['x'], pt['y']
        
        if add_mode:
            if not ((st.session_state.bc_df['X (in)'] == cx) & (st.session_state.bc_df['Y (in)'] == cy)).any():
                new_row = pd.DataFrame([[float(cx), float(cy), 4.0, 4.0, "Pinned"]], 
                                       columns=["X (in)", "Y (in)", "Width", "Height", "Type"])
                st.session_state.bc_df = pd.concat([st.session_state.bc_df, new_row], ignore_index=True)
                st.rerun()
                
        elif del_mode:
            to_drop = []
            for i, row in st.session_state.bc_df.iterrows():
                hx, hy = row['Width']/2, row['Height']/2
                if (row['X (in)']-hx <= cx <= row['X (in)']+hx) and (row['Y (in)']-hy <= cy <= row['Y (in)']+hy):
                    to_drop.append(i)
            
            if to_drop:
                st.session_state.bc_df = st.session_state.bc_df.drop(to_drop).reset_index(drop=True)
                st.rerun()

    # --- UI ORGANIZATION VIA TABS ---
    with st.expander("🛠️ Modify Boundary Conditions", expanded=False):
        tab_labels, tab_bc, tab_info = st.tabs(["👁️ Labels", "📋 Supports", "ℹ️ Info"])
        
        with tab_labels:
            st.checkbox("🏷️ Show Identifiers on Setup Plot", key="show_labels")
            
        with tab_bc:
            display_df = st.session_state.bc_df.copy()
            display_df.insert(0, "ID", [f"S{i+1}" for i in range(len(display_df))])
            
            edited_bc_df = st.data_editor(
                display_df, 
                num_rows="dynamic", use_container_width=True, hide_index=True, 
                column_config={"ID": st.column_config.TextColumn(disabled=True), "Type": st.column_config.SelectboxColumn("Type", options=["Pinned", "Fixed"])}
            )
            if not edited_bc_df.drop(columns=["ID"]).equals(st.session_state.bc_df):
                st.session_state.bc_df = edited_bc_df.drop(columns=["ID"])
                st.rerun()

        with tab_info:
            st.markdown("""
            Use the table below to manually edit the exact coordinates and dimensions of your supports.
            * **X & Y (in):** The center location of the support.
            * **Width & Height (in):** The dimensions of the rectangular support area.
            * **Type:** * *Pinned:* Prevents translation (movement) but allows rotation (bending).
                * *Fixed:* Prevents both translation and rotation.
            """)

# --- 3B. SOLVER / RUN COLUMN ---
with col_run:
    st.markdown('<div class="section-header">🚀 Solver</div>', unsafe_allow_html=True)
    
    solver_df = st.session_state.bc_df.copy()
    solver_df["Type"] = solver_df["Type"].map({"Pinned": 0, "Fixed": 1})
    BCMatrix = solver_df.to_numpy()

    run_pressed = st.button("🚀 Run Optimization", type="primary", use_container_width=True)
    
    # Placeholders for live plotting
    color_bar_spot = st.empty()
    live_plot_spot = st.empty()
    status_text = st.empty()

    def plot_2d_thickness_mpl(Z_matrix):
        x_range = dimx + 20
        y_range = dimy + 20
        aspect = y_range / x_range
        
        fig = plt.figure(figsize=(6, 6 * aspect), dpi=100)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis('off')
        
        custom_cmap = LinearSegmentedColormap.from_list("custom_blue", ['#cbd5e1', '#2563eb', '#08306b'])
        
        ax.imshow(Z_matrix, cmap=custom_cmap, extent=[0, dimx, 0, dimy], 
                  vmin=0, vmax=tmax, interpolation='nearest')
        
        border = patches.Rectangle((0, 0), dimx, dimy, linewidth=2, edgecolor='#0f172a', 
                                   facecolor='none', linestyle='--')
        ax.add_patch(border)
        
        for i, row in st.session_state.run_bc_df.iterrows():
            hx, hy = row['Width'] / 2.0, row['Height'] / 2.0
            x_min = row['X (in)'] - hx
            y_min = row['Y (in)'] - hy
            
            support = patches.Rectangle((x_min, y_min), row['Width'], row['Height'], 
                                        linewidth=1, edgecolor='darkred', facecolor='red', alpha=0.9)
            ax.add_patch(support)
            
        ax.set_xlim(-10, dimx + 10)
        ax.set_ylim(-10, dimy + 10)
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', transparent=True)
        plt.close(fig)
        buf.seek(0)
        return buf

    def plot_2d_thickness_plotly(Z_matrix):
        fig = go.Figure()
        
        custom_colorscale_plotly = [[0.0, '#cbd5e1'], [0.5, '#2563eb'], [1.0, '#08306b']]
        fig.add_trace(go.Heatmap(
            z=np.flipud(Z_matrix),
            x=np.linspace(0, dimx, Z_matrix.shape[1]),
            y=np.linspace(0, dimy, Z_matrix.shape[0]),
            colorscale=custom_colorscale_plotly,
            zmin=0, zmax=tmax, 
            showscale=True, 
            colorbar=dict(
                title='Thickness (in)', 
                orientation='h', x=0.5, y=1.05, 
                xanchor='center', yanchor='bottom', 
                thickness=15, len=0.8
            ), 
            hoverinfo='skip'
        ))
        
        fig.add_shape(type="rect", x0=0, y0=0, x1=dimx, y1=dimy, 
                      line=dict(color="#0f172a", width=2, dash="dash"), fillcolor="rgba(0,0,0,0)")
        
        for i, row in st.session_state.run_bc_df.iterrows():
            hx, hy = row['Width'] / 2.0, row['Height'] / 2.0
            x_min, x_max = row['X (in)'] - hx, row['X (in)'] + hx
            y_min, y_max = row['Y (in)'] - hy, row['Y (in)'] + hy
            
            fig.add_shape(type="rect", x0=x_min, y0=y_min, x1=x_max, y1=y_max, 
                          line=dict(color='red', width=1), fillcolor='rgba(255,0,0,0.4)')
            
        fig.update_layout(
            height=600,
            autosize=True,
            xaxis=dict(range=[-10, dimx+10], constrain='domain'), 
            yaxis=dict(range=[-10, dimy+10], scaleanchor="x", scaleratio=1, constrain='domain'), 
            margin=dict(l=0, r=0, t=0, b=0), showlegend=False
        )
        return fig

    if run_pressed:
        if len(BCMatrix) == 0:
            st.error("Please add at least one support!")
        else:
            st.session_state.run_bc_df = st.session_state.bc_df.copy()
            
            total_area = dimx * dimy
            target_volume = (total_area * tmin) + (vol_frac * total_area * (tmax - tmin))
            
            gradient_html = f"""
            <div style="text-align: center; margin-bottom: 5px; font-weight: bold; color: #475569; font-size: 0.9rem;">Thickness (in)</div>
            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 15px; padding: 0 10px;">
                <span style="font-size: 0.8rem; font-weight: bold;">0</span>
                <div style="flex-grow: 1; height: 12px; margin: 0 10px; background: linear-gradient(to right, #cbd5e1 0%, #2563eb 50%, #08306b 100%); border: 1px solid #cbd5e1; border-radius: 4px;"></div>
                <span style="font-size: 0.8rem; font-weight: bold;">{tmax}</span>
            </div>
            """
            color_bar_spot.markdown(gradient_html, unsafe_allow_html=True)
            
            def update_live_view(current_it, current_ch, current_Z):
                img_buffer = plot_2d_thickness_mpl(current_Z)
                live_plot_spot.image(img_buffer, use_container_width=True) 
                status_text.info(f"⚙️ Optimizing... Iteration: {current_it}")

            with st.spinner("Optimizing..."):
                SW_val = 1 if self_weight else 0
                X, Y, Thickness, history = logic.run_topology_optimization(
                    float(dimx), float(dimy), float(E), float(nu), float(rho), int(SW_val), 
                    BCMatrix, float(w_u), int(nelx), int(nely), float(target_volume), 
                    float(rmin), float(tmin), float(tmax), int(itmax), progress_callback=update_live_view
                )
                st.session_state.history, st.session_state.X, st.session_state.Y, st.session_state.run_finished = history, X, Y, True
                st.rerun()

    if st.session_state.run_finished and st.session_state.history is not None:
        color_bar_spot.empty() 
        final_plotly_fig = plot_2d_thickness_plotly(st.session_state.history[-1])
        live_plot_spot.plotly_chart(final_plotly_fig, use_container_width=True, key="final_result_plot", config=PLOTLY_CONFIG)
        status_text.success(f"✅ Optimization Complete! Iterations run: {len(st.session_state.history)}")


# ==========================================
# PART 4: INTERACTIVE 3D RESULTS & STL EXPORT
# ==========================================
if st.session_state.run_finished:
    st.markdown("---")
    st.markdown('<div class="section-header">🕒 Interactive 3D Results</div>', unsafe_allow_html=True)
    
    with st.expander("🖱️ How to interact with the 3D Plot", expanded=False):
        st.markdown("""
        **On a Computer (Mouse):**
        * **Rotate:** Left-click and drag.
        * **Pan:** Right-click and drag (or `Shift` + Left-click).
        * **Zoom:** Use the mouse scroll wheel.
        
        **On a Phone/Tablet (Touch):**
        * **Rotate:** Swipe with one finger.
        * **Pan:** Swipe with two fingers.
        * **Zoom:** Pinch in or out with two fingers.
        
        *Tip: You can use the menu in the top right corner of the plot to reset the view or download a snapshot!*
        """)

    steps = len(st.session_state.history)
    plot_placeholder = st.empty()
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    idx = st.slider("Iteration History", 0, steps - 1, steps - 1)
    
    col_cam, col_scale = st.columns(2)
    with col_cam:
        view_choice = st.selectbox("🎥 Camera View", ["Default", "Bottom (XY)", "Front (XZ)", "Side (YZ)"])
    with col_scale:
        use_true_scale = st.checkbox("📏 True Z-Scale", value=True)
        if use_true_scale:
            z_scale_pct = int(100*tmax/max(dimx, dimy))
        else:
            if "z_scale_val" not in st.session_state: st.session_state.z_scale_val = int(100*tmax/max(dimx, dimy))
            z_scale_pct = st.slider("Visual Z-Scale (%)", 0, 100, st.session_state.z_scale_val)
            st.session_state.z_scale_val = z_scale_pct

    if view_choice == "Bottom (XY)": cam_eye, cam_up = dict(x=0, y=0, z=-2.5), dict(x=0, y=1, z=0)
    elif view_choice == "Front (XZ)": cam_eye, cam_up = dict(x=0, y=-2.5, z=0), dict(x=0, y=0, z=1)
    elif view_choice == "Side (YZ)": cam_eye, cam_up = dict(x=-2.5, y=0, z=0), dict(x=0, y=0, z=1)
    else: cam_eye, cam_up = dict(x=1.2, y=-1.5, z=-0.8), dict(x=0, y=0, z=1)
    
    Z_raw = st.session_state.history[idx]
    Z_final = np.flipud(Z_raw) 
    
    x_coords = np.linspace(0, dimx, Z_final.shape[1])
    y_coords = np.linspace(0, dimy, Z_final.shape[0])
    X_mesh, Y_mesh = np.meshgrid(x_coords, y_coords)

    Z_plot_neg = -Z_final 
    custom_colorscale = [[0.0, '#08306b'], [0.4, '#2563eb'], [1.0, '#cbd5e1']]

    # Advanced lighting settings to make the 3D plot look solid and high-end
    lighting_effects = dict(ambient=0.6, diffuse=0.8, roughness=0.4, specular=0.5, fresnel=0.2)

    # Main Top and Bottom Surfaces
    roof_surface = go.Surface(
        z=np.zeros_like(Z_plot_neg), x=X_mesh, y=Y_mesh, 
        colorscale=[[0, '#cbd5e1'], [1, '#cbd5e1']], 
        showscale=False, hoverinfo='skip', opacity=1.0, lighting=lighting_effects
    )
    
    bottom_surface = go.Surface(
        z=Z_plot_neg, x=X_mesh, y=Y_mesh, 
        surfacecolor=Z_plot_neg,
        colorscale=custom_colorscale, cmin=-tmax, cmax=0, 
        lighting=lighting_effects,
        colorbar=dict(
            title='Thickness (in)', orientation='h',
            x=0.5, y=1.05, xanchor='center', yanchor='bottom',
            thickness=12, len=0.6
        )
    )

    # Generate 4 Side walls to close the geometry into a solid body
    side_surfaces = []
    
    # Front Wall (y=0)
    side_surfaces.append(go.Surface(
        x=np.vstack((X_mesh[0, :], X_mesh[0, :])),
        y=np.vstack((Y_mesh[0, :], Y_mesh[0, :])),
        z=np.vstack((np.zeros_like(Z_plot_neg[0, :]), Z_plot_neg[0, :])),
        surfacecolor=np.vstack((Z_plot_neg[0, :], Z_plot_neg[0, :])),
        colorscale=custom_colorscale, cmin=-tmax, cmax=0, showscale=False, hoverinfo='skip', lighting=lighting_effects
    ))
    # Back Wall (y=max)
    side_surfaces.append(go.Surface(
        x=np.vstack((X_mesh[-1, :], X_mesh[-1, :])),
        y=np.vstack((Y_mesh[-1, :], Y_mesh[-1, :])),
        z=np.vstack((np.zeros_like(Z_plot_neg[-1, :]), Z_plot_neg[-1, :])),
        surfacecolor=np.vstack((Z_plot_neg[-1, :], Z_plot_neg[-1, :])),
        colorscale=custom_colorscale, cmin=-tmax, cmax=0, showscale=False, hoverinfo='skip', lighting=lighting_effects
    ))
    # Left Wall (x=0)
    side_surfaces.append(go.Surface(
        x=np.vstack((X_mesh[:, 0], X_mesh[:, 0])).T,
        y=np.vstack((Y_mesh[:, 0], Y_mesh[:, 0])).T,
        z=np.vstack((np.zeros_like(Z_plot_neg[:, 0]), Z_plot_neg[:, 0])).T,
        surfacecolor=np.vstack((Z_plot_neg[:, 0], Z_plot_neg[:, 0])).T,
        colorscale=custom_colorscale, cmin=-tmax, cmax=0, showscale=False, hoverinfo='skip', lighting=lighting_effects
    ))
    # Right Wall (x=max)
    side_surfaces.append(go.Surface(
        x=np.vstack((X_mesh[:, -1], X_mesh[:, -1])).T,
        y=np.vstack((Y_mesh[:, -1], Y_mesh[:, -1])).T,
        z=np.vstack((np.zeros_like(Z_plot_neg[:, -1]), Z_plot_neg[:, -1])).T,
        surfacecolor=np.vstack((Z_plot_neg[:, -1], Z_plot_neg[:, -1])).T,
        colorscale=custom_colorscale, cmin=-tmax, cmax=0, showscale=False, hoverinfo='skip', lighting=lighting_effects
    ))

    fig = go.Figure(data=[roof_surface, bottom_surface] + side_surfaces)

    support_depth = -tmax * 1.2
    
    for i, row in st.session_state.run_bc_df.iterrows():
        hx, hy = row['Width'] / 2.0, row['Height'] / 2.0
        x_min, x_max = row['X (in)'] - hx, row['X (in)'] + hx
        y_min, y_max = row['Y (in)'] - hy, row['Y (in)'] + hy
        
        fig.add_trace(go.Mesh3d(
            x=[x_min, x_max, x_max, x_min, x_min, x_max, x_max, x_min],
            y=[y_min, y_min, y_max, y_max, y_min, y_min, y_max, y_max],
            z=[support_depth, support_depth, support_depth, support_depth, tmax * 0.1, tmax * 0.1, tmax * 0.1, tmax * 0.1],
            i=[7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
            j=[3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
            k=[0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
            color='red', 
            opacity=0.8, 
            flatshading=True,
            name=f"Support S{i+1}",
            showlegend=False
        ))

    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[-0.05 * dimx, 1.05 * dimx], title='X (in)'),
            yaxis=dict(range=[-0.05 * dimy, 1.05 * dimy], title='Y (in)'),
            zaxis=dict(range=[support_depth, tmax * 0.2], title='Z (in)'),
            aspectratio=dict(x=dimx/max(dimx, dimy), y=dimy/max(dimx, dimy), z=z_scale_pct/100.0),
            camera=dict(eye=cam_eye, up=cam_up)
        ),
        margin=dict(l=0, r=0, b=0, t=50), height=600
    )
    
    plot_placeholder.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
    
    # --- SOLID STL EXPORT FUNCTION ---
    st.markdown("---")
    st.subheader("💾 Export Geometry")
    
    def generate_solid_stl(X, Y, Z_bottom):
        ny, nx = Z_bottom.shape
        offset = nx * ny
        
        # Flatten coordinates
        x_flat = X.flatten()
        y_flat = Y.flatten()
        z_bot_flat = Z_bottom.flatten()
        z_top_flat = np.zeros_like(z_bot_flat)
        
        # Combine vertices: Top vertices first, then Bottom vertices
        vertices = np.zeros((2 * offset, 3))
        vertices[:offset, 0] = x_flat
        vertices[:offset, 1] = y_flat
        vertices[:offset, 2] = z_top_flat
        
        vertices[offset:, 0] = x_flat
        vertices[offset:, 1] = y_flat
        vertices[offset:, 2] = z_bot_flat
        
        faces = []
        
        # 1. Top and Bottom surfaces
        for i in range(ny - 1):
            for j in range(nx - 1):
                n0 = i * nx + j
                n1 = n0 + 1
                n2 = (i + 1) * nx + j
                n3 = n2 + 1
                
                # Top surface triangles
                faces.append([n0, n1, n2])
                faces.append([n1, n3, n2])
                
                # Bottom surface triangles
                b0, b1, b2, b3 = n0 + offset, n1 + offset, n2 + offset, n3 + offset
                faces.append([b0, b2, b1])
                faces.append([b1, b2, b3])
                
        # 2. Side Walls
        # Front Wall (y = 0)
        for j in range(nx - 1):
            n0 = j
            n1 = j + 1
            faces.append([n0, n0 + offset, n1])
            faces.append([n1, n0 + offset, n1 + offset])
            
        # Back Wall (y = max)
        for j in range(nx - 1):
            n0 = (ny - 1) * nx + j
            n1 = n0 + 1
            faces.append([n0, n1, n0 + offset])
            faces.append([n1, n1 + offset, n0 + offset])
            
        # Left Wall (x = 0)
        for i in range(ny - 1):
            n0 = i * nx
            n1 = (i + 1) * nx
            faces.append([n0, n1, n0 + offset])
            faces.append([n1, n1 + offset, n0 + offset])
            
        # Right Wall (x = max)
        for i in range(ny - 1):
            n0 = i * nx + (nx - 1)
            n1 = (i + 1) * nx + (nx - 1)
            faces.append([n0, n0 + offset, n1])
            faces.append([n1, n0 + offset, n1 + offset])
            
        faces = np.array(faces)
        
        # Create solid mesh
        solid_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
        for i, f in enumerate(faces):
            for j in range(3):
                solid_mesh.vectors[i][j] = vertices[f[j], :]
                
        buf = io.BytesIO()
        solid_mesh.save('solid.stl', fh=buf)
        return buf.getvalue()

    stl_data = generate_solid_stl(X_mesh, Y_mesh, Z_plot_neg)
    st.download_button(label="📥 Download as Solid .STL", data=stl_data, file_name=f"Optimized_Solid_Iter{idx}.stl", mime="model/stl", type="primary")

# ==========================================
# PART 5: AUTHOR & CONTACT INFO
# ==========================================
st.markdown("---")
st.markdown('<div class="section-header">📬 Contact & Info</div>', unsafe_allow_html=True)

col_info1, col_info2 = st.columns([2, 1])
with col_info1:
    st.markdown("""
    **Created by:** Sebastian Pozo Ocampo  
    **Contact:** [sebaspozo94@gmail.com](mailto:sebaspozo94@gmail.com)  
    For custom workflows, project-specific studies, or collaboration.
    """)
    # --- VISITOR BADGE ADDED HERE ---
    st.markdown("<br>", unsafe_allow_html=True) # Adds a little space
#    st.markdown(
#    "[![Visitors](https://api.visitorbadge.io/api/visitors?path=sebaspozo94.apps&countColor=%232563eb)](https://visitorbadge.io/status?path=sebaspozo94.apps)"
#    )
with col_info2:
    st.markdown("""
    * [Website](https://streamline-gallery-5d621e11.buildaispace.app)  
    * [LinkedIn](https://www.linkedin.com/in/sebastianpozo94/)
    * [GitHub](https://github.com/sebaspozo94)
    """)
