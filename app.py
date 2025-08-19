
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Set

EXCEL_PATH = "Penetration_POC.xlsx"   # change if needed
HIER_PATH = "Masterlist_HIER.csv"   # new: hierarchy/masterlist file

# =========================================================
# Data loading & preparation
# =========================================================
@st.cache_data(show_spinner=False)
def load_excel_data(file_path: str):
    """Load Penetration_POC.xlsx and prepare master/results/deals/stage_map."""
    # --- Load sheets
    master_df = pd.read_excel(file_path, sheet_name='Masterlist')
    vip_df = pd.read_excel(file_path, sheet_name='VIP')
    _ = pd.read_excel(file_path, sheet_name='Deal contact')  # kept for future
    deal_df = pd.read_excel(file_path, sheet_name='Deal')

    # --- PUBLIC LHN only for master page
    master_lhn = master_df[
        (master_df['Granularity'] == 'LHN') &
        (master_df['Sector'].astype(str).str.upper() == 'PUBLIC')
    ].copy()

    # Helpers / cleaning
    master_lhn['Name'] = master_lhn['Name'].astype(str).str.strip()
    master_lhn['State'] = master_lhn['State'].astype(str).str.strip()
    master_lhn['name_l'] = master_lhn['Name'].str.lower()
    staff_col = 'Number of clincial staff' if 'Number of clincial staff' in master_lhn.columns else 'Number of clinical staff'
    master_lhn['Number of Staff'] = pd.to_numeric(
        master_lhn.get(staff_col), errors='coerce'
    ).clip(lower=0)

    # --- Deals prep
    deal_df['assoc_comp_ids_list'] = deal_df['Associated Company IDs'].fillna('').astype(str).str.split(';')
    deal_df['assoc_comp_name_l'] = deal_df['Associated Company Name'].astype(str).str.lower().str.strip()
    for c in ['Deal Stage', 'Deal Name', 'Close Date', 'Deal owner', 'Amount',
              'Potential Total Contract Value', 'State']:
        if c not in deal_df.columns:
            deal_df[c] = np.nan

    # Build reverse indexes -> deal rows
    record_to_deals: Dict[str, list] = {}
    name_to_deals: Dict[str, list] = {}
    for i, r in deal_df.iterrows():
        for comp_id in r['assoc_comp_ids_list']:
            comp_id = str(comp_id).strip()
            if comp_id:
                record_to_deals.setdefault(comp_id, []).append(i)
        nm = r['assoc_comp_name_l']
        if isinstance(nm, str) and nm and nm != 'nan':
            name_to_deals.setdefault(nm, []).append(i)

    def deal_indices_for_lhn(name_l: str, record_id) -> list:
        out = []
        if pd.notna(record_id):
            rid = str(int(record_id)) if not isinstance(record_id, str) and not pd.isna(record_id) else str(record_id)
            out += record_to_deals.get(rid, [])
        out += name_to_deals.get(name_l, [])
        return sorted(set(out))

    # --- VIP mapping (name & record id keys)
    vip_df['org_name_l'] = vip_df['Org'].astype(str).str.lower().str.strip()
    vip_df['company_record_id'] = vip_df['Company Record ID'].astype(str)
    vip_map: Dict[str, Set[str]] = {}
    for _, r in vip_df.iterrows():
        nm = r.get('Name')
        if pd.notna(nm):
            vip_map.setdefault(r['org_name_l'], set()).add(nm)
            vip_map.setdefault(r['company_record_id'], set()).add(nm)

    # --- Build LHN rows + stage sets + deal index lists
    rows, stage_map = [], {}
    for _, l in master_lhn.iterrows():
        name_l = l['name_l']
        rid = l['Record ID']
        state = l.get('State')
        status = (str(l.get('Status', '')).strip() if pd.notna(l.get('Status', '')) else '')

        idxs = deal_indices_for_lhn(name_l, rid)

        # VIPs
        vip_set: Set[str] = set()
        if name_l in vip_map: vip_set |= set(vip_map[name_l])
        if pd.notna(rid):
            rid_key = str(int(rid)) if not isinstance(rid, str) else str(rid)
            vip_set |= set(vip_map.get(rid_key, set()))

        # Stages present for this LHN
        stages = set(str(deal_df.loc[i, 'Deal Stage']) for i in idxs if pd.notna(deal_df.loc[i, 'Deal Stage']))
        key = str(int(rid)) if pd.notna(rid) and not isinstance(rid, str) else str(rid)
        stage_map[key] = stages

        # Category from deal names
        cat = 'Untouched'
        if idxs:
            cats = []
            for i in idxs:
                dname = str(deal_df.loc[i, 'Deal Name']).lower()
                if '- h1' in dname: cats.append('H1')
                elif '- h2' in dname: cats.append('H2')
            if 'H1' in cats: cat = 'H1'
            elif 'H2' in cats: cat = 'H2'
            else: cat = 'Other'

        # Any-touch (status, VIP, or any stage)
        touched_any = (
            (status != '' and status.lower() != 'untouched') or
            bool(vip_set) or bool(stages)
        )

        rows.append({
            'LHN Name': l['Name'],
            'State': state,
            'Record ID': rid,
            'Status': status,
            'Category': cat,
            'Touched_Any': touched_any,
            'VIP Contacts': '; '.join(sorted(vip_set)) if vip_set else '',
            'Number of Staff': l['Number of Staff'],
            'Number of beds': l['Number of beds'] if 'Number of beds' in master_lhn.columns else np.nan,
            'Deal Indices': idxs,
        })

    results = pd.DataFrame(rows)
    # normalize State again on results
    results['State'] = results['State'].astype(str).str.strip()
    return results, deal_df, stage_map, master_lhn


@st.cache_data(show_spinner=False)
def load_hier_data(hier_path: str) -> pd.DataFrame:
    """Load Masterlist_HIER.csv; show raw file (no transforms)."""
    try:
        df = pd.read_csv(hier_path)
        return df
    except Exception as e:
        st.warning(f"Could not load {hier_path}: {e}")
        return pd.DataFrame()


# =========================================================
# App
# =========================================================
st.set_page_config(page_title="LHN Engagement", layout="wide")
st.title("Hospital LHN Engagement")

results_df, deal_df, lhn_stage_map, master_lhn_df = load_excel_data(EXCEL_PATH)
hier_df = load_hier_data(HIER_PATH)

# ---------------- PAGE SELECTOR (TOP OF SIDEBAR) ----------------
page = st.sidebar.selectbox(
    "Page",
    ["Engagement Dashboard", "Master List"],
    index=0
)

# ================================================================
# DYNAMIC FILTERS (depend on page)
# ================================================================
st.sidebar.header("Filters")

if page == "Engagement Dashboard":
    # --- existing Engagement Dashboard filters (States, price, Top/Bottom N)
    # Compute states from results_df only, normalized
    states_all = sorted([
        s for s in results_df['State'].dropna().astype(str).str.strip().unique()
        if s and s.lower() != 'nan'
    ])
    selected_states = st.sidebar.multiselect("States (clear = all)", states_all, default=states_all)
    if not selected_states:
        selected_states = states_all

    # This branch needs filtered for the dashboard page only
    filtered = results_df[results_df['State'].isin(selected_states)].copy()

    def touched_by_any_deal_stage(row) -> bool:
        rid = row['Record ID']
        key = str(int(rid)) if pd.notna(rid) and not isinstance(rid, str) else str(rid)
        stages = lhn_stage_map.get(key, set())
        return bool(stages)

    if not filtered.empty:
        filtered['Touched_By_Deals'] = filtered.apply(touched_by_any_deal_stage, axis=1)
        filtered['Number of Staff'] = pd.to_numeric(filtered['Number of Staff'], errors='coerce').clip(lower=0)
        unit_price = st.sidebar.number_input("Unit price per staff (AUD)", min_value=0.0, value=1299.0, step=1.0, key="unit_price")
        filtered['PTCV_custom'] = filtered['Number of Staff'].fillna(0) * unit_price
    else:
        unit_price = st.sidebar.number_input("Unit price per staff (AUD)", min_value=0.0, value=1299.0, step=1.0, key="unit_price")

    n_topbottom = st.sidebar.number_input("Top/Bottom N by (Staff × Unit Price)", 3, 50, 10, 1, key="n_topbottom")

elif page == "Master List":
    # --- New: Page 2 has State + LHN filters (cascading)
    # Create a display copy where "None" (string) and NaN are shown as blank
    hier_display = hier_df.copy()
    if not hier_display.empty:
        # Standardize for filtering but blank-out for display
        # Keep a normalized copy for filtering
        _norm = hier_display.copy()
        for c in _norm.columns:
            _norm[c] = _norm[c].astype(str)

        # Build state list from normalized data
        state_col = 'State' if 'State' in _norm.columns else None
        lhn_col = 'Local Hospital Network (LHN)' if 'Local Hospital Network (LHN)' in _norm.columns else None

        # Sidebar: State filter
        if state_col:
            states_all_p2 = sorted(
                s for s in _norm[state_col].dropna().astype(str).str.strip().unique()
                if s and s.lower() != 'nan' and s.lower() != 'none'
            )
        else:
            states_all_p2 = []

        sel_states_p2 = st.sidebar.multiselect("State (clear = all) ! Not working now", states_all_p2, default=states_all_p2)

        # Derive LHN list based on selected states (cascade)
        if lhn_col:
            if sel_states_p2:
                lhn_pool = _norm[_norm[state_col].isin(sel_states_p2)][lhn_col]
            else:
                lhn_pool = _norm[lhn_col]

            lhns_all_p2 = sorted(
                s for s in lhn_pool.dropna().astype(str).str.strip().unique()
                if s and s.lower() != 'nan' and s.lower() != 'none'
            )
        else:
            lhns_all_p2 = []

        sel_lhns_p2 = st.sidebar.multiselect("Local Hospital Network (LHN) (clear = all)", lhns_all_p2, default=lhns_all_p2)

        # Apply filters to the normalized frame
        filtered_p2 = _norm.copy()
        if state_col and sel_states_p2:
            filtered_p2 = filtered_p2[filtered_p2[state_col].isin(sel_states_p2)]
        if lhn_col and sel_lhns_p2:
            filtered_p2 = filtered_p2[filtered_p2[lhn_col].isin(sel_lhns_p2)]

        # Bring back to display copy (same rows) and blank out "None"
        if not filtered_p2.empty:
            hier_display = hier_display.loc[filtered_p2.index].copy()
        else:
            hier_display = hier_display.iloc[0:0].copy()

        # Display-only cleanup: NaN -> '', literal 'None' (any case) -> ''
        hier_display = hier_display.fillna('')
        hier_display = hier_display.replace(r'^\s*None\s*$', '', regex=True)
    else:
        hier_display = hier_df  # stays empty

# =========================================================================================
# PAGE 1: Engagement Dashboard
# =========================================================================================
if page == "Engagement Dashboard":
    if filtered.empty:
        st.info("No rows after filters.")
    else:
        # Split tables (now based on any deal presence; stage filter removed)
        touched_tbl = filtered[filtered['Touched_By_Deals']].copy()
        untouched_tbl = filtered[~filtered['Touched_By_Deals']].copy()

        # ----- Penetration (Any Touch) as proper lollipop (stems from 0 to value + marker)
        st.subheader("Penetration Rate by State")
        cov_any = filtered.groupby('State', as_index=False).agg(
            Total=('LHN Name', 'count'),
            Touched=('Touched_Any', 'sum')
        )
        cov_any['Penetration Rate'] = cov_any['Touched'] / cov_any['Total']
        cov_any_sorted = cov_any.sort_values('Penetration Rate')

        # Build broken-line arrays (None = segment break) for stems
        x_lines, y_lines = [], []
        for _, r in cov_any_sorted.iterrows():
            x_lines += [0, r['Penetration Rate'], None]
            y_lines += [r['State'], r['State'], None]

        fig_cov_any = go.Figure()
        # Stems
        fig_cov_any.add_trace(
            go.Scatter(
                x=x_lines,
                y=y_lines,
                mode='lines',
                line=dict(width=15),
                hoverinfo='skip',
                showlegend=False,
            )
        )
        # Lollipop heads (markers) + percentage labels
        fig_cov_any.add_trace(
            go.Scatter(
                x=cov_any_sorted['Penetration Rate'],
                y=cov_any_sorted['State'],
                mode='markers+text',
                marker=dict(size=10),
                text=cov_any_sorted['Penetration Rate'].map(lambda x: f"{x:.1%}"),
                textposition='middle right',
                showlegend=False,
                hovertemplate='State: %{y}<br>Penetration: %{x:.0%}<extra></extra>'
            )
        )
        fig_cov_any.update_layout(
            title='Penetration (Any Touch)',
            yaxis=dict(categoryorder='array', categoryarray=cov_any_sorted['State'].tolist()),
            margin=dict(l=60, r=20, t=50, b=40)
        )
        fig_cov_any.update_xaxes(range=[0, 1], title="Penetration Rate")
        fig_cov_any.update_yaxes(title="State")
        st.plotly_chart(fig_cov_any, use_container_width=True)

        # ----- Category counts (grouped) - uses ONLY filtered
        st.subheader("LHNs by Category per State (Counts)")
        cat_df = filtered.pivot_table(
            index='State', columns='Category', values='LHN Name', aggfunc='count', fill_value=0
        ).reset_index()

        # Ensure all expected columns exist
        for c in ['H1','H2','Other','Untouched']:
            if c not in cat_df.columns: cat_df[c] = 0

        fig_grp = go.Figure()
        for c in ['H1','H2','Other','Untouched']:
            fig_grp.add_bar(name=c, x=cat_df['State'], y=cat_df[c])
        fig_grp.update_layout(barmode='group', xaxis_title='State', yaxis_title='Number of LHNs',
                              title='LHNs by Category per State (Grouped Count)')
        st.plotly_chart(fig_grp, use_container_width=True)

        # ----- Top / Bottom N by staff × unit price (uses ONLY filtered)
        st.subheader(f"Top LHNs by Potential Value (Staff × ${unit_price:,.0f})")
        ptcv_df = filtered[['LHN Name','State','Number of Staff','PTCV_custom']].copy()
        top_n = ptcv_df.sort_values('PTCV_custom', ascending=False).head(int(n_topbottom))
        fig_top = px.bar(
            top_n.sort_values('PTCV_custom'),
            x='PTCV_custom', y='LHN Name', orientation='h', color='State',
            labels={'PTCV_custom':'Potential Value', 'LHN Name': 'LHN'}
        )
        fig_top.update_layout(xaxis_tickformat='$,.0f', title='Top N')
        fig_top.update_traces(text=top_n.sort_values('PTCV_custom')['PTCV_custom'].map(lambda v: f"${v:,.0f}"),
                              textposition='outside')
        st.plotly_chart(fig_top, use_container_width=True)


        # ----- Touched vs Untouched tables with selection (uses ONLY filtered)
        st.subheader("Which LHNs we have engaged and NOT engaged yet?")
        touched_tbl = touched_tbl[['LHN Name','State','Category','Number of Staff','PTCV_custom','Deal Indices']].sort_values(['State','LHN Name'])
        untouched_tbl = untouched_tbl[['LHN Name','State','Category','Number of Staff','PTCV_custom','Deal Indices']].sort_values(['State','LHN Name'])

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Touched (Has ≥1 Deal)**")
            sel_all_touched = st.checkbox("Select all touched (below)", value=False, key="sel_all_touched")
            t_view = touched_tbl.drop(columns=['Deal Indices']).copy()
            t_view.insert(0, "Pick", sel_all_touched)
            touched_out = st.data_editor(
                t_view, hide_index=True, use_container_width=True,
                disabled=['State','Category','Number of Staff','PTCV_custom','LHN Name'],
                column_config={"Pick": st.column_config.CheckboxColumn(required=False)},
                key="touched_editor"
            )
        with c2:
            st.markdown("**Untouched (No Deals)**")
            sel_all_untouched = st.checkbox("Select all untouched (below)", value=False, key="sel_all_ntchd")
            u_view = untouched_tbl.drop(columns=['Deal Indices']).copy()
            u_view.insert(0, "Pick", sel_all_untouched)
            untouched_out = st.data_editor(
                u_view, hide_index=True, use_container_width=True,
                disabled=['State','Category','Number of Staff','PTCV_custom','LHN Name'],
                column_config={"Pick": st.column_config.CheckboxColumn(required=False)},
                key="untouched_editor"
            )

        picked_lhns = []
        if 'Pick' in touched_out:
            picked_lhns += touched_out.loc[touched_out['Pick'], 'LHN Name'].tolist()
        if 'Pick' in untouched_out:
            picked_lhns += untouched_out.loc[untouched_out['Pick'], 'LHN Name'].tolist()
        picked_lhns = sorted(set(picked_lhns))

        colA, colB = st.columns([1,1])
        with colA:
            if st.button("Use ALL currently Touched (left table)"):
                picked_lhns = touched_tbl['LHN Name'].tolist()
        with colB:
            if st.button("Clear selection"):
                picked_lhns = []

        # ----- VIP + Deal list for selected LHNs (no stage filtering)
        st.subheader("VIP Contacts & Deals for Selected LHNs")
        if len(picked_lhns) == 0:
            st.info("Pick LHNs using the checkboxes above, or click **Use ALL currently Touched**.")
        else:
            vip_view = filtered[filtered['LHN Name'].isin(picked_lhns)][['LHN Name','State','VIP Contacts']]
            st.markdown("**VIP Contacts**")
            st.dataframe(vip_view.sort_values(['State','LHN Name']).reset_index(drop=True), use_container_width=True)

            st.markdown("**Deals**")
            selected_rows = filtered[filtered['LHN Name'].isin(picked_lhns)]
            all_indices = []
            for _, r in selected_rows.iterrows():
                all_indices += r['Deal Indices']
            all_indices = sorted(set(all_indices))
            if len(all_indices) == 0:
                st.info("No deals found for the selected LHNs.")
            else:
                deals_sel = deal_df.loc[all_indices].copy()
                cols = ['Deal Name','Deal Stage','Close Date','Deal owner','Amount',
                        'Potential Total Contract Value','Associated Company Name','State']
                cols = [c for c in cols if c in deals_sel.columns]
                st.dataframe(
                    deals_sel[cols].sort_values(['Deal Stage','Close Date'], na_position='last').reset_index(drop=True),
                    use_container_width=True
                )

# =========================================================================================
# PAGE 2: Master List (Public LHNs) + Masterlist_HIER
# =========================================================================================
if page == "Master List":
    tabs = st.tabs(["Public LHNs (Masterlist)", "To be added"])

    # ---------- Tab 1: Public LHNs (same filter by State applied) ----------
    with tabs[0]:
        if hier_df.empty:
            st.info("Masterlist_HIER.csv not found or empty.")
        else:
            st.caption("Showing the CSV exactly as-is (no filtering or transformations).")

            # Replace "None" (string) and None (Python object) with blank
            hier_df = hier_df.replace("None", "").fillna("")

            st.dataframe(hier_df, use_container_width=True)

    # ---------- Tab 2: Masterlist_HIER (CSV RAW, NO TRANSFORMS) ----------
    with tabs[1]:
        if hier_df.empty:
            st.info("Masterlist_HIER.csv not found or empty.")
        else:
            st.caption("Showing the CSV exactly as-is (no filtering or transformations).")

            # Apply the same replacement here too
            hier_df = hier_df.replace("None", "").fillna("")

            # st.dataframe(hier_df, use_container_width=True)