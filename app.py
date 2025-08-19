# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

EXCEL_PATH = "Coverage_POC.xlsx"  # change if needed

# =========================================================
# Data loading & preparation
# =========================================================
@st.cache_data
def load_data(file_path: str):
    # --- Load sheets
    master_df = pd.read_excel(file_path, sheet_name='Masterlist')
    vip_df = pd.read_excel(file_path, sheet_name='VIP')
    deal_contact_df = pd.read_excel(file_path, sheet_name='Deal contact')  # not used directly now
    deal_df = pd.read_excel(file_path, sheet_name='Deal')

    # --- PUBLIC LHN only
    lhn_df = master_df[
        (master_df['Granularity'] == 'LHN') &
        (master_df['Sector'].str.upper() == 'PUBLIC')
    ].copy()

    # Helpers / cleaning
    lhn_df['name_l'] = lhn_df['Name'].str.lower().str.strip()
    lhn_df['Number of Staff'] = pd.to_numeric(
        lhn_df.get('Number of clincial staff'), errors='coerce'
    ).clip(lower=0)

    # --- Deals prep
    deal_df['assoc_comp_ids_list'] = deal_df['Associated Company IDs'].fillna('').astype(str).str.split(';')
    deal_df['assoc_comp_name_l'] = deal_df['Associated Company Name'].str.lower().str.strip()
    # keep key fields clean
    for c in ['Deal Stage', 'Deal Name', 'Close Date', 'Deal owner', 'Amount', 'Potential Total Contract Value', 'State']:
        if c not in deal_df.columns:
            deal_df[c] = np.nan

    # Build reverse indexes -> deal rows
    record_to_deals, name_to_deals = {}, {}
    for i, r in deal_df.iterrows():
        for comp_id in r['assoc_comp_ids_list']:
            comp_id = comp_id.strip()
            if comp_id:
                record_to_deals.setdefault(comp_id, []).append(i)
        nm = r['assoc_comp_name_l']
        if pd.notna(nm):
            name_to_deals.setdefault(nm, []).append(i)

    def deal_indices_for_lhn(name_l: str, record_id):
        out = []
        if pd.notna(record_id):
            rid = str(int(record_id)) if not isinstance(record_id, str) else record_id
            out += record_to_deals.get(rid, [])
        out += name_to_deals.get(name_l, [])
        return sorted(set(out))

    # --- VIP mapping (name & record id keys)
    vip_df['org_name_l'] = vip_df['Org'].str.lower().str.strip()
    vip_df['company_record_id'] = vip_df['Company Record ID'].astype(str)
    vip_map = {}
    for _, r in vip_df.iterrows():
        if pd.notna(r['Name']):
            vip_map.setdefault(r['org_name_l'], []).append(r['Name'])
            vip_map.setdefault(r['company_record_id'], []).append(r['Name'])

    # --- Build LHN rows + stage sets + deal index lists
    rows, stage_map = [], {}
    for _, l in lhn_df.iterrows():
        name_l = l['name_l']
        rid = l['Record ID']
        state = l['State']
        status = (str(l['Status']).strip() if pd.notna(l['Status']) else '')

        idxs = deal_indices_for_lhn(name_l, rid)

        # VIPs
        vip_set = set()
        if name_l in vip_map: vip_set |= set(vip_map[name_l])
        if pd.notna(rid):
            rid_key = str(int(rid)) if not isinstance(rid, str) else str(rid)
            vip_set |= set(vip_map.get(rid_key, []))

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
            'Category': cat,
            'Touched_Any': touched_any,
            'VIP Contacts': '; '.join(sorted(vip_set)),
            'Number of Staff': l['Number of Staff'],
            'Deal Indices': idxs,  # keep for later deal listing
        })

    results = pd.DataFrame(rows)

    return results, deal_df, stage_map


# =========================================================
# App
# =========================================================
st.set_page_config(page_title="LHN Engagement Dashboard", layout="wide")
st.title("Hospital LHN Engagement Dashboard")

results_df, deal_df, lhn_stage_map = load_data(EXCEL_PATH)

# ---------------- Sidebar filters with "select all" + unit price + N ----------------
st.sidebar.header("Filters")

states_all = sorted(results_df['State'].dropna().unique())
stages_all = sorted({s for ss in lhn_stage_map.values() for s in ss})

sel_all_states = st.sidebar.checkbox("Select all States", value=True)
selected_states = st.sidebar.multiselect(
    "States", states_all, default=states_all if sel_all_states else []
)
if sel_all_states and set(selected_states) != set(states_all):
    selected_states = states_all

sel_all_stages = st.sidebar.checkbox("Select all Deal Stages", value=True)
selected_stages = st.sidebar.multiselect(
    "Deal Stages (controls 'Touched by Stage' + Deal list)",
    stages_all, default=stages_all if sel_all_stages else []
)
if sel_all_stages and set(selected_stages) != set(stages_all):
    selected_stages = stages_all

st.sidebar.markdown("---")
unit_price = st.sidebar.number_input("Unit price per staff (AUD)", min_value=0.0, value=1299.0, step=1.0)
n_topbottom = st.sidebar.number_input("Top/Bottom N by (Staff × Unit Price)", 3, 50, 10, 1)

# ---------------- Filter rows by states ----------------
filtered = results_df[results_df['State'].isin(selected_states)].copy()

# ---------------- Correct touched-by-stage logic ----------------
def touched_by_selected_stage(row) -> bool:
    rid = row['Record ID']
    key = str(int(rid)) if pd.notna(rid) and not isinstance(rid, str) else str(rid)
    stages = lhn_stage_map.get(key, set())
    return bool(stages.intersection(selected_stages)) if selected_stages else False

filtered['Touched_By_Stage'] = filtered.apply(touched_by_selected_stage, axis=1)

# Compute custom PTCV from sidebar price
for df_ in (filtered,):
    df_['Number of Staff'] = pd.to_numeric(df_['Number of Staff'], errors='coerce').clip(lower=0)
    df_['PTCV_custom'] = df_['Number of Staff'].fillna(0) * unit_price

# Split tables
touched_tbl = filtered[filtered['Touched_By_Stage']].copy()
untouched_tbl = filtered[~filtered['Touched_By_Stage']].copy()

# =========================================================
# Charts (computed AFTER filters so they reflect the view)
# =========================================================
# Coverage (any touch) for current states
cov_any = filtered.groupby('State', as_index=False).agg(
    Total=('LHN Name', 'count'),
    Touched=('Touched_Any', 'sum')
)
cov_any['Coverage Rate'] = cov_any['Touched'] / cov_any['Total']

# Coverage (by selected stages) for current states
cov_stage = filtered.groupby('State', as_index=False).agg(
    Total=('LHN Name', 'count'),
    Touched=('Touched_By_Stage', 'sum')
)
cov_stage['Coverage Rate'] = cov_stage['Touched'] / cov_stage['Total']

st.subheader("Coverage Rate by State (Any Touch)")
fig_cov_any = px.bar(
    cov_any.sort_values('Coverage Rate'),
    x='Coverage Rate', y='State', orientation='h', title='Coverage (Any Touch)'
)
fig_cov_any.update_xaxes(range=[0, 1])
fig_cov_any.update_traces(
    text=cov_any.sort_values('Coverage Rate')['Coverage Rate'].map(lambda x: f"{x:.1%}"),
    textposition='outside'
)
st.plotly_chart(fig_cov_any, use_container_width=True)


# Grouped counts by category for current states
st.subheader("LHNs by Category per State (Counts)")
cat_df = filtered.pivot_table(
    index='State', columns='Category', values='LHN Name', aggfunc='count', fill_value=0
).reset_index()
for c in ['H1','H2','Other','Untouched']:
    if c not in cat_df: cat_df[c] = 0
fig_grp = go.Figure()
for c in ['H1','H2','Other','Untouched']:
    fig_grp.add_bar(name=c, x=cat_df['State'], y=cat_df[c])
fig_grp.update_layout(
    barmode='group',
    xaxis_title='State', yaxis_title='Number of LHNs',
    title='LHNs by Category per State (Grouped Count)'
)
st.plotly_chart(fig_grp, use_container_width=True)

# Top / Bottom N by staff × unit price
st.subheader(f"Top LHNs by Potential Value (Staff × ${unit_price:,.0f})")
ptcv_df = filtered[['LHN Name','State','Number of Staff','PTCV_custom']].copy()
top_n = ptcv_df.sort_values('PTCV_custom', ascending=False).head(n_topbottom)
fig_top = px.bar(
    top_n.sort_values('PTCV_custom'),
    x='PTCV_custom', y='LHN Name', orientation='h', color='State',
    labels={'PTCV_custom':'Potential Value', 'LHN Name': 'LHN'}
)
fig_top.update_layout(xaxis_tickformat='$,.0f', title='Top N')
fig_top.update_traces(text=top_n.sort_values('PTCV_custom')['PTCV_custom'].map(lambda v: f"${v:,.0f}"),
                      textposition='outside')
st.plotly_chart(fig_top, use_container_width=True)

# =========================================================
# Touched vs Untouched tables (by selected stages) with selection
# =========================================================
st.subheader("Touched and Untouched LHNs by Selected Stage (with Staff × Unit Price)")
touched_tbl = touched_tbl[['LHN Name','State','Category','Number of Staff','PTCV_custom','Deal Indices']].sort_values(['State','LHN Name'])
untouched_tbl = untouched_tbl[['LHN Name','State','Category','Number of Staff','PTCV_custom','Deal Indices']].sort_values(['State','LHN Name'])

c1, c2 = st.columns(2)
with c1:
    st.markdown("**Touched LHNs**")
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
    st.markdown("**Untouched LHNs**")
    sel_all_untouched = st.checkbox("Select all untouched (below)", value=False, key="sel_all_untouched")
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

# =========================================================
# VIP + Deal list for selected LHNs
# =========================================================
st.subheader("VIP Contacts & Deals for Selected LHNs")
if len(picked_lhns) == 0:
    st.info("Pick LHNs using the checkboxes above, or click **Use ALL currently Touched**.")
else:
    # VIPs
    vip_view = filtered[filtered['LHN Name'].isin(picked_lhns)][['LHN Name','State','VIP Contacts']]
    st.markdown("**VIP Contacts**")
    st.dataframe(vip_view.sort_values(['State','LHN Name']).reset_index(drop=True), use_container_width=True)

    # Deal list (respect selected stages)
    st.markdown("**Deals**")
    # Gather all deal indices for the selected LHNs
    selected_rows = filtered[filtered['LHN Name'].isin(picked_lhns)]
    all_indices = []
    for _, r in selected_rows.iterrows():
        all_indices += r['Deal Indices']
    all_indices = sorted(set(all_indices))

    if len(all_indices) == 0:
        st.info("No deals found for the selected LHNs.")
    else:
        deals_sel = deal_df.loc[all_indices].copy()
        # Optional: filter by selected stages
        if selected_stages:
            deals_sel = deals_sel[deals_sel['Deal Stage'].isin(selected_stages)]
        # Attach LHN mapping by best-effort (via Associated Company Name to our picked set)
        # We’ll show the Associated Company Name as LHN-ish label
        cols = ['Deal Name','Deal Stage','Close Date','Deal owner','Amount',
                'Potential Total Contract Value','Associated Company Name','State']
        cols = [c for c in cols if c in deals_sel.columns]
        st.dataframe(
            deals_sel[cols].sort_values(['Deal Stage','Close Date'], na_position='last').reset_index(drop=True),
            use_container_width=True
        )
