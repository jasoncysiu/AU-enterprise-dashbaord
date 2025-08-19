# app.py
# ------------------------------------------------------------
# Streamlit app: State → LHN → Facility (PUBLIC) + VIP overlay
# - Google Sheets: local or uploaded service_account.json (no st.secrets)
# - CSV uploads as fallback
# - Preserves staff values (no aggregation)
# - Simple table views (no collapsible grid)
# - Record ID filter + mapping (Record ID ↔ Company Record ID)
# - Download buttons
#
# Install:
#   pip install streamlit pandas numpy gspread google-auth
# Run:
#   streamlit run app.py
# ------------------------------------------------------------

import json
import os
import re
import streamlit as st
import pandas as pd
import numpy as np

# ====== GOOGLE SHEETS CONFIG ======
import gspread
from google.oauth2.service_account import Credentials
from google.oauth2 import service_account as sa_mod

SCOPES = ["https://www.googleapis.com/auth/spreadsheets.readonly"]
DEFAULT_SHEET_ID = "1zX5r8SaqnNqOsQRxqSfB2tuRqoONQMhrtpHQUfVLwxs"   # <-- your Sheet ID
DEFAULT_SA_PATH = "service_account.json"                              # local path to service account JSON


def get_gspread_client_from_file(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Service account file not found at: {path}")
    creds = Credentials.from_service_account_file(path, scopes=SCOPES)
    return gspread.authorize(creds)


def get_gspread_client_from_uploaded_json(uploaded_file):
    if uploaded_file is None:
        raise RuntimeError("Please upload your service account JSON.")
    info = json.loads(uploaded_file.getvalue().decode("utf-8"))
    creds = sa_mod.Credentials.from_service_account_info(info, scopes=SCOPES)
    return gspread.authorize(creds)


@st.cache_data(show_spinner=True, ttl=600)
def load_from_google_sheets(sheet_id: str, auth_method: str, sa_path: str = None, sa_uploaded=None):
    """Load 'Masterlist' and 'VIP' worksheets as DataFrames."""
    if auth_method == "Local service_account.json":
        client = get_gspread_client_from_file(sa_path or DEFAULT_SA_PATH)
    elif auth_method == "Upload service_account.json":
        client = get_gspread_client_from_uploaded_json(sa_uploaded)
    else:
        raise ValueError("Unknown auth method.")

    sh = client.open_by_key(sheet_id)

    # Masterlist
    ws_master = sh.worksheet("Masterlist")
    rows_master = ws_master.get_all_values()
    if not rows_master:
        raise RuntimeError("Masterlist sheet is empty.")
    df_master = pd.DataFrame(rows_master[1:], columns=rows_master[0])

    # VIP (optional)
    try:
        ws_vip = sh.worksheet("VIP")
        rows_vip = ws_vip.get_all_values()
        df_vip = pd.DataFrame(rows_vip[1:], columns=rows_vip[0]) if rows_vip else pd.DataFrame(
            columns=['Org','State','Name','Company Record ID','Org_ID','Title','Organisation','Email','Owner','Note']
        )
    except gspread.WorksheetNotFound:
        df_vip = pd.DataFrame(columns=['Org','State','Name','Company Record ID','Org_ID','Title','Organisation','Email','Owner','Note'])

    return df_master, df_vip


# ====== HIERARCHY + VIP UTILS ======
def transform_hierarchy_preserve(df_in: pd.DataFrame, drop_empty_parents: bool = True) -> pd.DataFrame:
    """
    Build State → LHN → Facility hierarchy.
    Preserve staff values exactly as in source (no aggregation).
    """
    df = df_in.copy()

    # Normalize key columns
    for col in ['Granularity', 'State', 'Name', 'Local Hospital Network (LHN)']:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].astype(str).str.strip()

    g = df['Granularity'].str.casefold()

    # Include useful extra columns if present
    bed_col = 'Number of bed (Actual)' if 'Number of bed (Actual)' in df.columns else ('# of bed (Actual)' if '# of bed (Actual)' in df.columns else None)
    staff_col = 'Number of clincial staff' if 'Number of clincial staff' in df.columns else ('Number of clinical staff' if 'Number of clinical staff' in df.columns else None)
    extras = ['Sector', 'Status', 'Notes', 'Org_ID', 'Record ID', 'AIHW Ref#', 'Provider Number']
    extra_cols = [c for c in [bed_col, staff_col, *extras] if c and c in df.columns]

    # Preserve first-appearance order
    state_order = df.loc[g.eq('state'), 'State'].drop_duplicates().tolist()
    state_pos = {s: i for i, s in enumerate(state_order)}

    lhn_pos, seen_lhn = {}, {}
    for _, r in df[g.eq('lhn')][['State', 'Name']].iterrows():
        s, l = r['State'], r['Name']
        if (s, l) not in lhn_pos:
            seen_lhn.setdefault(s, 0)
            lhn_pos[(s, l)] = seen_lhn[s]
            seen_lhn[s] += 1

    fac_pos, seen_fac = {}, {}
    for _, r in df[g.eq('facility')][['State', 'Local Hospital Network (LHN)', 'Name']].iterrows():
        key = (r['State'], r['Local Hospital Network (LHN)'])
        name = r['Name']
        if (key[0], key[1], name) not in fac_pos:
            seen_fac.setdefault(key, 0)
            fac_pos[(key[0], key[1], name)] = seen_fac[key]
            seen_fac[key] += 1

    # Build layers (preserve provided values)
    states = (
        df[g.eq('state')]
        .drop_duplicates(subset=['State'])
        .assign(**{
            'Local Hospital Network (LHN)': '',
            'Hospital Name': '',
            'Granularity': 'State'
        })[['State', 'Local Hospital Network (LHN)', 'Hospital Name', 'Granularity'] + extra_cols]
    )

    lhns = (
        df[g.eq('lhn')][['State', 'Name'] + [c for c in extra_cols if c in df.columns]].drop_duplicates()
        .rename(columns={'Name': 'Local Hospital Network (LHN)'})
        .assign(**{
            'Hospital Name': '',
            'Granularity': 'LHN'
        })[['State', 'Local Hospital Network (LHN)', 'Hospital Name', 'Granularity'] + extra_cols]
    )

    facilities = (
        df[g.eq('facility')]
        .rename(columns={'Name': 'Hospital Name'})
        [['State', 'Local Hospital Network (LHN)', 'Hospital Name', 'Granularity'] + extra_cols]
        .assign(Granularity='Facility')
    )

    # Optionally drop parents with no children beneath
    if drop_empty_parents:
        if not facilities.empty:
            valid_lhns = set(map(tuple, facilities[['State', 'Local Hospital Network (LHN)']].drop_duplicates().to_numpy()))
            lhns = lhns[lhns.apply(lambda r: (r['State'], r['Local Hospital Network (LHN)']) in valid_lhns, axis=1)]
        valid_states = set(lhns['State']).union(set(facilities['State']))
        states = states[states['State'].isin(valid_states)]

    out = pd.concat([states, lhns, facilities], ignore_index=True)

    # Sort hierarchical order
    def sort_key(r):
        s, l, h, g_ = r['State'], r['Local Hospital Network (LHN)'], r['Hospital Name'], r['Granularity'].casefold()
        s_idx = state_pos.get(s, 10**9)
        if g_ == 'state':
            return (s_idx, -1, -1)
        elif g_ == 'lhn':
            return (s_idx, lhn_pos.get((s, l), 10**9), -1)
        else:
            return (s_idx, lhn_pos.get((s, l), 10**9), fac_pos.get((s, l, h), 10**9))

    out['_k'] = out.apply(sort_key, axis=1)
    out = out.sort_values('_k').drop(columns=['_k']).reset_index(drop=True)
    return out


def format_for_display(df_hier: pd.DataFrame) -> pd.DataFrame:
    """Flat table with parents blanked for readability."""
    df = df_hier.copy()
    df['State (display)'] = df['State']
    df['LHN (display)'] = df['Local Hospital Network (LHN)']
    df['Hospital (display)'] = df['Hospital Name']

    df.loc[df['Granularity'] != 'State', 'State (display)'] = ''
    df.loc[df['Granularity'] == 'Facility', 'LHN (display)'] = ''
    df.loc[df['Granularity'] != 'Facility', 'Hospital (display)'] = ''

    preferred = [
        'State (display)', 'LHN (display)', 'Hospital (display)',
        'Granularity', 'Sector',
        'Number of bed (Actual)', '# of bed (Actual)',
        'Number of clincial staff', 'Number of clinical staff',
        'Status', 'Notes', 'Org_ID', 'Record ID', 'Provider Number', 'AIHW Ref#', 'VIP'
    ]
    cols_front = [c for c in preferred if c in df.columns]
    cols_other = [c for c in df.columns if c not in cols_front + ['State', 'Local Hospital Network (LHN)', 'Hospital Name']]
    df = df[cols_front + cols_other].rename(columns={
        'State (display)': 'State',
        'LHN (display)': 'Local Hospital Network (LHN)',
        'Hospital (display)': 'Hospital Name'
    })
    return df


def hierarchy_as_indented_labels(df_hier: pd.DataFrame) -> pd.DataFrame:
    """One 'Label' column with ASCII indentation for a clean human-readable tree."""
    level = df_hier['Granularity'].str.lower()
    label = (df_hier['State'].where(level.eq('state'))
             .fillna(df_hier['Local Hospital Network (LHN)'].where(level.eq('lhn')))
             .fillna(df_hier['Hospital Name'].where(level.eq('facility'))))
    indent = np.where(level.eq('state'), '',
              np.where(level.eq('lhn'), '   └─ ',
              '      └─ '))
    out = pd.DataFrame({'Label': indent + label})
    # Bring along useful columns
    for c in ['Granularity','Sector','Number of bed (Actual)','# of bed (Actual)',
              'Number of clincial staff','Number of clinical staff','Status','Notes',
              'Org_ID','Record ID','Provider Number','AIHW Ref#','VIP']:
        if c in df_hier.columns:
            out[c] = df_hier[c]
    return out


def build_vip_flag(df_hier: pd.DataFrame, df_vip: pd.DataFrame, prefer_record_id: bool = True) -> pd.DataFrame:
    """
    Mark VIP rows. If prefer_record_id=True, mark VIPs primarily by Record ID match:
      df_hier['Record ID'] == df_vip['Company Record ID'].
    Falls back to Org_ID and name-based matches.
    """
    out = df_hier.copy()

    def _norm(s): return s.astype(str).str.strip()

    for c in ['Org_ID', 'State', 'Local Hospital Network (LHN)', 'Hospital Name', 'Record ID']:
        if c in out.columns:
            out[c] = _norm(out[c])

    for c in ['Org_ID','State','Name','Organisation','Org','Company Record ID']:
        if c in df_vip.columns:
            df_vip[c] = _norm(df_vip[c])

    if 'VIP' not in out.columns:
        out['VIP'] = False

    # 0) Preferred: Record ID ↔ Company Record ID
    if prefer_record_id and ('Record ID' in out.columns) and ('Company Record ID' in df_vip.columns):
        vip_rids = set(df_vip['Company Record ID'].dropna())
        out.loc[out['Record ID'].isin(vip_rids), 'VIP'] = True

    # 1) Org_ID
    if 'Org_ID' in out.columns and 'Org_ID' in df_vip.columns:
        vip_orgs = set(df_vip['Org_ID'].dropna())
        out.loc[out['Org_ID'].isin(vip_orgs), 'VIP'] = True

    # 2) Name-based fallbacks
    if 'Name' in df_vip.columns:
        name_set = set(df_vip['Name'].dropna().str.casefold())
        out.loc[out['Hospital Name'].str.casefold().isin(name_set), 'VIP'] = True
    if 'Organisation' in df_vip.columns:
        org_set = set(df_vip['Organisation'].dropna().str.casefold())
        out.loc[out['Local Hospital Network (LHN)'].str.casefold().isin(org_set), 'VIP'] = True
    if 'Org' in df_vip.columns:
        org2_set = set(df_vip['Org'].dropna().str.casefold())
        out.loc[out['Hospital Name'].str.casefold().isin(org2_set), 'VIP'] = True
        out.loc[out['Local Hospital Network (LHN)'].str.casefold().isin(org2_set), 'VIP'] = True

    # 3) Tighten by State if available
    if 'State' in df_vip.columns:
        vip_pairs = set(zip(df_vip['Name'].fillna('').str.casefold(),
                            df_vip['State'].fillna('').str.upper()))
        out_pairs = list(zip(out['Hospital Name'].fillna('').str.casefold(),
                             out['State'].fillna('').str.upper()))
        out.loc[[p in vip_pairs for p in out_pairs], 'VIP'] = True

    return out


def parse_record_id_input(text: str) -> list[str]:
    """Parse pasted Record IDs (comma/newline/space separated)."""
    if not text:
        return []
    tokens = re.split(r"[,\s]+", text.strip())
    return [t for t in tokens if t]


def build_record_id_mapping(df_master_public: pd.DataFrame, df_vip: pd.DataFrame) -> pd.DataFrame:
    """
    Join Masterlist ↔ VIP on Record ID ↔ Company Record ID.
    Returns a tidy mapping table for inspection/export.
    """
    m = df_master_public.copy()
    v = df_vip.copy()

    for c in ['Record ID','Org_ID','State','Name','Local Hospital Network (LHN)','Granularity',
              'Provider Number','AIHW Ref#','Status','Notes']:
        if c in m.columns:
            m[c] = m[c].astype(str).str.strip()

    for c in ['Company Record ID','Org_ID','State','Name','Organisation','Owner','Email','Title','Note','Org']:
        if c in v.columns:
            v[c] = v[c].astype(str).str.strip()

    left_cols = [
        'Record ID','State','Local Hospital Network (LHN)','Name','Granularity',
        'Org_ID','Provider Number','AIHW Ref#','Status','Notes'
    ]
    left_cols = [c for c in left_cols if c in m.columns]

    right_cols = [
        'Company Record ID','Org','Organisation','Name','State','Owner','Email','Title','Note','Org_ID'
    ]
    right_cols = [c for c in right_cols if c in v.columns]

    map_df = m[left_cols].merge(
        v[right_cols],
        left_on='Record ID', right_on='Company Record ID',
        how='left', suffixes=('', ' (VIP)')
    )

    # Helpful final ordering
    preferred = [
        'Record ID','Company Record ID',
        'State','Local Hospital Network (LHN)','Name','Granularity',
        'Org_ID','Org_ID (VIP)','Organisation','Org','Owner','Email','Title','Note',
        'Provider Number','AIHW Ref#','Status','Notes'
    ]
    cols = [c for c in preferred if c in map_df.columns] + [c for c in map_df.columns if c not in preferred]
    return map_df[cols]


# ====== STREAMLIT UI (simple) ======
st.set_page_config(page_title="Health Hierarchy + VIPs — Simple + Record ID filter", layout="wide")
st.title("🏥 State → LHN → Facility (PUBLIC) + VIP overlay")

# Sidebar: data source
st.sidebar.subheader("Data source")
source = st.sidebar.radio("Choose source", ["Google Sheets", "CSV upload"], index=0)

df_master, df_vip = None, None

if source == "Google Sheets":
    sheet_id = st.sidebar.text_input("Google Sheet ID", value=DEFAULT_SHEET_ID, help="The long ID from your Sheet URL.")
    auth_method = st.sidebar.radio("Auth method", ["Local service_account.json", "Upload service_account.json"], index=0)

    if auth_method == "Local service_account.json":
        sa_path = st.sidebar.text_input("Path to service_account.json", value=DEFAULT_SA_PATH)
        try:
            with st.spinner("Loading from Google Sheets..."):
                df_master, df_vip = load_from_google_sheets(sheet_id, auth_method, sa_path=sa_path, sa_uploaded=None)
            st.success(f"Loaded Masterlist ({len(df_master):,} rows) & VIP ({len(df_vip):,} rows).")
        except Exception as e:
            st.error(f"Google Sheets load failed: {e}")
            st.stop()
    else:
        sa_uploaded = st.sidebar.file_uploader("Upload service_account.json", type=["json"])
        if sa_uploaded is None:
            st.info("Upload your service account JSON to continue.")
            st.stop()
        try:
            with st.spinner("Loading from Google Sheets..."):
                df_master, df_vip = load_from_google_sheets(sheet_id, auth_method, sa_uploaded=sa_uploaded)
            st.success(f"Loaded Masterlist ({len(df_master):,} rows) & VIP ({len(df_vip):,} rows).")
        except Exception as e:
            st.error(f"Google Sheets load failed: {e}")
            st.stop()

else:
    st.sidebar.write("Upload your CSVs")
    up_master = st.sidebar.file_uploader("Masterlist CSV", type=["csv"], key="master")
    up_vip    = st.sidebar.file_uploader("VIP CSV (optional)", type=["csv"], key="vip")
    if up_master is None:
        st.info("Please upload the **Masterlist CSV** to continue.")
        st.stop()
    df_master = pd.read_csv(up_master)
    df_vip = (pd.read_csv(up_vip) if up_vip is not None
              else pd.DataFrame(columns=['Org','State','Name','Company Record ID','Org_ID','Title','Organisation','Email','Owner','Note']))
    st.success(f"Loaded Masterlist ({len(df_master):,}) and VIP ({len(df_vip):,}).")

# Filter to PUBLIC only
if 'Sector' in df_master.columns:
    df_master_public = df_master[df_master['Sector'].astype(str).str.strip().str.upper() == 'PUBLIC'].copy()
else:
    df_master_public = df_master.copy()

# Build hierarchy (preserving staff values as-is)
df_hier = transform_hierarchy_preserve(df_master_public, drop_empty_parents=True)

# VIP overlay (prefer Record ID matches)
df_hier = build_vip_flag(df_hier, df_vip, prefer_record_id=True)

# ---- Record ID filter UI ----
st.sidebar.subheader("Record ID filter")
all_record_ids = sorted(df_hier['Record ID'].dropna().astype(str).unique()) if 'Record ID' in df_hier.columns else []
sel_ids = st.sidebar.multiselect("Choose Record ID(s)", options=all_record_ids, default=[])

paste_ids = st.sidebar.text_area("…or paste Record ID(s) (comma / space / newline separated)", value="")
parsed_ids = parse_record_id_input(paste_ids)

# Combine both sources
selected_rids = set(sel_ids) | set(parsed_ids)

# Apply Record ID filter (if any)
if selected_rids and 'Record ID' in df_hier.columns:
    df_hier_f = df_hier[df_hier['Record ID'].astype(str).isin(selected_rids)].copy()
else:
    df_hier_f = df_hier.copy()

# Views toggle
st.sidebar.subheader("View")
view_style = st.sidebar.radio("Display as", ["Flat table (parents blanked)", "Indented 'Label' column"], index=0)
vip_only   = st.sidebar.checkbox("Show VIPs only", value=False)

# Format view
if view_style == "Flat table (parents blanked)":
    df_view = format_for_display(df_hier_f)
else:
    df_view = hierarchy_as_indented_labels(df_hier_f)

if vip_only and 'VIP' in df_view.columns:
    df_view_disp = df_view[df_view['VIP'] == True].copy()
else:
    df_view_disp = df_view.copy()

# ---- Main table ----
st.subheader("Hierarchy Table")
st.dataframe(df_view_disp, use_container_width=True)

# Download current view
csv = df_view_disp.to_csv(index=False).encode('utf-8')
st.download_button("⬇️ Download current view (CSV)", data=csv, file_name="hierarchy_view.csv", mime="text/csv")

# ---- Record ID Mapping view ----
st.subheader("Record ID Mapping (Masterlist ↔ VIP)")
map_df = build_record_id_mapping(df_master_public, df_vip)

# Filter mapping by selected rids too
if selected_rids and 'Record ID' in map_df.columns:
    map_df = map_df[map_df['Record ID'].astype(str).isin(selected_rids)]

st.dataframe(map_df, use_container_width=True)

# Download mapping
csv_map = map_df.to_csv(index=False).encode('utf-8')
st.download_button("⬇️ Download mapping (CSV)", data=csv_map, file_name="record_id_mapping.csv", mime="text/csv")

# VIP source preview
st.subheader("VIP List (source)")
st.dataframe(df_vip if not df_vip.empty else pd.DataFrame({'Info': ['(No VIP rows loaded)']}))
