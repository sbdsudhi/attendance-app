# streamlit_attendance_app.py
import streamlit as st
import pandas as pd
import os
from datetime import date, datetime

# ---------------- Config ----------------
# We will derive actual file names from the selected department
STUDENTS_MASTER = ""
ATTENDANCE_FILE = ""
DEPARTMENTS = ["ISE", "AIML"]

st.set_page_config(page_title="Attendance App (stable)", layout="wide")
st.title("📋 Attendance (stable) — counts, %, per-department")

# ----- Department selection -----
dept = st.selectbox("Select Department", DEPARTMENTS, index=0)
# Compute department-specific file names
STUDENTS_MASTER = f"students_master_{dept}.csv"
ATTENDANCE_FILE = f"attendance_{dept}.csv"

st.caption(f"Current department: **{dept}**  "
           f"(students: `{STUDENTS_MASTER}`, attendance: `{ATTENDANCE_FILE}`)")

# Reset per-department UI state if dept changed
if "current_dept" not in st.session_state:
    st.session_state["current_dept"] = dept
elif st.session_state["current_dept"] != dept:
    # Clear status selections when switching dept
    for k in list(st.session_state.keys()):
        if k.startswith("status_"):
            del st.session_state[k]
    st.session_state["current_dept"] = dept

# ---------------- Helpers ----------------
def read_students_from_upload(uploaded_file):
    try:
        fname = uploaded_file.name.lower()
        if fname.endswith((".xls", ".xlsx")):
            df = pd.read_excel(uploaded_file)
        else:
            df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Failed to read uploaded file: {e}")
        return None

    if df.empty:
        st.error("Uploaded file appears empty.")
        return None

    cols = list(df.columns)
    normalized = [str(c).strip().lower() for c in cols]
    col_map = dict(zip(normalized, cols))

    roll_candidates = [n for n in normalized if any(k in n for k in ("roll", "reg", "id"))]
    name_candidates = [n for n in normalized if any(k in n for k in ("name", "student", "full"))]

    if not roll_candidates or not name_candidates:
        st.error(
            "Could not auto-detect roll/name columns. Ensure your file has column names containing 'roll'/'reg'/'id' "
            "and 'name'/'student'/'full'.\n"
            f"Detected columns: {', '.join(cols)}"
        )
        return None

    roll_col = col_map[roll_candidates[0]]
    name_col = col_map[name_candidates[0]]

    df2 = df[[roll_col, name_col]].copy()
    df2.columns = ["roll", "name"]
    df2["roll"] = df2["roll"].astype(str).str.strip()
    df2["name"] = df2["name"].astype(str).str.strip()
    df2 = df2[~((df2["roll"] == "") & (df2["name"] == ""))]
    df2 = df2.reset_index(drop=True)
    return df2

def ensure_unique_rolls(df):
    df = df.copy().reset_index(drop=True)
    # fill blanks
    for i in range(len(df)):
        r = str(df.at[i, 'roll']).strip()
        if r == "" or r.lower() == "nan":
            df.at[i, 'roll'] = f"__R_{i+1}"
    # make unique suffixes for duplicates
    counts = {}
    for i in range(len(df)):
        r = str(df.at[i, 'roll'])
        if r not in counts:
            counts[r] = 1
        else:
            counts[r] += 1
            df.at[i, 'roll'] = f"{r}__{counts[r]}"
    df['roll'] = df['roll'].astype(str).str.strip()
    df['name'] = df['name'].astype(str).str.strip()
    return df

def load_master():
    if os.path.exists(STUDENTS_MASTER):
        df = pd.read_csv(STUDENTS_MASTER, dtype=str).fillna("")
        return ensure_unique_rolls(df)
    return None

def save_master(df):
    df.to_csv(STUDENTS_MASTER, index=False)

def load_attendance():
    if os.path.exists(ATTENDANCE_FILE):
        df = pd.read_csv(ATTENDANCE_FILE, dtype=str).fillna("")
        return df
    return None

def save_attendance(df):
    df.to_csv(ATTENDANCE_FILE, index=False)

def _recompute_counts_and_reorder(saved_df):
    """
    Recompute Total_P, Total_A, Total_Classes, Attendance_% and reorder columns.
    Uses pandas-safe .map instead of deprecated applymap.
    """
    df = saved_df.copy()
    # Remove old totals if present
    for col in ["Total_P", "Total_A", "Total_Classes", "Attendance_%"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Attendance columns are everything except id/name/totals
    att_cols = [c for c in df.columns if c not in ("roll", "name", "Total_P", "Total_A", "Total_Classes", "Attendance_%")]
    if att_cols:
        att_block = df[att_cols].fillna("").astype(str)

        # Use .map on each column (no applymap)
        bool_p = att_block.apply(lambda col: col.map(lambda x: x.upper().startswith('P')))
        bool_a = att_block.apply(lambda col: col.map(lambda x: x.upper().startswith('A')))

        p_counts = bool_p.sum(axis=1).astype(int)
        a_counts = bool_a.sum(axis=1).astype(int)
    else:
        p_counts = pd.Series([0] * len(df), index=df.index)
        a_counts = pd.Series([0] * len(df), index=df.index)

    df["Total_P"] = p_counts
    df["Total_A"] = a_counts
    df["Total_Classes"] = p_counts + a_counts

    # attendance percentage: P / classes * 100
    tc_nonzero = df["Total_Classes"].replace(0, pd.NA)
    df["Attendance_%"] = ((df["Total_P"] / tc_nonzero) * 100).round(1)
    df["Attendance_%"] = df["Attendance_%"].fillna(0)

    remaining = [
        c for c in df.columns
        if c not in ("roll", "name", "Total_P", "Total_A", "Total_Classes", "Attendance_%")
    ]
    new_order = ["roll", "name", "Total_P", "Total_A", "Total_Classes", "Attendance_%"] + remaining
    df = df.reindex(columns=new_order)
    return df

def do_save_statuses(statuses, datetime_col, master_df, working):
    """
    Save statuses (dict roll->'P'/'A') into attendance file under datetime_col.
    Also recomputes Total_P / Total_A / Total_Classes / Attendance_% and writes them into the CSV.
    If existing attendance.csv is missing 'roll' or 'name', it will be reset from working.
    """
    # ensure every student in working has a status
    for r in working['roll']:
        if r not in statuses:
            statuses[r] = 'A'

    saved = load_attendance()

    # If no file OR corrupted (missing roll/name), start fresh from working
    if (saved is None) or ('roll' not in saved.columns) or ('name' not in saved.columns):
        saved = working[['roll', 'name']].copy()
        saved = saved.set_index('roll')
    else:
        saved = saved.set_index('roll')
        m = master_df.set_index('roll')
        # add missing students and ensure name from master
        for r, n in m['name'].items():
            if str(r) not in saved.index:
                saved.loc[str(r), 'name'] = n
        for r in saved.index:
            if r in m.index:
                saved.at[r, 'name'] = m.at[r, 'name']
        saved = saved.reset_index()
        saved = saved.set_index('roll')

    # write the new column
    for r, s in statuses.items():
        saved.loc[r, datetime_col] = 'P' if str(s).upper().startswith('P') else 'A'

    saved = saved.reset_index()
    saved = _recompute_counts_and_reorder(saved)
    save_attendance(saved)
    return saved

def get_counts_for_roll(saved_att_df, roll):
    """
    Return (total_p, total_a, total_classes, attendance_percent) for a given roll.
    If saved_att_df is None or lacks 'roll', returns zeros.
    """
    if saved_att_df is None:
        return 0, 0, 0, 0.0
    df = saved_att_df.copy()

    # If roll column missing, we can't look it up safely
    if "roll" not in df.columns:
        return 0, 0, 0, 0.0

    try:
        row = df.set_index('roll').loc[str(roll)]
    except Exception:
        return 0, 0, 0, 0.0

    # Prefer totals if present
    tp = int(row.get("Total_P", 0) or 0)
    ta = int(row.get("Total_A", 0) or 0)
    tc = int(row.get("Total_Classes", tp + ta) or 0)

    if "Attendance_%" in df.columns:
        pct = float(row.get("Attendance_%", 0.0) or 0.0)
    else:
        pct = (tp * 100.0 / tc) if tc > 0 else 0.0
        pct = round(pct, 1)

    return tp, ta, tc, pct

# ---------------- Upload / master ----------------
st.header("1) Upload student list (one-time)")

col1, col2 = st.columns([3, 1])
with col1:
    uploaded = st.file_uploader(
        f"Upload CSV or Excel with roll & name columns for {dept}",
        type=["csv", "xls", "xlsx"]
    )
with col2:
    if st.button(f"Clear saved {dept} student list"):
        if os.path.exists(STUDENTS_MASTER):
            os.remove(STUDENTS_MASTER)
            st.success(f"Saved student list for {dept} cleared.")

master_df = None
if uploaded:
    dfu = read_students_from_upload(uploaded)
    if dfu is not None:
        dfu = ensure_unique_rolls(dfu)
        save_master(dfu)
        master_df = dfu
        st.success(f"Saved {len(dfu)} {dept} students to {STUDENTS_MASTER}")
else:
    master_df = load_master()
    if master_df is None:
        st.info(f"No saved student list found for {dept}. Please upload one.")

if master_df is not None:
    st.subheader(f"Saved students (master) — {dept}")
    st.dataframe(master_df, height=260)

# ---------------- Attendance UI ----------------
if master_df is None or master_df.empty:
    st.stop()

st.header(f"2) Take attendance (stable) — {dept}")

cdate, ctime = st.columns([2, 1])
with cdate:
    att_date = st.date_input("Select date", value=date.today(), key=f"date_{dept}")
with ctime:
    att_time = st.time_input(
        "Time of class",
        value=datetime.now().time().replace(second=0, microsecond=0),
        key=f"time_{dept}"
    )

datetime_col = f"{att_date.isoformat()}_{att_time.strftime('%H-%M')}"
st.caption(f"Attendance column will be: {datetime_col}  (Dept: {dept})")

saved_att = load_attendance()

# build working table (ordered by master)
if saved_att is None:
    working = master_df[['roll', 'name']].copy()
else:
    if 'roll' not in saved_att.columns or 'name' not in saved_att.columns:
        # ignore corrupted attendance, still use master for working
        working = master_df[['roll', 'name']].copy()
    else:
        saved_idx = saved_att.set_index('roll')
        master_idx = master_df.set_index('roll')
        for r, n in master_idx['name'].items():
            if str(r) not in saved_idx.index:
                saved_idx.loc[str(r), 'name'] = n
        saved_idx = saved_idx.reset_index().set_index('roll').reindex(master_idx.index)
        saved_idx['name'] = master_idx['name']
        working = saved_idx.reset_index()[['roll', 'name']]

# controls row
r1, r2, r3, r4 = st.columns([1, 1, 1, 2])
with r1:
    if st.button("🔄 Refresh (clear selections)", key=f"refresh_{dept}"):
        for k in list(st.session_state.keys()):
            if k.startswith("status_"):
                del st.session_state[k]
        st.success("Selections cleared.")
with r2:
    if st.button("🚫 Mark All Absent", key=f"all_abs_{dept}"):
        for i, r in enumerate(working['roll']):
            st.session_state[f"status_{i}_{r}"] = "Absent"
        st.success("All marked Absent (in UI).")
with r3:
    if st.button("✅ Mark All Present", key=f"all_pres_{dept}"):
        for i, r in enumerate(working['roll']):
            st.session_state[f"status_{i}_{r}"] = "Present"
        st.success("All marked Present (in UI).")
with r4:
    if st.button("🚫 All Absent & Save", key=f"all_abs_save_{dept}"):
        statuses_tmp = {r: 'A' for r in working['roll']}
        try:
            saved_df = do_save_statuses(statuses_tmp, datetime_col, master_df, working)
            st.success(f"[{dept}] Saved (all absent) to {ATTENDANCE_FILE} under '{datetime_col}'")
            saved_att = saved_df
        except Exception as e:
            st.error(f"Save failed: {e}")
    if st.button("✅ All Present & Save", key=f"all_pres_save_{dept}"):
        statuses_tmp = {r: 'P' for r in working['roll']}
        try:
            saved_df = do_save_statuses(statuses_tmp, datetime_col, master_df, working)
            st.success(f"[{dept}] Saved (all present) to {ATTENDANCE_FILE} under '{datetime_col}'")
            saved_att = saved_df
        except Exception as e:
            st.error(f"Save failed: {e}")

# compact rows (one-line each) with vertical counts box
st.markdown(f"### Students — compact view (counts + % at right) — {dept}")
for i, row in working.reset_index(drop=True).iterrows():
    r = str(row['roll'])
    n = str(row['name'])
    cols = st.columns([1.2, 3, 3, 1.6], gap="small")
    cols[0].markdown(f"**{r}**")
    cols[1].markdown(n)

    key = f"status_{i}_{r}"
    if key not in st.session_state:
        # default from saved_att if available else Present
        if saved_att is not None and "roll" in saved_att.columns and datetime_col in saved_att.columns:
            sa_idx = saved_att.set_index('roll')
            if str(r) in sa_idx.index:
                prev = sa_idx.at[str(r), datetime_col]
                st.session_state[key] = 'Present' if str(prev).upper().startswith('P') else 'Absent'
            else:
                st.session_state[key] = 'Present'
        else:
            st.session_state[key] = 'Present'

    # side-by-side Present / Absent / Status
    pcol, acol, scol = cols[2].columns([1, 1, 1], gap="small")
    if pcol.button("Present", key=f"p_{i}_{r}_{dept}"):
        st.session_state[key] = "Present"
    if acol.button("Absent", key=f"a_{i}_{r}_{dept}"):
        st.session_state[key] = "Absent"

    cur = st.session_state[key]
    color = "#0d6efd" if cur == "Present" else "#dc3545"
    scol.markdown(
        f"<div style='padding:6px 10px;border-radius:6px;background:{color};"
        f"color:white;text-align:center;font-weight:600'>{cur}</div>",
        unsafe_allow_html=True,
    )

    # vertical counts box at far right (P, A, Classes, %, Now)
    tp, ta, tc, pct = get_counts_for_roll(saved_att, r)
    now_status = 'P' if cur == 'Present' else 'A'
    v_html = (
        f"<div style='text-align:center;padding:6px;border-radius:6px;border:1px solid #e6e6e6'>"
        f"<div style='font-size:12px;color:#444;margin-bottom:6px'><b>Totals</b></div>"
        f"<div style='font-size:13px;margin-bottom:4px'>P: <b>{tp}</b></div>"
        f"<div style='font-size:13px;margin-bottom:4px'>A: <b>{ta}</b></div>"
        f"<div style='font-size:13px;margin-bottom:4px'>Classes: <b>{tc}</b></div>"
        f"<div style='font-size:13px;margin-bottom:6px'>%: <b>{pct}</b></div>"
        f"<div style='padding:6px 8px;border-radius:6px;background:#111827;color:white;font-weight:700'>Now: {now_status}</div>"
        f"</div>"
    )
    cols[3].markdown(v_html, unsafe_allow_html=True)

# collect statuses for manual Submit
statuses = {}
for i, r in enumerate(working['roll']):
    key = f"status_{i}_{r}"
    cur = st.session_state.get(key, 'Absent')
    statuses[r] = 'P' if cur == 'Present' else 'A'

# Manual submit -> save directly
if st.button("Submit & Save", key=f"submit_{dept}"):
    try:
        saved_df = do_save_statuses(statuses, datetime_col, master_df, working)
        st.success(f"[{dept}] Saved (from current selections) to {ATTENDANCE_FILE} under '{datetime_col}'")
        saved_att = saved_df
    except Exception as e:
        st.error(f"Save failed: {e}")

# Show attendance file (scrollable)
st.markdown(f"### Attendance file — {dept}")
att = load_attendance()
if att is not None:
    st.dataframe(att, height=420)
else:
    st.info(f"{ATTENDANCE_FILE} not created yet for {dept}.")
