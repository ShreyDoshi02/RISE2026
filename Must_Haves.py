import streamlit as st
import pandas as pd
import datetime
from io import BytesIO

st.set_page_config(page_title="CSV Tool")

if "step" not in st.session_state:
    st.session_state.step = 1

if st.session_state.step == 1:
    st.title("Upload File")

    file = st.file_uploader("Upload CSV", type=["csv"])

    if file:
        df = pd.read_csv(file)
        st.session_state.df = df

        st.write("Preview")
        st.dataframe(df.head(50))

        if st.button("Next"):
            st.session_state.step = 2

elif st.session_state.step == 2:
    st.title("Column Mapping")

    df = st.session_state.df
    cols = df.columns.tolist()

    user_id = st.selectbox("User ID column", cols)
    date = st.selectbox("Date column", cols)
    amount = st.selectbox("Amount column", cols)

    st.session_state.mapping = {
        "User_ID": user_id,
        "Transaction_Date": date,
        "Amount": amount
    }

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Back"):
            st.session_state.step = 1

    with col2:
        if st.button("Next"):
            st.session_state.step = 3

elif st.session_state.step == 3:
    st.title("Validation + Transform")

    df = st.session_state.df.copy()
    mapping = st.session_state.mapping

    df.rename(columns={
        mapping["User_ID"]: "User_ID",
        mapping["Transaction_Date"]: "Transaction_Date",
        mapping["Amount"]: "Amount"
    }, inplace=True)

    if pd.to_numeric(df["Amount"], errors="coerce").isnull().any():
        st.error("Amount has invalid values")

    try:
        pd.to_datetime(df["Transaction_Date"])
    except:
        st.error("Date format is wrong")

    remove_dup = st.checkbox("Remove duplicates")
    fill_null = st.checkbox("Fill null with 0")
    add_calc = st.checkbox("Add Adjusted Amount")

    if add_calc:
        multiplier = st.number_input("Multiplier", value=1.0)

    if remove_dup:
        df = df.drop_duplicates()

    if fill_null:
        df = df.fillna(0)

    if add_calc:
        df["Adjusted_Amount"] = df["Amount"] * multiplier

    st.session_state.final_df = df

    st.dataframe(df.head(50))

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Back"):
            st.session_state.step = 2

    with col2:
        if st.button("Next"):
            st.session_state.step = 4

elif st.session_state.step == 4:
    st.title("Download File")

    df = st.session_state.final_df

    st.dataframe(df.head(50))

    time = datetime.datetime.now().strftime("%Y%m%d_%H%M")

    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download CSV",
        data=csv,
        file_name=f"file_{time}.csv"
    )

    buffer = BytesIO()
    df.to_excel(buffer, index=False)
    buffer.seek(0)

    st.download_button(
        "Download Excel",
        data=buffer,
        file_name=f"file_{time}.xlsx"
    )

    if st.button("Back"):
        st.session_state.step = 3