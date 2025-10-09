from streamlit.runtime.media_file_storage import MediaFileStorageError
from vectordb_bench.frontend.config.styles import DB_SELECTOR_COLUMNS, DB_TO_ICON
from vectordb_bench.frontend.config.dbCaseConfigs import DB_LIST
import streamlit as st
import sys


def dbSelector(st: st):
    # DEBUG: Print to console AND show in UI
    db_names = [db.name for db in DB_LIST]
    icon_names = [db.name for db in DB_TO_ICON.keys()]

    debug_msg = f"""
    DEBUG dbSelector called!
    - DB_LIST length: {len(DB_LIST)}
    - DB_LIST: {db_names}
    - DB_TO_ICON keys: {icon_names}
    - Antfly in DB_LIST: {'Antfly' in db_names}
    - Antfly in DB_TO_ICON: {'Antfly' in icon_names}
    """
    print(debug_msg, file=sys.stderr)

    st.markdown(
        "<div style='height: 12px;'></div>",
        unsafe_allow_html=True,
    )
    st.subheader("STEP 1: Select the database(s)")

    # Show debug info prominently
    with st.expander("🐛 DEBUG INFO - Click to expand", expanded=True):
        st.code(debug_msg)

    st.markdown(
        "<div style='color: #647489; margin-bottom: 24px; margin-top: -12px;'>Choose at least one case you want to run the test for. </div>",
        unsafe_allow_html=True,
    )

    dbContainerColumns = st.columns(DB_SELECTOR_COLUMNS, gap="small")
    dbIsActived = {db: False for db in DB_LIST}

    for i, db in enumerate(DB_LIST):
        column = dbContainerColumns[i % DB_SELECTOR_COLUMNS]
        dbIsActived[db] = column.checkbox(db.name)
        image_src = DB_TO_ICON.get(db, None)
        if image_src:
            column.markdown(
                f'<img src="{image_src}" style="width:100px;height:100px;object-fit:contain;object-position:center;margin-bottom:10px;">',
                unsafe_allow_html=True,
            )
        else:
            column.warning(f"{db.name} image not available")
    activedDbList = [db for db in DB_LIST if dbIsActived[db]]

    return activedDbList
