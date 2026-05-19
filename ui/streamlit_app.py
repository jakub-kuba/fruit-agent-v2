import uuid
import httpx
import streamlit as st

API_URL = "http://localhost:8080"

st.set_page_config(
    page_title="Fruit Market Agent",
    page_icon="🍎",
    layout="centered"
)

st.title("🍎🍋 Fruit Market AI Agent 🍇🍊")
st.caption("Powered by LangGraph & Azure OpenAI")

# --- Session state initialization ---
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Sidebar ---
with st.sidebar:
    st.header("Session")
    st.code(st.session_state.session_id[:8] + "...", language=None)

    if st.button("🗑️ New Conversation"):
        # Clear history in FastAPI/MySQL
        httpx.delete(f"{API_URL}/chat/{st.session_state.session_id}")
        # Reset session
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.markdown("**What can I do?**")
    st.markdown("""
    - 🏷️ Latest fruit prices in PLN
    - 💱 Convert prices to other currencies
    - 📊 Generate monthly/yearly reports
    - 📚 Answer questions about fruits
    """)

# --- Chat history display ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- Chat input ---
if prompt := st.chat_input("Ask about fruits..."):

    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Call FastAPI
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = httpx.post(
                    f"{API_URL}/chat",
                    json={
                        "message": prompt,
                        "session_id": st.session_state.session_id
                    },
                    timeout=30.0
                )
                response.raise_for_status()
                data = response.json()

                ai_response = data["response"]
                report_csv = data.get("report_csv")
                intent = data["intent"]

                st.markdown(ai_response)

                if report_csv:
                    st.download_button(
                        label="⬇ Download Report (CSV)",
                        data=report_csv,
                        file_name="fruit_report.csv",
                        mime="text/csv"
                    )
                # Show intent badge
                intent_colors = {
                    "knowledge": "🟢 Knowledge",
                    "tool": "🔵 Tool",
                    "off_topic": "🔴 Off-topic",
                    "unknown": "⚪ Unknown"
                }
                st.caption(intent_colors.get(intent, intent))

            except httpx.TimeoutException:
                ai_response = "Request timed out. Please try again."
                st.error(ai_response)
            except Exception as e:
                ai_response = f"Error: {str(e)}"
                st.error(ai_response)

    st.session_state.messages.append({
        "role": "assistant",
        "content": ai_response
    })