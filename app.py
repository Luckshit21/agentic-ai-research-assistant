
import asyncio
import streamlit as st

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from my_agent.agent import root_agent


APP_NAME = "research_agent_app"
USER_ID = "local_user"


st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="🔎",
    layout="centered",
)


@st.cache_resource
def setup_agent():
    session_service = InMemorySessionService()

    runner = Runner(
        app_name=APP_NAME,
        agent=root_agent,
        session_service=session_service,
    )

    return runner, session_service


async def research(topic):
    runner, session_service = setup_agent()

    session = await session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
    )

    prompt = (
        f"Research {topic}. Prepare a detailed research report "
        "using reliable web sources. Explain the main findings "
        "in simple terms, include source URLs, and clearly state "
        "any uncertainties. Do not invent facts or citations. "
        "Save the report with the title 'Research Report'."
    )

    message = types.Content(
        role="user",
        parts=[types.Part(text=prompt)],
    )

    answer = ""

    async for event in runner.run_async(
        user_id=USER_ID,
        session_id=session.id,
        new_message=message,
    ):
        if event.is_final_response() and event.content:
            if event.content.parts:
                answer = "\n".join(
                    part.text
                    for part in event.content.parts
                    if getattr(part, "text", None)
                )

    return answer


st.title("🔎 AI Research Assistant")
st.write(
    "Enter any topic to generate a research report "
    "with source URLs and simple explanations."
)

topic = st.text_area(
    "What do you want to research?",
    placeholder="Example: How does artificial intelligence help in healthcare?",
    height=120,
)

if st.button("Generate Research Report", type="primary"):
    if not topic.strip():
        st.warning("Please enter a research topic first.")
    else:
        try:
            with st.spinner(
                "Researching sources and preparing your report..."
            ):
                result = asyncio.run(research(topic.strip()))

            if result:
                st.subheader("Research Report")
                st.markdown(result)

                st.download_button(
                    "Download Report (.md)",
                    data=result,
                    file_name="research_report.md",
                    mime="text/markdown",
                )
            else:
                st.warning(
                    "No final answer was received. "
                    "Check the terminal for errors."
                )

        except Exception as error:
            st.error(f"Something went wrong: {error}")
            st.info(
                "Check that your venv is active and your API "
                "configuration is available."
            )