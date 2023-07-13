import streamlit as st
from streamlit_chat import message
from streamlit_extras.colored_header import colored_header
from streamlit_extras.add_vertical_space import add_vertical_space
from backend.backend import run_g_llm
import os


def get_text(instruction: str = "You: "):
    input_text = st.text_input(instruction, "", key=f"input-{instruction}")
    return input_text


st.set_page_config(page_title="infoBot")
with st.sidebar:
    st.title("Wiki infoBot")
    
    st.markdown(
    """
    ## Demo
    This is a demo for LLM to query info from wikipedia
    - wiki: [Hunter X Hunter](https://en.wikipedia.org/wiki/Hunter_%C3%97_Hunter)

    ## Referece
    - git: [github-assistant](https://github.com/g-emarco/github-assistant)
    """
    )

    
    add_vertical_space(10)

    st.write("Made by [James]")

if "generated" not in st.session_state:
    st.session_state["generated"] = ["Hi! I'm a infoBot"]
if "past" not in st.session_state:
    st.session_state["past"] = ["Hi!"]
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []


input_container = st.container()
colored_header(label="", description="", color_name="blue-30")
response_container = st.container()


with input_container:
    user_input = get_text(instruction="You: ")


## Conditional display of AI generated responses as a function of user provided prompts
with response_container:
    if user_input:
        with st.spinner("Generating response..."):
            response = run_g_llm(
                query=user_input, chat_history=st.session_state["chat_history"]
            )
            st.session_state.past.append(user_input)
            st.session_state.generated.append(response["answer"])
            st.session_state["chat_history"].append((user_input, response["answer"]))

    if st.session_state["generated"]:
        for i in range(len(st.session_state["generated"])):
            message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
            message(st.session_state["generated"][i], key=str(i))
