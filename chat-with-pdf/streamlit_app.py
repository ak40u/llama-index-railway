import os, streamlit as st
from llama_cloud import LlamaCloud
from llama_index.core import Document, Settings, VectorStoreIndex

# Streamlit app config
st.set_page_config(page_title="Chat with PDF", page_icon="📄")
st.subheader("Chat with PDF")

# Initialize session state
if "openai_api_key" not in st.session_state:
    st.session_state.openai_api_key = ""
if "llama_cloud_api_key" not in st.session_state:
    st.session_state.llama_cloud_api_key = ""
if "loaded_doc" not in st.session_state:
    st.session_state.loaded_doc = None
if "query_engine" not in st.session_state:
    st.session_state.query_engine = None

with st.sidebar:
    st.header("Settings")
    openai_api_key = st.text_input("OpenAI API key", type="password")
    if openai_api_key and openai_api_key != st.session_state.openai_api_key:
        st.session_state.openai_api_key = openai_api_key
 
    llama_cloud_api_key = st.text_input("LlamaCloud API key", type="password")
    if llama_cloud_api_key and llama_cloud_api_key != st.session_state.llama_cloud_api_key:
        st.session_state.llama_cloud_api_key = llama_cloud_api_key
 
    source_doc = st.file_uploader("Source document", type="pdf")

col1, col2 = st.columns([4,1])
query = col1.text_input("Query", placeholder="Ask a question about your PDF...", label_visibility="collapsed")
submit = col2.button("Submit", use_container_width=True)

if submit:
    if not st.session_state.openai_api_key.strip():
        st.error("Please provide the OpenAI API key.")
    elif not st.session_state.llama_cloud_api_key.strip():
        st.error("Please provide the LlamaCloud API key.")
    elif not query.strip():
        st.error("Please provide a query.")
    elif not source_doc:
        st.error("Please upload a PDF document.")
    else:
        os.environ["OPENAI_API_KEY"] = st.session_state.openai_api_key
        os.environ["LLAMA_CLOUD_API_KEY"] = st.session_state.llama_cloud_api_key

        # Only re-parse if a new document has been uploaded
        if st.session_state.loaded_doc != source_doc:
            with st.spinner("Uploading and indexing document, please wait...", show_time=True):
                try:
                    client = LlamaCloud(api_key=st.session_state.llama_cloud_api_key)
                    result = client.parsing.parse(
                        tier="cost_effective",
                        version="latest",
                        upload_file=(source_doc.name, source_doc.getvalue(), "application/pdf"),
                        processing_options={"ocr_parameters": {"languages": ["en"]}},
                        expand=["markdown"],
                    )

                    # One document per successfully parsed page
                    pages = result.markdown.pages if result.markdown else []
                    documents = [
                        Document(text=page.markdown, metadata={"page_number": page.page_number, "file_name": source_doc.name})
                        for page in pages
                        if page.success and page.markdown.strip()
                    ]
                    if not documents:
                        raise ValueError("LlamaCloud returned no text for this PDF.")

                    index = VectorStoreIndex.from_documents(documents)
                    st.session_state.query_engine = index.as_query_engine(
                        similarity_top_k=5,
                        response_mode="tree_summarize",
                    )
                    st.session_state.loaded_doc = source_doc
                    st.success("Document indexed successfully.")
                except Exception as e:
                    st.error(f"An error occurred while indexing: {e}")
                    st.stop()
        
        with st.spinner("Searching, please wait..."):
            try:
                response = st.session_state.query_engine.query(query)
                st.markdown("**Answer**")
                st.info(response)
            except Exception as e:
                st.error(f"An error occurred: {e}")
