from state.question_state import GraphState

# 청크 합치기
def format_docs(docs):
    return "\n".join(
        [
            f"{doc.page_content}"
            for doc in docs
        ]
    )
    
# 관련성 체크하는 함수(router)
def is_relevant(state: GraphState):
    if state["relevance"] == "yes":
        return "relevant"
    else:
        return "not_relevant"