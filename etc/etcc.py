from state.summary_state import SummaryState

# 청크 합치기
def format_docs(docs):
    return "\n".join(
        [
            f"{doc.page_content}"
            for doc in docs
        ]
    )
    
# 사실 여부 체크하는 함수(router)
def is_fact(state: SummaryState):
    if state["yes_or_no"] == "yes":
        return "fact"
    else:
        return "not_fact"