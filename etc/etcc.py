from state.question_state import QuestionState

# 청크 합치기
def format_docs(docs):
    return "\n".join(
        [
            f"{doc.page_content}"
            for doc in docs
        ]
    )
    
# 관련성을 확인하는 함수    
def is_relevant(state: QuestionState, key: str):
    print(f'{key}: ', state[key])
    if state[key] == "yes":
        return "relevant"
    else:
        return "not_relevant"   
    
# 사실 여부 체크하는 함수(router)
def is_fact(state: QuestionState):
    print('fact: ', state['fact'])
    #print('resume: ', state['resume'])
    #print('evaluation: ', state['evaluation'])
    if state["fact"] == "yes":
        return "fact"
    else:
        return "not_fact"
    
    