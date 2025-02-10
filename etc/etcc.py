from state.question_state import QuestionState
from state.score_state import ScoreState
from state.summary_state import SummaryState

# 청크 합치기
def format_docs(docs):
    return "\n".join(
        [
            f"{doc.page_content}"
            for doc in docs
        ]
    )
    
# 관련성을 확인하는 함수    
def question_is_relevant(state: QuestionState, key: str):
    print(f'{key}: ', state[key])
    if state[key] == "yes":
        return "relevant"
    else:
        return "not_relevant"   

# 관련성 체크하는 함수(router)
def score_is_relevant(state: ScoreState):
    if state["yes_or_no"] == "yes":
        return "relevant"
    else:
        return "not_relevant"

    
# 사실 여부 체크하는 함수(router)
def question_is_fact(state: QuestionState):
    print('fact: ', state['fact'])
    #print('resume: ', state['resume'])
    #print('evaluation: ', state['evaluation'])
    if state["fact"] == "yes":
        return "fact"
    else:
        return "not_fact"
    
# 사실 여부 체크하는 함수(router)
def score_is_fact(state: ScoreState):
    if state["yes_or_no"] == "yes":
        return "fact"
    else:
        return "not_fact"
    
# 사실 여부 체크하는 함수(router)
def summary_is_fact(state: SummaryState):
    if state["yes_or_no"] == "yes":
        return "fact"
    else:  
        return "not_fact"