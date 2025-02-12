from state.question_state import QuestionState
from state.score_state import ScoreState
from state.summary_state import SummaryState

def format_docs(docs):
    """
    검색한 청크들을 하나의 청크로 합치는 함수.

    Args:
        docs (list): zilliz로 부터 검색한 내용을 담은 리스트

    Returns:
        list: 하나의 청크로 합친 리스트
    """
    return "\n".join(
        [
            f"{doc.page_content}"
            for doc in docs
        ]
    )
    
def question_is_relevant(state: QuestionState, key: str):
    """
    Query와 검색한 Chunk의 관련성을 확인하는 함수.

    Args:
        state (QuestionState): 질문 State
        key (str): State의 Key 값

    Returns:
        str : "relevant" (관련이 있는 경우) 또는 "not_relevant" (관련이 없는 경우).
    """
    print(f'{key}: ', state[key])
    if state[key] == "yes":
        return "relevant"
    else:
        return "not_relevant"   

def score_is_relevant(state: ScoreState):
    """
    Query와 검색한 Chunk의 관련성을 확인하는 함수.

    Args:
        state (ScoreState): 점수평가 State

    Returns:
        str : "relevant" (관련이 있는 경우) 또는 "not_relevant" (관련이 없는 경우).
    """
    if state["yes_or_no"] == "yes":
        return "relevant"
    else:
        return "not_relevant"

    
# 사실 여부 체크하는 함수(router)
def question_is_fact(state: QuestionState):
    """
    Query와 검색한 Chunk의 관련성을 확인하는 함수.

    Args:
        state (QuestionState): 점수평가 State

    Returns:
        str : "fact" (사실인 경우) 또는 "not_fact" (사실이 아닌 경우).
    """
    print('fact: ', state['fact'])
    #print('resume: ', state['resume'])
    #print('evaluation: ', state['evaluation'])
    if state["fact"] == "yes":
        return "fact"
    else:
        return "not_fact"
    
# 사실 여부 체크하는 함수(router)
def score_is_fact(state: ScoreState):
    """
    평가 근거의 사실 여부를 검증하는 함수.

    Args:
        state (ScoreState): 평가된 문서의 상태 정보를 포함하는 객체.

    Returns:
        str: "fact" (사실인 경우) 또는 "not_fact" (사실이 아닌 경우).
    """
    if state["yes_or_no"] == "yes":
        return "fact"
    else:
        return "not_fact"
    
# 사실 여부 체크하는 함수(router)
def summary_is_fact(state: SummaryState):
    """
    요약된 내용의 사실 여부를 검증하는 함수.

    Args:
        state (SummaryState): 요약된 문서의 상태 정보를 포함하는 객체.

    Returns:
        str: "fact" (사실인 경우) 또는 "not_fact" (사실이 아닌 경우).
    """
    
    if state["yes_or_no"] == "yes":
        return "fact"
    else:  
        return "not_fact"