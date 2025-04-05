
# Graph
from langgraph.graph import END, StateGraph
# Node
from node.score_node import retrieve_document, score_resume, fact_checking, relevance_check, no_relevance
# etc
from etc.validator import score_is_fact, score_is_relevant
from prompt.score_prompt import score_prompt

def score_stategraph(workflow: StateGraph):
     # 1. Node 추가
    workflow.add_node(
        "retrieve_document",
        lambda state: {"resume": retrieve_document(state, "resume", 'applicant_id')},
    )
    workflow.add_node(
        "score_resume",
        lambda state: {"eval_resume": score_resume(state, score_prompt)},
    )
    workflow.add_node("relevance_check", relevance_check)
    workflow.add_node("no_relevance", no_relevance)
    workflow.add_node("fact_checking", fact_checking)
    
    # 2. Edge 연결
    workflow.add_edge('retrieve_document','relevance_check')
    workflow.add_edge('no_relevance',END)
    workflow.add_edge('score_resume','fact_checking')
    
    #3. 조건부 엣지 추가
    workflow.add_conditional_edges(
        "relevance_check",  # 사실 체크 노드에서 나온 결과를 is_relevant 함수에 전달합니다.
        score_is_relevant,
        {"relevant": 'score_resume',
        "not_relevant": 'no_relevance',  # 관련이 없으면 0점입니다.
        },
    )
    workflow.add_conditional_edges(
        "fact_checking",  # 사실 체크 노드에서 나온 결과를 is_relevant 함수에 전달합니다.
        score_is_fact,
        {"fact": END,
        "not_fact": "score_resume",  # 사실이 아니면 다시 평가합니다.
        },
    )

    # 4. 그래프 진입점 설정
    workflow.set_entry_point("retrieve_document")
    
    return workflow