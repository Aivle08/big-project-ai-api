# Graph
from langgraph.graph import END, StateGraph
# Node
from node.summary_node import resume_load, resume_summary, fact_checking, retrieve_document, combine_prompt
# etc
from etc.validator import summary_is_fact
from etc.graphs import visualize_graph
from etc.messages import invoke_graph, random_uuid
from prompt.summary_prompt import summary_prompt, extraction_prompt

def summary_stategraph(workflow: StateGraph):
     # 1. Node 추가
    workflow.add_node(
        "resume_document",
        lambda state: {"resume": resume_load(state, "resume", 'applicant_id')},
    )
    workflow.add_node(
        "summary_document",
        lambda state: {"summary_result": resume_summary(state, summary_prompt)},
    )
    workflow.add_node("fact_checking", fact_checking)
    
    # 2. Edge 연결
    workflow.add_edge('resume_document','summary_document')
    workflow.add_edge('summary_document','fact_checking')
    
    # 3. 조건부 엣지 추가
    workflow.add_conditional_edges(
        "fact_checking",  # 사실 체크 노드에서 나온 결과를 is_relevant 함수에 전달합니다.
        summary_is_fact,
        {"fact": END,
            "not_fact": "summary_document",  # 사실이 아니면 다시 요약합니다.
        },
    )

    # 4. 그래프 진입점 설정
    workflow.set_entry_point("resume_document")
    
    return workflow