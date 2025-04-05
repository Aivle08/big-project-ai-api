# Graph
from langgraph.graph import END, StateGraph
# Node
from node.question_node import input, retrieve_document, relevance_check, combine_prompt, fact_checking, rewrite_question, experience_work_fact_checking
# etc
from etc.validator import question_is_relevant, question_is_fact
from prompt.question_prompt import tecnology_prompt, rewrite_prompt, experience_prompt, work_prompt

def tech_stategraph(workflow: StateGraph):
    # 1. Node 추가
    workflow.add_node("input", input)
    ## Retriever
    workflow.add_node(
        "retrieve_1_document",
        lambda state: {"resume": retrieve_document(state, "resume", 'applicant_id')},
    )
    workflow.add_node(
        "retrieve_2_document",
        lambda state: {"evaluation": retrieve_document(state, "evaluation", 'company_id')},
    )
    ## Relevance
    workflow.add_node(
        "relevance_check_1",
        lambda state: {"relevance_1": relevance_check(state, 'resume')},
    )
    workflow.add_node(
        "relevance_check_2",
        lambda state: {"relevance_2": relevance_check(state, 'evaluation')},
    )
    workflow.add_node(
        "rewrite_question_1",
        lambda state: {"resume_query": rewrite_question(state, rewrite_prompt, 'resume')},
    )
    workflow.add_node(
        "rewrite_question_2",
        lambda state: {"evaluation_query": rewrite_question(state, rewrite_prompt, 'evaluation')},
    )
    workflow.add_node(
        "combine_prompt",
        lambda state: {"final_question": combine_prompt(state, tecnology_prompt)},
    )
    workflow.add_node("fact_checking", fact_checking)
    # 2. Edge 연결
    workflow.add_edge("input", "retrieve_1_document")
    workflow.add_edge("input", "retrieve_2_document")
    workflow.add_edge("retrieve_1_document", "relevance_check_1")
    workflow.add_edge("retrieve_2_document", "relevance_check_2")
    
    # 3. 조건부 엣지 추가
    workflow.add_conditional_edges(
        "relevance_check_1",  # 관련성 체크 노드에서 나온 결과를 is_relevant 함수에 전달합니다.
        lambda state: question_is_relevant(state, key='relevance_1'),
        {
            "relevant": "combine_prompt",  # 관련성이 있으면 답변을 생성합니다.
            "not_relevant": "rewrite_question_1",  # 관련성이 없으면 다시 검색합니다.
        },
    )
    
    # 3. 조건부 엣지 추가
    workflow.add_conditional_edges(
        "relevance_check_2",  # 관련성 체크 노드에서 나온 결과를 is_relevant 함수에 전달합니다.
        lambda state: question_is_relevant(state, key = 'relevance_2'),
        {
            "relevant": "combine_prompt",  # 관련성이 있으면 답변을 생성합니다.
            "not_relevant": "rewrite_question_2",  # 관련성이 없으면 다시 검색합니다.
        },
    )
    workflow.add_edge("rewrite_question_1", "retrieve_1_document")
    workflow.add_edge("rewrite_question_2", "retrieve_2_document")

    workflow.add_edge('combine_prompt','fact_checking')
    
    # 3. 조건부 엣지 추가
    workflow.add_conditional_edges(
        "fact_checking",  # 관련성 체크 노드에서 나온 결과를 is_fact 함수에 전달합니다.
        question_is_fact,
        {
            "fact": END,  # 관련성이 있으면 답변을 생성합니다.
            "not_fact": "combine_prompt",  # 관련성이 없으면 다시 검색합니다.
        },
    )
    
    # 4. 그래프 진입점 설정
    workflow.set_entry_point("input")
    
    return workflow

def experience_stategraph(workflow: StateGraph):
     # 1. Node 추가
    workflow.add_node(
        "retrieve_1_document",
        lambda state: {"resume": retrieve_document(state, "resume", 'applicant_id')},
    )
    workflow.add_node(
        "relevance_check",
        lambda state: {"relevance_1": relevance_check(state, 'resume')},
    )
    workflow.add_node(
        "rewrite_question",
        lambda state: {"resume_query": rewrite_question(state, rewrite_prompt, 'resume')},
    )
    ## Relevance
    workflow.add_node(
        "fact_check",
        lambda state: {"fact": experience_work_fact_checking(state, 'resume')},
    )
    workflow.add_node(
        "combine_prompt",
        lambda state: {"final_question": combine_prompt(state, experience_prompt)},
    )

    # 2. Edge 추가
    workflow.add_edge('retrieve_1_document','relevance_check')
    workflow.add_edge('combine_prompt','fact_check')
    workflow.add_edge('rewrite_question','retrieve_1_document')
    
    # 3. 조건부 엣지 추가
    workflow.add_conditional_edges(
        "relevance_check",  # 관련성 체크 노드에서 나온 결과를 is_relevant 함수에 전달합니다.
        lambda state: question_is_relevant(state, 'relevance_1'),
        {
            "relevant": 'combine_prompt',  # 관련성이 있으면 답변을 생성합니다.
            "not_relevant": "rewrite_question",  # 관련성이 없으면 다시 검색합니다.
        },
    )
    
    workflow.add_conditional_edges(
        "fact_check",  # 관련성 체크 노드에서 나온 결과를 is_relevant 함수에 전달합니다.
        lambda state: question_is_fact(state),
        {
            "fact": END,  # 관련성이 있으면 답변을 생성합니다.
            "not_fact": "combine_prompt",  # 관련성이 없으면 다시 검색합니다.
        },
    )

    # 4. 그래프 진입점 설정
    workflow.set_entry_point("retrieve_1_document")

    return workflow

def work_stategraph(workflow: StateGraph):
     # 1. Node 추가
    workflow.add_node(
        "retrieve_1_document",
        lambda state: {"resume": retrieve_document(state, "resume", 'applicant_id')},
    )
    workflow.add_node(
        "relevance_check",
        lambda state: {"relevance_1": experience_work_fact_checking(state, 'resume')},
    )
    workflow.add_node(
        "rewrite_question",
        lambda state: {"resume_query": rewrite_question(state, rewrite_prompt, 'resume')},
    )
    ## Relevance
    workflow.add_node(
        "fact_check",
        lambda state: {"fact": experience_work_fact_checking(state, 'resume')},
    )
    
    workflow.add_node(
        "combine_prompt",
        lambda state: {"final_question": combine_prompt(state, work_prompt)},
    )

    # 2. Edge 추가
    workflow.add_edge('retrieve_1_document','relevance_check')
    workflow.add_edge('rewrite_question','retrieve_1_document')
    workflow.add_edge('combine_prompt','fact_check')
    
    # 3. 조건부 엣지 추가
    workflow.add_conditional_edges(
        "relevance_check",  # 관련성 체크 노드에서 나온 결과를 is_relevant 함수에 전달합니다.
        lambda state: question_is_relevant(state, 'relevance_1'),
        {
            "relevant": "combine_prompt",  # 관련성이 있으면 답변을 생성합니다.
            "not_relevant": "rewrite_question",  # 관련성이 없으면 다시 검색합니다.
        },
    )
    
    workflow.add_conditional_edges(
        "fact_check",  # 관련성 체크 노드에서 나온 결과를 is_relevant 함수에 전달합니다.
        lambda state: question_is_fact(state),
        {
            "fact": END,  # 관련성이 있으면 답변을 생성합니다.
            "not_fact": "combine_prompt",  # 관련성이 없으면 다시 검색합니다.
        },
    )

    # 4. 그래프 진입점 설정
    workflow.set_entry_point("retrieve_1_document")
    
    return workflow