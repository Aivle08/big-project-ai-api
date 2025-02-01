#########################################################################################
# Basic
import os
import pandas as pd
import traceback
# Fastapi
from fastapi import APIRouter, HTTPException, status, File, UploadFile

# LangChain
from langchain_core.runnables import RunnableConfig

# Graph
from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.memory import MemorySaver

## Module
# State
from state.question_state import GraphState
# Node
from node.question_node import retrieve_document, relevance_check, combine_prompt
# etc
from etc.etcc import is_relevant
from etc.graphs import visualize_graph
from etc.messages import invoke_graph, random_uuid
from prompt.question_prompt import tecnology_prompt, experience_prompt, work_prompt
#########################################################################################
question = APIRouter(prefix='/question')

# 기술 중심 Prompt
@question.post("/tech", status_code = status.HTTP_200_OK, tags=['question'])
async def tech_prompt():
    print('\n\033[36m[AI-API] \033[32m 질문 추출(기술)')
    try:
        workflow = StateGraph(GraphState)

        # 1. Node 추가
        workflow.add_node(
            "retrieve_1_document",
            lambda state: {"resume": retrieve_document(state, "resume", 'applicant_id')},
        )
        workflow.add_node(
            "retrieve_2_document",
            lambda state: {"evaluation": retrieve_document(state, "evaluation", 'company_id')},
        )
        
        
        workflow.add_node("relevance_check", relevance_check)
        workflow.add_node(
            "combine_prompt",
            lambda state: {"final_question": combine_prompt(state, tecnology_prompt)},
        )

        # 2. Edge 연결
        workflow.add_edge(START, "retrieve_1_document")
        workflow.add_edge(START, "retrieve_2_document")
        
        workflow.add_edge("retrieve_1_document", "combine_prompt")
        workflow.add_edge("retrieve_2_document", "combine_prompt")
    
        workflow.add_edge('combine_prompt','relevance_check')
        
        # 3. 조건부 엣지 추가
        workflow.add_conditional_edges(
            "relevance_check",  # 관련성 체크 노드에서 나온 결과를 is_relevant 함수에 전달합니다.
            is_relevant,
            {
                "relevant": END,  # 관련성이 있으면 답변을 생성합니다.
                "not_relevant": "combine_prompt",  # 관련성이 없으면 다시 검색합니다.
            },
        )

        # 4. 그래프 진입점 설정
        workflow.set_entry_point("retrieve_1_document")

        # 5. 체크포인터 설정
        memory = MemorySaver()

        # 6. 그래프 컴파일
        app = workflow.compile(checkpointer=memory)

        # 7. 그래프 시각화
        visualize_graph(app,'tech_graph')
            
        # 8. config 설정(재귀 최대 횟수, thread_id)
        config = RunnableConfig(recursion_limit=10, configurable={"thread_id": random_uuid()})


        # 9. 질문 입력
        input_job = 'IT영업'
        inputs = GraphState(job=input_job, 
                            company_id = 1,
                            applicant_id = 2,
                            relevance='no',
                            query_main=f'{input_job}의 기술 중심으로 생성해줘' )


        # 10. 그래프 실행 출력
        invoke_graph(app, inputs, config)

        # 11. 최종 출력 확인
        outputs = app.get_state(config).values

        print("===" * 20)
        print(f'job:\n{outputs["job"]}')
        print(f'query_main:\n{outputs["query_main"]}')
        print(f'resume:\n{outputs["resume"]}')
        print(f'evaluation:\n{outputs["evaluation"]}')
        print(f'relevance:\n{outputs["relevance"]}')
        print(f'final_question:\n{outputs["final_question"]}')
        
        return outputs["final_question"]
    except Exception as e:
            traceback.print_exc()
            return {
                "status": "error",
                "message": f"에러 발생: {str(e)}"
            }
            
# 경험 중심 Prompt
@question.post("/experience", status_code = status.HTTP_200_OK, tags=['question'])
async def experience_prompt():
    print('\n\033[36m[AI-API] \033[32m 질문 추출(경험)')
    try:
        workflow = StateGraph(GraphState)

        # 1. Node 추가
        workflow.add_node(
            "retrieve_1_document",
            lambda state: {"resume": retrieve_document(state, "resume", 'applicant_id')},
        )
        workflow.add_node("relevance_check", relevance_check)
        workflow.add_node(
            "combine_prompt",
            lambda state: {"final_question": combine_prompt(state, experience_prompt)},
        )

        # 2. Edge 추가
        workflow.add_edge('retrieve_1_document','combine_prompt')
        workflow.add_edge('combine_prompt','relevance_check')
        
        # 3. 조건부 엣지 추가
        workflow.add_conditional_edges(
            "relevance_check",  # 관련성 체크 노드에서 나온 결과를 is_relevant 함수에 전달합니다.
            is_relevant,
            {
                "relevant": END,  # 관련성이 있으면 답변을 생성합니다.
                "not_relevant": "combine_prompt",  # 관련성이 없으면 다시 검색합니다.
            },
        )

        # 4. 그래프 진입점 설정
        workflow.set_entry_point("retrieve_1_document")

        # 5. 체크포인터 설정
        memory = MemorySaver()

        # 6. 그래프 컴파일
        app = workflow.compile(checkpointer=memory)

        visualize_graph(app,'experience_graph')
            
        # config 설정(재귀 최대 횟수, thread_id)
        config = RunnableConfig(recursion_limit=10, configurable={"thread_id": random_uuid()})

        input_job = 'IT영업'

        # 질문 입력
        inputs = GraphState(job=input_job, 
                            company_id = 1,
                            applicant_id = 3,
                            evaluation = """5. 공고
                                            수행업무
                                            • 대기업/중견기업 고객 대상 ICT 관련 영업활동
                                            o Industry 별 ICT, AI, DX 시장정보 수집
                                            o 고객사 니즈 파악을 통한 세일즈 전략 수립
                                            o 영업 기회 발굴 및 매출화/이익 관리
                                            o 고객 디지털 혁신 니즈를 선제안을 통한 사업화 추진
                                            o 고객 Care 등 고객 만족 활동 지원
                                            o 다양한 내외부 관계자들과의 소통, 이해관계 조율 등의 제반업무
                                            수행
                                            • 공공, 국방, 금융고객 대상 B2B 세일즈 및 고객 관리
                                            o 공공, 국방, 금융 고객 신규 시장 및 영업 기회 발굴, 고객 Care
                                            활동
                                            o 관련 분야 시장 정보 수집을 통한 세일즈 전략 수립, 솔루션/서비스
                                            제안
                                            o 영업 활동을 통한 사업목표 달성 및 매출/이익 관리
                                            우대요건
                                            • ICT 기술 전반에 대한 지식 보유 및 시장 분석 역량
                                            • 비즈니스 Writing& Speeching 역량
                                            • 원활한 커뮤니케이션 능력
                                            KT 의 IT 영업 직무는 대기업/중견기업, 공공, 국방, 금융 고객을 대상으로 ICT
                                            관련 솔루션과 서비스를 제공하며, 고객의 디지털 혁신을 선도하는 역할을
                                            수행합니다. 신입 지원자는 해당 직무의 특성을 반영하여 기술적 이해, 영업
                                            전략 수립, 고객 니즈 파악 및 매출화 등을 잘 수행할 수 있는 역량을 갖춰야
                                            합니다.
                                            기본 요구 사항
                                            1. 학력 및 전공
                                            o 최소 학사 학위 보유 (ICT 관련 전공 우대: 컴퓨터공학, 전자공학,
                                            경영학 등)
                                            o 전공과 관계없이 ICT 기술 및 디지털 혁신에 관심이 있는 지원자
                                            우대
                                            2. 기본 기술적 이해
                                            o ICT(정보통신기술), AI, DX(디지털 트랜스포메이션)에 대한 기본적인
                                            이해
                                            o 클라우드, 빅데이터, IoT, AI 등 최신 IT 기술에 대한 관심 및
                                            기본적인 지식 보유
                                            3. 영업 전략 및 고객 니즈 분석
                                            o 고객의 요구를 파악하고 이에 맞는 세일즈 전략 수립
                                            경험(대외활동, 프로젝트, 인턴 경험 등)
                                            o 고객 만족을 위해 영업 활동을 통한 문제 해결 및 서비스 개선 경험
                                            우대 사항
                                            1. ICT 및 관련 산업 지식
                                            o ICT 시장에 대한 이해(산업별 시장 정보 수집 경험, 트렌드 분석
                                            등)
                                            o 디지털 혁신 관련 활동 또는 경험 (AI, 클라우드, 데이터 분석 등
                                            활용 경험)
                                            o B2B 세일즈 경험(공공, 금융, 국방 분야 관련 경험 우대)
                                            2. 영업 및 커뮤니케이션 능력
                                            o 비즈니스 Writing(제안서 작성) 및 Speeching(발표 및
                                            프레젠테이션) 역량 보유
                                            o 고객 니즈 분석 및 그에 맞는 솔루션 제안 경험
                                            o 팀워크 및 협업 경험이 있는 경우 우대 (팀 프로젝트, 대외활동 등)
                                            3. 고객 관리 및 사업화 경험
                                            o 고객 Care 활동을 통해 고객 만족을 증대시킨 경험
                                            o 영업 기회 발굴 및 매출화/이익 관리 경험(학업 또는
                                            대외활동에서의 경험 포함 가능)
                                            o 사업 목표 달성 및 고객과의 장기적 관계 구축을 위한 노하우 보유
                                            4. 전문적 자격증
                                            o IT 관련 자격증 (정보처리기사, AWS, Azure 등)
                                            o 영업 관련 자격증(CRM, 마케팅 관련 자격증) 보유자 우대
                                            경력/대외활동 경험
                                            • 인턴 경험: IT 영업, ICT 관련 기업, 공공/금융/국방 관련 기관에서의 인턴
                                            경험.
                                            • 프로젝트 경험: IT 관련 프로젝트(예: 클라우드 도입, 데이터 분석 시스템
                                            구축 등)에서 고객 니즈 파악, 솔루션 제안, 매출화 등 직무와 연계된 활동
                                            경험.
                                            • 대외활동 경험: IT 관련 경진대회, 공모전, 세미나 참여 등 ICT 관련
                                            산업에 대한 이해를 증명할 수 있는 경험""",
                            query_main=f'{input_job}의 기술 중심으로 생성해줘' )


        # 그래프 실행
        invoke_graph(app, inputs, config)

        # 최종 출력 확인
        outputs = app.get_state(config).values

        print("===" * 20)
        print(f'job:\n{outputs["job"]}')
        print(f'query_main:\n{outputs["query_main"]}')
        print(f'resume:\n{outputs["resume"]}')
        print(f'evaluation:\n{outputs["evaluation"]}')
        print(f'relevance:\n{outputs["relevance"]}')
        print(f'final_question:\n{outputs["final_question"]}')
        
        return outputs["final_question"]
    except Exception as e:
            traceback.print_exc()
            return {
                "status": "error",
                "message": f"에러 발생: {str(e)}"
            }
            
# 경력 중심 Prompt
@question.post("/work", status_code = status.HTTP_200_OK, tags=['question'])
async def work_prompt():
    print('\n\033[36m[AI-API] \033[32m 질문 추출(경력)')
    try:
        workflow = StateGraph(GraphState)

        # 1. Node 추가
        workflow.add_node(
            "retrieve_1_document",
            lambda state: {"resume": retrieve_document(state, "resume", 'applicant_id')},
        )
        workflow.add_node("relevance_check", relevance_check)
        workflow.add_node(
            "combine_prompt",
            lambda state: {"final_question": combine_prompt(state, work_prompt)},
        )

        # 2. Edge 추가
        workflow.add_edge('retrieve_1_document','combine_prompt')
        workflow.add_edge('combine_prompt','relevance_check')
        
        # 3. 조건부 엣지 추가
        workflow.add_conditional_edges(
            "relevance_check",  # 관련성 체크 노드에서 나온 결과를 is_relevant 함수에 전달합니다.
            is_relevant,
            {
                "relevant": END,  # 관련성이 있으면 답변을 생성합니다.
                "not_relevant": "combine_prompt",  # 관련성이 없으면 다시 검색합니다.
            },
        )

        # 4. 그래프 진입점 설정
        workflow.set_entry_point("retrieve_1_document")

        # 5. 체크포인터 설정
        memory = MemorySaver()

        # 6. 그래프 컴파일
        app = workflow.compile(checkpointer=memory)

        visualize_graph(app,'work_graph')
            
        # config 설정(재귀 최대 횟수, thread_id)
        config = RunnableConfig(recursion_limit=10, configurable={"thread_id": random_uuid()})

        input_job = 'IT영업'

        # 질문 입력
        inputs = GraphState(job=input_job, 
                            company_id = 1,
                            applicant_id = 1,
                            evaluation = """5. 공고
                                            수행업무
                                            • 대기업/중견기업 고객 대상 ICT 관련 영업활동
                                            o Industry 별 ICT, AI, DX 시장정보 수집
                                            o 고객사 니즈 파악을 통한 세일즈 전략 수립
                                            o 영업 기회 발굴 및 매출화/이익 관리
                                            o 고객 디지털 혁신 니즈를 선제안을 통한 사업화 추진
                                            o 고객 Care 등 고객 만족 활동 지원
                                            o 다양한 내외부 관계자들과의 소통, 이해관계 조율 등의 제반업무
                                            수행
                                            • 공공, 국방, 금융고객 대상 B2B 세일즈 및 고객 관리
                                            o 공공, 국방, 금융 고객 신규 시장 및 영업 기회 발굴, 고객 Care
                                            활동
                                            o 관련 분야 시장 정보 수집을 통한 세일즈 전략 수립, 솔루션/서비스
                                            제안
                                            o 영업 활동을 통한 사업목표 달성 및 매출/이익 관리
                                            우대요건
                                            • ICT 기술 전반에 대한 지식 보유 및 시장 분석 역량
                                            • 비즈니스 Writing& Speeching 역량
                                            • 원활한 커뮤니케이션 능력
                                            KT 의 IT 영업 직무는 대기업/중견기업, 공공, 국방, 금융 고객을 대상으로 ICT
                                            관련 솔루션과 서비스를 제공하며, 고객의 디지털 혁신을 선도하는 역할을
                                            수행합니다. 신입 지원자는 해당 직무의 특성을 반영하여 기술적 이해, 영업
                                            전략 수립, 고객 니즈 파악 및 매출화 등을 잘 수행할 수 있는 역량을 갖춰야
                                            합니다.
                                            기본 요구 사항
                                            1. 학력 및 전공
                                            o 최소 학사 학위 보유 (ICT 관련 전공 우대: 컴퓨터공학, 전자공학,
                                            경영학 등)
                                            o 전공과 관계없이 ICT 기술 및 디지털 혁신에 관심이 있는 지원자
                                            우대
                                            2. 기본 기술적 이해
                                            o ICT(정보통신기술), AI, DX(디지털 트랜스포메이션)에 대한 기본적인
                                            이해
                                            o 클라우드, 빅데이터, IoT, AI 등 최신 IT 기술에 대한 관심 및
                                            기본적인 지식 보유
                                            3. 영업 전략 및 고객 니즈 분석
                                            o 고객의 요구를 파악하고 이에 맞는 세일즈 전략 수립
                                            경험(대외활동, 프로젝트, 인턴 경험 등)
                                            o 고객 만족을 위해 영업 활동을 통한 문제 해결 및 서비스 개선 경험
                                            우대 사항
                                            1. ICT 및 관련 산업 지식
                                            o ICT 시장에 대한 이해(산업별 시장 정보 수집 경험, 트렌드 분석
                                            등)
                                            o 디지털 혁신 관련 활동 또는 경험 (AI, 클라우드, 데이터 분석 등
                                            활용 경험)
                                            o B2B 세일즈 경험(공공, 금융, 국방 분야 관련 경험 우대)
                                            2. 영업 및 커뮤니케이션 능력
                                            o 비즈니스 Writing(제안서 작성) 및 Speeching(발표 및
                                            프레젠테이션) 역량 보유
                                            o 고객 니즈 분석 및 그에 맞는 솔루션 제안 경험
                                            o 팀워크 및 협업 경험이 있는 경우 우대 (팀 프로젝트, 대외활동 등)
                                            3. 고객 관리 및 사업화 경험
                                            o 고객 Care 활동을 통해 고객 만족을 증대시킨 경험
                                            o 영업 기회 발굴 및 매출화/이익 관리 경험(학업 또는
                                            대외활동에서의 경험 포함 가능)
                                            o 사업 목표 달성 및 고객과의 장기적 관계 구축을 위한 노하우 보유
                                            4. 전문적 자격증
                                            o IT 관련 자격증 (정보처리기사, AWS, Azure 등)
                                            o 영업 관련 자격증(CRM, 마케팅 관련 자격증) 보유자 우대
                                            경력/대외활동 경험
                                            • 인턴 경험: IT 영업, ICT 관련 기업, 공공/금융/국방 관련 기관에서의 인턴
                                            경험.
                                            • 프로젝트 경험: IT 관련 프로젝트(예: 클라우드 도입, 데이터 분석 시스템
                                            구축 등)에서 고객 니즈 파악, 솔루션 제안, 매출화 등 직무와 연계된 활동
                                            경험.
                                            • 대외활동 경험: IT 관련 경진대회, 공모전, 세미나 참여 등 ICT 관련
                                            산업에 대한 이해를 증명할 수 있는 경험""",
                            query_main=f'경력 사항, 인턴 및 대외활동' )

        # 그래프 실행
        invoke_graph(app, inputs, config)

        # 최종 출력 확인
        outputs = app.get_state(config).values

        print("===" * 20)
        print(f'job:\n{outputs["job"]}')
        print(f'query_main:\n{outputs["query_main"]}')
        print(f'resume:\n{outputs["resume"]}')
        print(f'evaluation:\n{outputs["evaluation"]}')
        print(f'relevance:\n{outputs["relevance"]}')
        print(f'final_question:\n{outputs["final_question"]}')
        
        return outputs["final_question"]
    except Exception as e:
            traceback.print_exc()
            return {
                "status": "error",
                "message": f"에러 발생: {str(e)}"
            }