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
from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver

## Module
# State
from state.score_state import ScoreState
# Node
from node.score_node import retrieve_document, score_resume, fact_checking
# etc
from etc.etcc import is_fact
from etc.graphs import visualize_graph
from etc.messages import invoke_graph, random_uuid
from prompt.score_prompt import score_prompt
#########################################################################################
score = APIRouter(prefix='/score')

# 요약 Prompt
@score.post("", status_code = status.HTTP_200_OK, tags=['score'])
def summary_graph():
    print('\n\033[36m[AI-API] \033[32m 점수 측정')
    try:
        workflow = StateGraph(ScoreState)

        # 1. Node 추가
        workflow.add_node(
            "retrieve_document",
            lambda state: {"resume": retrieve_document(state, "resume", 'applicant_id')},
        )
        workflow.add_node(
            "score_resume",
            lambda state: {"eval_resume": score_resume(state, score_prompt)},
        )
        #workflow.add_node("fact_checking", fact_checking)
        
        # 2. Edge 연결
        workflow.add_edge('retrieve_document','score_resume')
        #workflow.add_edge('summary_document','fact_checking')
        
        # 3. 조건부 엣지 추가
        # workflow.add_conditional_edges(
        #     "fact_checking",  # 사실 체크 노드에서 나온 결과를 is_relevant 함수에 전달합니다.
        #     is_fact,
        #     {"fact": END,
        #         "not_fact": "summary_document",  # 사실이 아니면 다시 요약합니다.
        #     },
        # )

        # 4. 그래프 진입점 설정
        workflow.set_entry_point("retrieve_document")

        # 5. 체크포인터 설정
        memory = MemorySaver()

        # 6. 그래프 컴파일
        app = workflow.compile(checkpointer=memory)

        # 7. 그래프 시각화
        visualize_graph(app,'eval_graph')
            
        # 8. config 설정(재귀 최대 횟수, thread_id)
        config = RunnableConfig(recursion_limit=10, configurable={"thread_id": random_uuid()})

        input_job = 'IT영업'
        input_eval_item = '인재상'
        input_eval_item_content = """
        인재상
        1. 고객(Customer Value)
        • 기준:
        o 고객의 요구를 깊이 이해하고, 문제 해결을 위해 적극적으로 노력한 경험.
        o 고객 중심 사고를 증명할 수 있는 사례(프로젝트, 대외활동 등).
        • 평가 요소:
        o 대외활동, 인턴 경험, 프로젝트에서 고객의 니즈를 반영한 문제 해결 경험.
        o 새로운 고객 경험 제공을 위한 아이디어를 실제로 실행한 사례.
        2. 역량(Excellence)
        • 기준:
        o 직무와 관련된 전문 지식 및 기술 보유(예: 5G, AI, IoT, 데이터 분석).
        o 문제 해결 능력을 입증할 수 있는 수상 내역 또는 프로젝트 경험.
        • 평가 요소:
        o 자격증: 직무와 연관된 자격증 보유(정보처리기사, 네트워크관리사 등).
        o 경진대회: 직무 관련 공모전이나 대회에서의 수상 경험.
        o 기술 스택: 프로그래밍, 데이터 분석, 클라우드 기술 등 실무에 필요한 기술 활용 능력.
        3. 실질(Practical Outcome)
        • 기준:
        o 화려한 스펙보다는 실제 성과를 창출한 경험 강조.
        o 본인의 역할과 기여도를 명확히 설명할 수 있는 사례.
        • 평가 요소:
        o 실무 경험: 인턴십, 산학 협력, 실질적인 성과를 창출한 프로젝트 경험.
        o 성과 중심 활동: KPI를 설정하고 달성한 경험.
        o 문서화 능력: 실질적 결과물을 보고서, 발표자료로 정리한 능력.
        4. 화합(Togetherness)
        • 기준:
        o 팀워크와 협업 경험이 풍부하며, 다양한 사람들과 조화를 이룬 사례.
        o 갈등 상황에서도 문제를 해결하고 합의점을 찾은 경험.
        • 평가 요소:
        o 팀 프로젝트에서 역할 수행과 성과 기여 사례.
        o 협력 과정에서 나타난 리더십, 커뮤니케이션 능력.
        o 다문화 또는 다양한 배경의 사람들과 함께 일한 경험(해외 활동, 글로벌 프로젝트).
        """

        # 9. 질문 입력
        input_job = 'IT영업'
        inputs = ScoreState(job=input_job, 
                            applicant_id = 1,
                            eval_item = input_eval_item,
                            eval_item_content = input_eval_item_content,
                            query_main=f'{input_job} 직무에 대한 {input_eval_item}을 평가해주세요.')

        # 10. 그래프 실행 출력
        invoke_graph(app, inputs, config)

        # 11. 최종 출력 확인
        outputs = app.get_state(config).values

        print("===" * 20)
        print(f'job:\n{outputs["job"]}')
        print(f'applicant_id:\n{outputs["applicant_id"]}')
        print(f'resume:\n{outputs["resume"]}')
        #print(f'yes_or_no:\n{outputs["yes_or_no"]}')
        print(f'eval_resume:\n{outputs["eval_resume"]}')
        
        return outputs["eval_resume"]
    except Exception as e:
            traceback.print_exc()
            return {
                "status": "error",
                "message": f"에러 발생: {str(e)}"
            }

