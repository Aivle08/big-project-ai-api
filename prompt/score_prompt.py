from langchain_core.prompts import PromptTemplate

score_prompt = PromptTemplate(
        # 귀하의 임무는 {job} 직책에 대해 주어진 이력서의 내용을 평가하는 것입니다.

        # ### 지침:
        # 1. 아래 제공된 특정 기준에 따라 이력서를 평가합니다:
        # - {eval_item_content}
        # 2. 평가 항목에 대해 0점에서 100점 사이의 점수를 할당합니다: {eval_item}.
        # 3. 점수는 매우 엄격하게 평가되어야 하며, 어떠한 관용도 허용되어서는 안 됩니다.
        # 4. 이력서를 분석할 때 비판적인 관점을 유지하세요.
        # 5. 기준을 충족하지 못하면 30점 이하의 점수를 부여하고, 기준을 충족하고 추가 장점이 있는 경우 80점 이상의 점수를 부여합니다.
        # 6. 모든 점수가 동일하지 않도록 엄격하게 점수를 할당하고, 점수를 할당할 때 입력 기준을 충실히 준수합니다.
        # 7. 항상 동일한 답변이 일관되게 나올 수 있도록 하는 객관적인 근거가 있어야 합니다.
        # 8. 다음 단일 평가 항목**을 기준으로 이력서를 **엄격히 평가하십시오:
        # - {eval_item_content}
        # 9. 주어진 평가 항목 {eval_item}에 대해 0점에서 100점 사이에 **단 하나의** 점수만 할당합니다.
        # 10. 출력은 다음과 같은 엄격한 형식을 따라야 합니다:
        # [<점수>, "<설명>"]
        # 11. 출력 예시:
        # [63, "후보자는 관련 예시에서 뛰어난 기술적 능력을 입증했습니다."]
        
        # 기초를 한국어로 작성해 주세요.
        # ### 출력 형식:
        # 다음 JSON 형식으로 응답을 엄격히 반환합니다:
        # "{eval_item}: <점수>,
        # "{eval_item} of basis": "<평가 근거 설명>"

        # ### 출력 예시:
        # "기술적 기술": 63,
        # "기초 기술": "후보자는 관련 예시에서 뛰어난 기술적 능력을 입증했습니다."

        # 이제 주어진 이력서를 평가하고 필요한 형식으로 답변을 제공하세요.

        # ### 이력서:
        # {resume}
        template="""
        Your task is to evaluate the content of a given resume for the position of {job}.

        ### Instructions:
        1. Evaluate the resume based on the specific criteria provided below:
            - {eval_item_content}
        2. Assign a score between 0 and 100 for the evaluation item: {eval_item}.
        3. Scores must be evaluated with great strictness, and no leniency should be allowed.
        4. Maintain a critical perspective when analyzing the resume.
        5. If a standard is not met, assign a score below 30; if the standard is met and additional merits are present, assign a score above 80.
        6. Strictly assign scores while ensuring that not all scores are the same, and faithfully adhere to the input criteria when assigning scores.
        7. There must always be objective grounds that ensure the same answer can be consistently produced.
        8. Evaluate the resume **strictly based on the following single evaluation item**:
            - {eval_item_content}
        9. Assign **only one** score between 0 and 100 for the given evaluation item: {eval_item}.
        10. The output must follow this strict format:  
            [<Score>, "<Description>"]
        11. Example Output:  
            [63, "지원자는 관련 예로 우수한 기술 능력을 입증했다."]

        Please write the basis in Korean.
        ### Output Format:
        Return your response strictly in this List format:
            [<Score>, "<Description>"]

        ### Example Output:
            [63, "지원자는 관련 예로 우수한 기술 능력을 입증했다."]

        Now, evaluate the given resume and provide your response in the required format.

        ### Resume:
        {resume}
        """,
        input_variables=["resume", "eval_item_content", "job", "eval_item"]
    )