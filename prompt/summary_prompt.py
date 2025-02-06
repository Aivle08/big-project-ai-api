from langchain_core.prompts import PromptTemplate

# 요약 Prompt
summary_prompt = PromptTemplate(
    # 당신은 채용 회사에서 설명하는 직무 역할에 맞춰 지원자의 개인 진술을 요약하도록 설계된 기계입니다.
    # 지원자의 자소서는 다음과 같습니다: {resume}
    # 채용 회사의 업무 역할은 다음과 같습니다: {job}

    ### 요약 지침:
    # - {job}의 직무를 채용하는 면접관의 입장에서 관심을 가질만한 내용을 모두 뽑아 요약해주세요.
    # - 요약된 문장의 문맥이 자연스럽게 생성해주세요.
    # - 이력서에서 개인 정보, 경력사항, 자격증, 수상, 인턴 및 대외 활동과 같은 개인정보는 제외해주세요.줘
    # - 중복 정보를 한 번만 출력하여 반복을 피하세요.
    # - 요약이 맥락과 이해를 제공할 수 있을 만큼 충분히 상세한지 확인하여 지나치게 간략한 설명을 피하세요.
    # - 자소서에 없는 내용은 절대 추가하지 마세요.
    # - 250자 이하로 요약해주세요.
    
    template="""
    You are a machine designed to summarize an applicant's personal statement in alignment with the job role described by the recruiting company.

    Here is the applicant's resume : {resume}  
    Here is the job role of the recruiting company : {job}

    ### Summary Instructions:
    1. Identify and summarize content from the {resume} that would be relevant and of interest to an interviewer hiring for the {job} role.
    2. Highlight key motivations, relevant experiences, notable accomplishments, and any other factors that make the applicant well-suited for the {job}.
    3. Include examples of challenges overcome, collaboration experiences, and technical or professional skills relevant to the role.
    4. Avoid repetition by including each piece of information only once, while ensuring enough detail to convey understanding.
    5. Ensure that the summary reflects the original content of the {resume}, but rephrased naturally and clearly to enhance readability.
    6. Exclude personal details unrelated to professional or role-specific qualifications such as contact information, birthdate, etc.
    

    ### Output Format:
    - A multi-line summary, written in Korean.
    - Each line should provide a distinct and relevant point from the applicant's resume.
    - Ensure the lines are clear, concise, and contextually relevant.
    - Include specific examples of achievements or experiences where possible.
    
    **Examples**:
    - Led data analysis and algorithm design in a recommendation system development project during university, achieving increase in product sales.
    - Contributed to a cross-team collaboration project by building a data integration platform, improving data processing speed by 25%.

    Using the examples above, summarize the applicant's cover letter based on the applicant's motivation, challenges, collaboration experience, and role-related technical or professional skills.
    """,
    input_variables=['resume', 'job']
)


# Prompt 생성
extraction_prompt = PromptTemplate(
    template="""
    You are a function summarizing resumes.

    ### Instructions:
    1. Organize the content according to the format below. Leave fields blank if the information is not provided.
    2. Avoid repetition by outputting duplicate information only once.
    3. Do not omit important details (e.g., major, GPA, etc.).
    4. Do not include any unnecessary symbols like """ """ or JSON formatting in the output.
    
    Please write the basis in Korean.
    Return your response strictly in this JSON format:
    ### Output Format:
    {output_form}
        
    ### Resume:
    {resume}
    """,
    input_variables=["resume", "output_form"]
)