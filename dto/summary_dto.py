from pydantic import Field, BaseModel

class SummaryDTO(BaseModel):
    """
    SummaryDTO (Summary Data Transfer Object)

    지원자 정보를 요약하기 위한 데이터 객체.

    Attributes:
        job (str): 지원자가 지원하는 직무명.
        applicant_id (int): 지원자의 고유 ID.
    """
    job: str = Field('IT영업', description='직무')
    applicant_id: int = Field(101, description='지원자 id')
    
class ExtractionDTO(BaseModel):
    """
    ExtractionDTO (Extraction Data Transfer Object)

    지원자 정보 추출하기 위한 데이터 객체.

    Attributes:
        applicant_id (int): 지원자의 고유 ID, 정보 추출 시 사용됩니다.
    """
    applicant_id: int = Field(101, description='지원자 id')