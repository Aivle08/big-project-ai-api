from pydantic import Field, BaseModel

class SummaryDTO(BaseModel):
    job: str = Field('IT영업', description='직무')
    applicant_id: int = Field(1, description='지원자 id')
    
class ExtractionDTO(BaseModel):
    applicant_id: int = Field(1, description='지원자 id')