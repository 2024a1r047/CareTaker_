from typing import Generic, Optional, TypeVar,  List
from pydantic.generics import GenericModel
from pydantic import BaseModel, Field, EmailStr, constr, validator
from enum import Enum

T = TypeVar('T')

#Login
class Login(BaseModel):
    username: str
    password: str
    role: Optional[str] = "caretaker" # Hint for client, but backend will verify

#Register
# ---------- Gender Enum ----------
class GenderEnum(str, Enum):
    male = "Male"
    female = "Female"
    other = "Other"

# ---------- CareRecipient Input Model ----------
class CareRecipientCreate(BaseModel):
    full_name: str = Field(..., example="Alice Smith")
    preferred_name: Optional[str] = Field(None, example="Amma")  # name Saathi/Asha uses to call them
    email: EmailStr = Field(..., example="alice@example.com")
    phone_number: str = Field(..., min_length=10, max_length=10, example="9876543210")
    age: int = Field(..., example=70)
    gender: GenderEnum = Field(..., example="Female")
    city: Optional[str] = Field(None, example="Jammu")
    respiratory_condition_status: bool = Field(default=False)
    
    # New Profile Fields
    height: Optional[float] = Field(None, example=170.5)
    weight: Optional[float] = Field(None, example=75.0)
    blood_group: Optional[str] = Field(None, example="O+")
    emergency_contact: Optional[str] = Field(None, example="9876543211")

    @validator('gender', pre=True)
    def normalize_gender(cls, v):
        """Accept case-insensitive gender values or enum member names/values.

        Examples accepted: 'female', 'Female', 'FEMALE', GenderEnum.female
        Returns the corresponding GenderEnum member.
        """
        if isinstance(v, GenderEnum):
            return v
        if isinstance(v, str):
            s = v.strip()
            for member in GenderEnum:
                if s.lower() == member.name.lower() or s.lower() == member.value.lower():
                    return member
        raise ValueError("Invalid gender; expected one of: Male, Female, Other")

# ---------- Registration Model ----------
class Register(BaseModel):
    email: EmailStr = Field(..., example="john@example.com")
    username: str = Field(..., example="john_doe")
    phone_number: str = Field(..., min_length=10, max_length=10, example="9999999999")
    password: str = Field(...,min_length=3, max_length=72,example="strongpassword123")
    full_name: str = Field(..., example="John Doe")
    role: str = Field("caretaker", example="caretaker") # 'caretaker' or 'doctor'
    specialization: Optional[str] = Field(None, example="Cardiologist")
    # Optional for doctors, required for caretakers (checked in route logic)
    care_recipients: Optional[List[CareRecipientCreate]] = Field(default=[], min_items=0)

#response model
class ResponseSchema(BaseModel):
    code: int
    status: str
    message: str
    result: Optional[T]= None

#token
class TokenResponse(BaseModel):
    access_token: str
    token_type: str

# --- Update Models ---
class RecipientUpdate(BaseModel):
    full_name: Optional[str] = None
    preferred_name: Optional[str] = None
    email: Optional[str] = None
    phone_number: Optional[str] = None
    age: Optional[int] = None
    gender: Optional[str] = None
    city: Optional[str] = None
    height: Optional[float] = None
    weight: Optional[float] = None
    blood_group: Optional[str] = None
    emergency_contact: Optional[str] = None
    respiratory_condition_status: Optional[bool] = None

class CaretakerUpdate(BaseModel):
    full_name: Optional[str] = None
    email: Optional[str] = None
    phone_number: Optional[str] = None
    password: Optional[str] = None

# --- Medical Record Input Models ---

class ConditionInput(BaseModel):
    disease_name: str
    disease_code: Optional[str] = "CUSTOM"
    status: Optional[str] = "active"
    severity: Optional[str] = "moderate"
    first_detected: Optional[str] = None # ISO format date

class MedicationInput(BaseModel):
    medicine_name: str
    dosage: Optional[str] = None
    frequency: Optional[str] = None
    schedule_time: Optional[str] = None
    duration_days: Optional[int] = None
    start_date: Optional[str] = None  # ISO date YYYY-MM-DD, defaults to today
    status: Optional[str] = "active"
    current_stock: Optional[int] = 0
    doses_per_day: Optional[int] = 1
    auto_order_enabled: Optional[bool] = True


class AllergyInput(BaseModel):
    allergen: str
    allergy_type: Optional[str] = "other"
    reaction: Optional[str] = None
    severity: Optional[str] = "moderate"
    status: Optional[str] = "active"
