"""
Pydantic is a Python library used for data validation and parsing using type hints.
It ensures that the data you receive (from APIs, user input, JSON, etc.) matches the expected structure and types.

Why Pydantic is Useful :----
✅ Automatic data validation
✅ Type checking using Python type hints
✅ Converts data to correct types (e.g., "25" → int)
✅ Clear error messages
✅ Widely used with FastAPI

"""


## Code 

from pydantic import BaseModel

class Student(BaseModel):
    name: str
    age: int

new_student = Student(name='Himanshu', age='23')

print(new_student)