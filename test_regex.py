import re

text = "Q: What are the professional conduct expectations?\n A: Maintain discipline, respect all team members, meet deadlines, and communicate professionally. Disrespect or unprofessional behaviour may lead to disciplinary measures."

qa_match = re.search(r'(.*?\?)\s*(.*)', text, re.DOTALL)
if qa_match:
    print("Match found!")
    print(f"Q: '{qa_match.group(1)}'")
    print(f"A: '{qa_match.group(2)}'")
else:
    print("Match NOT found!")

# Try without DOTALL
qa_match2 = re.search(r'(.*?\?)\s*(.*)', text)
if qa_match2:
    print("Match2 found!")
else:
    print("Match2 NOT found!")
