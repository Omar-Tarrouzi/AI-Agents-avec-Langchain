from langchain.messages import HumanMessage
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv  # Fixed import (removed .ipython)
from langchain_community.vectorstores import Chroma
from langchain.tools import tool

# Load environment variables
load_dotenv()

# Define tools
@tool
def get_employee_info(name: str):
    """
    Get information about a given employee (name, salary, seniority)
    """
    print(f"get_employee_info invoked for {name}")
    return {"name": name, "salary": 12000, "seniority": 5}  # Fixed: i2000 -> 12000

@tool
def send_email(email: str, subject: str, content: str):
    """
    Send email with subject and content
    """
    print(f"Sending email to {email}, subject: {subject}, content: {content}")
    return f"Email successfully sent to {email}, subject: {subject}, content: {content}"

# Initialize the LLM
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# Create the agent
agent = create_agent(
    model=llm,  
    tools=[get_employee_info, send_email],
    system_prompt="répond au mode prophétie aux question du user",
    debug=True  # Optional: for debugging
)

resp=agent.invoke(input={"messages":[HumanMessage(content="hdfs c'est quoi ?")]})
print(resp['messages'][-1].contenr)