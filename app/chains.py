import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from dotenv import load_dotenv
import json
import re

load_dotenv()

class Chain:
    def __init__(self):
        self.llm = ChatGroq(temperature=0, groq_api_key=os.getenv("GROQ_API_KEY"), model_name="llama3-8b-8192")

    def extract_jobs(self, cleaned_text):
        prompt_extract = PromptTemplate.from_template(
            """
            ### SCRAPED TEXT FROM WEBSITE:
            {page_data}
            ### INSTRUCTION:
            The scraped text is from the career's page of a website.
            Your job is to extract the job postings and return them in JSON format containing the following keys: `role`, `experience`, `skills` and `description`.
            Only return the valid JSON.
            ### VALID JSON (NO PREAMBLE):
            """
        )

        chain_extract = prompt_extract | self.llm
        res = chain_extract.invoke(input={"page_data": cleaned_text})

        # Raw response from the LLM
        raw_response = res.content
        
        # Use regex to extract valid JSON
        match = re.search(r'\{.*\}', raw_response, re.DOTALL)
        if match:
            json_part = match.group()  # Extract the JSON part
            try:
                extracted_json = json.loads(json_part)  # Parse JSON
                print(json.dumps(extracted_json, indent=4))  # Pretty-print the JSON
            except json.JSONDecodeError as e:
                print("Error: The extracted text is not valid JSON.", e)
        else:
            print("Error: No valid JSON found in the response.")

        try:
            json_parser = JsonOutputParser()
            res = json_parser.parse(raw_response)
        except OutputParserException:
            raise OutputParserException("Context too big. Unable to parse jobs.")
        return res if isinstance(res, list) else [res]
    

    def write_mail(self, job, links):
        prompt_email = PromptTemplate.from_template(
            """
            ### JOB DESCRIPTION:
            {job_description}

            ### INSTRUCTION:
            You are Phanendra Sai Sree Vinay Alapaka, a software engineer with extensive experience in developing scalable systems, building REST APIs, and working on both frontend and backend technologies. Your job is to write a cold email to the client regarding the job mentioned above, describing how your skills align with their requirements. You should also mention your technical experience in various domains such as cloud, databases, UI development, and backend systems.

            Remember you are Phanendra Sai Sree Vinay Alapaka. 
            Do not provide a preamble.

            ### EMAIL (NO PREAMBLE):
            """
        )
        chain_email = prompt_email | self.llm
        res = chain_email.invoke({"job_description": str(job), "link_list": links})
        return res.content

if __name__ == "__main__":
    print(os.getenv("GROQ_API_KEY"))