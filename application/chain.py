from langchain_groq import ChatGroq
import chromadb
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
import pandas as pd
import uuid
from dotenv import load_dotenv
import os

load_dotenv()


class Chain:

    def __init__(self):
        self.llm = ChatGroq(temperature=0, groq_api_key = os.getenv("GROQ_API_KEY"))     #model_name="llama-3.1-70b-versatile")

    def extract_job(self, processed_text):
        prompt_extract = PromptTemplate.from_template(
            """
            ### Scraped Text from Website:
            {page_data}
            
            ### Instructions:
            This data is coming from a career's page of a website.
            Your Job is to extract the Job postings and return them in the JSON format containing the following keys:
            'role', 'experience', 'skills' and 'description'.
            Only return the valid JSON.
            
            ###VALID JSON (NO PREAMBLE):  
            
            """
        )
        chain_extract = prompt_extract | self.llm
        response = chain_extract.invoke(input={'page_data':processed_text})

        try:
            json_parser = JsonOutputParser()
            json_data = json_parser.parse(response.content)
        except OutputParserException:
            raise OutputParserException("output parser exception...")


        return json_data if isinstance(json_data, list) else [json_data]


    def write_mail(self, job_data, link_data):
        prompt_email = PromptTemplate.from_template(
        """
        ### JOB DESCRIPTION:
        {job_description}
        
        ### INSTRUCTION:
        You are business development executive at ABC org. ABC is an AI & Software Consulting company dedicated to facilitating
        the seamless integration of business processes through automated tools. 
        Over our experience, we have empowered numerous enterprises with tailored solutions, fostering scalability, 
        process optimization, cost reduction, and heightened overall efficiency. 
        Your job is to write a cold email to the client regarding the job mentioned above describing the capability of ABC 
        in fulfilling their needs.
        Also add the most relevant ones from the following links to showcase ABC's portfolio: {link_list}
        Remember you are XYZ, BDE at ABC org. 
        Do not provide a preamble.
        ### EMAIL (NO PREAMBLE):
        
        """
        )
        chain_email = prompt_email | self.llm
        response = chain_email.invoke({"job_description": str(job_data), "link_list": link_data})
        return response.content

if __name__=="__main__":
    print(os.getenv("GROQ_API_KEY"))