import streamlit as slt
from chain import Chain
from portfolio import Portfolio
from langchain_community.document_loaders import WebBaseLoader
from utils import clean_text





def create_app(chain, portfolio):
    slt.title("COLD EMAIL... using GEN AI")
    input = slt.text_input("Enter a Job posting URL:")
    submit_button = slt.button("Generate Email")
    if submit_button:
        try:

            loader = WebBaseLoader([input])

            data = clean_text(loader.load().pop().page_content)

            jobs = chain.extract_job(data)

            print(jobs)

            portfolio.load_portfolios()

            for job in jobs:
                
                skills = job.get('skills', [])
                links = portfolio.query_link(skills)
                email = chain.write_mail(job, links)

                slt.code(email, language='markdown')
        
        except Exception as e:

            slt.error(e)

if __name__ == "__main__":

    chain = Chain()
    portfolio = Portfolio("resource/my_portfolio.csv")

    create_app(chain, portfolio)
