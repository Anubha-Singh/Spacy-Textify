import streamlit as st 
import os


# NLP Pkgs
from textblob import TextBlob 
import spacy
from gensim.summarization.summarizer import summarize 
from gensim.summarization import keywords
from spacy.lang.en import English

# Sumy Summary Pkg
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer


# Function for Sumy Summarization
def sumy_summarizer(docx):
	parser = PlaintextParser.from_string(docx,Tokenizer("english"))
	lex_summarizer = LexRankSummarizer()
	summary = lex_summarizer(parser.document,3)
	summary_list = [str(sentence) for sentence in summary]
	result = ' '.join(summary_list)
	return result

# Function to Analyse Tokens and Lemma
@st.cache
def text_analyzer(my_text):
	nlp = English()
	docx = nlp(my_text)
	# tokens = [ token.text for token in docx]
	allData = [('"Token":{},\n"Lemma":{}'.format(token.text,token.lemma_))for token in docx ]
	return allData

# Function For Extracting Entities
@st.cache
def entity_analyzer(my_text):
	nlp = English()
	docx = nlp(my_text)
	tokens = [ token.text for token in docx]
	entities = [(entity.text,entity.label_)for entity in docx.ents]
	allData = ['"Token":{},\n"Entities":{}'.format(tokens,entities)]
	return allData


def main():
	""" NLP Based App with Streamlit """

	# Title
	st.title("Spacy Textify")
	
	
	menu = ["Home","About"]
	choice = st.sidebar.selectbox("Menu",menu)
	if choice == "Home":
		st.subheader("Home")
		if st.checkbox("Show Tokens and Lemma"):
			st.subheader("Tokenize Your Text")
			message = st.text_area("Enter Text","Type Here ..")
			if st.button("Analyze",key="Analyze"):
				nlp_result = text_analyzer(message)
				st.json(nlp_result)
		if st.checkbox("Show Named Entities"):
			st.subheader("Analyze Your Text")
			message = st.text_area("Enter Text","Please Type..")
			if st.button("Extract",key="Extract"):
				entity_result = entity_analyzer(message)
				st.json(entity_result)
		if st.checkbox("Show Sentiment Analysis"):
			st.subheader("Analyse Your Text")
			message = st.text_area("Enter Text","Type Here ")
			if st.button("Analyze",key="Sentiment"):
				blob = TextBlob(message)
				result_sentiment = blob.sentiment
				st.success(result_sentiment)
		if st.checkbox("Show Text Summarization"):
			st.subheader("Summarize Your Text")
			message = st.text_area("Enter Text","Type..If you want to use gensim summarizer kindly ensure you have long paragraphs..")
			summary_options = st.selectbox("Choose Summarizer",['sumy','gensim'])
			if st.button("Summarize",key="Summarize"):
				if summary_options == 'sumy':
					st.text("Using Sumy Summarizer ..")
					summary_result = sumy_summarizer(message)
				elif summary_options == 'gensim':
					st.text("Using Gensim Summarizer ..")
					summary_result = summarize(message,ratio=0.05)
				else:
					st.warning("Using Default Summarizer")
					st.text("Using Gensim Summarizer ..")
					summary_result = summarize(message,ratio=0.05)
				st.success(summary_result)



	if choice== "About":
		
		st.markdown("""#### Description""")
		st.success("This  App is useful for basic NLP task Tokenization, NER, Sentiment, Summarization.")
		st.markdown("Made with :heartbeat: By Anubha Singh")
		#st.success("Made with love By Anubha Singh")


		
		
	
if __name__ == '__main__':
	main()
