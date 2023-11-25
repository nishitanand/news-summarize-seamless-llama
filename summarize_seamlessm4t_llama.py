from indicnlp.tokenize import sentence_tokenize
from seamless_communication.models.inference import Translator
import pandas as pd
import transformers
import torch
from langchain import HuggingFacePipeline
from langchain import PromptTemplate
from langchain.chains import LLMChain
import argparse
import gc
import os


msg='''3 arguments are required: 
    --input_csv_path    Input CSV File Path
    --output_csv_path   Output CSV File Path
    --lang              3-letter source language code
    '''

parser = argparse.ArgumentParser(description=msg)

parser.add_argument("--input_csv_path", help = "Input CSV File Path", required=True)
parser.add_argument("--output_csv_path", help = "Output CSV File Path", required=True, default=".")
parser.add_argument("--language", help = "3-letter source language code", required=True)

args = parser.parse_args()

input_lang=str(args.language)


source_df = pd.read_csv(str(args.input_csv_path))
df = source_df['text']
dataframe = pd.DataFrame(df)


# Initialize a Translator object with a multitask model, vocoder on the GPU.
translator = Translator("seamlessM4T_medium", vocoder_name_or_card="vocoder_36langs", device=torch.device("cuda:0"))


def t2t(text, source_lang, target_lang):
    try:
        translated_text, _, _ = translator.predict(text, "t2tt",target_lang , src_lang=source_lang)
        #print(translated_text)
        return translated_text
    except Exception as e:
        return str(e)

def split_summarise(text):
    try:
        sentences = sentence_tokenize.sentence_split(text, input_lang, delim_pat='auto')
        summ_sentences = [str(t2t(i, input_lang, "eng")) for i in sentences]
        result_string = ' '.join(summ_sentences)
    except Exception as e:
        return str(e)
    return result_string


dataframe['english_translation'] = dataframe['text'].apply(split_summarise)

del translator
gc.collect()
torch.cuda.empty_cache()

dataframe.to_csv('intermediate_files/trans_eng.csv')



from ctransformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("TheBloke/Llama-2-7b-Chat-GGUF", model_file="llama-2-7b-chat.Q3_K_M.gguf", model_type="llama",hf=True, gpu_layers=32)

# model = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation", #task
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
    max_length=1500,
    # max_new_tokens=512,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id
)

llm = HuggingFacePipeline(pipeline = pipeline, model_kwargs = {'temperature':0})
template = """
            Write a concise summary of the following delimited by triple backquotes
            Return your response as a single paragraph summarising the key points of the text in no more than 100-150 words.
           ```{text}```
           SINGLE PARAGRAPH SUMMARY:

           """
prompt = PromptTemplate(template=template, input_variables=["text"])
llm_chain = LLMChain(prompt=prompt, llm=llm)


# Your input text
def split_text_into_chunks(text):
    sentences = text.split(". ")
    # Initialize variables to keep track of chunks
    # max_chunk_length = 700
    max_chunk_length = 400
    current_chunk = ""
    chunks = []
    
    # Iterate through the sentences
    for sentence in sentences:
        # Check if adding the current sentence to the current chunk would exceed the maximum token length
        if len(tokenizer(current_chunk + sentence)["input_ids"]) > max_chunk_length:
            # If yes, start a new chunk
            chunks.append(current_chunk)
            current_chunk = ""
    
        # Add the current sentence to the current chunk
        if current_chunk:
            current_chunk += ". "
        current_chunk += sentence
    
    if current_chunk:
        chunks.append(current_chunk)

    return chunks

def get_text_summarization(text):
    try:
        # Initialize an empty string to store the concatenated summary
        concatenated_summary = ""
        
        # Split the input text into chunks as you did previously
        # (assuming you have already split the text into chunks)
        chunks = split_text_into_chunks(text)

        # Iterate through the chunks
        print(len(chunks))
        for chunk in chunks:
            summary = llm_chain.run(concatenated_summary + chunk)
            
            concatenated_summary += summary + " "
        
        # Return the summary of the final chunk
        print(summary)
        return summary.strip()  # Remove trailing whitespace

    except Exception as e:
        #print(f"Error processing text: {str(e)}")
        return None  # You can choose to return None or some default value for rows with errors


df = pd.read_csv('intermediate_files/trans_eng.csv')


df['english_summary'] = None

for i, row in df.iterrows():
    print(f"Processing row {i}/{len(df)}")
    llama_output = get_text_summarization(row['english_translation'])
    df.at[i, 'english_summary'] = llama_output


df.to_csv('intermediate_files/eng_summary.csv')

del model
del tokenizer
del pipeline
gc.collect()
torch.cuda.empty_cache()


df = pd.read_csv('intermediate_files/eng_summary.csv')
df['english_summary']

df.drop('Unnamed: 0', axis=1, inplace=True)
df.drop('Unnamed: 0.1', axis=1, inplace=True)

# Initialize a Translator object with a multitask model, vocoder on the GPU.
translator = Translator("seamlessM4T_medium", vocoder_name_or_card="vocoder_36langs", device=torch.device("cuda:0"))

def t2t(text, source_lang, target_lang):
    try:
        translated_text, _, _ = translator.predict(text, "t2tt",target_lang , src_lang=source_lang)
        #print(translated_text)
        return translated_text
    except Exception as e:
        return str(e)

def split_summarise(text):
    try:
        sentences = sentence_tokenize.sentence_split(text, 'eng', delim_pat='auto')
        summ_sentences = [str(t2t(i, "eng", input_lang)) for i in sentences]
        result_string = ' '.join(summ_sentences)
    except Exception as e:
        return str(e)
    return result_string


df['source_lang_summary'] = df['english_summary'].apply(split_summarise)

finaldf = df.drop(columns=['english_translation', 'english_summary'])

df.to_csv('intermediate_files/pipeline_summary.csv')
# finaldf.to_csv('final_summary.csv')
finaldf.to_csv(os.path.join(str(args.output_csv_path), 'final_summary.csv'))


del translator
gc.collect()
torch.cuda.empty_cache()
