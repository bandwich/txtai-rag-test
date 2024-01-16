from txtai.embeddings import Embeddings
from txtai.pipeline import LLM
from script import data

# Create embeddings
embeddings = Embeddings(content=True, autoid="uuid5")

# Create an index for the list of text
embeddings.index(data)

# Create LLM with llama.cpp - GGUF file is automatically downloaded
llm = LLM("llama_cpp/models/7Bf/ggml-model-q4_0.gguf", method="llama.cpp")

def execute(question, data):
  prompt = f"""<|im_start|>system
  You answer questions completely, using the dialogue in the context as a reference for speaking style.<|im_end|>
  <|im_start|>user
  Answer the following question using general language skills and knowledge, using the context as a reference.
  
  question: {question}
  context: {data} <|im_end|>
  <|im_start|>assistant
  """

  return llm(prompt, maxlength=1024)

print(execute("If a real estate agent calls you offering their services, and you're not interested, what are ten things you might say?", data))

# Create and run extractor instance

# extractor = Extractor(embeddings, llm, separator="\n", template=template)
# print("Done with setup. Ready for input.\n")

# result = extractor("You're a property owner looking to sell a home. If you were to list at some point, what are 10 things you expect a real-estate agent to do to get your home sold?")

# print("ANSWER:", result["answer"])
# print("REFERENCE:", embeddings.search("select id, text from txtai where id = :id", parameters={"id": result["reference"]}))

# def io_loop():
#     userInput = input()
#     result = extractor(userInput)
#     print(result["answer"])

# io_loop()