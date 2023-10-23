# -*- coding: utf-8 -*-



import torch
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
model_name = 'tuner007/pegasus_paraphrase'
torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = PegasusTokenizer.from_pretrained(model_name)
model = PegasusForConditionalGeneration.from_pretrained(model_name).to(torch_device)

def get_response(input_text,num_return_sequences):
  batch = tokenizer([input_text],truncation=True,padding='longest',max_length=60, return_tensors="pt").to(torch_device)
  translated = model.generate(**batch,max_length=60, num_return_sequences=num_return_sequences, temperature=1.5)
  tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
  return tgt_text

# Input text you want to paraphrase
text = " enter your here."

get_response(text, 5)

#processing  a paragraph of text
context = "    Honesty is the practice of speaking and acting truthfully and with integrity. it is essential to building trust and respect in our relationships. Whether it's being honest with ourselves and others about our thoughts and feelings, admitting our mistakes and shortcomings, or communicating clearly and transparently. honesty can create a sense of authenticity and connection in our interactions with others."

print(context)

#takes the input paragraph and splits it into list
from sentence_splitter import SentenceSplitter, split_text_into_sentences


splitter = SentenceSplitter(language='en')



sentence_list = splitter.split(context)
sentence_list

paraphrase = []


for i in sentence_list:
  a = get_response(i,1)
  paraphrase.append(a)

#this is a paraphrased text
paraphrase

paraphrase2 = [' '.join(x) for x in paraphrase ]
paraphrase2

#combines the above list into paragraph
paraphrase3 = [' '.join(x for x in paraphrase2) ]

paraphrase4 = (str(paraphrase3).strip('[]'))

#comapring the original and paraphrased version
print(context)
print(paraphrase4.strip("'"))

