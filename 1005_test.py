from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

model_name_or_path = "TheBloke/LongChat-13B-GPTQ"
# To use a different branch, change revision
# For example: revision="main"
model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                             device_map="auto",
                                             trust_remote_code=False,
                                             revision="main")

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

prompt = '''TASK INSTRUCTIONS:\n\nGiven a list of clusters of texts describing people's experiences, classify which clusters show a shared personal experience across the different texts, and those that do not. If a cluster shows a shared experience, give it the tag 'shared'. if a cluster does not show a shared experience, and rather shows a mix of experiences, give it the tag 'mix'. Finally, if a cluster seems to have been repeated, give it a tag 'x similar to y' and provide the cluster numbers for x and y respectively.\n\nCLUSTERS TO CLASSIFY:\n\n0\nWhat was the use of fabric rings applied to the corners of the top of the backpack?\nIs this waterproof backpack the one they were asking about?\nIs it okay to carry the second outdoor O bag?\nWhat is the size and weight of the bag?\n1\nWill it be durable after that?\nHow long does the zipper last?\nHow much weight can the product handle?\nIs it really genuine? Dealers should apologize.\n2\nWhat size is the room?\nWhat size is the compartment?\nIs item size specified as small, small-medium, or medium?\nHow do I know which size is right for me?\n3\nHas been registered.\nFirst tried on Intersport, but heard the Kanken mini didn't work anymore.\nCan't find the whistle.\nIs registration card lost?\n4\nIs there a waterproof barrier?\nIs product waterproof?\nHow waterproof is it?\nHas user tried the waterproof rating yet?\n\n###\n\n'''
prompt_template=f'''You are a helpful AI assistant.

USER: {prompt}
ASSISTANT:

'''

print("\n\n*** Generate:")

# input_ids = tokenizer(prompt_template, return_tensors='pt').input_ids.cuda()
# output = model.generate(inputs=input_ids, temperature=0.7, do_sample=True, top_p=0.95, top_k=40, max_new_tokens=4096)
# # print(output[0])

# Inference can also be done using transformers' pipeline

print("*** Pipeline:")
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=2048,
    do_sample=True,
    temperature=0.7,
    top_p=0.95,
    top_k=40,
    repetition_penalty=1.1
)

print(pipe(prompt_template)[0]['generated_text'])