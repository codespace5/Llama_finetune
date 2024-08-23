import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer


# model_name = "NousResearch/llama-2-7b-chat-hf"
model_name = "TheBloke/LongChat-13B-GPTQ"
# The instruction dataset to use
# dataset_name = "dataset/data1"
# dataset_name = "mlabonne/guanaco-llama2-1k"
dataset_name = "./dataset"
# Fine-tuned model name
new_model = "llama-2-13b"

################################################################################
# QLoRA parameters
################################################################################

# LoRA attention dimension
lora_r = 64

# Alpha parameter for LoRA scaling
lora_alpha = 16

# Dropout probability for LoRA layers
lora_dropout = 0.1

################################################################################
# bitsandbytes parameters
################################################################################

# Activate 4-bit precision base model loading
use_4bit = True

# Compute dtype for 4-bit base models
bnb_4bit_compute_dtype = "float16"

# Quantization type (fp4 or nf4)
bnb_4bit_quant_type = "nf4"

# Activate nested quantization for 4-bit base models (double quantization)
use_nested_quant = False

################################################################################
# TrainingArguments parameters
################################################################################

# Output directory where the model predictions and checkpoints will be stored
output_dir = "./results"

# Number of training epochs
num_train_epochs = 1

# Enable fp16/bf16 training (set bf16 to True with an A100)
fp16 = False
bf16 = True

# Batch size per GPU for training
per_device_train_batch_size = 2

# Batch size per GPU for evaluation
per_device_eval_batch_size = 2

# Number of update steps to accumulate the gradients for
gradient_accumulation_steps = 1

# Enable gradient checkpointing
gradient_checkpointing = True

# Maximum gradient normal (gradient clipping)
max_grad_norm = 0.3

# Initial learning rate (AdamW optimizer)
learning_rate = 2e-4

# Weight decay to apply to all layers except bias/LayerNorm weights
weight_decay = 0.001

# Optimizer to use
optim = "paged_adamw_32bit"

# Learning rate schedule (constant a bit better than cosine)
lr_scheduler_type = "constant"

# Number of training steps (overrides num_train_epochs)
max_steps = -1

# Ratio of steps for a linear warmup (from 0 to learning rate)
warmup_ratio = 0.03

# Group sequences into batches with same length
# Saves memory and speeds up training considerably
group_by_length = True

# Save checkpoint every X updates steps
save_steps = 500

# Log every X updates steps
logging_steps = 500

################################################################################
# SFT parameters
################################################################################

# Maximum sequence length to use
max_seq_length = None

# Pack multiple short examples in the same input sequence to increase efficiency
packing = False

# Load the entire model on the GPU 0
device_map = {"": 0}

# dataset = load_dataset(dataset_name, split="train")
dataset = load_dataset("json", data_files="./data/5.json", split="train")

# print(dataset)
# # Load tokenizer and model with QLoRA configuration
compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=use_nested_quant,
)

# Check GPU compatibility with bfloat16
if compute_dtype == torch.float16 and use_4bit:
    major, _ = torch.cuda.get_device_capability()
    if major >= 8:
        print("=" * 80)
        print("Your GPU supports bfloat16: accelerate training with bf16=True")
        print("=" * 80)

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map=device_map
)
model.config.use_cache = False
model.config.pretraining_tp = 1

# Load LLaMA tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training

# Load LoRA configuration
peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",
    task_type="CAUSAL_LM",
)

# Set training parameters
training_arguments = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    save_steps=save_steps,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    fp16=fp16,
    bf16=bf16,
    max_grad_norm=max_grad_norm,
    max_steps=max_steps,
    warmup_ratio=warmup_ratio,
    group_by_length=group_by_length,
    lr_scheduler_type=lr_scheduler_type,
    report_to="tensorboard"
)

# Set supervised fine-tuning parameters
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=8192,
    # max_seq_length=509,
    tokenizer=tokenizer,
    args=training_arguments,
    packing=packing,
)

# Train model
trainer.train()

# Save trained model
trainer.model.save_pretrained(new_model)

print("traning is finished1111111111111111111111111111111")

print("----------------Test------------------")
prompt = "TASK INSTRUCTIONS:\n\nGiven a list of clusters of texts describing people's experiences, classify which clusters show a shared personal experience across the different texts, and those that do not. If a cluster shows a shared experience, give it the tag 'shared'. if a cluster does not show a shared experience, and rather shows a mix of experiences, give it the tag 'mix'. Finally, if a cluster seems to have been repeated, give it a tag 'x similar to y' and provide the cluster numbers for x and y respectively.\n\nCLUSTERS TO CLASSIFY:\n\n0\nUser asks for a password reset.\nUser asks to reset two-factor authentication.\nCustomer asks for more information regarding what issue exactly is when trying to log in.\nWants a password reset.\nCan you please help me to reset password?\nRequests reset of password.\nWants help logging into her account.\nWants help with password.\n1\nLocked account was unlocked.\nAccount got locked due to unsuccessful login attempts, so they unlocked it.\nAccount unlocked due to unsuccessful login attempts.\nAccount got locked due to unsuccessful login attempts.\n2\nTo complete the request, they require a description outlining the purpose of the account.\nUser requests a description outlining the purpose of the new account.\nTo complete request, customer needs to describe the purpose of a new checking account.\n3\nCustomer asks Rho bank team to reset password.\nWould like Rho account help to recover credentials.\nUser wants to update their address with Rho.\n4\nCustomer service representative asks customer to try to log in again, to confirm personal details.\nUser asks Boris to check phone for an SMS with a code, to try it to log in.\nCustomer service representative unlocks customer's account after it had been locked due to entering the wrong password.\n5\nUser asks to add a checking account to Bonside Advisor LLC.\nUser wants to add a checking account to their service.\n6\nUser asks for a reset password link one more time, to save to all devices.\nUser wants reset password link sent again.\n7\nUser requests help clearing browsing history, cache and cookies.\nUser asks customer to clear browsing history, cache and cookies, then close and reopen browser.\n\n###\n\n"
pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=4000)
result = pipe(f"{prompt}")

print("result is:111111111")
print(result[0]['generated_text'])




print("----------------Test------------------")
prompt = "TASK INSTRUCTIONS:\n\nGiven a list of clusters of texts describing people's experiences, classify which clusters show a shared personal experience across the different texts, and those that do not. If a cluster shows a shared experience, give it the tag 'shared'. if a cluster does not show a shared experience, and rather shows a mix of experiences, give it the tag 'mix'. Finally, if a cluster seems to have been repeated, give it a tag 'x similar to y' and provide the cluster numbers for x and y respectively.\n\nCLUSTERS TO CLASSIFY:\n\n0\nUser asks for a password reset.\nUser asks to reset two-factor authentication.\nCustomer asks for more information regarding what issue exactly is when trying to log in.\nWants a password reset.\nCan you please help me to reset password?\nRequests reset of password.\nWants help logging into her account.\nWants help with password.\n1\nLocked account was unlocked.\nAccount got locked due to unsuccessful login attempts, so they unlocked it.\nAccount unlocked due to unsuccessful login attempts.\nAccount got locked due to unsuccessful login attempts.\n2\nTo complete the request, they require a description outlining the purpose of the account.\nUser requests a description outlining the purpose of the new account.\nTo complete request, customer needs to describe the purpose of a new checking account.\n3\nCustomer asks Rho bank team to reset password.\nWould like Rho account help to recover credentials.\nUser wants to update their address with Rho.\n4\nCustomer service representative asks customer to try to log in again, to confirm personal details.\nUser asks Boris to check phone for an SMS with a code, to try it to log in.\nCustomer service representative unlocks customer's account after it had been locked due to entering the wrong password.\n5\nUser asks to add a checking account to Bonside Advisor LLC.\nUser wants to add a checking account to their service.\n6\nUser asks for a reset password link one more time, to save to all devices.\nUser wants reset password link sent again.\n7\nUser requests help clearing browsing history, cache and cookies.\nUser asks customer to clear browsing history, cache and cookies, then close and reopen browser.\n\n###\n\n"
pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=4000)
result = pipe(f"<s>[INST] {prompt} [/INST]")

print("result is:111111111")
print(result[0]['generated_text'])


print("----------------Test------------------")

print("----------------Test------------------")
prompt = "TASK INSTRUCTIONS:\n\nGiven a list of clusters of texts describing people's experiences, classify which clusters show a shared personal experience across the different texts, and those that do not. If a cluster shows a shared experience, give it the tag 'shared'. if a cluster does not show a shared experience, and rather shows a mix of experiences, give it the tag 'mix'. Finally, if a cluster seems to have been repeated, give it a tag 'x similar to y' and provide the cluster numbers for x and y respectively.\n\nCLUSTERS TO CLASSIFY:\n\n0\nIs Batman experience suitable for a child?\nIs Batman experience available, or only the regular one?\nAre these events for kids 6-12?\nWhat time does the Batman experience stop?\nWhy is the Batman experience not recommended for children?\nCan user be admitted to Batman experience at 7:05?\n1\nIs there a way to still obtain a refund, or a voucher to complete visit on a later date?\nHow to get refund for cancelled flight?\nIs a ticket refundable if purchased?\nWhat is the refund policy?\nCan I get a refund for three tickets bought for today?\n2\nCan the gift pass be used to visit the SpyScape exhibit?\nIs Spyscape open every day during a trip to NYC from Dec 24-28?\nWill Spyscape experience be interesting for someone with basic English?\nWhat's the difference between the All Access Pass and the Spy Date Experience Pass?\nIs Spyscape suitable for age 6?\n3\nCan customer come in by 1:30, or will have to reschedule?\nIf he purchases the tickets with a specific date, can his friends reschedule for another date in December?\nIs it too late to change a group's time slot, or will they be denied entrance?\nCan user change the date of tickets she bought?\n4\nHow to use 2 for 1 ticket offer through NYCgo?\nCan school buy tickets upon arrival?\nHow to buy the discounted tickets?\nCan I buy my child's ticket online in a separate order?\n5\nDo you have a place to store their backpacks while they take the tour?\nDoes the venue offer bag checks?\nDoes company offer a coat check or locker to put items\/coats away?\n6\nCan they make a reservation with a \"Sightseeing Pass\"?\nHow to make reservations for a visit from Jan 1st to 3rd, with Go City Explorer Pass?\nDoes user need to make reservations in advance for the Sightseeing Pass, or can they just show up and scan when they get there?\n7\nCan they buy tickets when they arrive at the museum, or should they buy them in advance?\nDo I need to make a reservation in advance to visit the museum?\nDo I have to buy tickets in advance if I want to visit with my family on December 24?\n8\nIs there a problem purchasing tickets on the website?\nCan date of gift tickets be extended?\nIf it is closed, why would it allow purchase of tickets for a closed day?\n9\nCustomer asks how to get the tickets to ensure they have the same timeslot as the larger group.\nIs there an age minimum for the VIP?\nWhat ticket type did customer want to purchase?\nIf group purchased VIP tickets together, can they all go in together, or will they be split up?\n10\nIs it OK if a reservation is late, due to traffic from New Jersey?\nIs it okay if a customer arrives a few minutes late for a scheduled meeting?\nIs being late for a scheduled appointment a problem?\nHow late arrival to reservation affects reservation?\n11\nCan they find out if there are open spots without having to show up in person?\nCan they pre-book a time\/date, or need to wait in line?\n\n###\n\n"
pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=4000)
result = pipe(f"<s>[INST] {prompt} [/INST]")

print("result is:111111111")
print(result[0]['generated_text'])


print("----------------Test------------------")


print("----------------Test------------------")
prompt = "TASK INSTRUCTIONS:\n\nGiven a list of clusters of texts describing people's experiences, classify which clusters show a shared personal experience across the different texts, and those that do not. If a cluster shows a shared experience, give it the tag 'shared'. if a cluster does not show a shared experience, and rather shows a mix of experiences, give it the tag 'mix'. Finally, if a cluster seems to have been repeated, give it a tag 'x similar to y' and provide the cluster numbers for x and y respectively.\n\nCLUSTERS TO CLASSIFY:\n\n0\nThe more quickly something can be done, the more quickly the user will learn to do more.\nCommunication\/coordination worked well for family's needs.\nUnlocking the use of the site for French-speaking people helps them use it optimally.\nPwC's consulting arm, ProEdge, sells ProEdge tooling and services.\n1\nJefit includes nutrition tracking, goals, coaching, etc.\nJefit is the top exercise tracker.\nWill help user put more accurate win\/loss in sport events.\nAllows better \"at a glance\" analytics.\n2\nThe product provides the ability to run a series of actions all at once, and efficiently print certain pages of a document without needing to point to everyone where the print button is.\nEnables user to only expose information their external partner needs to know.\nFeature is valuable because it automatically creates a link back to the source.\nGoogle Docs feature of seeing which changes were made is valuable for team collaboration, allowing decisions to be more thought-out.\n3\nWebCatalog appifies many of these sorts of apps, and it works really well.\nComplete agreement with previous feedback about the need for more powerful iOS app, full capabilities on the iPad, and more nativeness or independence per platform.\nThe product allows a more seamless user experience when using external websites for functionality.\nRuns webpage in standalone Chrome instance.\n4\nSoftware adds a lot of value to user.\nThinks design is sleek and like notion.\nFeature allows startup to personalize workspace, reinforcing brand, becoming an expectation for tools alike.\nSo far, product is great, user loves it, and would like to be part of the team one day.\n5\nUser finds the product easy to use.\nSays product is great.\nThe rest of the product is perfect.\nSays product is great.\n6\nOnce user understood that all building blocks were nested in the Explore button, it was easier to use.\nNew features make project trackers more useful, fresh.\nMove line and Insert line are very welcome additions to the context menu.\nMakes more matching dashboards for clients.\n7\nDecides to move personal and professional websites to Coda.\nAbility to move graphical objects around on the canvas to set values in coda tables.\nSurprised to find Coda can be tailored to replace or complement existing systems.\nCODA is the solution for small businesses.\n8\nUser loves Coda so far.\nPrefers Notion's templates over Coda's.\nStill use Coda.\nGreat to hear that Max's customer loves the flexibility of most other aspects of Coda.\n9\nCoda is so versatile, user is impressed with the way the other person found a way to do anything in it.\nCoda allows managing large datasets spread across multiple views of the same presentation more manageable.\nCoda is intuitive to learn, and helpful videos are useful.\nCoda is much better than anything else, and customer support is outstanding.\n10\nLikes Notion for personal use.\nLikes Notion's page structure.\nNotion's capabilities are also profoundly useful and gratifying.\nWas helpful to keep all content in one place with Super.so and Notion as a data backend.\n11\nInteresting idea!\nUser likes the tool, but would like more freedom, such as the ability to remove emojis from the title if desired, and more font sizes and colors.\nUser finds it cool that they can make their own graphs.\nFound it very interesting to read about the work-around you built using Kumu, and enjoyed both of your thoughts on the topic.\n12\nUser thinks the join function does exactly what they expect, and that's because their expectation is shaped by other \"joiny\" functions they're familiar with.\nTaking ambiguity out of data visualization.\nThe attachment column is better in Airtable than in Basecamp.\nThinks Detail view is a nice way to show data.\n13\nMethod works as long as rows are never sorted or manually changed to cause RowID values to be out of order.\nThe formula works for the user quite well.\nThe product adds value to understanding and maintaining formulas.\nThe formulas are easy to learn.\n\n###\n\n"
pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=4000)
result = pipe(f"<s>[INST] {prompt} [/INST]")

print("result is:111111111")
print(result[0]['generated_text'])


print("----------------Test------------------")