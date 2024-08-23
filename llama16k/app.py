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
model_name = "togethercomputer/Llama-2-7B-32K-Instruct"
# The instruction dataset to use
# dataset_name = "dataset/data1"
# dataset_name = "mlabonne/guanaco-llama2-1k"
dataset_name = "./dataset"
# Fine-tuned model name
new_model = "llama-2-7b"

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
num_train_epochs = 10

# Enable fp16/bf16 training (set bf16 to True with an A100)
fp16 = False
bf16 = False

# Batch size per GPU for training
per_device_train_batch_size = 4

# Batch size per GPU for evaluation
per_device_eval_batch_size = 4

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
save_steps = 25

# Log every X updates steps
logging_steps = 25

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
dataset = load_dataset("json", data_files="./dataset/5.json", split="train")

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
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=training_arguments,
    packing=packing,
)

# Train model
trainer.train()

# Save trained model
trainer.model.save_pretrained(new_model)


prompt = "TASK INSTRUCTIONS:\n\nGiven a list of clusters of texts describing people's experiences, classify which clusters show a shared personal experience across the different texts, and those that do not. If a cluster shows a shared experience, give it the tag 'shared'. if a cluster does not show a shared experience, and rather shows a mix of experiences, give it the tag 'mix'. Finally, if a cluster seems to have been repeated, give it a tag 'x similar to y' and provide the cluster numbers for x and y respectively.\n\nCLUSTERS TO CLASSIFY:\n\n0\nLightweight, comfortable clothing fits perfectly.\nSuper comfy and very slenderizing. Feels like a second skin.\nLightweight and comfortable feel like yoga pants.\n1\nLikes the product, especially the location of the labels.\nLikes the material, everything about the product.\nLikes the quality of the product.\nLikes the quality of the product.\n2\nPants are durable and fit well.\nQuality of pants is great.\nPants are durable, breathable, and water repellent.\nPants of superior quality will be worn for years to come.\n3\nLikes pants, great fit, zip on and off easily, holding up well, best snaps.\nSuper quiet and comfortable. Love full-length zippers.\nMedium fit just right, except for the zipper being on the left.\nZippers work great, and length of shorts is just right.\n4\nHas had several of these pants for years, and loves them.\nBest all-around pants they've found, purchased three pairs in different colors.\nLoves the pants, found them after a long search for a fit for their curves.\nPleased with quality of pants.\n5\nThe pants are light weight, and recommended for skiing.\nPants fit perfectly to sizing chart, are super lightweight, extremely comfortable, and easy to move around in.\nPants are very comfortable, with belt and side elastic, light for summer days.\nGood lightweight, waterproof pants.\n6\nBest weight, pockets, durability, and look of any shorts user has owned, will be going with these as go-to running shorts.\nFlexible fit, elastic waistband, and detachable lightweight belt make shorts very functional.\nShorts are very comfortable and great for active wear on land and water.\nQuality of shorts is good, and they're reasonably priced and comfortable.\n7\nPack small, save room for travel.\nPacked in carry-on luggage in case of emergency.\n8\nJacket is thin and works well as an insulating layer.\nThe jacket is thin, but still absolutely windproof.\nJacket is thin and warm, keeps the wind out, color on photo corresponds to reality.\n9\nPants are perfect for traveling, repel moisture, are cool and comfortable, and dry quickly after washing.\nPants fit great, move with you, no binding, soft and light, wash and dry well, dry quickly if wet.\nPants do their job, keeping wearer dry and comfortable, even when humidity is 100% outside.\nPants were very comfortable, dried quickly, retained their shape, and worked well for the conditions.\n10\nSuper comfortable, beautiful and versatile jacket could become wearer's favorite.\nJacket is beautiful, can be worn on any occasion, goes well with casual and office wear.\nThe jacket is stylish, and can be worn in the office and outside.\nJacket is comfortable and looks chic, will become user's favorite.\n11\nFabric durable, suited well, great ventilation.\nSize fits, material ok, sufficiently durable.\nMaterial is good, withstands all weathers.\n12\nBest vest wearer has owned.\nFinds vest warm and smart, like the previous one.\n13\nGreat quality.\nSuper quality helps keep prices affordable.\nGreat quality.\nBest quality, as always.\n16\nLight, qualitative, nice product.\nLight and warm for in and around town.\nLight and warm product.\n17\nGood product.\nGreat product.\nGreat product.\nGreat product.\n18\nBest fitting pants of all Mountain Fox's pants, even if they're a little loose in the waist.\nBest fitting of all Mountain Fox pants, even if a little loose in waist.\n19\nVery comfortable, relaxed fit.\nProduct comfortable and fitting well.\nComfortable and functional, perfect fit, comfortable and stylish.\nComfortable fit.\n20\nJacket is versatile, stylish, and well-functioning, and customer doesn't regret buying it.\nJacket is versatile and of very nice quality, as one would expect from Arcteryx.\nJacket was a good buy, functions well, is stylish and handy.\nJacket is excellent, useful in all situations.\n21\nLoves the rain jacket, finds it a necessity for living in the Northwest.\nLoves the rain jacket, wore it several times in heavy rain without getting wet.\nLoves the jacket, which suits every occasion, with jeans, perfect for cycling, is breathable, and great as a rain jacket.\nLoves the jacket, finds it keeps them dry, has a nice fit, and isn't too bulky.\n22\nHigh-quality, well-designed shell pants.\nFirst pair of ski pants that actually fit me, and they're flattering.\nWill purchase North Face ski pants again, as they are very warm and comfortable, and fit the same as the first pair.\nExcellent fit, color, and performance of shell ski pant.\n23\nGood fit and quality after a couple of weeks of use.\nJacket fit right away and is still in good shape after several washes.\n25\nLikes capris from Columbia because they fit well, are comfortable, and are good for hiking and everyday activities.\nThe Capri pants fit nicely, and the customer would recommend them to anyone looking for a nice pair of pants that move easily and keep you cool.\nOrdering a pair of capris, find them to feel and fit great.\nReplacing non-Columbia capris with Columbia ones, finds fit and look nicer.\n26\nJacket is suitable for outdoor activities, keeps the wind well, keeps user warm.\nThe outdoor jacket is perfect.\nJacket is perfect for winter frost, with adjustment possibilities and high quality and durability.\nJacket is perfect for cold winter outdoor activities.\n27\nExtra length at bottom keeps legs warm.\nEven when kneeling, the shorts are long enough to stay in the right place.\n28\nJacket is versatile, low-maintenance, warm enough for town and trails, and stylish enough for town.\nJacket meets requirements of being light, cozy, and warm during breaks, and a second layer in winter.\nJacket works well as reinforcement clothing or intermediate layer in nature and the city.\nJacket is part of all outdoor activities, now user's go-to jacket for everything.\n29\nSuper lightweight, wicks sweat, resists staining.\nLightweight material is moisture wicking, so user expects to be comfortable.\nLightweight, dries fast, cool in summer months.\n30\nThe trousers are very comfortable, elastic, and not too tight.\nThe quality of the trousers is very good, and they're very comfortable.\nTrousers very comfortable and fit perfectly.\nTrousers are light and robust, comfortable to wear. Would buy again.\n32\nOutdoor equipment is simple, but chic, comfortable to wear, with a soothing color, and partly dirt and water-repellent.\nCute, comfortable rainwear will be great for upcoming trip.\n33\nPants are a nice quality, keep user dry on hikes.\nPants are heavy duty, built like a tank, very comfortable for hiking.\nPants are perfect for hiking along the California coast, with drastic shifts in temperature.\nPants are nice, lightweight, and great for hiking.\n34\nSuper pants, with reinforced material in right places.\nThe purchase was the right decision, perfect autumn hiking\/running pants, thick fabric, reinforced on the buttocks and knees, very soft and comfortable to wear.\nSuperb trousers for training and leisure in autumn, spring, and winter.\nPockets where they're needed, robust, perfect outdoor trousers.\n35\nPants are great - comfortable, fit well, look great. Will purchase more Columbia pants in the future.\nLove these go-to pants, hope Columbia doesn't discontinue them.\n36\nClimate in pants is perfect.\nPants are comfortable and versatile, perfect for cool mornings and evenings with hot days in between.\nPants are comfortable to wear, relaxed and comfortable to move, but still stylish.\nPants are the most comfortable ever, lightweight enough to not get hot, but heavy enough to not freeze when it's chilly.\n37\nJacket is the best in the world.\nBest-looking training jacket ever.\nThe best jacket I've ever owned.\nBest choice ever, couldn't be more satisfied with warm jacket after years of hand me downs.\n38\nFast delivery.\nSuper fast delivery.\nDelivery was fast.\nFast delivery.\n39\nProduct looks smart\/slim, keeps you warm.\nGood quality, warm.\nBest Buy! Stylish, warm enough, and true to size.\nFluffy, great quality, keeps you warm.\n40\nGreat two-in-one pants, allowing you to pack easily with pants and shorts in one.\nCan fit two pairs of trousers underneath in summer.\n42\nThe fit for my waist is great!\nFits well on someone with wider hips than many others. Doesn't ride up the back.\nFavorite feature is tie string inside waist band for perfect fit.\nThe product works very well, and the waist is larger than another size 4 they purchased in a different style, but the flexibility and smoothness of how it hangs on the body's frame is nice.\n43\nLikes the color, quality, weight, and waterproofing.\nLikes the color and fit.\nLikes the color, fit, warmth, and quality of the product.\nLikes the color and fit, and the good mobility and durability of the product.\n44\nBest pants I've ever owned.\nBest pants ever bought.\nThe best pants on the market, better than any others.\n45\nLikes the coat, comfortable and warm.\nUser loves a jacket.\nLikes the jacket, finds it comfortable.\nUser likes the jacket.\n46\nLikes fit, warmth, price of pants.\nLikes quality and comfort of product.\nLikes fit and warmth of padding.\n47\nLikes the capris, very comfortable and look great.\nLikes the material of the capris, which is stretchy, lightweight, and has sun protection.\nLikes Saturday capris, nice fit, easy to care for.\nLoves the capris' material and flattering fit.\n48\nJacket fits perfectly.\nJacket is great, fits well, is comfortable.\nGot a nice jacket that fit well.\nJacket is perfect, fits well.\n49\nJacket has nice cut, color is darker than photos.\nGreat jacket, great color.\nJacket is comfortable, love the color.\nJacket is very nice, good model and nice color.\n50\nLoves the North Face product, finds it keeps them dry and warm, the texture of the material comfortable and durable, the design and color selection amazing.\nGreat Northface fit and quality as usual.\nCustomer very happy with purchase, North Face quality is excellent, lasts a lifetime.\nNorth Face makes one feel confident and special.\n51\nLighter weight than other pants, fits perfectly. Bought another pair for daughter, who is taller than user, and they fit both of them great.\nGreat, lightweight, stretchy pants that fit perfectly. Had to size up one size.\n52\nThe design of the jacket was loved, and it fit perfectly.\nLiked the jacket, surprised at how cute it was, falls at waist, boxy cut and oversized.\nLoved the fit and material of coat.\nThrilled with the jacket's fit and fabric.\n53\nJacket is very warm, kept user warm on snow trips for ten days with only a t-shirt.\nJacket keeps you warm in coldest temperatures.\nFleece jacket keeps you super warm.\nJacket keeps user warm except for coldest days.\n54\nThe material is pleasant, light, and airy.\nThe material is soft, warm, and well-insulated.\nMaterial is quiet, dries quickly, perfect for a Scanian autumn and winter when it's rain rather than snow.\n55\nGood big pockets, size s fits perfectly, as customer is 181 and normal built.\n180cm, 88-90kg, with enough space to wear wool underneath, finds size 50l fits very well.\n56\nProduct is incredibly beautiful, better in reality than in the picture, fits equally well in city as well as forest and countryside.\nProduct is very nice and beautiful, good fabric and exactly as it should be.\nProduct is nice and pretty.\n57\nPants are exactly what user was looking for, wear them for working outdoors, hiking and casually.\nThe perfect pant for outdoor activities.\nShell pants work well for outdoor activities, look good while cycling.\nPant works for everything, suitable for everyday life and outdoor activities.\n58\nTook them rock climbing, got them wet, and they dried out quickly.\nDurable, climbed rocks and sat on the ground.\n59\nUser found that a small fit perfectly, when usually they wear medium.\nUser normally wears size small, and size small fit perfectly on the model.\nBoth sizes fit, but Small worked better for wearer's preferred layering.\nThe sizing was done according to very small people, so the XL size jacket was just right and right now isn't too small.\n60\nSnow pants fit well.\nSnow pants are great fit, very comfortable. Not too bulky.\nSnow pants fit perfectly.\nSnow pants are beautiful, lightweight, stylish, and unique.\n61\nLikes the product since purchase, finds it comfortable, beautiful and cozy.\nLikes product very much, thinks it's warm and cozy, well made.\nLikes the product, many small details and comfortable to wear.\nSon loves the product, finds it very comfortable, stylish, and cozy.\n62\nFit a tad snug on arrival, but they also stretch out a bit, so after washing and wearing them, they fit fine.\nThe fit is supposed to be loose, but since customer should be a size bigger, it fits him as a more slim fit and hugs his body, which he prefers.\n63\nGreat choice, would order again.\nLikes item, color, fit, would order again.\n64\nSleeves are too long, but customer likes the jacket so much, they'll have it shortened.\nLoves the carefully sewn, lovely, comfortable item, but sleeves are too long.\nSleeves are pretty long.\nSleeves are long to avoid gap between mitten and sleeve when arms are over head or moving around a lot.\n65\nLikes the quality of the pants and the fabric.\nLikes the pants.\nLikes the pants, they fit perfectly.\nLikes the fit and look of the pants, and the features of the pockets.\n66\nLikes the price point.\nFair price point for a product of this quality.\nPrice was very good.\n67\nCapris are very comfortable, cool, light, and come out of wash looking like new.\nCapris are very comfortable, lightweight, and water-repellant.\nCapris are very comfortable, you have to get a size bigger than usual.\nCapris are so comfortable, love the fit and feel.\n68\nJacket is warm, nice, and well-fitting.\nJacket is very warm, very windproof, and comfortable.\nJacket is super, warm and comfortable.\nGreat, warm, stylish jacket.\n69\nFits perfectly.\nFits perfectly, looks cute.\nFits perfectly, great comfort.\nFits great.\n71\nLight weight, washable, and good looking.\nGreat fit and very lightweight.\nStylish, cozy, lightweight makes it easy to wear and look good.\nLight weight, durable.\n72\nCut, color, and comfort of product is great.\nLight and comfortable, nice cut and quality of fabric.\nComfortable and easy to wear, cut is great.\n74\nConsistently high quality in materials and details.\nProduct is good, in delicious materials.\nQuality of material and raw material is good.\nProduct is good quality, with several good details.\n76\nCoat is normal size, good fit, warm, and very pleased with it.\nCoat fits well and length is good.\nCoat is not meant for cold climates, but customer is pleased with it.\nCoat is a good fit for user, despite being larger than recommended.\n77\nWaist fit well, but legs were super baggy.\nPants fit well, good for a pear-shaped body.\nPants fit well, not baggy.\nPants fit perfectly, slightly big but in a comfortable way.\n78\nProduct quality and warmth and comfort are fantastic.\nProduct quality, construction, warmth are all very good.\n\n###\n\n"
pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=4095)
result = pipe(f"<s>[INST] {prompt} [/INST]")

print("result is:111111111")
print(result[0]['generated_text'])
