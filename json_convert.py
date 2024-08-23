import json

# Read data from 1.jsonl and transform it into the desired format
output_data = []
with open("2.jsonl", "r") as input_file:
    for line in input_file:
        entry = json.loads(line.strip())
        prompt = entry["prompt"]
        completion = entry["completion"]
        transformed_entry = {
            "text": f"<s><INST>{prompt}</INST><INST>{completion}</INST></s>"
        }
        output_data.append(transformed_entry)

# Write the transformed data to 2.json
with open("1.json", "w") as output_file:
    json.dump(output_data, output_file, indent=4)

print("Conversion completed. Data written to 2.json.")