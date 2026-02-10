import json
import sys

def convert_to_alpaca_format(input_path, output_path):
    with open(input_path, "r") as f:
        data = json.load(f)

    alpaca_dataset = []

    for key in data:
        entries = data[key]
        for entry in entries:
            sample = entry.get("sample", {})
            question = sample.get("question", "").strip()
            context = sample.get("context", "").strip()
            answer = sample.get("answer", "").strip()

            if question and answer:
                alpaca_dataset.append({
                    "instruction": question,
                    "input": context,
                    "output": answer
                })

    with open(output_path, "w") as f:
        json.dump(alpaca_dataset, f, indent=2)

    print(f"Converted {len(alpaca_dataset)} samples to Alpaca format.")
    print(f"📄 Output saved to: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert_to_alpaca.py <input_json_path> <output_json_path>")
        sys.exit(1)

    convert_to_alpaca_format(sys.argv[1], sys.argv[2])
