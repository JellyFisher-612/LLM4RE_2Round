import json

def is_valid_element(element):
    """
    Checks if a single element in the 'output' list is valid based on the rules:
    - 'subject' and 'object', if present, must be non-empty lists.
    - 'relationship' can be anything or even missing.
    """
    subject = element.get("subject")
    object_val = element.get("object")

    # Check if subject exists and is empty
    if subject is not None and (not isinstance(subject, list) or len(subject) == 0):
        return False
    # Check if object exists and is empty
    if object_val is not None and (not isinstance(object_val, list) or len(object_val) == 0):
        return False

    # If both checks pass, the element is valid
    return True

def filter_output(json_data):
    """
    Filters the 'output' field in the given JSON data according to the rules.
    """
    # Assume the input is a list of items like the example provided
    for item in json_data:
        if "output" in item and isinstance(item["output"], list):
            # Filter the 'output' list, keeping only valid elements
            filtered_output = [elem for elem in item["output"] if is_valid_element(elem)]
            # Assign the filtered list back. If empty, it becomes []
            item["output"] = filtered_output
    return json_data

def main():
    """
    Main function to load JSON, filter it, and save the result.
    """
    input_file_path = "/root/autodl-tmp/LLM4RE_2Round/Prediction/LAMMA_RAW.json" # Replace with your input file path
    output_file_path = "/root/autodl-tmp/LLM4RE_2Round/Prediction/LAMMA_RAW_del.json" # Replace with your desired output file path

    try:
        # Read the JSON file
        with open(input_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File '{input_file_path}' not found.")
        return
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return

    # Filter the data
    filtered_data = filter_output(data)

    # Write the filtered data back to a new file
    try:
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(filtered_data, f, ensure_ascii=False, indent=2)
        print(f"Filtering complete. Results saved to '{output_file_path}'.")
    except IOError as e:
        print(f"Error writing to file: {e}")

if __name__ == "__main__":
    main()