import json

class TestData:
    def __init__(self):
        pass

    def load(self, file_path):
        test_data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        cleaned_test = {}
                        test = json.loads(line)
                        if test.get('messages'):
                            for message in test['messages']:
                                if message['role'] == 'user':
                                    cleaned_test['inputs'] = message['content']
                                if message['role'] == 'assistant':
                                    cleaned_test['model_predictions'] = message['content']
                        if test.get('ground_truth'):
                            cleaned_test['ground_truth'] = test['ground_truth'][0]['content']
                        test_data.append(cleaned_test)
                    except json.JSONDecodeError as e:
                        print(f"Error decoding line: {e}")
        return test_data
