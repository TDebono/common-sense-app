def preprocess_function(examples, ending_names=["OptionA", "OptionB", "OptionC"]):
    # Repeat each first sentence three times to go with the three possibilities of second sentences.
    first_sentences = [[context] * 3 for context in examples["FalseSent"]]
    # Grab all second sentences possible for each context.
    question_headers = examples["sent2"]
    second_sentences = [[f"{examples[end][i]}" for end in ending_names] for i, header in enumerate(question_headers)]
    # Flatten everything
    first_sentences = sum(first_sentences, [])
    second_sentences = sum(second_sentences, [])
    sentences = [first + " " + second for first, second in zip(first_sentences, second_sentences)]

    
    # Tokenize
    tokenized_examples = tokenizer(sentences, truncation=True)
    # Un-flatten
    return {k: [v[i:i+3] for i in range(0, len(v), 3)] for k, v in tokenized_examples.items()}