def get_numerical_tokens(model):
    # Get the vocabulary
    vocab = model.tokenizer.get_vocab()

    # Search for numerical tokens
    numerical_tokens = {}
    for token, id in vocab.items():
        # Skip superscript numbers and other special characters
        if token in ['¹', '²', '³']:
            continue
            
        # Check if token is a number (integer or decimal)
        # Only include ASCII digits and decimal point
        if token.strip().replace('.','').replace('-','').isdigit() and all(c in '0123456789.-' for c in token):
            numerical_tokens[token] = id
    
    return numerical_tokens