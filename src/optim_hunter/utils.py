import pandas as pd
import torch

def prepare_prompt(x_train, y_train, x_test):
    """
    Prepare the prompt
    """
    # Format numeric columns to 3 sig figs
    x_train = x_train.round(3)
    y_train = y_train.round(3)
    x_test = x_test.round(3)

    # Get input variables (features)
    input_variables = x_train.columns.to_list()

    # Create examples list of dicts combining x and y values
    examples = [{**x1, y_train.name: x2} for x1, x2 in zip(x_train.to_dict('records'), y_train)]

    # Create the template for examples
    template = [f"{feature}: {{{feature}}}" for feature in x_train.columns]
    template.append(f"{y_train.name}: {{{y_train.name}}}")
    template = "\n".join(template)

    # Create suffix (test case format)
    suffix = [f"{feature}: {{{feature}}}" for feature in x_train.columns]
    suffix.append(f"{y_train.name}: ")
    suffix = "\n".join(suffix)

    # Format all examples using the template
    formatted_examples = [template.format(**example) for example in examples]
    examples_text = "\n\n".join(formatted_examples)

    # Format the test case using the suffix
    test_case = suffix.format(**x_test.to_dict('records')[0])

    # Add instruction prefix
    prefix_instruction = 'The task is to provide your best estimate for "Output". Please provide that and only that, without any additional text.\n\n\n\n\n'

    # Combine everything
    final_prompt = f"{prefix_instruction}{examples_text}\n\n{test_case}"

    return final_prompt

def prepare_prompt_from_tokens(model, x_train_tokens, y_train_tokens, x_test_tokens, prepend_bos=True, prepend_inst=True):
    """
    Prepare a prompt tensor from pre-tokenized numeric data.

    Args:
        model: The language model with tokenizer
        x_train_tokens (pd.DataFrame): DataFrame containing tokenized training features
        y_train_tokens (pd.Series): Series containing tokenized training labels
        x_test_tokens (pd.DataFrame): DataFrame containing tokenized test features
        prepend_bos (bool): Whether to prepend the beginning of sequence token. Defaults to True.

    Returns:
        torch.Tensor: A tensor of tokens representing the complete prompt with shape [1, sequence_length]
    """
    # Get tokens for static text elements
    instruction = model.to_tokens(
        'The task is to provide your best estimate for "Output". Please provide that and only that, without any additional text.\n\n\n\n\n',
        prepend_bos=prepend_bos
    )[0]

    newline = model.to_tokens('\n', prepend_bos=False)[0]
    double_newline = model.to_tokens('\n\n', prepend_bos=False)[0]
    colon_space = model.to_tokens(': ', prepend_bos=False)[0]

    # Initialize list to store all tokens
    all_tokens = []

    # Add instruction tokens
    if prepend_inst:
        all_tokens.extend(instruction.tolist())

    # Process training examples
    for idx in range(len(x_train_tokens)):
        if idx > 0:
            # Add separator between examples
            all_tokens.extend(double_newline.tolist())

        # Add features
        for col in x_train_tokens.columns:
            # Add feature name
            feature_tokens = model.to_tokens(f"{col}", prepend_bos=False)[0]
            all_tokens.extend(feature_tokens.tolist())

            # Add colon and space
            all_tokens.extend(colon_space.tolist())

            # Add feature value
            all_tokens.extend(x_train_tokens[col].iloc[idx].tolist())

            # Add newline
            all_tokens.extend(newline.tolist())

        # Add output label
        output_name_tokens = model.to_tokens("Output", prepend_bos=False)[0]
        all_tokens.extend(output_name_tokens.tolist())

        # Add colon and space
        all_tokens.extend(colon_space.tolist())

        # Add output value
        all_tokens.extend(y_train_tokens.iloc[idx].tolist())

    # Add separator before test case
    all_tokens.extend(double_newline.tolist())

    # Add test features
    for col in x_test_tokens.columns:
        # Add feature name
        feature_tokens = model.to_tokens(f"{col}", prepend_bos=False)[0]
        all_tokens.extend(feature_tokens.tolist())

        # Add colon and space
        all_tokens.extend(colon_space.tolist())

        # Add feature value
        all_tokens.extend(x_test_tokens[col].iloc[0].tolist())

        # Add newline
        all_tokens.extend(newline.tolist())

    # Add final "Output: "
    output_tokens = model.to_tokens("Output: ", prepend_bos=False)[0]
    all_tokens.extend(output_tokens.tolist())

    # Convert to tensor and ensure on correct device
    prompt_tensor = torch.tensor(all_tokens, device=model.cfg.device)

    return prompt_tensor.unsqueeze(0)  # Add batch dimension

def slice_dataset(x_train, y_train, x_test, y_test, n=10):
    """
    Slice the first n items from each dataset while preserving DataFrame structure

    Args:
        x_train (pd.DataFrame): Training features
        y_train (pd.Series): Training labels
        x_test (pd.DataFrame): Test features
        y_test (pd.Series): Test labels
        n (int): Number of items to keep

    Returns:
        tuple: (x_train_slice, y_train_slice, x_test_slice, y_test_slice)
    """
    x_train_slice = x_train.iloc[:n]
    y_train_slice = y_train.iloc[:n]
    x_test_slice = x_test.iloc[:n]
    y_test_slice = y_test.iloc[:n]

    return x_train_slice, y_train_slice, x_test_slice, y_test_slice

def pad_numeric_tokens(model, x_train, y_train, x_test):
    """
    Create new dataframes/lists with tokenized and padded numeric values.

    Args:
        model: The language model with tokenizer
        x_train: DataFrame or list of DataFrames with training features
        y_train: Series or list of Series with training labels
        x_test: DataFrame or list of DataFrames with test features

    Returns:
        tuple: (x_train_tokens, y_train_tokens, x_test_tokens) matching input type
    """
    # Convert to lists if single DataFrame/Series
    is_single = not isinstance(x_train, list)
    if is_single:
        x_train = [x_train]
        y_train = [y_train]
        x_test = [x_test]

    # Get zero token for padding
    zero_token = model.to_tokens('0', truncate=True)[0][-1].cpu()  # Move to CPU

    # Format numeric columns to 3 sig figs
    x_train = [x.round(3) for x in x_train]
    y_train = [y.round(3) for y in y_train]
    x_test = [x.round(3) for x in x_test]

    # Function to tokenize a single number
    def tokenize_number(num):
        return model.to_tokens(str(num), prepend_bos=False, truncate=True)[0].cpu()  # Move to CPU

    # Function to pad tokens to target length
    def pad_tokens(tokens, max_len):
        if len(tokens) < max_len:
            padding = torch.tensor([zero_token] * (max_len - len(tokens)))
            return torch.cat([tokens, padding])
        return tokens

    # Get all numeric values (both x and y)
    all_values = []
    for x_df in x_train + x_test:
        for col in x_df.columns:
            all_values.extend(x_df[col].values)
    for y_series in y_train:
        all_values.extend(y_series.values)

    # Tokenize all values and find global maximum length
    all_tokenized = [tokenize_number(val) for val in all_values]
    global_max_len = max(len(tokens) for tokens in all_tokenized)

    # Process X training data
    x_train_tokens = []
    for x_df in x_train:
        x_tokens_df = pd.DataFrame(index=x_df.index)
        for col in x_df.columns:
            x_tokens_df[col] = [
                pad_tokens(tokenize_number(val), global_max_len)
                for val in x_df[col]
            ]
        x_train_tokens.append(x_tokens_df)

    # Process X test data
    x_test_tokens = []
    for x_df in x_test:
        x_tokens_df = pd.DataFrame(index=x_df.index)
        for col in x_df.columns:
            x_tokens_df[col] = [
                pad_tokens(tokenize_number(val), global_max_len)
                for val in x_df[col]
            ]
        x_test_tokens.append(x_tokens_df)

    # Process y values
    y_train_tokens = []
    for y_series in y_train:
        y_tokens = pd.Series([
            pad_tokens(tokenize_number(val), global_max_len)
            for val in y_series
        ], index=y_series.index)
        y_train_tokens.append(y_tokens)

    # Return single items if input was single items
    if is_single:
        return x_train_tokens[0], y_train_tokens[0], x_test_tokens[0]

    return x_train_tokens, y_train_tokens, x_test_tokens
