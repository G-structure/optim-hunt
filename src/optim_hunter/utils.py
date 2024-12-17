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
    suffix.append(f"{y_train.name}:")
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