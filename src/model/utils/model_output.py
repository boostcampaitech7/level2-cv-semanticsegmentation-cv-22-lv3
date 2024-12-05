def get_model_output(model, input):
    outputs = model(input)['out'] if isinstance(model(input), dict) else model(input)
    return outputs