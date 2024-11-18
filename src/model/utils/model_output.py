def get_model_output(model, library, input):
    if library == 'smp':
        return model(input)
    else:
        return model(input)['out']