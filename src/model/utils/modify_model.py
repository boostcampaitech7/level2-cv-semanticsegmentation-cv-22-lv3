def update_last_layer(model, new_out_channels):
    if hasattr(model, 'classifier'):
        classifier = model.classifier

        if hasattr(classifier, '4') and isinstance(classifier[4], nn.Conv2d):
            classifier[4].out_channels = new_out_channels
        
        elif hasattr(classifier, 'low_classifier') and hasattr(classifier, 'high_classifier'):
            classifier.low_classifier.out_channels = new_out_channels
            classifier.high_classifier.out_channels = new_out_channels
        
        else:
            raise ValueError('Unsupported classifier structure.')
    else:
        raise AttributeError('The model does not have a "classifier" attribute.')
    
    return model