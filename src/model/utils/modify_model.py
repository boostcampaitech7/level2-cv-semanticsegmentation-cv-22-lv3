import torch.nn as nn

def modify_model(model, num_classes):
    # classifier 수정
    if hasattr(model, 'classifier'):
        _set_num_classes(model.classifier, num_classes)
    else:
        print("Model has no 'classifier' attribute.")

    # aux_classifier 수정 (존재하는 경우)
    if hasattr(model, 'aux_classifier'):
        _set_num_classes(model.aux_classifier, num_classes)
    else:
        print("Model has no 'aux_classifier' attribute.")

    return model

def _set_num_classes(module, num_classes):
    last_conv = None
    last_conv_name = None

    # 모듈 내의 모든 하위 모듈 탐색
    for name, child in module.named_children():
        if isinstance(child, nn.Conv2d):
            last_conv = child
            last_conv_name = name
        else:
            _set_num_classes(child, num_classes)  # 재귀적으로 탐색

    # 마지막 Conv2d 레이어 수정
    if last_conv is not None:
        new_conv = nn.Conv2d(
            in_channels=last_conv.in_channels,
            out_channels=num_classes,
            kernel_size=last_conv.kernel_size,
            stride=last_conv.stride,
            padding=last_conv.padding,
            dilation=last_conv.dilation,
            groups=last_conv.groups,
            bias=(last_conv.bias is not None),
            padding_mode=last_conv.padding_mode
        )
        setattr(module, last_conv_name, new_conv)
        print(f"Modified {module.__class__.__name__} '{last_conv_name}' out_channels to {num_classes}.")


import torch.nn as nn

def modify_deeplabv3_model(model, num_classes):
    # classifier 수정
    if hasattr(model, 'classifier'):
        classifier = model.classifier

        # 마지막 Conv2d 레이어를 찾아 수정합니다.
        if isinstance(classifier, nn.Sequential):
            # nn.Sequential인 경우
            for idx in reversed(range(len(classifier))):
                if isinstance(classifier[idx], nn.Conv2d):
                    in_channels = classifier[idx].in_channels
                    kernel_size = classifier[idx].kernel_size
                    stride = classifier[idx].stride
                    padding = classifier[idx].padding
                    dilation = classifier[idx].dilation

                    # 새로운 Conv2d 레이어 생성
                    classifier[idx] = nn.Conv2d(
                        in_channels, num_classes,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                        dilation=dilation
                    )
                    print(f"Modified classifier Conv2d layer at index {idx} to have {num_classes} output channels.")
                    break
        else:
            # nn.Sequential이 아닌 경우
            for name, module in classifier.named_modules():
                if isinstance(module, nn.Conv2d):
                    in_channels = module.in_channels
                    kernel_size = module.kernel_size
                    stride = module.stride
                    padding = module.padding
                    dilation = module.dilation

                    # 새로운 Conv2d 레이어 생성
                    new_conv = nn.Conv2d(
                        in_channels, num_classes,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                        dilation=dilation
                    )
                    setattr(classifier, name, new_conv)
                    print(f"Modified classifier Conv2d layer '{name}' to have {num_classes} output channels.")
                    break

    else:
        print("Model has no 'classifier' attribute.")

    # aux_classifier 수정 (존재하는 경우)
    if hasattr(model, 'aux_classifier'):
        aux_classifier = model.aux_classifier

        # 마지막 Conv2d 레이어를 찾아 수정합니다.
        if isinstance(aux_classifier, nn.Sequential):
            for idx in reversed(range(len(aux_classifier))):
                if isinstance(aux_classifier[idx], nn.Conv2d):
                    in_channels = aux_classifier[idx].in_channels
                    kernel_size = aux_classifier[idx].kernel_size
                    stride = aux_classifier[idx].stride
                    padding = aux_classifier[idx].padding
                    dilation = aux_classifier[idx].dilation

                    # 새로운 Conv2d 레이어 생성
                    aux_classifier[idx] = nn.Conv2d(
                        in_channels, num_classes,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                        dilation=dilation
                    )
                    print(f"Modified aux_classifier Conv2d layer at index {idx} to have {num_classes} output channels.")
                    break
        else:
            for name, module in aux_classifier.named_modules():
                if isinstance(module, nn.Conv2d):
                    in_channels = module.in_channels
                    kernel_size = module.kernel_size
                    stride = module.stride
                    padding = module.padding
                    dilation = module.dilation

                    # 새로운 Conv2d 레이어 생성
                    new_conv = nn.Conv2d(
                        in_channels, num_classes,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                        dilation=dilation
                    )
                    setattr(aux_classifier, name, new_conv)
                    print(f"Modified aux_classifier Conv2d layer '{name}' to have {num_classes} output channels.")
                    break

    else:
        print("Model has no 'aux_classifier' attribute.")

    return model

def modify_lraspp_model(model, num_classes):
    # low_classifier 수정
    if hasattr(model.classifier, 'low_classifier'):
        low_classifier = model.classifier.low_classifier
        if isinstance(low_classifier, nn.Conv2d):
            new_low_classifier = nn.Conv2d(
                in_channels=low_classifier.in_channels,
                out_channels=num_classes,
                kernel_size=low_classifier.kernel_size,
                stride=low_classifier.stride,
                padding=low_classifier.padding,
                bias=(low_classifier.bias is not None),
                padding_mode=low_classifier.padding_mode
            )
            model.classifier.low_classifier = new_low_classifier
            print(f"Modified low_classifier out_channels to {num_classes}.")
        else:
            print("low_classifier is not a Conv2d")
    else:
        print("Model has no 'low_classifier' attribute.")

    # high_classifier 수정
    if hasattr(model.classifier, 'high_classifier'):
        high_classifier = model.classifier.high_classifier
        if isinstance(high_classifier, nn.Conv2d):
            new_high_classifier = nn.Conv2d(
                in_channels=high_classifier.in_channels,
                out_channels=num_classes,
                kernel_size=high_classifier.kernel_size,
                stride=high_classifier.stride,
                padding=high_classifier.padding,
                bias=(high_classifier.bias is not None),
                padding_mode=high_classifier.padding_mode
            )
            model.classifier.high_classifier = new_high_classifier
            print(f"Modified high_classifier out_channels to {num_classes}.")
        else:
            print("high_classifier is not a Conv2d")
    else:
        print("Model has no 'high_classifier' attribute.")

    # cbr 모듈 수정
    if hasattr(model.classifier, 'cbr'):
        cbr = model.classifier.cbr
        if isinstance(cbr, nn.Sequential):
            # Conv2d 수정
            conv = cbr[0]
            if isinstance(conv, nn.Conv2d):
                new_conv = nn.Conv2d(
                    in_channels=conv.in_channels,
                    out_channels=128,
                    kernel_size=conv.kernel_size,
                    stride=conv.stride,
                    padding=conv.padding,
                    dilation=conv.dilation,
                    groups=conv.groups,
                    bias=(conv.bias is not None),
                    padding_mode=conv.padding_mode
                )
                cbr[0] = new_conv
                print("Reset cbr Conv2d out_channels to 128.")
            # BatchNorm2d 수정
            bn = cbr[1]
            if isinstance(bn, nn.BatchNorm2d):
                new_bn = nn.BatchNorm2d(num_features=128)
                cbr[1] = new_bn
                print("Reset cbr BatchNorm2d num_features to 128.")

    return model
