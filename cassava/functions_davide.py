from sklearn.metrics import confusion_matrix

def ConfusionMatrix(valid, model, bNormalize = True, bGradient = True):
    """
    :param valid: the validation set;
    :param model: the model (use learn.model.eval() to get it);
    :param bNormalize: a Boolean parameter to set the normalization (or not) of data;
    :param bGradient: a Boolean parameter to set the Gradient (or not) in the return;
    :return: a Pandas DataFrame with the Confusion Matrix.

    """

    label = []
    pred = []

    for i, j in valid:
        pred.append(torch.argmax(model(i), axis=1).tolist())
        label.append(j.tolist())

    if bNormalize:
        df = pd.DataFrame(confusion_matrix(np.array(list(itertools.chain.from_iterable(label))),
                                                    np.array(list(itertools.chain.from_iterable(pred))),
                                                    normalize="true")).mul(100).round(2)
    else:
        df = pd.DataFrame(confusion_matrix(np.array(list(itertools.chain.from_iterable(label))),
                                           np.array(list(itertools.chain.from_iterable(pred)))))

    if bGradient:
        df = df.style.set_properties(**{"font-size": "10pt"}).background_gradient("Blues").format('{0:,.2f}%')

    return df