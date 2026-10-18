def find_s(training_data):
    hypothesis = None

    for features, label in training_data:
        if label != 1:
            continue

        if hypothesis is None:
            hypothesis = features.copy()
        else:
            for i in range(len(hypothesis)):
                if hypothesis[i] != features[i]:
                    hypothesis[i] = '?'

    return hypothesis

training_data = [
    (['High', 'Good', 'Stable', 'No'], 1),
    (['Medium', 'Good', 'Stable', 'No'], 1),
    (['Low', 'Poor', 'Stable', 'No'], 1),
    (['High', 'Poor', 'Stable', 'No'], 0),
    (['High', 'Good', 'Unstable', 'No'], 0),
    (['Medium', 'Good', 'Stable', 'Yes'], 0)
]

result = find_s(training_data)
print("Final hypothesis:", result)