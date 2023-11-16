from ext.classifier import Classifier
from utils.string_helper import StringHelper, Word2VecModels, Model


def main() -> None:
    # Build str helper (string processing)
    all_models: Word2VecModels = Word2VecModels()
    w2v_model: Model = all_models.get_smallest_model()
    str_helper: StringHelper = StringHelper(w2v_model)

    classifier: Classifier = Classifier(str_helper,
                                        ['dataset/intents.json'])
    # classifier.train(epochs=250)
    classifier.load_model('classifier-1700162899.0588.h5')

    test: str = 'hello'
    print(classifier.classify(test))


if __name__ == '__main__':
    main()
