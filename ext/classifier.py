from utils.data import warning_sentences, max_token_length, info_sentences
import time
import numpy as np
import logging
from tensorflow import keras
import json
import os
from utils.exceptions import *
import utils.string_checker as string_checker
from utils.string_helper import StringHelper
from typing import *
from dataclasses import dataclass


@dataclass
class PredictionData:
    tag: str
    responses: List[str]
    confidence: float


class Classifier:
    def __init__(self, str_helper: StringHelper, datasets: List[str]) -> None:
        self.__string_helper: StringHelper = str_helper
        self.__datasets: List[str] = datasets
        self.__tags: List[str] = []

        # Create model
        self.__model: Union[keras.Sequential, None] = None
        # self.__init_tags()

        self.__responses_by_tags: Dict[str, List[str]] = {}
        self.__init_responses_by_tags()

        self.__tags = list(self.__responses_by_tags.keys())

    def __init_responses_by_tags(self) -> None:
        """
        Inits all responses
        :return: None
        """
        for dataset in self.__datasets:
            if not os.path.exists(dataset):
                raise FileNotFoundError(f'Unable to find file {dataset}')
            if not os.access(dataset, os.R_OK):
                raise PermissionError(f'Unable to read file {dataset}')
            with open(dataset, 'r', encoding='utf-8') as f:
                r_dataset: dict = json.load(f)
            tags: List[str] = [x['tag'] for x in r_dataset['intents']]
            for tag in tags:
                self.__responses_by_tags[tag] = r_dataset['intents'][tags.index(tag)]['responses']

    def get_data_by_prediction(self, prediction: np.ndarray) -> Optional[PredictionData]:
        tag: str = self.__tags[int(np.argmax(prediction))]
        responses: List[str] = self.__responses_by_tags.get(tag, None)
        if responses:
            return PredictionData(tag, responses, prediction.max())

    def load_model(self, path: str) -> None:
        """
        Loads given model into instance
        :param path: The path to the model
        :return: None
        """
        if not os.path.isfile(path):
            raise FileNotFoundError(f'Unable to find model at {path}.')
        if not os.access(path, os.R_OK):
            raise PermissionError(f'Unable to read model at {path}. Reason: No Read Privileges.')
        self.__model = keras.models.load_model(path)

    @staticmethod
    def __is_tag_invalid(tag: Any, stored_tags: List[str]) -> Tuple[bool, str]:
        """
        Checks if a given tag is valid or not
        :param tag: The given tag
        :param stored_tags: List which should contain all already added tags
        :return: True if the tag is invalid, else False. Second is error str
        """
        if string_checker.is_empty(tag):
            return True, 'no_tag'

        if tag in stored_tags:
            return True, 'duplicated_tag'
        return False, ''

    def train(self, epochs: int = 250, save_model: bool = True) -> None:
        """
        Trains the model
        :param epochs: The amount of epochs to train on
        :param save_model: If true the model is saved locally after training
        :return:
        """
        logging.info(info_sentences['prepare_data'])
        features, labels = self.__get_features_and_labels()

        if not self.__model:
            self.__init_model()

        logging.info(info_sentences['start_training'])
        self.__model.compile(optimizer='adam', loss=keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
        self.__model.fit(features, labels, epochs=epochs)

        if save_model:
            os.makedirs('classifier_models', exist_ok=True)
            self.__model.save(f'classifier_models/classifier-{time.time()}.h5')

    def classify(self, s: str) -> np.ndarray:
        if not self.__model:
            raise Exception('Please train or load your model first, before calling the classify function.')
        return self.__model.predict(self.__string_helper.get_insertable(s, max_token_length, is_input=True))[0]

    def __get_features_and_labels(self) -> List[np.ndarray]:
        """
        prepares training data, given at the constructor
        :return: returns the converted data in order: features, labels
        """
        features: List[np.ndarray] = []
        labels: List[float] = []

        stored_tags: List[str] = []  # just to prevent duplicates

        for dataset in self.__datasets:
            if not os.path.exists(dataset):
                raise FileNotFoundError(f'Unable to find file {dataset}')
            if not os.access(dataset, os.R_OK):
                raise PermissionError(f'Unable to read file {dataset}')
            with open(dataset, 'r', encoding='utf-8') as f:
                r_dataset: dict = json.load(f)
            for idx, intent in enumerate(r_dataset['intents']):
                tag: str = intent['tag']
                if tag not in self.__tags:
                    self.__tags.append(tag)

                is_invalid, error_str = self.__is_tag_invalid(tag, stored_tags)
                if is_invalid:
                    if not string_checker.is_empty(error_str):
                        logging.warning(
                            f'Something went wrong. Skipped tag {tag} from dataset {dataset}. Error: {error_str}')
                    continue

                patterns: List[str] = intent['patterns']
                if len(patterns) <= 0:
                    logging.warning(warning_sentences['no_patterns'].format(tag=tag, dataset=dataset))
                    continue

                stored_tags.append(tag)

                # get label value for patterns
                if len(features) < 1:
                    # there are no features, start from 1
                    tag_numeric_value = 1
                else:
                    # already values in, old value +1
                    tag_numeric_value = labels[-1] + 1

                # preparing patterns and store good ones
                for pattern in patterns:
                    if string_checker.is_empty(pattern):
                        # Pattern is invalid, bc. it's empty
                        logging.warning(
                            warning_sentences['empty_pattern'].format(pattern=pattern, dataset=dataset, tag=tag))
                        continue
                    if self.__string_helper.get_token_length(pattern) > max_token_length:
                        # The string_helper follows a max_token_length to have all data normalized
                        # your pattern can go below, but not over it
                        logging.warning(
                            warning_sentences['max_token_length_exceeded'].format(pattern=pattern, dataset=dataset,
                                                                                  tag=tag,
                                                                                  max_token_length=max_token_length),
                            stack_info=False)
                        continue

                    n_pattern: np.ndarray = self.__string_helper.get_insertable(pattern, 25)

                    features.append(n_pattern)
                    labels.append(tag_numeric_value)

        max_label_val: float = labels[-1]
        labels = [x / max_label_val for x in labels]
        labels = keras.utils.to_categorical(labels, num_classes=len(self.__tags))
        return [np.array(features), np.array(labels)]

    def __init_model(self) -> None:
        """
        init the model
        :return: None
        """
        if not self.__tags:
            raise TagsNotInitializedException('init_model function can\'t be called before init_datasets function')
        self.__model: keras.Sequential = keras.Sequential([
            keras.layers.LSTM(128, return_sequences=True),
            keras.layers.Dense(128),
            keras.layers.Dropout(.2),
            keras.layers.LSTM(128, return_sequences=True),
            keras.layers.Dense(128),
            keras.layers.Dropout(.2),
            keras.layers.LSTM(128),
            keras.layers.Dense(len(self.__tags), activation='softmax')
        ])
