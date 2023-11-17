import json
import socket
from threading import Thread

from ext.classifier import Classifier, PredictionData
from utils import string_checker
from utils.data import prefix
from utils.string_helper import StringHelper, Word2VecModels, Model


class SocketServer:
    def __init__(self, classifier: Classifier, host: str = '0.0.0.0', port: int = 3030) -> None:
        self.__host: str = host
        self.__port: int = port
        self.__classifier: Classifier = classifier

        self.__socket: socket.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.__socket.bind((host, port))

    def start(self) -> None:
        self.__socket.listen()

        print(prefix, 'Server is ready to use. Listening ...')

        while True:
            conn, _addr = self.__socket.accept()
            try:
                Thread(target=self.__handle_connection, args=(conn,)).start()
            except socket.error:
                # socket error, return nothing
                continue
            except (Exception,):
                # send error msg
                conn.sendall(bytes(json.dumps(self.__build_error_msg()), 'utf-8'))

    @staticmethod
    def __build_error_msg() -> dict:
        return {'status': 'error', 'error_msg': 'Sorry, an unknown error happened.'}

    @staticmethod
    def __recv(conn: socket.socket) -> bytes:
        rec_data: bytes = b''
        while True:
            tmp_data: bytes = conn.recv(1024)
            if tmp_data:
                if not str(tmp_data).encode(encoding='utf-8'):
                    break
                rec_data += tmp_data
            break
        return rec_data

    def __handle_connection(self, conn: socket.socket) -> None:
        rec_data: bytes = self.__recv(conn)
        data: dict = json.loads(rec_data)
        resp: dict = self.__get_response(data)
        conn.sendall(bytes(json.dumps(resp), 'utf-8'))
        conn.close()

    def __get_response(self, mc_input: dict) -> dict:
        """
        Creates a response for the minecraft plugin
        :param mc_input: The Minecraft Request data (MUST have "sentence" key)
        :return: Dictionary with tag and responses
        """
        sentence: str = mc_input.get('sentence', None)
        if not sentence or string_checker.is_empty(sentence):
            return {'status': 'error', 'error_msg': 'No sentence were given!'}
        data: PredictionData = self.__classifier.get_data_by_prediction(self.__classifier.classify(sentence))
        return {
            'status': 'ok',
            'sentence': sentence,
            'tag': data.tag,
            'responses': data.responses,
            'confidence': str(data.confidence)
        }

    def __del__(self) -> None:
        self.__socket.close()


def main() -> None:
    # Build str helper (string processing)
    all_models: Word2VecModels = Word2VecModels()
    w2v_model: Model = all_models.get_smallest_model()
    str_helper: StringHelper = StringHelper(w2v_model)

    classifier: Classifier = Classifier(str_helper,
                                        ['dataset/intents.json'])
    # classifier.train(epochs=300)
    classifier.load_model('classifier_models/classifier-1700166883.6944025.h5')

    server: SocketServer = SocketServer(classifier)
    server.start()


if __name__ == '__main__':
    main()
