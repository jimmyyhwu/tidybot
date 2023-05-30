import argparse
import time
from multiprocessing.connection import Listener
from threading import Thread
from constants import CONN_AUTHKEY

class Publisher:
    def __init__(self, hostname='localhost', port=6000):
        self.listener = Listener((hostname, port), authkey=CONN_AUTHKEY)
        self.data = None
        self.timestamp = None

    def get_data(self):
        time.sleep(0.033)
        return {'timestamp': time.time()}

    def worker(self):
        while True:
            data = self.get_data()
            self.data = data
            self.timestamp = time.time()

    def clean_up(self):
        print('Cleaning up')

    def handle_conn(self, conn):
        last_timestamp = self.timestamp
        try:
            while True:
                # Wait for new data
                while self.timestamp == last_timestamp:
                    time.sleep(0.0001)
                last_timestamp = self.timestamp

                # Send new data
                conn.recv()
                conn.send(self.data)
        except (ConnectionResetError, EOFError, BrokenPipeError):
            pass

    def run(self):
        try:
            Thread(target=self.worker, daemon=True).start()
            address, port = self.listener.address
            print(f'Waiting for connections ({address}:{port})')
            while True:
                conn = self.listener.accept()
                print(f'Connected! ({address}:{port})')
                Thread(target=self.handle_conn, args=(conn,), daemon=True).start()
        finally:
            self.clean_up()

def main(args):
    Publisher(port=args.port).run()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=6000)
    main(parser.parse_args())
